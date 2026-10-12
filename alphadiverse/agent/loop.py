from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from ..llm import ModelEndpointError
from ..plotting import plot_final_results as _plot
from ..experiment import load_panel, RunRequest
from ..factors.profiles import LEVEL1 as LEVEL1, resolve_base_features
from ..model_eval import (
    ModelValidationConfig,
    inner_search_panel,
    run_model_validation,
    run_final_heldout_evaluation,
)
from .specification import SpecificationModule
from .portfolio_policy import ResearchPortfolioPolicy
from .implementation import ImplementationModule
from .validation import ValidationModule
from .analysis import AnalysisModule
from .history import _research_plan_history, _model_retained_factor_cards
from .utils import _json_dumps, _compact_model_result


def _write(path, payload):
    path.write_text(_json_dumps(payload), encoding="utf-8")


def run_agent_loop(
    cfg,
    data,
    output,
    rounds=20,
    membership_data=None,
    model_cfg=ModelValidationConfig(),
):
    if rounds < 1:
        raise ValueError("rounds must be positive")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory for each run")
    output.mkdir(parents=True)
    stage = "input_preparation"
    try:
        (panel, layers, label) = load_panel(data, cfg, membership_data)
        base = resolve_base_features(cfg.base_features)
        missing = set(base) - set(panel)
        if missing:
            raise ValueError("Missing base factors: " + str(sorted(missing)))
        inner = inner_search_panel(panel, cfg)
        request = RunRequest(
            cfg.search_validation.inner_start, cfg.search_validation.outer_end
        )
        spec = SpecificationModule().build(
            cfg, request, inner, layers, label, base_model_features=base
        )
        del inner
        (output / "rounds").mkdir()
        _write(
            output / "run.json",
            {
                "status": "running",
                "rounds": rounds,
                "market": cfg.evaluation.active_markets[0],
                "label": label,
                "base_factors": base,
                "validation": asdict(cfg.search_validation),
                "feature_mask_pct": cfg.agent.available_feature_mask_pct,
            },
        )
        (policy, implementation, validator, analysis) = (
            ResearchPortfolioPolicy(),
            ImplementationModule(spec),
            ValidationModule(cfg),
            AnalysisModule(),
        )
        (accepted, retained, history, retained_cards) = ([], [], [], [])
        (all_specs, all_metrics) = ({}, {})
        with TemporaryDirectory(prefix="alphadiverse-") as workspace:
            scratch = Path(workspace)
            stage = "baseline_validation"
            baseline = run_model_validation(
                panel,
                label,
                base,
                cfg,
                scratch / "baseline",
                "baseline",
                model_cfg=model_cfg,
            )
            if baseline.get("status") != "ok":
                raise RuntimeError(
                    "Baseline validation failed: " + baseline.get("reason", "unknown")
                )
            sota = baseline["sota"]
            for i in range(1, rounds + 1):
                stage = f"round_{i}"
                portfolio = policy.propose(
                    spec, history, i, rounds, sota, retained_cards, set(all_specs)
                )
                raw = [
                    f for p in portfolio["research_plans"] for f in p["factor_specs"]
                ]
                (panel, implemented) = implementation.implement(panel, raw)
                all_specs.update({f["name"]: f for f in implemented})
                names = [f["name"] for f in implemented]
                validation = validator.validate(
                    panel, all_specs, names, label, accepted
                )
                all_metrics.update({r["factor"]: r for r in validation["metrics"]})
                new = [f for f in validation["accepted_factors"] if f not in retained]
                accepted = list(
                    dict.fromkeys([*accepted, *validation["accepted_factors"]])
                )
                candidate = {
                    "status": "skipped",
                    "reason": "no_new_accepted_factors",
                    "sota_improved": False,
                    "new_factors": [],
                }
                if new:
                    with TemporaryDirectory(dir=scratch) as temp:
                        candidate = run_model_validation(
                            panel,
                            label,
                            [*base, *retained, *new],
                            cfg,
                            Path(temp),
                            f"round_{i}",
                            new_factors=new,
                            previous_sota=sota,
                            model_cfg=model_cfg,
                        )
                    if candidate.get("status") != "ok":
                        raise RuntimeError("Candidate model validation failed")
                    if candidate.get("sota_improved"):
                        retained = list(dict.fromkeys([*retained, *new]))
                        sota = candidate["sota"]
                cumulative = {
                    "accepted_factors": accepted,
                    "accepted": [all_metrics[f] for f in accepted],
                }
                retained_cards = _model_retained_factor_cards(
                    cumulative, all_specs, retained
                )
                feedback = analysis.analyze(
                    portfolio,
                    validation,
                    candidate,
                    history,
                    True,
                    i,
                    rounds,
                    sota,
                    retained,
                )
                record = dict(
                    round=i,
                    research_plans=_research_plan_history(
                        portfolio, validation, retained
                    ),
                    feedback=feedback["feedback"],
                    llm_feedback=feedback.get("llm_feedback", {}),
                    model_validation=_compact_model_result(candidate),
                    model_retained_factors=list(retained),
                )
                record["llm_feedback"].pop("raw_response", None)
                history.append(record)
                _write(
                    output / "rounds" / f"{i:03d}.json",
                    dict(
                        round=i,
                        plans=[
                            {k: v for (k, v) in p.items() if k != "factor_specs"}
                            for p in portfolio["research_plans"]
                        ],
                        factors=[
                            dict(formula=f["formula"], name=f["name"])
                            for f in implemented
                        ],
                        decisions=validation["decisions"],
                        model_retained=candidate.get("sota_improved", False),
                        feedback=record["llm_feedback"].get(
                            "feedback", record["feedback"]
                        ),
                    ),
                )
                print(
                    f"Round {i}/{rounds}: accepted={len(validation['accepted_factors'])}, retained_total={len(retained)}",
                    flush=True,
                )
            frozen_factors = [
                dict(
                    **all_specs[name],
                    accepted=name in accepted,
                    model_retained=name in retained,
                )
                for name in all_specs
            ]
            _write(output / "factors.json", frozen_factors)
            stage = "outer_evaluation"
            final_dir = scratch / "final"
            outer = run_final_heldout_evaluation(
                panel,
                label,
                base,
                [*base, *retained],
                cfg,
                final_dir,
                sota,
                baseline["sota"],
                model_cfg,
            )
            if outer.get("status") != "ok":
                raise RuntimeError("Final held-out evaluation failed")
            for filename, target in [
                ("outer_model_summary.csv", "metrics.csv"),
                ("outer_daily_ic.csv", "daily_ic.csv"),
                ("outer_daily_long_short.csv", "returns.csv"),
                ("final_groups.csv", "final_groups.csv"),
                ("final_returns.csv", "final_returns.csv"),
            ]:
                source = final_dir / filename
                if source.exists():
                    frame = pd.read_csv(source)
                    if target == "metrics.csv":
                        columns = [
                            "model",
                            "market",
                            "ic_mean",
                            "icir",
                            "rank_ic_mean",
                            "rank_icir",
                            "long_short_annual",
                            "sharpe",
                            "max_drawdown",
                            "sota_score",
                        ]
                        frame = frame[[c for c in columns if c in frame]]
                    frame.to_csv(output / target, index=False)
            stage = "plotting"
            _plot(output)
            summary = dict(
                status="complete",
                rounds=rounds,
                label=label,
                generated_factors=len(all_specs),
                accepted_factors=len(accepted),
                model_retained_factors=retained,
                inner_metrics=sota.get("search_aggregate", sota.get("best_test", {})),
                outer_event_count=1,
            )
            _write(output / "summary.json", summary)
            _write(
                output / "run.json",
                {
                    "status": "complete",
                    "rounds": rounds,
                    "market": cfg.evaluation.active_markets[0],
                    "label": label,
                    "base_factors": base,
                    "validation": asdict(cfg.search_validation),
                    "feature_mask_pct": cfg.agent.available_feature_mask_pct,
                },
            )
        return summary
    except Exception as exc:
        failure = {"status": "failed", "error_type": type(exc).__name__, "stage": stage}
        if stage == "input_preparation" and isinstance(exc, ValueError):
            failure["message"] = str(exc)
        if isinstance(exc, ModelEndpointError):
            failure["model_call"] = exc.diagnostic
        _write(output / "status.json", failure)
        run_path = output / "run.json"
        if run_path.exists():
            import json

            run = json.loads(run_path.read_text(encoding="utf-8"))
            run["status"] = "failed"
            _write(run_path, run)
        raise
