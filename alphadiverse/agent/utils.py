from __future__ import annotations
from typing import Any
import json
import math
import numpy as np
from ..llm import resolve_chat_backend


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for (k, v) in value.items()}
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_sanitize(value), indent=2, ensure_ascii=False, default=str)


def _json_compact(value: Any) -> str:
    return json.dumps(
        _json_sanitize(value), ensure_ascii=False, separators=(",", ":"), default=str
    )


def _message_token_count(messages: list[dict[str, str]]) -> int:
    return _text_token_count(_json_compact(messages))


def _json_token_count(value: Any) -> int:
    return _text_token_count(_json_compact(value))


def _pack_json_items(
    items: list[Any], token_budget: int, *, newest_first: bool
) -> tuple[list[Any], dict[str, int]]:
    if token_budget <= 0 or not items:
        return (
            [],
            {
                "input_count": len(items),
                "included_count": 0,
                "omitted_count": len(items),
                "estimated_tokens": 0,
            },
        )
    indexed = list(enumerate(items))
    if newest_first:
        indexed = list(reversed(indexed))
    selected: list[tuple[int, Any]] = []
    used = 2
    for idx, item in indexed:
        item_tokens = _json_token_count(item) + 1
        if used + item_tokens > token_budget:
            continue
        selected.append((idx, item))
        used += item_tokens
    selected.sort(key=lambda pair: pair[0])
    packed = [item for (_, item) in selected]
    return (
        packed,
        {
            "input_count": len(items),
            "included_count": len(packed),
            "omitted_count": len(items) - len(packed),
            "estimated_tokens": used if packed else 0,
        },
    )


def _shrink_to_message_budget(
    prompt: dict[str, Any],
    messages_factory: Any,
    budget: int,
    shrink_paths: list[tuple[str, ...]],
) -> dict[str, Any]:
    while _message_token_count(messages_factory(prompt)) > budget:
        changed = False
        for path in shrink_paths:
            target: Any = prompt
            for key in path[:-1]:
                target = target.get(key, {}) if isinstance(target, dict) else {}
            key = path[-1]
            values = target.get(key, []) if isinstance(target, dict) else []
            if isinstance(values, list) and values:
                values.pop(0)
                changed = True
                break
        if not changed:
            break
    return prompt


def _compact_history(
    history: list[dict[str, Any]],
    limit: int,
    *,
    model_sota: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    current_best = _llm_model_sota(model_sota or {})
    compact = []
    for item in history[-limit:]:
        analysis_feedback = _compact_analysis_feedback(item.get("llm_feedback", {}))
        if not analysis_feedback and item.get("feedback"):
            analysis_feedback = {"feedback": item.get("feedback")}
        model_result = _llm_model_candidate(item.get("model_validation", {}))
        metric_key = (
            "search_metrics" if "search_metrics" in current_best else "test_metrics"
        )
        model_result = _dedupe_candidate_metrics(
            model_result,
            current_best,
            reference=f"model_context.current_best.{metric_key}",
        )
        compact.append(
            _drop_none_values(
                {
                    "round": item.get("round"),
                    "research_plans": _compact_research_plans(
                        item.get("research_plans", [])
                    ),
                    "model_result": model_result,
                    "analysis_feedback": analysis_feedback,
                }
            )
        )
    return compact


def _compact_research_plans(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    plans = []
    for plan in value:
        if not isinstance(plan, dict) or not plan.get("plan_id"):
            continue
        outcomes = []
        for row in plan.get("factor_outcomes", []):
            if not isinstance(row, dict) or not row.get("factor"):
                continue
            outcomes.append(
                _drop_none_values(
                    {
                        "factor": row.get("factor"),
                        "operator": row.get("operator"),
                        "input_features": row.get("input_features", []),
                        "formula": row.get("formula"),
                        "rationale": row.get("rationale"),
                        "factor_decision": row.get("factor_decision"),
                        "factor_reason": row.get("factor_reason"),
                        "duplicate_of": row.get("duplicate_of"),
                        "rank_ic_mean": _round_for_llm(row.get("rank_ic_mean")),
                        "factor_score": _round_for_llm(row.get("factor_score")),
                        "coverage": _round_for_llm(row.get("coverage")),
                        "model_retained": row.get("model_retained"),
                    }
                )
            )
        plans.append(
            _drop_none_values(
                {
                    "plan_id": plan.get("plan_id"),
                    "mechanism_id": plan.get("mechanism_id"),
                    "theme": plan.get("theme"),
                    "event": plan.get("event"),
                    "economic_mechanism": plan.get("economic_mechanism"),
                    "horizon": plan.get("horizon"),
                    "conditioning": plan.get("conditioning"),
                    "expected_direction": plan.get("expected_direction"),
                    "relation_to_base": plan.get("relation_to_base"),
                    "factor_outcomes": outcomes,
                }
            )
        )
    return plans


def _compact_analysis_feedback(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return _drop_none_values(
        {
            "portfolio_supported": value.get("portfolio_supported"),
            "sota_decision": value.get("sota_decision"),
            "feedback": value.get("feedback"),
            "next_steps": value.get("next_steps", []),
            "error": value.get("error"),
        }
    )


def _compact_portfolio_for_analysis(portfolio: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_plans": [
            {key: value for (key, value) in plan.items() if key != "factor_outcomes"}
            for plan in portfolio.get("research_plans", [])
            if isinstance(plan, dict)
        ]
    }


def _compact_factor_validation(validation: dict[str, Any]) -> dict[str, Any]:
    decisions = {
        row.get("factor"): row
        for row in validation.get("decisions", [])
        if isinstance(row, dict) and row.get("factor")
    }
    factor_results = []
    seen: set[str] = set()
    for metric in validation.get("metrics", []):
        if not isinstance(metric, dict) or not metric.get("factor"):
            continue
        factor = str(metric["factor"])
        decision = decisions.get(factor, {})
        factor_results.append(
            _drop_none_values(
                {
                    "factor": factor,
                    "layer": metric.get("layer"),
                    "decision": decision.get("decision"),
                    "reason": decision.get("reason"),
                    "rank_ic_mean": _round_for_llm(
                        metric.get("ic", {}).get("rank_ic_mean")
                    ),
                    "ic_mean": _round_for_llm(metric.get("ic", {}).get("ic_mean")),
                    "coverage": _round_for_llm(metric.get("coverage")),
                    "sharpe": _round_for_llm(
                        metric.get("long_short", {}).get("sharpe")
                    ),
                }
            )
        )
        seen.add(factor)
    for factor, decision in decisions.items():
        if factor in seen:
            continue
        factor_results.append(
            _drop_none_values(
                {
                    "factor": factor,
                    "decision": decision.get("decision"),
                    "reason": decision.get("reason"),
                }
            )
        )
    return {
        "accepted_count": sum(
            (row.get("decision") == "accept" for row in factor_results)
        ),
        "factor_results": factor_results,
    }


def _llm_model_context(
    model_sota: dict[str, Any] | None,
    model_retained_factors: list[str] | list[dict[str, Any]],
    *,
    round_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_best = _llm_model_sota(model_sota or {})
    context: dict[str, Any] = {
        "current_best": current_best,
        "model_retained_factors": model_retained_factors,
    }
    if round_candidate is not None:
        candidate = _llm_model_candidate(round_candidate)
        metric_key = (
            "search_metrics" if "search_metrics" in current_best else "test_metrics"
        )
        candidate = _dedupe_candidate_metrics(
            candidate, current_best, reference=f"current_best.{metric_key}"
        )
        context["round_candidate"] = candidate
    return context


def _dedupe_candidate_metrics(
    candidate: dict[str, Any], current_best: dict[str, Any], *, reference: str
) -> dict[str, Any]:
    if (
        candidate.get("sota_improved")
        and candidate.get("run_name")
        and (candidate.get("run_name") == current_best.get("run_name"))
    ):
        candidate = dict(candidate)
        candidate.pop("test_metrics", None)
        candidate.pop("search_metrics", None)
        candidate[f"{reference.rsplit('.', 1)[-1]}_ref"] = reference
    return candidate


def _llm_model_sota(model_sota: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(model_sota, dict) or not model_sota:
        return {}
    best_test = model_sota.get("search_aggregate") or model_sota.get("best_test", {})
    out = {
        "run_name": model_sota.get("run_name"),
        "search_protocol": model_sota.get("protocol"),
        "model": best_test.get("model") if isinstance(best_test, dict) else None,
    }
    if model_sota.get("protocol") == "inner_multifold" or model_sota.get(
        "search_aggregate"
    ):
        out["search_metrics"] = _llm_model_metrics(
            best_test, split="inner_fold_aggregate"
        )
        out["fold_metrics"] = _llm_fold_metrics(model_sota.get("fold_metrics", []))
    else:
        out["test_metrics"] = _llm_model_metrics(best_test, split="test")
    return _drop_none_values(out)


def _llm_model_candidate(model_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(model_payload, dict) or not model_payload:
        return {}
    out = {
        "run_name": model_payload.get("run_name"),
        "status": model_payload.get("status"),
        "reason": model_payload.get("reason"),
        "new_factors": model_payload.get("new_factors", []),
        "sota_improved": model_payload.get("sota_improved"),
    }
    if model_payload.get("protocol") == "inner_multifold" or model_payload.get(
        "search_aggregate"
    ):
        aggregate = model_payload.get("search_aggregate") or model_payload.get(
            "best_test", {}
        )
        out["search_protocol"] = "inner_multifold"
        out["search_metrics"] = _llm_model_metrics(
            aggregate, split="inner_fold_aggregate"
        )
        out["fold_metrics"] = _llm_fold_metrics(model_payload.get("fold_metrics", []))
        out["retain_decision"] = _llm_retain_decision(
            model_payload.get("retain_decision", {})
        )
    else:
        out["test_metrics"] = _llm_model_metrics(
            model_payload.get("best_test", {}), split="test"
        )
    return _drop_none_values(out)


def _llm_model_metrics(row: Any, *, split: str) -> dict[str, Any]:
    if not isinstance(row, dict) or not row:
        return {}
    metric_names = (
        "sota_score",
        "rank_ic_mean",
        "rank_icir",
        "ic_mean",
        "icir",
        "long_short_annual",
        "sharpe",
        "max_drawdown",
        "turnover_mean",
        "win_rate",
    )
    out = {
        "model": row.get("model"),
        "split": split,
        "market": row.get("market"),
        "markets": row.get("markets", []),
    }
    out.update({name: _round_for_llm(row.get(name)) for name in metric_names})
    market_metrics = []
    for market_row in row.get("market_metrics", []):
        if not isinstance(market_row, dict):
            continue
        compact_market = {"market": market_row.get("market")}
        compact_market.update(
            {
                name: _round_for_llm(market_row.get(name))
                for name in metric_names
                if name != "sota_score"
            }
        )
        market_metrics.append(_drop_none_values(compact_market))
    if market_metrics:
        out["market_metrics"] = market_metrics
    return _drop_none_values(out)


def _llm_fold_metrics(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out = []
    for row in value:
        if not isinstance(row, dict):
            continue
        test = row.get("test", {}) if isinstance(row.get("test"), dict) else {}
        out.append(
            _drop_none_values(
                {
                    "fold_id": row.get("fold_id"),
                    "weight": _round_for_llm(row.get("weight")),
                    "sota_score": _round_for_llm(test.get("sota_score")),
                    "rank_ic_mean": _round_for_llm(test.get("rank_ic_mean")),
                    "rank_icir": _round_for_llm(test.get("rank_icir")),
                    "ic_mean": _round_for_llm(test.get("ic_mean")),
                    "icir": _round_for_llm(test.get("icir")),
                    "long_short_annual": _round_for_llm(test.get("long_short_annual")),
                }
            )
        )
    return out


def _llm_retain_decision(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return _drop_none_values(
        {
            "retain": value.get("retain"),
            "aggregate_delta": _round_for_llm(value.get("aggregate_delta")),
            "aggregate_improved": value.get("aggregate_improved"),
            "improved_fold_count": value.get("improved_fold_count"),
            "required_improved_folds": value.get("required_improved_folds"),
            "fold_majority_passed": value.get("fold_majority_passed"),
        }
    )


def _round_for_llm(value: Any, digits: int = 6) -> Any:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return round(float(value), digits)
    return value


def _drop_none_values(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item for (key, item) in value.items() if item is not None and item != {}
    }


def _metric_float(row: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = row
    for key in path:
        value = value.get(key, {}) if isinstance(value, dict) else math.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if not math.isnan(value) else math.nan


def _compact_model_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    out = {
        "model": row.get("model"),
        "split": row.get("split"),
        "market": row.get("market"),
        "markets": row.get("markets", []),
        "missing_markets": row.get("missing_markets", []),
        "market_count": row.get("market_count"),
        "expected_market_count": row.get("expected_market_count"),
        "sota_score": row.get("sota_score"),
        "sota_score_raw": row.get("sota_score_raw"),
        "sota_score_fields": row.get("sota_score_fields", []),
        "sota_score_scales": row.get("sota_score_scales", {}),
        "sota_score_components": row.get("sota_score_components", {}),
        "label": row.get("label"),
        "rank_ic_mean": row.get("rank_ic_mean"),
        "rank_icir": row.get("rank_icir"),
        "ic_mean": row.get("ic_mean"),
        "icir": row.get("icir"),
        "long_short_mean": row.get("long_short_mean"),
        "long_short_annual": row.get("long_short_annual"),
        "sharpe": row.get("sharpe"),
        "max_drawdown": row.get("max_drawdown"),
        "turnover_mean": row.get("turnover_mean"),
        "win_rate": row.get("win_rate"),
        "group_monotonic_spearman": row.get("group_monotonic_spearman"),
        "group_daily_monotonic_mean": row.get("group_daily_monotonic_mean"),
        "group_daily_monotonic_positive_ratio": row.get(
            "group_daily_monotonic_positive_ratio"
        ),
        "group_top_minus_bottom": row.get("group_top_minus_bottom"),
        "group_top_minus_bottom_annual": row.get("group_top_minus_bottom_annual"),
        "group_top_minus_bottom_icir": row.get("group_top_minus_bottom_icir"),
        "group_middle_adjacent_mean": row.get("group_middle_adjacent_mean"),
    }
    market_metrics = row.get("market_metrics", [])
    if isinstance(market_metrics, list) and market_metrics:
        out["market_metrics"] = [
            {
                "market": item.get("market"),
                "rank_ic_mean": item.get("rank_ic_mean"),
                "rank_icir": item.get("rank_icir"),
                "ic_mean": item.get("ic_mean"),
                "icir": item.get("icir"),
                "long_short_annual": item.get("long_short_annual"),
                "sharpe": item.get("sharpe"),
                "max_drawdown": item.get("max_drawdown"),
                "turnover_mean": item.get("turnover_mean"),
                "group_top_minus_bottom_icir": item.get("group_top_minus_bottom_icir"),
            }
            for item in market_metrics
            if isinstance(item, dict)
        ]
    return out


def _compact_model_result(model_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not model_payload:
        return {}
    best_valid = (
        model_payload.get("best_valid", {}) if isinstance(model_payload, dict) else {}
    )
    best_test = (
        model_payload.get("best_test", {}) if isinstance(model_payload, dict) else {}
    )
    return {
        "run_name": model_payload.get("run_name"),
        "protocol": model_payload.get("protocol"),
        "status": model_payload.get("status"),
        "reason": model_payload.get("reason", ""),
        "feature_count": model_payload.get("feature_count"),
        "new_factors": model_payload.get("new_factors", []),
        "sota_improved": model_payload.get("sota_improved"),
        "best_valid": _compact_model_metric_row(best_valid),
        "best_test": _compact_model_metric_row(best_test),
        "search_aggregate": _compact_model_metric_row(
            model_payload.get("search_aggregate", {})
        ),
        "fold_metrics": model_payload.get("fold_metrics", []),
        "retain_decision": model_payload.get("retain_decision", {}),
        "best_test_artifacts": model_payload.get("best_test_artifacts", []),
    }


def _model_feedback_sentence(model_payload: dict[str, Any]) -> str:
    compact = _compact_model_result(model_payload)
    if not compact:
        return "not run."
    if compact.get("status") != "ok":
        return f"{compact.get('status')} ({compact.get('reason', '')})."
    is_multifold = compact.get("protocol") == "inner_multifold" or bool(
        compact.get("search_aggregate")
    )
    best = (
        compact.get("search_aggregate", {})
        if is_multifold
        else compact.get("best_test", {})
    )
    label = "inner_fold_aggregate" if is_multifold else "best_test"
    decision = compact.get("retain_decision", {}) if is_multifold else {}
    return f"{label}={best.get('model')} market={best.get('market')} markets={best.get('markets', [])} sota_score={best.get('sota_score')} rank_ic={best.get('rank_ic_mean')} rank_icir={best.get('rank_icir')} ic={best.get('ic_mean')} icir={best.get('icir')} long_short_annual={best.get('long_short_annual')} sharpe={best.get('sharpe')} group_ls_icir={best.get('group_top_minus_bottom_icir')} max_drawdown={best.get('max_drawdown')} improved={compact.get('sota_improved')} improved_folds={decision.get('improved_fold_count')}/{decision.get('required_improved_folds')}."


def _text_token_count(text):
    return len(text.encode("utf-8"))


def _llm_prompt_token_budget(call_site="llm"):
    cfg = resolve_chat_backend(call_site)
    reserve = cfg.max_tokens + 1024
    return max(1, cfg.context_window_tokens - reserve)
