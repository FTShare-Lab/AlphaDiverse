import math
from .factor_metrics import (
    _evaluate_factors_parallel,
    _batch_mean_abs_daily_corr,
    _factor_score,
)
from .utils import _metric_float
from .types import AgentValidationConfig
from ..model_eval import inner_search_panel, evaluation_panel


class ValidationModule:
    def __init__(self, cfg, validation=AgentValidationConfig()):
        (self.cfg, self.validation) = (cfg, validation)

    def validate(self, panel, specs, names, label, existing=()):
        scoped = evaluation_panel(panel, [*names, *existing], label)
        inner = inner_search_panel(scoped, self.cfg)
        (metrics, decisions) = _evaluate_factors_parallel(
            inner,
            specs,
            names,
            label,
            self.cfg.evaluation.long_short_quantile,
            self.cfg.evaluation.min_assets,
            1,
        )
        metrics.sort(key=_factor_score, reverse=True)
        candidates = [r["factor"] for r in metrics]
        (correlations, _) = _batch_mean_abs_daily_corr(
            inner,
            candidates,
            list(dict.fromkeys([*existing, *candidates])),
            self.cfg.evaluation.min_assets,
        )
        (accepted, previous) = ([], list(existing))
        for row in metrics:
            reason = ""
            for path, threshold, message in [
                (("coverage",), self.validation.min_coverage, "low_coverage"),
                (("ic", "days"), self.validation.min_ic_days, "too_few_ic_days"),
                (
                    ("factor_score",),
                    self.validation.min_factor_score,
                    "weak_factor_score",
                ),
            ]:
                value = _metric_float(row, path)
                if math.isnan(value) or value < threshold:
                    reason = message
                    break
            duplicate = next(
                (
                    p
                    for p in previous
                    if correlations.get((row["factor"], p), math.nan)
                    > self.validation.correlation_threshold
                ),
                "",
            )
            reason = reason or ("duplicate" if duplicate else "")
            decisions.append(
                dict(
                    factor=row["factor"],
                    decision="reject" if reason else "accept",
                    reason=reason or "passed",
                )
            )
            if not reason:
                accepted.append(row)
                previous.append(row["factor"])
        return dict(
            metrics=metrics,
            accepted=accepted,
            accepted_factors=[r["factor"] for r in accepted],
            decisions=decisions,
            evidence_scope="inner_only",
        )
