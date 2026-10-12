from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    family: str
    signature: str
    min_inputs: int
    max_inputs: int
    uses_window: bool = False
    min_window: int = 2
    uses_quantile: bool = False
    uses_weights: bool = False
    description: str = ""
    capabilities: tuple[str, ...] = ()

    def prompt_card(self) -> dict[str, Any]:
        input_count: int | str = (
            self.min_inputs
            if self.min_inputs == self.max_inputs
            else f"{self.min_inputs}-{self.max_inputs}"
        )
        parameters = [
            name
            for (name, enabled) in (
                ("window", self.uses_window),
                ("quantile", self.uses_quantile),
                ("weights", self.uses_weights),
            )
            if enabled
        ]
        card: dict[str, Any] = {
            "name": self.name,
            "signature": self.signature,
            "input_count": input_count,
        }
        if parameters:
            card["parameters"] = parameters
        if self.description:
            card["description"] = self.description
        return card


@dataclass(frozen=True)
class TransformDefinition:
    name: str
    uses_window: bool = False
    min_window: int = 2
    description: str = ""
    capabilities: tuple[str, ...] = ()

    def prompt_card(self) -> dict[str, Any]:
        card: dict[str, Any] = {"name": self.name}
        if self.description:
            card["description"] = self.description
        return card


def _op(
    name: str,
    family: str,
    signature: str,
    inputs: int | tuple[int, int],
    *,
    window: bool = False,
    min_window: int = 2,
    quantile: bool = False,
    weights: bool = False,
    description: str = "",
    capabilities: tuple[str, ...] = (),
) -> OperatorDefinition:
    (minimum, maximum) = (inputs, inputs) if isinstance(inputs, int) else inputs
    return OperatorDefinition(
        name=name,
        family=family,
        signature=signature,
        min_inputs=minimum,
        max_inputs=maximum,
        uses_window=window,
        min_window=min_window,
        uses_quantile=quantile,
        uses_weights=weights,
        description=description,
        capabilities=capabilities,
    )


OPERATOR_DEFINITIONS = (
    _op(
        "linear_combo",
        "algebra",
        "linear_combo(x1,...,x5; weights)",
        (1, 5),
        weights=True,
        description="Cross-sectionally standardize each input, then take the weighted sum.",
        capabilities=("combination", "relative_value"),
    ),
    _op("at_multiply", "algebra", "at_multiply(x,y)", 2, capabilities=("interaction",)),
    _op(
        "at_subtract",
        "algebra",
        "at_subtract(x,y)",
        2,
        capabilities=("spread", "divergence"),
    ),
    _op(
        "at_divide",
        "algebra",
        "at_divide(x,y)",
        2,
        capabilities=("ratio", "normalization"),
    ),
    _op("at_max", "algebra", "at_max(x,y)", 2, capabilities=("extreme",)),
    _op("at_min", "algebra", "at_min(x,y)", 2, capabilities=("extreme",)),
    _op(
        "ts_mean",
        "time_series",
        "ts_mean(x,window)",
        1,
        window=True,
        capabilities=("smoothing", "level"),
    ),
    _op(
        "ts_std",
        "time_series",
        "ts_std(x,window)",
        1,
        window=True,
        capabilities=("volatility", "dispersion"),
    ),
    _op(
        "ts_sum",
        "time_series",
        "ts_sum(x,window)",
        1,
        window=True,
        capabilities=("accumulation", "momentum"),
    ),
    _op(
        "ts_max",
        "time_series",
        "ts_max(x,window)",
        1,
        window=True,
        capabilities=("extreme", "breakout"),
    ),
    _op(
        "ts_min",
        "time_series",
        "ts_min(x,window)",
        1,
        window=True,
        capabilities=("extreme", "breakout"),
    ),
    _op(
        "ts_median",
        "time_series",
        "ts_median(x,window)",
        1,
        window=True,
        capabilities=("robust_level",),
    ),
    _op(
        "ts_mad",
        "time_series",
        "ts_mad(x,window)",
        1,
        window=True,
        description="Rolling mean absolute deviation from the rolling mean.",
        capabilities=("robust_dispersion", "volatility"),
    ),
    _op(
        "ts_quantile",
        "time_series",
        "ts_quantile(x,window,quantile)",
        1,
        window=True,
        quantile=True,
        capabilities=("distribution", "tail"),
    ),
    _op(
        "ts_rank",
        "time_series",
        "ts_rank(x,window)",
        1,
        window=True,
        capabilities=("historical_extreme",),
    ),
    _op(
        "ts_zscore",
        "time_series",
        "ts_zscore(x,window)",
        1,
        window=True,
        description="Standard rolling z-score: (current value - rolling mean) / rolling std.",
        capabilities=("historical_extreme", "normalization"),
    ),
    _op(
        "ts_skewness",
        "time_series",
        "ts_skewness(x,window)",
        1,
        window=True,
        min_window=3,
        capabilities=("asymmetry", "tail"),
    ),
    _op(
        "ts_kurtosis",
        "time_series",
        "ts_kurtosis(x,window)",
        1,
        window=True,
        min_window=4,
        capabilities=("tail", "fragility"),
    ),
    _op(
        "ts_ir",
        "time_series",
        "ts_ir(x,window)",
        1,
        window=True,
        description="Rolling mean divided by rolling standard deviation.",
        capabilities=("risk_adjusted_trend", "stability"),
    ),
    _op(
        "ts_decay_linear",
        "time_series",
        "ts_decay_linear(x,window)",
        1,
        window=True,
        description="Linearly weighted rolling mean with larger weight on recent observations.",
        capabilities=("decay", "smoothing"),
    ),
    _op(
        "ts_ema",
        "time_series",
        "ts_ema(x,window)",
        1,
        window=True,
        capabilities=("decay", "smoothing"),
    ),
    _op(
        "ts_delta",
        "time_series",
        "ts_delta(x,window)",
        1,
        window=True,
        min_window=1,
        capabilities=("change", "momentum"),
    ),
    _op(
        "ts_delay",
        "time_series",
        "ts_delay(x,window)",
        1,
        window=True,
        min_window=1,
        capabilities=("lag",),
    ),
    _op(
        "ts_argmax",
        "time_series",
        "ts_argmax(x,window)",
        1,
        window=True,
        capabilities=("event_timing", "extreme"),
    ),
    _op(
        "ts_argmin",
        "time_series",
        "ts_argmin(x,window)",
        1,
        window=True,
        capabilities=("event_timing", "extreme"),
    ),
    _op(
        "ts_correlation",
        "dependence",
        "ts_correlation(x,y,window)",
        2,
        window=True,
        description="Rolling Pearson correlation between two input series.",
        capabilities=("dependence", "confirmation", "divergence"),
    ),
    _op(
        "ts_covariance",
        "dependence",
        "ts_covariance(x,y,window)",
        2,
        window=True,
        description="Rolling covariance; scale-sensitive by design.",
        capabilities=("dependence", "co_movement"),
    ),
    _op(
        "ts_slope",
        "regression",
        "ts_slope(x,window)",
        1,
        window=True,
        min_window=3,
        capabilities=("trend", "path_shape"),
    ),
    _op(
        "ts_rsquare",
        "regression",
        "ts_rsquare(x,window)",
        1,
        window=True,
        min_window=3,
        capabilities=("trend_quality", "path_efficiency"),
    ),
    _op(
        "ts_residual",
        "regression",
        "ts_residual(x,window)",
        1,
        window=True,
        min_window=3,
        capabilities=("trend_residual", "reversal"),
    ),
    _op(
        "ts_regression_beta",
        "regression",
        "ts_regression_beta(y,x,window)",
        2,
        window=True,
        min_window=3,
        description="Rolling OLS slope from regressing the first input y on the second input x.",
        capabilities=("beta", "sensitivity", "neutralization"),
    ),
    _op(
        "ts_regression_residual",
        "regression",
        "ts_regression_residual(y,x,window)",
        2,
        window=True,
        min_window=3,
        description="Latest rolling OLS residual from regressing the first input y on the second input x.",
        capabilities=("residualization", "orthogonalization", "relative_value"),
    ),
)
OPERATOR_REGISTRY = {definition.name: definition for definition in OPERATOR_DEFINITIONS}
TRANSFORM_DEFINITIONS = (
    TransformDefinition("identity"),
    TransformDefinition("at_abs", capabilities=("magnitude",)),
    TransformDefinition("at_reverse", capabilities=("direction",)),
    TransformDefinition("at_sign", capabilities=("direction",)),
    TransformDefinition(
        "at_sqrt",
        description="Signed square root: sign(x)*sqrt(abs(x)).",
        capabilities=("compression",),
    ),
    TransformDefinition("at_square", capabilities=("convexity",)),
    TransformDefinition("at_log1p", capabilities=("compression",)),
    TransformDefinition(
        "at_signed_log1p",
        description="Signed log compression: sign(x)*log(1+abs(x)).",
        capabilities=("compression",),
    ),
    TransformDefinition("ts_delta_1", capabilities=("change",)),
    TransformDefinition("ts_return_1", capabilities=("return", "change")),
    TransformDefinition("ts_abs_delta_1", capabilities=("change", "magnitude")),
    TransformDefinition("ts_abs_return_1", capabilities=("return", "volatility")),
    TransformDefinition("ts_positive_delta_1", capabilities=("change", "asymmetry")),
    TransformDefinition("ts_negative_delta_1", capabilities=("change", "asymmetry")),
    TransformDefinition(
        "ts_positive_indicator_1", capabilities=("direction", "asymmetry")
    ),
    TransformDefinition(
        "ts_negative_indicator_1", capabilities=("direction", "asymmetry")
    ),
    TransformDefinition("ts_log_ratio_1", capabilities=("return", "change")),
    TransformDefinition(
        "ts_zscore",
        uses_window=True,
        min_window=2,
        description="Standard rolling z-score using the factor window.",
        capabilities=("historical_extreme", "normalization"),
    ),
    TransformDefinition(
        "ts_rank",
        uses_window=True,
        min_window=2,
        description="Rolling percentile rank using the factor window.",
        capabilities=("historical_extreme",),
    ),
    TransformDefinition(
        "cs_demean", capabilities=("market_relative", "neutralization")
    ),
    TransformDefinition("cs_zscore", capabilities=("cross_sectional", "normalization")),
    TransformDefinition("cs_rank", capabilities=("cross_sectional", "relative_value")),
    TransformDefinition("cs_winsorize", capabilities=("robustness",)),
    TransformDefinition(
        "cs_mean",
        description="Broadcast the same-date equal-weight cross-sectional mean to every stock.",
        capabilities=("market_aggregate", "common_component"),
    ),
    TransformDefinition(
        "cs_std",
        description="Broadcast the same-date cross-sectional standard deviation to every stock.",
        capabilities=("market_state", "cross_sectional_dispersion"),
    ),
)
TRANSFORM_REGISTRY = {
    definition.name: definition for definition in TRANSFORM_DEFINITIONS
}


def operator_prompt_catalog() -> list[dict[str, Any]]:
    return [definition.prompt_card() for definition in OPERATOR_DEFINITIONS]


def transform_prompt_catalog() -> list[dict[str, Any]]:
    return [definition.prompt_card() for definition in TRANSFORM_DEFINITIONS]


def operator_capability_summary() -> list[dict[str, Any]]:
    families: dict[str, set[str]] = {}
    for definition in OPERATOR_DEFINITIONS:
        families.setdefault(definition.family, set()).update(definition.capabilities)
    summary = [
        {"family": family, "capabilities": sorted(capabilities)}
        for (family, capabilities) in sorted(families.items())
    ]
    transform_capabilities = {
        capability
        for definition in TRANSFORM_DEFINITIONS
        for capability in definition.capabilities
    }
    summary.append(
        {"family": "input_transform", "capabilities": sorted(transform_capabilities)}
    )
    return summary
