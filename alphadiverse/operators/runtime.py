from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .cross_sectional import (
    cs_demean,
    cs_mean,
    cs_rank,
    cs_std,
    cs_winsorize,
    cs_zscore,
    numeric,
)
from .regression import ts_pair_regression, ts_time_regression
from .time_series import (
    rolling_pair,
    rolling_unary,
    ts_arg_position,
    ts_decay_linear,
    ts_delay,
    ts_delta,
    ts_ema,
    ts_ir,
    ts_mad,
    ts_zscore,
)


def apply_transform(
    panel: pd.DataFrame, series: pd.Series, transform: str, *, window: int | None
) -> pd.Series:
    values = numeric(series)
    if transform == "identity":
        return values
    if transform == "at_abs":
        return values.abs()
    if transform == "at_reverse":
        return -values
    if transform == "at_sign":
        return np.sign(values)
    if transform == "at_sqrt":
        return np.sign(values) * np.sqrt(values.abs())
    if transform == "at_square":
        return values.pow(2)
    if transform == "at_log1p":
        return np.log1p(values.where(values > -1.0))
    if transform == "at_signed_log1p":
        return np.sign(values) * np.log1p(values.abs())
    if transform == "ts_delta_1":
        return ts_delta(panel, values, 1)
    if transform == "ts_return_1":
        previous = ts_delay(panel, values, 1).replace(0, np.nan)
        return values / previous - 1.0
    if transform == "ts_abs_delta_1":
        return ts_delta(panel, values, 1).abs()
    if transform == "ts_abs_return_1":
        previous = ts_delay(panel, values, 1).replace(0, np.nan)
        return (values / previous - 1.0).abs()
    if transform == "ts_positive_delta_1":
        return ts_delta(panel, values, 1).clip(lower=0)
    if transform == "ts_negative_delta_1":
        return (-ts_delta(panel, values, 1)).clip(lower=0)
    if transform == "ts_positive_indicator_1":
        delta = ts_delta(panel, values, 1)
        return delta.gt(0).astype(float).where(delta.notna())
    if transform == "ts_negative_indicator_1":
        delta = ts_delta(panel, values, 1)
        return delta.lt(0).astype(float).where(delta.notna())
    if transform == "ts_log_ratio_1":
        previous = ts_delay(panel, values, 1).replace(0, np.nan)
        ratio = values / previous
        return np.sign(ratio) * np.log(ratio.abs().replace(0, np.nan))
    if transform == "ts_zscore":
        if window is None:
            raise ValueError("ts_zscore input transform requires window")
        return ts_zscore(panel, values, window)
    if transform == "ts_rank":
        if window is None:
            raise ValueError("ts_rank input transform requires window")
        return rolling_unary(panel, values, window, "rank")
    if transform == "cs_demean":
        return cs_demean(panel, values)
    if transform == "cs_zscore":
        return cs_zscore(panel, values)
    if transform == "cs_rank":
        return cs_rank(panel, values)
    if transform == "cs_winsorize":
        return cs_winsorize(panel, values)
    if transform == "cs_mean":
        return cs_mean(panel, values)
    if transform == "cs_std":
        return cs_std(panel, values)
    raise ValueError(f"Unsupported input transform: {transform}")


def apply_operator(
    panel: pd.DataFrame,
    operator: str,
    components: list[pd.Series],
    *,
    window: int | None,
    quantile: float | None,
    weights: list[float],
) -> pd.Series:
    if operator == "linear_combo":
        standardized = [cs_zscore(panel, component) for component in components]
        return sum(
            (
                weight * component
                for (weight, component) in zip(weights, standardized, strict=True)
            )
        )
    if operator == "at_multiply":
        return components[0] * components[1]
    if operator == "at_subtract":
        return components[0] - components[1]
    if operator == "at_divide":
        return components[0] / components[1].replace(0, np.nan)
    if operator == "at_max":
        return pd.concat(components, axis=1).max(axis=1, skipna=False)
    if operator == "at_min":
        return pd.concat(components, axis=1).min(axis=1, skipna=False)
    if window is None:
        raise ValueError(f"Operator {operator} requires window")
    unary_operations = {
        "ts_mean": "mean",
        "ts_std": "std",
        "ts_sum": "sum",
        "ts_max": "max",
        "ts_min": "min",
        "ts_median": "median",
        "ts_quantile": "quantile",
        "ts_rank": "rank",
        "ts_skewness": "skew",
        "ts_kurtosis": "kurt",
    }
    if operator in unary_operations:
        return rolling_unary(
            panel,
            components[0],
            window,
            unary_operations[operator],
            quantile=0.5 if quantile is None else quantile,
        )
    if operator == "ts_zscore":
        return ts_zscore(panel, components[0], window)
    if operator == "ts_mad":
        return ts_mad(panel, components[0], window)
    if operator == "ts_ir":
        return ts_ir(panel, components[0], window)
    if operator == "ts_decay_linear":
        return ts_decay_linear(panel, components[0], window)
    if operator == "ts_ema":
        return ts_ema(panel, components[0], window)
    if operator == "ts_delta":
        return ts_delta(panel, components[0], window)
    if operator == "ts_delay":
        return ts_delay(panel, components[0], window)
    if operator == "ts_argmax":
        return ts_arg_position(panel, components[0], window, "max")
    if operator == "ts_argmin":
        return ts_arg_position(panel, components[0], window, "min")
    if operator == "ts_correlation":
        return rolling_pair(panel, components[0], components[1], window, "corr")
    if operator == "ts_covariance":
        return rolling_pair(panel, components[0], components[1], window, "cov")
    if operator in {"ts_slope", "ts_rsquare", "ts_residual"}:
        output = {
            "ts_slope": "slope",
            "ts_rsquare": "rsquare",
            "ts_residual": "residual",
        }[operator]
        return ts_time_regression(panel, components[0], window, output)
    if operator == "ts_regression_beta":
        return ts_pair_regression(panel, components[0], components[1], window, "beta")
    if operator == "ts_regression_residual":
        return ts_pair_regression(
            panel, components[0], components[1], window, "residual"
        )
    raise ValueError(f"Unsupported operator: {operator}")


def execute_factor_spec(panel: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    components = [
        apply_transform(panel, panel[feature], transform, window=spec.get("window"))
        for (feature, transform) in zip(
            spec["input_features"], spec["input_transforms"], strict=True
        )
    ]
    raw = apply_operator(
        panel,
        spec["operator"],
        components,
        window=spec.get("window"),
        quantile=spec.get("quantile"),
        weights=spec.get("weights", []),
    )
    finite_raw = raw.replace([np.inf, -np.inf], np.nan)
    return cs_zscore(panel, finite_raw).rename(spec["name"])
