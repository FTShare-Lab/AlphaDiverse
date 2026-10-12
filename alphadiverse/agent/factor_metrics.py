from __future__ import annotations
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np
import pandas as pd
from ..evaluation import evaluate_factor
from .utils import _metric_float

FACTOR_SCORE_FIELDS = ("rank_ic_mean", "rank_icir", "ic_mean", "icir", "annual_return")
FACTOR_SCORE_SCALES = {
    "rank_ic_mean": 0.05,
    "rank_icir": 0.35,
    "ic_mean": 0.03,
    "icir": 0.25,
    "annual_return": 0.3,
}


def _evaluate_factors_parallel(
    panel: pd.DataFrame,
    factor_specs: dict[str, dict[str, Any]],
    factor_names: list[str],
    label: str,
    quantile: float,
    min_assets: int,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def evaluate_one(
        factor: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if (
            factor not in panel
            or panel[factor].notna().sum() < 2
            or panel[factor].nunique(dropna=True) < 2
        ):
            return (
                None,
                {"factor": factor, "decision": "reject", "reason": "empty_or_constant"},
            )
        row = evaluate_factor(
            panel, factor, label, quantile=quantile, min_assets=min_assets
        )
        row["layer"] = factor_specs.get(factor, {}).get("layer", "unknown")
        _attach_factor_score(row)
        return (row, None)

    if not factor_names:
        return ([], [])
    if workers <= 1 or len(factor_names) <= 1:
        pairs = [evaluate_one(factor) for factor in factor_names]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pairs = list(executor.map(evaluate_one, factor_names))
    rows = [row for (row, _) in pairs if row is not None]
    rejected = [decision for (_, decision) in pairs if decision is not None]
    return (rows, rejected)


def _batch_mean_abs_daily_corr(
    df: pd.DataFrame, candidates: list[str], baselines: list[str], min_assets: int
) -> tuple[dict[tuple[str, str], float], list[dict[str, Any]]]:
    candidates = [factor for factor in candidates if factor in df]
    baselines = [factor for factor in baselines if factor in df]
    if not candidates or not baselines:
        return ({}, [])
    sums = np.zeros((len(candidates), len(baselines)), dtype="float64")
    counts = np.zeros((len(candidates), len(baselines)), dtype="int32")
    cols = list(dict.fromkeys([*candidates, *baselines]))
    for _, group in df[["date", *cols]].groupby("date", sort=False):
        if len(group) < min_assets:
            continue
        block = _pairwise_abs_corr_block(
            group[candidates].to_numpy(dtype="float64", copy=False),
            group[baselines].to_numpy(dtype="float64", copy=False),
            min_assets=min_assets,
        )
        valid = np.isfinite(block)
        sums[valid] += block[valid]
        counts[valid] += 1
    lookup: dict[tuple[str, str], float] = {}
    rows: list[dict[str, Any]] = []
    for i, candidate in enumerate(candidates):
        for j, baseline in enumerate(baselines):
            if candidate == baseline:
                continue
            mean_corr = float(sums[i, j] / counts[i, j]) if counts[i, j] else math.nan
            lookup[candidate, baseline] = mean_corr
            rows.append(
                {
                    "factor_a": candidate,
                    "factor_b": baseline,
                    "mean_abs_daily_corr": mean_corr,
                }
            )
    return (lookup, rows)


def _pairwise_abs_corr_block(
    left: np.ndarray, right: np.ndarray, min_assets: int
) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.full((left.shape[1], right.shape[1]), math.nan, dtype="float64")
    left_valid = np.isfinite(left)
    right_valid = np.isfinite(right)
    left0 = np.where(left_valid, left, 0.0)
    right0 = np.where(right_valid, right, 0.0)
    left_mask = left_valid.astype("float64", copy=False)
    right_mask = right_valid.astype("float64", copy=False)
    n = left_mask.T @ right_mask
    sum_left = left0.T @ right_mask
    sum_right = left_mask.T @ right0
    sum_left2 = (left0 * left0).T @ right_mask
    sum_right2 = left_mask.T @ (right0 * right0)
    sum_cross = left0.T @ right0
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sum_cross - sum_left * sum_right / n
        var_left = sum_left2 - sum_left * sum_left / n
        var_right = sum_right2 - sum_right * sum_right / n
        corr = cov / np.sqrt(var_left * var_right)
    corr[(n < min_assets) | (var_left <= 0) | (var_right <= 0)] = math.nan
    return np.abs(corr)


def _factor_score(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _finite_metric(row, ("factor_score",), default=0.0),
        abs(_finite_metric(row, ("ic", "rank_ic_mean"), default=0.0)),
        abs(_finite_metric(row, ("ic", "ic_mean"), default=0.0)),
    )


def _finite_metric(
    row: dict[str, Any], path: tuple[str, ...], default: float = 0.0
) -> float:
    value = _metric_float(row, path)
    return value if not math.isnan(value) else default


def _attach_factor_score(row: dict[str, Any]) -> None:
    raw_values = []
    normalized_values = []
    components: dict[str, float] = {}
    for field in FACTOR_SCORE_FIELDS:
        value = _factor_score_input(row, field)
        if math.isnan(value):
            continue
        raw_values.append(value)
        normalized = value / FACTOR_SCORE_SCALES[field]
        normalized_values.append(normalized)
        components[field] = normalized
    row["factor_score_raw"] = float(np.mean(raw_values)) if raw_values else -math.inf
    row["factor_score"] = (
        float(np.mean(normalized_values)) if normalized_values else -math.inf
    )
    row["factor_score_fields"] = list(FACTOR_SCORE_FIELDS)
    row["factor_score_scales"] = dict(FACTOR_SCORE_SCALES)
    row["factor_score_components"] = components


def _factor_score_input(row: dict[str, Any], field: str) -> float:
    if field in {"rank_ic_mean", "rank_icir", "ic_mean", "icir"}:
        value = _metric_float(row, ("ic", field))
    elif field == "annual_return":
        value = _metric_float(row, ("long_short", "annual_return"))
    else:
        value = math.nan
    return abs(value) if not math.isnan(value) else math.nan
