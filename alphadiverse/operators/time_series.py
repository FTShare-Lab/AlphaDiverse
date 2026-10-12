from __future__ import annotations
from collections.abc import Callable
import numpy as np
import pandas as pd
from .cross_sectional import numeric


def ordered_frame(panel: pd.DataFrame, **series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"symbol": panel["symbol"], "date": panel["date"]}, index=panel.index
    )
    for name, values in series.items():
        frame[name] = numeric(values)
    return frame.sort_values(["symbol", "date"])


def minimum_periods(window: int, minimum: int = 2) -> int:
    return max(int(minimum), int(window) // 2)


def rolling_unary(
    panel: pd.DataFrame,
    series: pd.Series,
    window: int,
    operation: str,
    *,
    quantile: float = 0.5,
) -> pd.Series:
    work = ordered_frame(panel, value=series)
    min_periods = minimum_periods(
        window, 4 if operation == "kurt" else 3 if operation == "skew" else 2
    )
    if operation in {"skew", "kurt"}:
        return _rolling_shape(work, window, min_periods, operation).reindex(panel.index)
    grouped = work.groupby("symbol", sort=False)["value"]
    roller: Callable[[pd.Series], pd.Series]
    if operation == "mean":
        roller = lambda values: values.rolling(window, min_periods=min_periods).mean()
    elif operation == "std":
        roller = lambda values: values.rolling(window, min_periods=min_periods).std()
    elif operation == "sum":
        roller = lambda values: values.rolling(window, min_periods=min_periods).sum()
    elif operation == "max":
        roller = lambda values: values.rolling(window, min_periods=min_periods).max()
    elif operation == "min":
        roller = lambda values: values.rolling(window, min_periods=min_periods).min()
    elif operation == "median":
        roller = lambda values: values.rolling(window, min_periods=min_periods).median()
    elif operation == "quantile":
        roller = lambda values: values.rolling(
            window, min_periods=min_periods
        ).quantile(float(quantile))
    elif operation == "rank":
        roller = lambda values: values.rolling(window, min_periods=min_periods).rank(
            pct=True
        )
    else:
        raise ValueError(f"Unsupported rolling operation: {operation}")
    return grouped.transform(roller).reindex(panel.index)


def _rolling_shape(
    work: pd.DataFrame, window: int, min_periods: int, operation: str
) -> pd.Series:
    result = pd.Series(np.nan, index=work.index, dtype=float)
    for _, group in work.groupby("symbol", sort=False):
        values = group["value"].to_numpy(dtype=float)
        padded = np.pad(values, (window - 1, 0), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, window)
        valid = np.isfinite(windows)
        count = valid.sum(axis=1)
        safe_count = np.where(count > 0, count, 1)
        mean = np.where(valid, windows, 0.0).sum(axis=1) / safe_count
        centered = np.where(valid, windows - mean[:, None], 0.0)
        m2 = np.square(centered).sum(axis=1) / safe_count
        with np.errstate(divide="ignore", invalid="ignore"):
            if operation == "skew":
                m3 = np.power(centered, 3).sum(axis=1) / safe_count
                values_out = (
                    np.sqrt(count * (count - 1)) / (count - 2) * m3 / np.power(m2, 1.5)
                )
            elif operation == "kurt":
                m4 = np.power(centered, 4).sum(axis=1) / safe_count
                g2 = m4 / np.square(m2) - 3.0
                values_out = (
                    (count - 1) / ((count - 2) * (count - 3)) * ((count + 1) * g2 + 6.0)
                )
            else:
                raise ValueError(f"Unsupported rolling shape operation: {operation}")
        values_out[(count < min_periods) | ~np.isfinite(values_out) | (m2 <= 0)] = (
            np.nan
        )
        result.loc[group.index] = values_out
    return result


def ts_zscore(panel: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    mean = rolling_unary(panel, series, window, "mean")
    std = rolling_unary(panel, series, window, "std").replace(0, np.nan)
    return (numeric(series) - mean) / std


def ts_mad(panel: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    work = ordered_frame(panel, value=series)
    min_periods = minimum_periods(window)

    def mad(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if len(finite) < min_periods:
            return np.nan
        return float(np.mean(np.abs(finite - np.mean(finite))))

    out = work.groupby("symbol", sort=False)["value"].transform(
        lambda values: values.rolling(window, min_periods=min_periods).apply(
            mad, raw=True
        )
    )
    return out.reindex(panel.index)


def ts_ir(panel: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    mean = rolling_unary(panel, series, window, "mean")
    std = rolling_unary(panel, series, window, "std").replace(0, np.nan)
    return mean / std


def ts_decay_linear(panel: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    work = ordered_frame(panel, value=series)
    min_periods = minimum_periods(window)
    weights = np.arange(1, window + 1, dtype=float)

    def weighted(values: np.ndarray) -> float:
        valid = np.isfinite(values)
        if valid.sum() < min_periods:
            return np.nan
        used_weights = weights[-len(values) :][valid]
        return float(np.dot(values[valid], used_weights) / used_weights.sum())

    out = work.groupby("symbol", sort=False)["value"].transform(
        lambda values: values.rolling(window, min_periods=min_periods).apply(
            weighted, raw=True
        )
    )
    return out.reindex(panel.index)


def ts_ema(panel: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    work = ordered_frame(panel, value=series)
    out = work.groupby("symbol", sort=False)["value"].transform(
        lambda values: values.ewm(
            span=window, adjust=False, min_periods=minimum_periods(window)
        ).mean()
    )
    return out.reindex(panel.index)


def ts_delta(panel: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    work = ordered_frame(panel, value=series)
    return work.groupby("symbol", sort=False)["value"].diff(window).reindex(panel.index)


def ts_delay(panel: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    work = ordered_frame(panel, value=series)
    return (
        work.groupby("symbol", sort=False)["value"].shift(window).reindex(panel.index)
    )


def ts_arg_position(
    panel: pd.DataFrame, series: pd.Series, window: int, mode: str
) -> pd.Series:
    work = ordered_frame(panel, value=series)
    min_periods = minimum_periods(window)

    def position(values: np.ndarray) -> float:
        valid = np.isfinite(values)
        if valid.sum() < min_periods:
            return np.nan
        comparable = np.where(valid, values, -np.inf if mode == "max" else np.inf)
        offset = int(np.argmax(comparable) if mode == "max" else np.argmin(comparable))
        return float(offset + 1) / len(values)

    out = work.groupby("symbol", sort=False)["value"].transform(
        lambda values: values.rolling(window, min_periods=min_periods).apply(
            position, raw=True
        )
    )
    return out.reindex(panel.index)


def rolling_pair(
    panel: pd.DataFrame, left: pd.Series, right: pd.Series, window: int, operation: str
) -> pd.Series:
    work = ordered_frame(panel, left=left, right=right)
    min_periods = minimum_periods(window)
    chunks = []
    for _, group in work.groupby("symbol", sort=False):
        if operation == "corr":
            values = (
                group["left"]
                .rolling(window, min_periods=min_periods)
                .corr(group["right"])
            )
        elif operation == "cov":
            values = (
                group["left"]
                .rolling(window, min_periods=min_periods)
                .cov(group["right"])
            )
        else:
            raise ValueError(f"Unsupported rolling pair operation: {operation}")
        chunks.append(values)
    return (
        pd.concat(chunks).sort_index().reindex(panel.index)
        if chunks
        else pd.Series(np.nan, index=panel.index)
    )
