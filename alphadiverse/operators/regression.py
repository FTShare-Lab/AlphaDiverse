from __future__ import annotations
import numpy as np
import pandas as pd
from .cross_sectional import numeric
from .time_series import minimum_periods, ordered_frame


def ts_time_regression(
    panel: pd.DataFrame, series: pd.Series, window: int, output: str
) -> pd.Series:
    work = ordered_frame(panel, value=series)
    min_periods = minimum_periods(window, 3)
    result = pd.Series(np.nan, index=work.index, dtype=float)
    for _, group in work.groupby("symbol", sort=False):
        values = numeric(group["value"])
        positions = np.arange(len(group), dtype=float)
        frame = pd.DataFrame(
            {"y": values.to_numpy(dtype=float), "x": positions}, index=group.index
        )
        valid = frame["y"].notna().astype(float)
        y_values = frame["y"].where(valid.eq(1), 0.0)
        x_values = frame["x"].where(valid.eq(1), 0.0)
        count = valid.rolling(window, min_periods=1).sum()
        safe_count = count.replace(0, np.nan)
        sum_y = y_values.rolling(window, min_periods=1).sum()
        sum_x = x_values.rolling(window, min_periods=1).sum()
        sum_xx = x_values.pow(2).rolling(window, min_periods=1).sum()
        sum_xy = (x_values * y_values).rolling(window, min_periods=1).sum()
        denom = sum_xx - sum_x.pow(2) / safe_count
        beta = (sum_xy - sum_x * sum_y / safe_count) / denom.replace(0, np.nan)
        alpha = sum_y / safe_count - beta * sum_x / safe_count
        enough = count >= min_periods
        if output == "slope":
            values_out = beta
        elif output == "residual":
            values_out = frame["y"] - (alpha + beta * frame["x"])
        elif output == "rsquare":
            centered_yy = (
                y_values.pow(2).rolling(window, min_periods=1).sum()
                - sum_y.pow(2) / safe_count
            )
            values_out = (
                (sum_xy - sum_x * sum_y / count).pow(2)
                / (denom * centered_yy).replace(0, np.nan)
            ).clip(0, 1)
        else:
            raise ValueError(f"Unsupported time regression output: {output}")
        result.loc[group.index] = values_out.where(enough)
    return result.reindex(panel.index)


def ts_pair_regression(
    panel: pd.DataFrame, y: pd.Series, x: pd.Series, window: int, output: str
) -> pd.Series:
    work = ordered_frame(panel, y=y, x=x)
    min_periods = minimum_periods(window, 3)
    result = pd.Series(np.nan, index=work.index, dtype=float)
    for _, group in work.groupby("symbol", sort=False):
        valid = group[["x", "y"]].notna().all(axis=1).astype(float)
        x_values = numeric(group["x"]).where(valid.eq(1), 0.0)
        y_values = numeric(group["y"]).where(valid.eq(1), 0.0)
        count = valid.rolling(window, min_periods=1).sum()
        sum_x = x_values.rolling(window, min_periods=1).sum()
        sum_y = y_values.rolling(window, min_periods=1).sum()
        sum_xx = x_values.pow(2).rolling(window, min_periods=1).sum()
        sum_xy = (x_values * y_values).rolling(window, min_periods=1).sum()
        safe_count = count.replace(0, np.nan)
        denom = sum_xx - sum_x.pow(2) / safe_count
        beta = (sum_xy - sum_x * sum_y / safe_count) / denom.replace(0, np.nan)
        alpha = sum_y / safe_count - beta * sum_x / safe_count
        enough = count >= min_periods
        if output == "beta":
            values_out = beta
        elif output == "residual":
            values_out = numeric(group["y"]) - (alpha + beta * numeric(group["x"]))
        else:
            raise ValueError(f"Unsupported pair regression output: {output}")
        result.loc[group.index] = values_out.where(enough)
    return result.reindex(panel.index)
