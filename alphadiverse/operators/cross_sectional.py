from __future__ import annotations
import numpy as np
import pandas as pd


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def cs_demean(panel: pd.DataFrame, series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values - values.groupby(panel["date"], sort=False).transform("mean")


def cs_zscore(panel: pd.DataFrame, series: pd.Series) -> pd.Series:
    values = numeric(series)
    centered = values - values.groupby(panel["date"], sort=False).transform("mean")
    std = values.groupby(panel["date"], sort=False).transform("std").replace(0, np.nan)
    return centered / std


def cs_rank(panel: pd.DataFrame, series: pd.Series) -> pd.Series:
    return numeric(series).groupby(panel["date"], sort=False).rank(pct=True)


def cs_winsorize(
    panel: pd.DataFrame, series: pd.Series, n_sigma: float = 3.0
) -> pd.Series:
    values = numeric(series)
    mean = values.groupby(panel["date"], sort=False).transform("mean")
    std = values.groupby(panel["date"], sort=False).transform("std")
    return values.clip(
        lower=mean - float(n_sigma) * std, upper=mean + float(n_sigma) * std
    )


def cs_mean(panel: pd.DataFrame, series: pd.Series) -> pd.Series:
    return numeric(series).groupby(panel["date"], sort=False).transform("mean")


def cs_std(panel: pd.DataFrame, series: pd.Series) -> pd.Series:
    return numeric(series).groupby(panel["date"], sort=False).transform("std")
