from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
from .portfolio import (
    daily_portfolio_summary,
    one_day_label,
    resolve_holding_period_days,
    staggered_bucket_returns,
)


def daily_ic(
    df: pd.DataFrame, factor: str, label: str, min_assets: int = 3
) -> pd.DataFrame:
    rows = []
    for date, group in df[[factor, label, "date"]].dropna().groupby("date"):
        if (
            len(group) < min_assets
            or group[factor].nunique() < 2
            or group[label].nunique() < 2
        ):
            continue
        ic = group[factor].corr(group[label], method="pearson")
        rank_ic = group[factor].rank().corr(group[label].rank(), method="pearson")
        rows.append({"date": date, "ic": ic, "rank_ic": rank_ic, "n": len(group)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = out["date"].astype(str)
    return out


def summarize_ic(ic_df: pd.DataFrame) -> dict[str, float]:
    if ic_df.empty:
        return {
            "ic_mean": math.nan,
            "ic_std": math.nan,
            "icir": math.nan,
            "ic_positive_ratio": math.nan,
            "ic_t_stat": math.nan,
            "rank_ic_mean": math.nan,
            "rank_ic_std": math.nan,
            "rank_icir": math.nan,
            "rank_ic_positive_ratio": math.nan,
            "rank_ic_t_stat": math.nan,
            "days": 0.0,
        }
    n = len(ic_df)
    ic_std = ic_df["ic"].std(ddof=1)
    rank_std = ic_df["rank_ic"].std(ddof=1)
    return {
        "ic_mean": float(ic_df["ic"].mean()),
        "ic_std": float(ic_std) if not np.isnan(ic_std) else math.nan,
        "icir": _daily_ir(ic_df["ic"]),
        "ic_positive_ratio": float((ic_df["ic"] > 0).mean()),
        "ic_t_stat": float(ic_df["ic"].mean() / ic_std * math.sqrt(n))
        if ic_std and (not np.isnan(ic_std))
        else math.nan,
        "rank_ic_mean": float(ic_df["rank_ic"].mean()),
        "rank_ic_std": float(rank_std) if not np.isnan(rank_std) else math.nan,
        "rank_icir": _daily_ir(ic_df["rank_ic"]),
        "rank_ic_positive_ratio": float((ic_df["rank_ic"] > 0).mean()),
        "rank_ic_t_stat": float(ic_df["rank_ic"].mean() / rank_std * math.sqrt(n))
        if rank_std and (not np.isnan(rank_std))
        else math.nan,
        "days": float(n),
    }


def long_short_returns(
    df: pd.DataFrame,
    factor: str,
    label: str,
    quantile: float = 0.2,
    min_assets: int = 3,
    daily_return_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    holding_period_days = resolve_holding_period_days(label)
    if holding_period_days > 1:
        daily_label = one_day_label(label)
        daily_source = daily_return_frame if daily_return_frame is not None else df
        if daily_label not in daily_source:
            raise ValueError(
                f"{holding_period_days}-day staggered factor backtest requires {daily_label}"
            )
        formations: dict[str, dict[str, set[str]]] = {}
        eligible = df[[factor, label, "date", "symbol"]].dropna().copy()
        for date, group in eligible.groupby("date", sort=True):
            if len(group) < min_assets or group[factor].nunique() < 2:
                continue
            low_cut = group[factor].quantile(quantile)
            high_cut = group[factor].quantile(1.0 - quantile)
            formations[str(date)] = {
                "long": set(group.loc[group[factor] >= high_cut, "symbol"].astype(str)),
                "short": set(group.loc[group[factor] <= low_cut, "symbol"].astype(str)),
            }
        staggered = staggered_bucket_returns(
            formations,
            daily_source,
            daily_label=daily_label,
            holding_period_days=holding_period_days,
        )
        if staggered.empty:
            return staggered
        returns = staggered.pivot(index="date", columns="bucket", values="return")
        meta = staggered.groupby("date").agg(
            active_sleeves=("active_sleeves", "max"),
            capital_fraction=("capital_fraction", "max"),
            return_coverage=("return_coverage", "mean"),
        )
        out = meta.join(returns).reset_index()
        out["long"] = out.get("long", 0.0)
        out["short"] = out.get("short", 0.0)
        out["long_short"] = out["long"] - out["short"]
        out["holding_period_days"] = holding_period_days
        out["periods_per_year"] = 252.0
        out["portfolio_mode"] = "staggered_daily_mark_to_market"
        out["daily_return_label"] = daily_label
        return out
    rows: list[dict[str, Any]] = []
    eligible = df[[factor, label, "date"]].dropna().copy()
    for date, group in eligible.groupby("date", sort=True):
        if len(group) < min_assets or group[factor].nunique() < 2:
            continue
        low_cut = group[factor].quantile(quantile)
        high_cut = group[factor].quantile(1.0 - quantile)
        bottom = group[group[factor] <= low_cut][label].mean()
        top = group[group[factor] >= high_cut][label].mean()
        rows.append(
            {
                "date": str(date),
                "holding_period_days": holding_period_days,
                "periods_per_year": 252.0,
                "portfolio_mode": "daily_rebalance",
                "long": top,
                "short": bottom,
                "long_short": top - bottom,
            }
        )
    return pd.DataFrame(rows)


def summarize_long_short(returns: pd.DataFrame) -> dict[str, Any]:
    holding_period_days = 1
    if not returns.empty and "holding_period_days" in returns:
        values = pd.to_numeric(returns["holding_period_days"], errors="coerce").dropna()
        holding_period_days = int(values.iloc[0]) if not values.empty else 1
    if returns.empty:
        return {
            "annual_return": math.nan,
            "sharpe": math.nan,
            "max_drawdown": math.nan,
            "days": 0.0,
            "holding_period_days": float(holding_period_days),
            "annualization_periods": 252.0,
        }
    ordered = (
        returns.sort_values("date").copy() if "date" in returns else returns.copy()
    )
    series = pd.to_numeric(ordered["long_short"], errors="coerce").fillna(0.0)
    summary = daily_portfolio_summary(series)
    result = {
        "annual_return": summary["annual"],
        "sharpe": summary["sharpe"],
        "max_drawdown": summary["max_drawdown"],
        "days": float(len(series)),
        "holding_period_days": float(holding_period_days),
        "annualization_periods": 252.0,
        "portfolio_mode": str(ordered["portfolio_mode"].iloc[0])
        if "portfolio_mode" in ordered
        else "daily_rebalance",
    }
    if holding_period_days > 1:
        result["staggered_sleeve_count"] = float(holding_period_days)
    return result


def evaluate_factor(
    df: pd.DataFrame,
    factor: str,
    label: str,
    quantile: float = 0.2,
    min_assets: int = 3,
    daily_return_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    ic_df = daily_ic(df, factor, label, min_assets=min_assets)
    ls_df = long_short_returns(
        df,
        factor,
        label,
        quantile=quantile,
        min_assets=min_assets,
        daily_return_frame=daily_return_frame if daily_return_frame is not None else df,
    )
    return {
        "factor": factor,
        "coverage": float(df[[factor, label]].dropna().shape[0] / max(len(df), 1)),
        "ic": summarize_ic(ic_df),
        "long_short": summarize_long_short(ls_df),
    }


def _daily_ir(series: pd.Series) -> float:
    std = series.std(ddof=1)
    return float(series.mean() / std) if std and (not np.isnan(std)) else math.nan
