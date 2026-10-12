from __future__ import annotations
from collections.abc import Hashable, Mapping
import math
import re
import numpy as np
import pandas as pd

_LABEL_HORIZON_RE = re.compile("_(\\d+)d$")


def one_day_label(label: str) -> str:
    match = _LABEL_HORIZON_RE.search(str(label))
    if match is None:
        raise ValueError(f"Cannot infer a one-day return label from {label!r}")
    return f"{str(label)[: match.start()]}_1d"


def label_horizon_days(label: str) -> int:
    match = _LABEL_HORIZON_RE.search(str(label))
    return int(match.group(1)) if match else 1


def resolve_holding_period_days(label: str | None, value: int | None = None) -> int:
    inferred = label_horizon_days(label) if label else None
    resolved = int(value if value is not None else inferred or 1)
    if resolved < 1:
        raise ValueError("holding period must be at least one day")
    if inferred is not None and value is not None and (resolved != inferred):
        raise ValueError(
            f"Holding period {resolved} does not match label horizon {inferred}: {label}"
        )
    return resolved


def staggered_bucket_returns(
    formations: Mapping[str, Mapping[Hashable, set[str]]],
    daily_returns: pd.DataFrame,
    *,
    daily_label: str,
    holding_period_days: int,
) -> pd.DataFrame:
    horizon = int(holding_period_days)
    if horizon < 1:
        raise ValueError("holding_period_days must be at least one")
    required = {"date", "symbol", daily_label}
    missing = sorted(required - set(daily_returns.columns))
    if missing:
        raise ValueError(f"Daily return frame is missing columns: {missing}")
    if not formations:
        return pd.DataFrame()
    source_dates = daily_returns["date"].astype(str)
    calendar = sorted(source_dates.unique().tolist())
    calendar_index = {date: index for (index, date) in enumerate(calendar)}
    missing_dates = sorted(set(formations) - set(calendar_index))
    if missing_dates:
        raise ValueError(
            f"Daily return frame does not cover formation dates: {missing_dates[:10]}"
        )
    incomplete = [
        date for date in formations if calendar_index[date] + horizon > len(calendar)
    ]
    if incomplete:
        raise ValueError(
            f"Daily return frame does not contain every holding-period leg for formation dates: {sorted(incomplete)[:10]}"
        )
    first_return_date = min(formations, key=calendar_index.__getitem__)
    last_return_date = calendar[
        max((calendar_index[date] + horizon - 1 for date in formations))
    ]
    window = source_dates.between(first_return_date, last_return_date)
    returns = daily_returns.loc[window, ["date", "symbol", daily_label]].copy()
    returns["date"] = returns["date"].astype(str)
    returns["symbol"] = returns["symbol"].astype(str)
    returns[daily_label] = pd.to_numeric(returns[daily_label], errors="coerce")
    if returns.duplicated(["date", "symbol"]).any():
        raise ValueError("Daily return frame contains duplicate date/symbol keys")
    by_date = {
        str(date): frame.set_index("symbol")[daily_label]
        for (date, frame) in returns.groupby("date", sort=False)
    }
    formation_by_index = {
        calendar_index[str(date)]: {
            bucket: set((str(symbol) for symbol in members))
            for (bucket, members) in buckets.items()
        }
        for (date, buckets) in formations.items()
    }
    first_index = min(formation_by_index)
    last_index = max((index + horizon - 1 for index in formation_by_index))
    buckets = list(
        dict.fromkeys(
            (
                bucket
                for bucket_map in formation_by_index.values()
                for bucket in bucket_map
            )
        )
    )
    previous_weights: dict[Hashable, dict[str, float]] = {
        bucket: {} for bucket in buckets
    }
    rows: list[dict[str, object]] = []
    for return_index in range(first_index, last_index + 1):
        active = [
            formation_by_index[formation_index]
            for formation_index in range(return_index - horizon + 1, return_index + 1)
            if formation_index in formation_by_index
        ]
        return_date = calendar[return_index]
        daily = by_date.get(return_date, pd.Series(dtype="float64"))
        for bucket in buckets:
            cohort_members = [
                bucket_map.get(bucket, set())
                for bucket_map in active
                if bucket_map.get(bucket)
            ]
            target_weights = _aggregate_sleeve_weights(cohort_members, horizon)
            turnover = _weight_turnover(previous_weights[bucket], target_weights)
            previous_weights[bucket] = target_weights
            cohort_returns: list[float] = []
            observed = 0
            expected = 0
            for members in cohort_members:
                values = pd.to_numeric(daily.reindex(sorted(members)), errors="coerce")
                expected += len(members)
                observed += int(values.notna().sum())
                cohort_returns.append(
                    float(values.fillna(0.0).sum() / len(members)) if members else 0.0
                )
            contribution = float(sum(cohort_returns) / horizon)
            rows.append(
                {
                    "date": return_date,
                    "bucket": bucket,
                    "return": contribution,
                    "active_sleeves": int(len(cohort_members)),
                    "capital_fraction": float(len(cohort_members) / horizon),
                    "member_observations": int(observed),
                    "member_observations_expected": int(expected),
                    "return_coverage": float(observed / expected)
                    if expected
                    else math.nan,
                    "target_name_count": int(len(target_weights)),
                    "turnover": turnover,
                    "holding_period_days": horizon,
                    "periods_per_year": 252.0,
                    "portfolio_mode": "staggered_daily_mark_to_market",
                }
            )
    return pd.DataFrame(rows)


def _aggregate_sleeve_weights(
    cohort_members: list[set[str]], holding_period_days: int
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for members in cohort_members:
        if not members:
            continue
        weight = 1.0 / holding_period_days / len(members)
        for symbol in members:
            weights[symbol] = weights.get(symbol, 0.0) + weight
    return weights


def _weight_turnover(
    previous: Mapping[str, float], current: Mapping[str, float]
) -> float:
    symbols = set(previous) | set(current)
    if not symbols:
        return math.nan
    return float(
        0.5
        * sum(
            (
                abs(float(current.get(symbol, 0.0)) - float(previous.get(symbol, 0.0)))
                for symbol in symbols
            )
        )
    )


def attach_portfolio_equity(
    returns: pd.DataFrame, value_col: str = "long_short"
) -> pd.DataFrame:
    if returns.empty:
        return returns
    out = returns.sort_values("date").reset_index(drop=True).copy()
    series = pd.to_numeric(out[value_col], errors="coerce").fillna(0.0)
    out["equity"] = (1.0 + series).cumprod()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    return out


def daily_portfolio_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if values.empty:
        return {"annual": math.nan, "sharpe": math.nan, "max_drawdown": math.nan}
    std = values.std(ddof=1)
    equity = (1.0 + values).cumprod()
    return {
        "annual": float(values.mean() * 252.0),
        "sharpe": float(values.mean() / std * math.sqrt(252.0))
        if len(values) > 1 and std and (not np.isnan(std))
        else math.nan,
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
    }
