from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .factors.daily import ALPHA158_FEATURES, _CORE_DAILY_FORMULAS
from .factors.minute import MINUTE_FEATURE_COLUMNS


@dataclass(frozen=True)
class RunRequest:
    start: str
    end: str
    layers: tuple[str, ...] = ("daily", "minute")


def market_benchmark_map(cfg):
    return dict((value.split(":", 1) for value in cfg.evaluation.market_benchmarks))


def active_market_names(cfg):
    return list(cfg.evaluation.active_markets)


def market_member_column(market):
    return "in_" + market


def market_label_column(market, horizon_days):
    return f"label_{market}_excess_ret_{horizon_days}d"


def label_mode_market(mode):
    return mode.removesuffix("_excess")


def _shsz(symbol):
    return bool(re.fullmatch("(?:6\\d{5}\\.XSHG|[03]\\d{5}\\.XSHE)", str(symbol)))


def _symbol(symbol):
    return (
        str(symbol).replace(".SH", ".XSHG")
        if str(symbol).endswith(".SH")
        else str(symbol).replace(".SZ", ".XSHE")
        if str(symbol).endswith(".SZ")
        else str(symbol)
    )


def _read(path, start, end):
    path = Path(path)
    if path.is_file():
        paths = [path]
    else:
        paths = sorted(path.rglob("*.parquet"))
    if not paths:
        raise ValueError("No local Parquet files found")
    frames = []
    for item in paths:
        year = next((p[5:] for p in item.parts if p.startswith("year=")), None)
        if year and (not start[:4] <= year <= end[:4]):
            continue
        frame = pq.ParquetFile(item).read().to_pandas()
        if "date" not in frame:
            raise ValueError("Each input partition must contain date")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime(
            "%Y-%m-%d"
        )
        selected = frame.date.between(start, end)
        frames.append(frame if selected.all() else frame.loc[selected])
    if not frames:
        raise ValueError("No observations in the requested date range")
    return pd.concat(frames, ignore_index=True)


def load_panel(data, cfg, membership_data=None):
    protocol = cfg.search_validation
    warmup_start = protocol.inner_start
    source = Path(data)
    panel = _read(
        source / "base_panel" if (source / "base_panel").is_dir() else source,
        warmup_start,
        protocol.outer_end,
    )
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "turnover"}
    if not required <= set(panel):
        raise ValueError(
            "Missing price-volume panel columns: " + str(sorted(required - set(panel)))
        )
    panel["symbol"] = panel.symbol.map(_symbol)
    if panel.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate stock-date observations")
    panel = panel.loc[panel.symbol.map(_shsz)].copy()
    market = cfg.evaluation.active_markets[0]
    if membership_data:
        weights = _read(membership_data, warmup_start, protocol.outer_end)
        required = {"date", "market", "component_symbol"}
        if not required <= set(weights):
            raise ValueError("Membership table requires date, market, component_symbol")
        selected = weights.loc[weights.market.eq(market)].copy()
        selected["component_symbol"] = selected.component_symbol.map(_symbol)
        if market != "ashare":
            missing = set(panel.date) - set(selected.date)
            if missing:
                raise ValueError(
                    "Missing daily membership dates; static fallback is forbidden"
                )
            index = pd.MultiIndex.from_frame(selected[["date", "component_symbol"]])
            panel["in_" + market] = pd.MultiIndex.from_frame(
                panel[["date", "symbol"]]
            ).isin(index)
    if market == "ashare":
        panel["in_ashare"] = True
    member = "in_" + market
    if member not in panel or panel[member].isna().any():
        raise ValueError("Provide daily historical " + member + " or --membership-data")
    if not set(panel[member].unique()) <= {True, False, 0, 1}:
        raise ValueError("Membership flags must be Boolean or 0/1")
    panel = panel.loc[panel[member].astype(bool)].copy()
    if "universe_eligible" in panel:
        if not set(panel.universe_eligible.dropna().unique()) <= {True, False, 0, 1}:
            raise ValueError("Invalid universe_eligible flag")
        panel = panel.loc[panel.universe_eligible.fillna(False).astype(bool)].copy()
    label = market_label_column(market, 1)
    if label not in panel:
        raise ValueError("Missing precomputed O2O target " + label)
    known = (
        set(ALPHA158_FEATURES) | set(_CORE_DAILY_FORMULAS) | set(MINUTE_FEATURE_COLUMNS)
    )
    inputs = [
        c
        for c in panel
        if c in known or re.fullmatch("(close_ma|ret|volatility|volume_ma)_\\d+", c)
    ]
    columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        member,
        label,
    ]
    panel = (
        panel[list(dict.fromkeys([*columns, *inputs]))]
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    for c in inputs:
        if not pd.api.types.is_numeric_dtype(panel[c].dtype):
            panel[c] = pd.to_numeric(panel[c], errors="raise")
        values = (
            panel[c].to_numpy(copy=False)
            if isinstance(panel[c].dtype, np.dtype)
            else panel[c].to_numpy(dtype="float64", na_value=np.nan)
        )
        if np.isinf(values).any():
            panel[c] = panel[c].replace([np.inf, -np.inf], np.nan)
    for start, end in [
        (protocol.inner_start, protocol.inner_end),
        (protocol.outer_start, protocol.outer_end),
    ]:
        observed = panel.loc[panel.date.between(start, end)]
        if observed.empty or observed[label].notna().sum() < 100:
            raise ValueError(
                "Insufficient labeled observations in an evaluation interval"
            )
    minute = [c for c in inputs if c in set(MINUTE_FEATURE_COLUMNS)]
    daily = [c for c in inputs if c not in set(minute)]
    return (panel, {"daily": daily, "minute": minute}, label)
