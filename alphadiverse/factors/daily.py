from __future__ import annotations
import re
import pandas as pd
from .minute import minute_factor_formula

BASE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "turnover"]
EPS = 1e-12
ALPHA158_WINDOWS = (5, 10, 20, 30, 60)
ALPHA158_KBAR_FEATURES = (
    "KMID",
    "KLEN",
    "KMID2",
    "KUP",
    "KUP2",
    "KLOW",
    "KLOW2",
    "KSFT",
    "KSFT2",
)
ALPHA158_PRICE_FEATURES = ("OPEN0", "HIGH0", "LOW0", "VWAP0")
ALPHA158_ROLLING_PREFIXES = (
    "ROC",
    "MA",
    "STD",
    "BETA",
    "RSQR",
    "RESI",
    "MAX",
    "MIN",
    "QTLU",
    "QTLD",
    "RANK",
    "RSV",
    "IMAX",
    "IMIN",
    "IMXD",
    "CORR",
    "CORD",
    "CNTP",
    "CNTN",
    "CNTD",
    "SUMP",
    "SUMN",
    "SUMD",
    "VMA",
    "VSTD",
    "WVMA",
    "VSUMP",
    "VSUMN",
    "VSUMD",
)
ALPHA158_FEATURES = (
    *ALPHA158_KBAR_FEATURES,
    *ALPHA158_PRICE_FEATURES,
    *(
        f"{prefix}{window}"
        for prefix in ALPHA158_ROLLING_PREFIXES
        for window in ALPHA158_WINDOWS
    ),
)
ALPHA20_FORMULAS: dict[str, str] = {
    "RESI5": "Resi(close, 5) / close",
    "WVMA5": "Std(abs(close / Ref(close, 1) - 1) * volume, 5) / Mean(abs(close / Ref(close, 1) - 1) * volume, 5)",
    "RSQR5": "Rsquare(close, 5)",
    "KLEN": "(high - low) / open",
    "RSQR10": "Rsquare(close, 10)",
    "CORR5": "Corr(close, log(volume + 1), 5)",
    "CORD5": "Corr(close / Ref(close, 1), log(volume / Ref(volume, 1) + 1), 5)",
    "CORR10": "Corr(close, log(volume + 1), 10)",
    "ROC60": "Ref(close, 60) / close",
    "RESI10": "Resi(close, 10) / close",
    "VSTD5": "Std(volume, 5) / volume",
    "RSQR60": "Rsquare(close, 60)",
    "CORR60": "Corr(close, log(volume + 1), 60)",
    "WVMA60": "Std(abs(close / Ref(close, 1) - 1) * volume, 60) / Mean(abs(close / Ref(close, 1) - 1) * volume, 60)",
    "STD5": "Std(close, 5) / close",
    "RSQR20": "Rsquare(close, 20)",
    "CORD60": "Corr(close / Ref(close, 1), log(volume / Ref(volume, 1) + 1), 60)",
    "CORD10": "Corr(close / Ref(close, 1), log(volume / Ref(volume, 1) + 1), 10)",
    "CORR20": "Corr(close, log(volume + 1), 20)",
    "KLOW": "(min(open, close) - low) / open",
}
ALPHA20_FEATURES = tuple(ALPHA20_FORMULAS)
_CORE_DAILY_FORMULAS: dict[str, str] = {
    "ret_1": "close / Ref(close, 1) - 1",
    "open_close_ret": "close / open - 1",
    "high_low_range": "high / low - 1",
    "close_position": "(close - low) / (high - low)",
    "vwap": "turnover / volume",
    "vwap_deviation": "close / vwap - 1",
    "log_turnover": "log(1 + turnover)",
    "log_volume": "log(1 + volume)",
    "KMID": "(close - open) / open",
    "KLEN": "(high - low) / open",
    "KMID2": "(close - open) / (high - low)",
    "KUP": "(high - max(open, close)) / open",
    "KUP2": "(high - max(open, close)) / (high - low)",
    "KLOW": "(min(open, close) - low) / open",
    "KLOW2": "(min(open, close) - low) / (high - low)",
    "KSFT": "(2 * close - high - low) / open",
    "KSFT2": "(2 * close - high - low) / (high - low)",
    "OPEN0": "open / close",
    "HIGH0": "high / close",
    "LOW0": "low / close",
    "VWAP0": "vwap / close",
}


def base_factor_formula(name: str) -> str:
    if name in ALPHA20_FORMULAS:
        return ALPHA20_FORMULAS[name]
    if name in _CORE_DAILY_FORMULAS:
        return _CORE_DAILY_FORMULAS[name]
    patterns = (
        ("close_ma(\\d+)_ratio", lambda d: f"close / Mean(close, {d}) - 1"),
        ("ret_(\\d+)", lambda d: f"close / Ref(close, {d}) - 1"),
        ("volatility_(\\d+)", lambda d: f"Std(close / Ref(close, 1) - 1, {d})"),
        ("volume_ma(\\d+)_ratio", lambda d: f"volume / Mean(volume, {d}) - 1"),
        ("ROC(\\d+)", lambda d: f"Ref(close, {d}) / close"),
        ("MA(\\d+)", lambda d: f"Mean(close, {d}) / close"),
        ("STD(\\d+)", lambda d: f"Std(close, {d}) / close"),
        ("BETA(\\d+)", lambda d: f"Slope(close, {d}) / close"),
        ("RSQR(\\d+)", lambda d: f"Rsquare(close, {d})"),
        ("RESI(\\d+)", lambda d: f"Resi(close, {d}) / close"),
        ("MAX(\\d+)", lambda d: f"Max(high, {d}) / close"),
        ("MIN(\\d+)", lambda d: f"Min(low, {d}) / close"),
        ("QTLU(\\d+)", lambda d: f"Quantile(close, {d}, 0.8) / close"),
        ("QTLD(\\d+)", lambda d: f"Quantile(close, {d}, 0.2) / close"),
        ("RANK(\\d+)", lambda d: f"Rank(close, {d})"),
        (
            "RSV(\\d+)",
            lambda d: f"(close - Min(low, {d})) / (Max(high, {d}) - Min(low, {d}))",
        ),
        ("IMAX(\\d+)", lambda d: f"IdxMax(high, {d}) / {d}"),
        ("IMIN(\\d+)", lambda d: f"IdxMin(low, {d}) / {d}"),
        ("IMXD(\\d+)", lambda d: f"(IdxMax(high, {d}) - IdxMin(low, {d})) / {d}"),
        ("CORR(\\d+)", lambda d: f"Corr(close, log(volume + 1), {d})"),
        (
            "CORD(\\d+)",
            lambda d: f"Corr(close / Ref(close, 1), log(volume / Ref(volume, 1) + 1), {d})",
        ),
        ("CNTP(\\d+)", lambda d: f"Mean(close > Ref(close, 1), {d})"),
        ("CNTN(\\d+)", lambda d: f"Mean(close < Ref(close, 1), {d})"),
        (
            "CNTD(\\d+)",
            lambda d: f"Mean(close > Ref(close, 1), {d}) - Mean(close < Ref(close, 1), {d})",
        ),
        (
            "SUMP(\\d+)",
            lambda d: f"Sum(max(close - Ref(close, 1), 0), {d}) / Sum(abs(close - Ref(close, 1)), {d})",
        ),
        (
            "SUMN(\\d+)",
            lambda d: f"Sum(max(Ref(close, 1) - close, 0), {d}) / Sum(abs(close - Ref(close, 1)), {d})",
        ),
        (
            "SUMD(\\d+)",
            lambda d: f"(Sum(max(close - Ref(close, 1), 0), {d}) - Sum(max(Ref(close, 1) - close, 0), {d})) / Sum(abs(close - Ref(close, 1)), {d})",
        ),
        ("VMA(\\d+)", lambda d: f"Mean(volume, {d}) / volume"),
        ("VSTD(\\d+)", lambda d: f"Std(volume, {d}) / volume"),
        (
            "WVMA(\\d+)",
            lambda d: f"Std(abs(close / Ref(close, 1) - 1) * volume, {d}) / Mean(abs(close / Ref(close, 1) - 1) * volume, {d})",
        ),
        (
            "VSUMP(\\d+)",
            lambda d: f"Sum(max(volume - Ref(volume, 1), 0), {d}) / Sum(abs(volume - Ref(volume, 1)), {d})",
        ),
        (
            "VSUMN(\\d+)",
            lambda d: f"Sum(max(Ref(volume, 1) - volume, 0), {d}) / Sum(abs(volume - Ref(volume, 1)), {d})",
        ),
        (
            "VSUMD(\\d+)",
            lambda d: f"(Sum(max(volume - Ref(volume, 1), 0), {d}) - Sum(max(Ref(volume, 1) - volume, 0), {d})) / Sum(abs(volume - Ref(volume, 1)), {d})",
        ),
    )
    for pattern, render in patterns:
        match = re.fullmatch(pattern, name)
        if match:
            return render(match.group(1))
    return minute_factor_formula(name)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "symbol",
        "stock_code",
        "name",
        "board",
        "status",
        "primary_sector",
        "mcap_bucket",
        "date",
        "datetime",
        "ts_millis",
        "ts_millis_open",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "shares",
        "float_a_shares",
        "mcap",
        "float_mcap",
        "log_mcap",
        "st",
        "is_st",
        "listing_date",
        "delisting_date",
        "listing_age_days",
        "is_new_listing",
        "prev_close",
        "limit_move_threshold",
        "limit_move_today",
        "entry_open_limit_move",
        "exit_open_limit_move",
        "universe_eligible",
    }
    return [
        c
        for c in df.columns
        if c not in excluded and (not c.startswith(("label_", "benchmark_", "in_")))
    ]
