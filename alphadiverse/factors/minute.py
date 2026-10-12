from __future__ import annotations
import re

MINUTE_SEGMENT_WINDOWS = (1, 3, 5, 10, 15, 20, 30)
MINUTE_SEGMENT_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "vwap",
    "ret",
    "range",
    "volume_share",
    "turnover_share",
    "close_vwap_deviation",
)
MINUTE_SEGMENT_30M_DUPLICATE_FIELDS = frozenset(
    {"ret", "volume_share", "close_vwap_deviation"}
)


def _minute_segment_fields(window: int) -> tuple[str, ...]:
    if window == 30:
        return tuple(
            (
                field
                for field in MINUTE_SEGMENT_FIELDS
                if field not in MINUTE_SEGMENT_30M_DUPLICATE_FIELDS
            )
        )
    return MINUTE_SEGMENT_FIELDS


MINUTE_SEGMENT_FEATURE_COLUMNS = [
    f"minute_{side}_{window}m_{field}"
    for side in ("open", "close")
    for window in MINUTE_SEGMENT_WINDOWS
    for field in _minute_segment_fields(window)
]
MINUTE_FEATURE_COLUMNS = [
    "minute_bar_count",
    "intraday_ret",
    "intraday_volatility",
    "intraday_range",
    "first30_volume_share",
    "last30_volume_share",
    "close_vwap_deviation",
    "high_time_ratio",
    "low_time_ratio",
    "minute_KMID",
    "minute_KLEN",
    "minute_KMID2",
    "minute_KSFT",
    "minute_KSFT2",
    "minute_close_position",
    "minute_up_bar_ratio",
    "minute_down_bar_ratio",
    "minute_ret_skew",
    "minute_ret_kurt",
    "first_half_volume_share",
    "second_half_volume_share",
    "first_quarter_volume_share",
    "last_quarter_volume_share",
    "volume_u_shape",
    "first30_ret",
    "last30_ret",
    "first_half_ret",
    "second_half_ret",
    "first_quarter_ret",
    "last_quarter_ret",
    "ret_second_minus_first_half",
    "ret_last_minus_first30",
    "ret_last_quarter_minus_first_quarter",
    "first30_vwap_deviation",
    "last30_vwap_deviation",
    *MINUTE_SEGMENT_FEATURE_COLUMNS,
]
_MINUTE_FACTOR_FORMULAS = {
    "minute_bar_count": "Count(intraday minute bars)",
    "intraday_ret": "minute_close_last / minute_open_first - 1",
    "intraday_volatility": "Std(minute_return)",
    "intraday_range": "Max(minute_high) / Min(minute_low) - 1",
    "first30_volume_share": "Sum(first 30m volume) / Sum(day volume)",
    "last30_volume_share": "Sum(last 30m volume) / Sum(day volume)",
    "close_vwap_deviation": "minute_close_last / day_vwap - 1",
    "high_time_ratio": "ArgMax(minute_high) / (minute_bar_count - 1)",
    "low_time_ratio": "ArgMin(minute_low) / (minute_bar_count - 1)",
    "minute_KMID": "minute_close_last / minute_open_first - 1",
    "minute_KLEN": "(Max(minute_high) - Min(minute_low)) / minute_open_first",
    "minute_KMID2": "(minute_close_last - minute_open_first) / (Max(minute_high) - Min(minute_low))",
    "minute_KSFT": "(2 * minute_close_last - Max(minute_high) - Min(minute_low)) / minute_open_first",
    "minute_KSFT2": "(2 * minute_close_last - Max(minute_high) - Min(minute_low)) / (Max(minute_high) - Min(minute_low))",
    "minute_close_position": "(minute_close_last - Min(minute_low)) / (Max(minute_high) - Min(minute_low))",
    "minute_up_bar_ratio": "Mean(minute_return > 0)",
    "minute_down_bar_ratio": "Mean(minute_return < 0)",
    "minute_ret_skew": "Skew(minute_return)",
    "minute_ret_kurt": "Kurtosis(minute_return)",
    "first_half_volume_share": "Sum(first-half volume) / Sum(day volume)",
    "second_half_volume_share": "Sum(second-half volume) / Sum(day volume)",
    "first_quarter_volume_share": "Sum(first-quarter volume) / Sum(day volume)",
    "last_quarter_volume_share": "Sum(last-quarter volume) / Sum(day volume)",
    "volume_u_shape": "(Sum(first 30m volume) + Sum(last 30m volume)) / Sum(day volume)",
    "first30_ret": "first30_close / first30_open - 1",
    "last30_ret": "last30_close / last30_open - 1",
    "first_half_ret": "first_half_close / first_half_open - 1",
    "second_half_ret": "second_half_close / second_half_open - 1",
    "first_quarter_ret": "first_quarter_close / first_quarter_open - 1",
    "last_quarter_ret": "last_quarter_close / last_quarter_open - 1",
    "ret_second_minus_first_half": "second_half_ret - first_half_ret",
    "ret_last_minus_first30": "last30_ret - first30_ret",
    "ret_last_quarter_minus_first_quarter": "last_quarter_ret - first_quarter_ret",
    "first30_vwap_deviation": "first30_close / first30_vwap - 1",
    "last30_vwap_deviation": "last30_close / last30_vwap - 1",
}


def minute_factor_formula(name: str) -> str:
    if name in _MINUTE_FACTOR_FORMULAS:
        return _MINUTE_FACTOR_FORMULAS[name]
    match = re.fullmatch(
        "minute_(open|close)_(\\d+)m_(open|high|low|close|volume|turnover|vwap|ret|range|volume_share|turnover_share|close_vwap_deviation)",
        name,
    )
    if not match:
        return name
    (side, window, field) = match.groups()
    segment = f"{side} {window}m"
    formulas = {
        "open": f"First({segment} minute_open)",
        "high": f"Max({segment} minute_high)",
        "low": f"Min({segment} minute_low)",
        "close": f"Last({segment} minute_close)",
        "volume": f"Sum({segment} minute_volume)",
        "turnover": f"Sum({segment} minute_turnover)",
        "vwap": f"Sum({segment} minute_turnover) / Sum({segment} minute_volume)",
        "ret": f"Last({segment} minute_close) / First({segment} minute_open) - 1",
        "range": f"Max({segment} minute_high) / Min({segment} minute_low) - 1",
        "volume_share": f"Sum({segment} minute_volume) / Sum(day minute_volume)",
        "turnover_share": f"Sum({segment} minute_turnover) / Sum(day minute_turnover)",
        "close_vwap_deviation": f"Last({segment} minute_close) / VWAP({segment}) - 1",
    }
    return formulas[field]
