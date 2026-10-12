from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MechanismDefinition:
    id: str
    theme: str
    summary: str
    required_feature_capabilities: tuple[str, ...]
    required_construction_capabilities: tuple[str, ...] = ()
    typical_horizons: tuple[str, ...] = ("1d",)

    def prompt_card(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "theme": self.theme,
            "summary": self.summary,
            "required_feature_capabilities": list(self.required_feature_capabilities),
            "required_construction_capabilities": list(
                self.required_construction_capabilities
            ),
            "typical_horizons": list(self.typical_horizons),
        }


def _m(
    identifier: str,
    theme: str,
    summary: str,
    feature_capabilities: tuple[str, ...],
    horizons: tuple[str, ...] = ("1d",),
    construction: tuple[str, ...] = (),
) -> MechanismDefinition:
    return MechanismDefinition(
        identifier, theme, summary, feature_capabilities, construction, horizons
    )


MECHANISMS = (
    _m(
        "short_horizon_momentum",
        "trend_momentum",
        "Recent relative return strength persists over the next short horizon.",
        ("return",),
    ),
    _m(
        "medium_horizon_momentum",
        "trend_momentum",
        "Multi-day price trends persist after excluding the most recent noisy move.",
        ("return", "trend"),
        ("1d", "2-5d", "5-20d"),
    ),
    _m(
        "trend_strength_efficiency",
        "trend_momentum",
        "A smooth directional path is more persistent than a noisy path with the same total return.",
        ("return",),
        construction=("path_efficiency",),
    ),
    _m(
        "breakout_continuation",
        "trend_momentum",
        "Price leaving its recent range continues when the breakout is supported by participation.",
        ("return", "range"),
        construction=("breakout",),
    ),
    _m(
        "return_acceleration",
        "trend_momentum",
        "Increasing return speed identifies strengthening or weakening trend pressure.",
        ("return",),
        construction=("change",),
    ),
    _m(
        "short_horizon_reversal",
        "reversal",
        "Very recent cross-sectional winners and losers partially reverse because of temporary price pressure.",
        ("return",),
    ),
    _m(
        "overreaction_exhaustion",
        "reversal",
        "Extreme moves with weak path quality or excessive volatility are followed by correction.",
        ("return", "volatility"),
    ),
    _m(
        "opening_gap_reversal",
        "reversal",
        "An opening dislocation unsupported by the subsequent intraday path partially closes.",
        ("price_bar", "return"),
    ),
    _m(
        "late_session_pressure_reversal",
        "reversal",
        "Crowded late-session price pressure mean-reverts after temporary liquidity demand fades.",
        ("time_segment", "return"),
    ),
    _m(
        "vwap_dislocation_reversal",
        "reversal",
        "A close far from the trading-volume reference price reverts when the move lacks confirmation.",
        ("vwap", "return"),
    ),
    _m(
        "range_oscillation",
        "reversal",
        "Prices rotate within a stable range instead of continuing through its boundaries.",
        ("range", "location"),
    ),
    _m(
        "liquidity_shock",
        "liquidity_volume",
        "An abrupt participation or traded-amount shock changes short-horizon price pressure.",
        ("volume",),
        construction=("change",),
    ),
    _m(
        "illiquidity_price_impact",
        "liquidity_volume",
        "Large absolute return per unit of traded amount measures costly liquidity and price impact.",
        ("return", "amount"),
    ),
    _m(
        "turnover_surprise",
        "liquidity_volume",
        "Activity far above or below its own history identifies attention and liquidity regimes.",
        ("volume",),
        construction=("historical_extreme",),
    ),
    _m(
        "volume_price_confirmation",
        "liquidity_volume",
        "Participation confirms whether a directional price move is informative and persistent.",
        ("return", "volume"),
    ),
    _m(
        "volume_price_disagreement",
        "liquidity_volume",
        "Price direction and participation disagree, indicating weak conviction or absorption.",
        ("return", "volume"),
    ),
    _m(
        "liquidity_dry_up",
        "liquidity_volume",
        "Low participation combined with unstable prices signals fragile liquidity and reversal risk.",
        ("volume", "volatility"),
    ),
    _m(
        "opening_pressure_persistence",
        "intraday_structure",
        "Opening-session pressure continues when later trading confirms the move.",
        ("time_segment", "return"),
    ),
    _m(
        "late_session_persistence",
        "intraday_structure",
        "Late-session return and activity contain informed demand that carries into the next open.",
        ("time_segment", "return"),
    ),
    _m(
        "intraday_return_asymmetry",
        "intraday_structure",
        "Early and late returns have different information content and jointly reveal pressure transfer.",
        ("time_segment", "return"),
    ),
    _m(
        "intraday_volume_concentration",
        "intraday_structure",
        "Unusual concentration of trading in a session segment measures attention and crowding.",
        ("time_segment", "volume"),
    ),
    _m(
        "high_low_timing",
        "intraday_structure",
        "The time at which the intraday high or low occurs reveals persistent versus rejected price pressure.",
        ("event_timing", "range"),
    ),
    _m(
        "intraday_vwap_pressure",
        "intraday_structure",
        "Price displacement from VWAP across session segments measures demand pressure and absorption.",
        ("time_segment", "vwap"),
    ),
    _m(
        "realized_volatility_regime",
        "volatility_range",
        "The level of recent realized volatility changes the payoff of trend and reversal mechanisms.",
        ("volatility",),
    ),
    _m(
        "range_expansion",
        "volatility_range",
        "Expanding high-low or intraday ranges indicate information arrival and directional repricing.",
        ("range",),
        construction=("change",),
    ),
    _m(
        "range_compression",
        "volatility_range",
        "Compressed ranges identify latent pressure before expansion or continued low-volatility drift.",
        ("range",),
        construction=("historical_extreme",),
    ),
    _m(
        "volatility_of_volatility",
        "volatility_range",
        "Instability in realized volatility separates persistent regimes from transient shocks.",
        ("volatility",),
        construction=("volatility", "change"),
    ),
    _m(
        "downside_upside_asymmetry",
        "volatility_range",
        "Negative and positive moves carry different volatility and reversal implications.",
        ("return",),
        construction=("asymmetry",),
    ),
    _m(
        "tail_risk_extremes",
        "volatility_range",
        "Extreme-return frequency and tail thickness identify fragility and lottery-like behavior.",
        ("return",),
        construction=("tail",),
    ),
    _m(
        "intraday_distribution_shape",
        "volatility_range",
        "Intraday skewness and kurtosis distinguish directional pressure from noisy dispersion.",
        ("time_segment", "return"),
        construction=("asymmetry", "tail"),
    ),
    _m(
        "market_relative_strength",
        "relative_crowding",
        "Stock returns relative to the same-day common market move capture idiosyncratic strength.",
        ("return",),
        construction=("market_relative",),
    ),
    _m(
        "market_beta_exposure",
        "relative_crowding",
        "Sensitivity to the same-day equal-weight market return identifies changing common-factor exposure.",
        ("return",),
        construction=("market_aggregate", "beta"),
    ),
    _m(
        "idiosyncratic_volatility",
        "relative_crowding",
        "Volatility of market-demeaned returns measures stock-specific uncertainty.",
        ("return",),
        construction=("market_relative", "volatility"),
    ),
    _m(
        "market_residual_continuation",
        "relative_crowding",
        "Return pressure left after removing the common market component persists when it reflects stock-specific information.",
        ("return",),
        construction=("market_aggregate", "residualization"),
    ),
    _m(
        "market_residual_reversal",
        "relative_crowding",
        "Temporary stock-specific pressure left after removing the common market component reverses.",
        ("return",),
        construction=("market_aggregate", "residualization"),
    ),
    _m(
        "cross_sectional_dispersion",
        "relative_crowding",
        "Wide same-day return dispersion changes the payoff of stock-specific trend and reversal signals.",
        ("return",),
        construction=("cross_sectional_dispersion", "interaction"),
    ),
    _m(
        "co_movement_crowding",
        "relative_crowding",
        "High dependence on the same-day common market return indicates crowded common positioning.",
        ("return",),
        construction=("market_aggregate", "dependence"),
    ),
    _m(
        "dependence_regime",
        "relative_crowding",
        "The strength of price-volume or stock-market dependence identifies a changing information regime.",
        ("return", "volume"),
        construction=("dependence",),
    ),
)
MECHANISM_REGISTRY = {mechanism.id: mechanism for mechanism in MECHANISMS}
THEMES = tuple(dict.fromkeys((mechanism.theme for mechanism in MECHANISMS)))


def feasible_mechanisms(
    feature_capabilities: set[str], construction_capabilities: set[str]
) -> list[dict[str, Any]]:
    return [
        mechanism.prompt_card()
        for mechanism in MECHANISMS
        if set(mechanism.required_feature_capabilities).issubset(feature_capabilities)
        and set(mechanism.required_construction_capabilities).issubset(
            construction_capabilities
        )
    ]
