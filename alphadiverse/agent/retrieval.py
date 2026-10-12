from __future__ import annotations
from collections import Counter
from typing import Any
import re
from ..operators import OPERATOR_DEFINITIONS
from .ontology import MECHANISM_REGISTRY

RAW_CAPABILITIES = {
    "open": ("price_bar", "price_level", "return"),
    "high": ("price_bar", "price_level", "range", "location"),
    "low": ("price_bar", "price_level", "range", "location"),
    "close": ("price_bar", "price_level", "return", "location"),
    "volume": ("volume", "liquidity"),
    "turnover": ("amount", "volume", "liquidity"),
}


def feature_capabilities(name: str, layer: str) -> tuple[str, ...]:
    lower = name.lower()
    capabilities = set(RAW_CAPABILITIES.get(lower, ()))
    if any((token in lower for token in ("ret", "return", "roc", "kmid"))):
        capabilities.add("return")
    if any(
        (
            token in lower
            for token in ("ma", "momentum", "trend", "slope", "rsqr", "rsv")
        )
    ):
        capabilities.update(("trend", "trend_quality"))
    if any(
        (token in lower for token in ("std", "volatility", "variance", "var_", "rvol"))
    ):
        capabilities.update(("volatility", "dispersion"))
    if any((token in lower for token in ("range", "high_low", "klen", "atr"))):
        capabilities.update(("range", "volatility"))
    if any(
        (
            token in lower
            for token in ("volume", "vma", "bar_count", "up_bar", "down_bar")
        )
    ):
        capabilities.update(("volume", "liquidity"))
    if any((token in lower for token in ("turnover", "amount"))):
        capabilities.update(("amount", "volume", "liquidity"))
    if "vwap" in lower:
        capabilities.update(("vwap", "location"))
    if any(
        (
            token in lower
            for token in (
                "minute",
                "first",
                "last",
                "quarter",
                "half",
                "open_",
                "close_",
            )
        )
    ):
        capabilities.add("time_segment")
    if any((token in lower for token in ("high_time", "low_time", "argmax", "argmin"))):
        capabilities.add("event_timing")
    if any((token in lower for token in ("skew", "kurt", "tail", "moment"))):
        capabilities.update(("distribution_shape", "tail", "asymmetry"))
    if any((token in lower for token in ("corr", "cov", "beta", "cord"))):
        capabilities.add("dependence")
    if any((token in lower for token in ("resi", "residual", "alpha"))):
        capabilities.add("residual")
    if any((token in lower for token in ("market", "benchmark", "relative", "excess"))):
        capabilities.add("market_relative")
    if any(
        (token in lower for token in ("position", "percentile", "rank", "imax", "imin"))
    ):
        capabilities.update(("location", "historical_extreme"))
    if any((token in lower for token in ("delta", "change", "diff", "surprise"))):
        capabilities.add("change")
    if layer in {"daily", "minute"}:
        capabilities.add("cross_sectional")
    return tuple(sorted(capabilities))


def feature_capability_summary(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = {}
    for feature in features:
        layer = str(feature.get("layer") or "daily")
        layer_counts = counts.setdefault(layer, Counter())
        layer_counts.update(feature_capabilities(str(feature.get("name") or ""), layer))
    return [
        {
            "layer": layer,
            "feature_count": sum(
                (1 for feature in features if feature.get("layer") == layer)
            ),
            "capabilities": dict(sorted(layer_counts.items())),
        }
        for (layer, layer_counts) in sorted(counts.items())
    ]


def available_capabilities(features: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for feature in features:
        result.update(
            feature_capabilities(
                str(feature.get("name") or ""), str(feature.get("layer") or "daily")
            )
        )
    return result


class CapabilityRetriever:
    def __init__(
        self,
        specification: dict[str, Any],
        *,
        feature_limit: int = 16,
        operator_limit: int = 10,
    ):
        self.features = [
            feature
            for feature in specification.get("available_features", [])
            if isinstance(feature, dict)
        ]
        self.feature_limit = feature_limit
        self.operator_limit = operator_limit

    def retrieve(self, plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._retrieve_plan(plan) for plan in plans]

    def _retrieve_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        mechanism = MECHANISM_REGISTRY[plan["mechanism_id"]]
        required_features = set(mechanism.required_feature_capabilities)
        required_construction = set(mechanism.required_construction_capabilities)
        plan_tokens = _tokens(
            " ".join(
                (
                    str(plan.get(field) or "")
                    for field in ("event", "economic_mechanism", "conditioning")
                )
            )
        )
        scored_features = []
        for feature in self.features:
            name = str(feature.get("name") or "")
            layer = str(feature.get("layer") or "daily")
            capabilities = set(feature_capabilities(name, layer))
            overlap = len(required_features & capabilities)
            lexical = len(plan_tokens & _tokens(name.replace("_", " ")))
            coverage = float(feature.get("coverage") or 0.0)
            raw_bonus = 0.25 if name in RAW_CAPABILITIES else 0.0
            score = overlap * 4.0 + lexical + min(max(coverage, 0.0), 1.0) + raw_bonus
            if overlap or lexical or name in RAW_CAPABILITIES:
                scored_features.append((score, name, feature, capabilities))
        scored_features.sort(key=lambda row: (-row[0], row[1]))
        selected_features = [
            {
                "name": name,
                "layer": feature.get("layer"),
                "capabilities": sorted(capabilities),
                **(
                    {"description": feature["description"]}
                    if feature.get("description")
                    else {}
                ),
            }
            for (_, name, feature, capabilities) in scored_features[
                : self.feature_limit
            ]
        ]
        if len(selected_features) < 2:
            raise ValueError(
                f"Plan {plan['plan_id']} has fewer than two retrievable features"
            )
        scored_operators = []
        for definition in OPERATOR_DEFINITIONS:
            overlap = len(required_construction & set(definition.capabilities))
            core = definition.family == "algebra"
            if overlap or core:
                scored_operators.append(
                    (overlap * 3 + (1 if core else 0), definition.name, definition)
                )
        scored_operators.sort(key=lambda row: (-row[0], row[1]))
        operators = [
            definition.prompt_card()
            for (_, _, definition) in scored_operators[: self.operator_limit]
        ]
        return {
            "plan_id": plan["plan_id"],
            "features": selected_features,
            "operators": operators,
        }


def _tokens(value: str) -> set[str]:
    return {token for token in re.split("[^a-z0-9]+", value.lower()) if len(token) >= 3}
