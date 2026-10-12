from __future__ import annotations
from typing import Any


def _research_plan_history(
    portfolio: dict[str, Any],
    validation: dict[str, Any],
    model_retained_factors: list[str],
) -> list[dict[str, Any]]:
    decisions = {
        row.get("factor"): row
        for row in validation.get("decisions", [])
        if isinstance(row, dict) and row.get("factor")
    }
    metrics = {
        row.get("factor"): row
        for row in validation.get("metrics", [])
        if isinstance(row, dict) and row.get("factor")
    }
    retained = set(model_retained_factors)
    plans = []
    for plan in portfolio.get("research_plans", []):
        if not isinstance(plan, dict):
            continue
        outcomes = []
        for spec in plan.get("factor_specs", []):
            if not isinstance(spec, dict) or not spec.get("name"):
                continue
            factor = str(spec["name"])
            decision = decisions.get(factor, {})
            metric = metrics.get(factor, {})
            outcomes.append(
                {
                    "factor": factor,
                    "operator": spec.get("operator"),
                    "input_features": spec.get("input_features", []),
                    "formula": spec.get("formula"),
                    "rationale": spec.get("rationale"),
                    "factor_decision": decision.get("decision"),
                    "factor_reason": decision.get("reason"),
                    "duplicate_of": decision.get("duplicate_of"),
                    "rank_ic_mean": metric.get("ic", {}).get("rank_ic_mean"),
                    "factor_score": metric.get("factor_score"),
                    "coverage": metric.get("coverage"),
                    "model_retained": factor in retained,
                }
            )
        plans.append(
            {
                **{
                    key: value for (key, value) in plan.items() if key != "factor_specs"
                },
                "factor_outcomes": outcomes,
            }
        )
    return plans


def _best_factor_cards(
    validation: dict[str, Any], factor_specs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    cards = []
    accepted = set(validation.get("accepted_factors", []))
    for row in validation.get("accepted", []):
        factor = row.get("factor")
        if factor not in accepted:
            continue
        spec = factor_specs.get(factor, {})
        cards.append(
            {
                "factor": factor,
                "layer": spec.get("layer", row.get("layer", "unknown")),
                "operator": spec.get("operator", ""),
                "plan_id": spec.get("plan_id", ""),
                "mechanism_id": spec.get("mechanism_id", ""),
                "theme": spec.get("theme", ""),
                "input_features": spec.get("input_features", []),
                "formula": spec.get("formula", ""),
                "rank_ic_mean": row.get("ic", {}).get("rank_ic_mean"),
                "ic_mean": row.get("ic", {}).get("ic_mean"),
                "factor_score": row.get("factor_score"),
                "coverage": row.get("coverage"),
                "sharpe": row.get("long_short", {}).get("sharpe"),
            }
        )
    return cards


def _model_retained_factor_cards(
    validation: dict[str, Any],
    factor_specs: dict[str, dict[str, Any]],
    model_retained_factors: list[str],
) -> list[dict[str, Any]]:
    retained = set(model_retained_factors)
    return [
        card
        for card in _best_factor_cards(validation, factor_specs)
        if card.get("factor") in retained
    ]
