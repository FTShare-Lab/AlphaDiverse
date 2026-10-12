from __future__ import annotations
from collections import Counter
from typing import Any


def build_research_memory(history: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = Counter()
    successful = Counter()
    factor_only = Counter()
    failed = Counter()
    transitions = []
    previous_mechanisms: tuple[str, ...] = ()
    previous_success = False
    for item in history:
        plans = [
            plan for plan in item.get("research_plans", []) if isinstance(plan, dict)
        ]
        current_mechanisms = tuple(
            (
                str(plan.get("mechanism_id"))
                for plan in plans
                if plan.get("mechanism_id")
            )
        )
        retained_names = set(item.get("model_retained_factors", []))
        current_success = False
        for plan in plans:
            mechanism = str(plan.get("mechanism_id") or "")
            if not mechanism:
                continue
            attempted[mechanism] += 1
            outcomes = [
                row for row in plan.get("factor_outcomes", []) if isinstance(row, dict)
            ]
            retained = any(
                (
                    row.get("factor") in retained_names
                    or row.get("model_retained") is True
                    for row in outcomes
                )
            )
            accepted = any((row.get("factor_decision") == "accept" for row in outcomes))
            if retained:
                successful[mechanism] += 1
                current_success = True
            elif accepted:
                factor_only[mechanism] += 1
            else:
                failed[mechanism] += 1
        if previous_mechanisms and (not previous_success):
            transitions.append(
                {
                    "from_mechanisms": list(previous_mechanisms),
                    "to_mechanisms": list(current_mechanisms),
                    "switched": set(previous_mechanisms) != set(current_mechanisms),
                    "next_round_model_retain": current_success,
                }
            )
        previous_mechanisms = current_mechanisms
        previous_success = current_success
    crowded = [
        {
            "mechanism_id": mechanism,
            "attempts": count,
            "model_retain": successful[mechanism],
        }
        for (mechanism, count) in attempted.most_common(12)
        if count >= 2
    ]
    return {
        "successful_mechanisms": [
            {"mechanism_id": mechanism, "model_retain_count": count}
            for (mechanism, count) in successful.most_common(12)
        ],
        "factor_only_mechanisms": [
            {"mechanism_id": mechanism, "factor_accept_count": count}
            for (mechanism, count) in factor_only.most_common(12)
        ],
        "failed_mechanisms": [
            {"mechanism_id": mechanism, "factor_failure_count": count}
            for (mechanism, count) in failed.most_common(12)
        ],
        "crowded_mechanisms": crowded,
        "recent_failure_transitions": transitions[-2:],
    }
