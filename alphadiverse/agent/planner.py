from __future__ import annotations
from typing import Any
import json
from ..operators import operator_capability_summary
from ..llm import resolve_chat_backend
from .llm_json import call_validated_json
from .memory import build_research_memory
from .ontology import MECHANISM_REGISTRY, feasible_mechanisms
from .retrieval import available_capabilities, feature_capability_summary
from .utils import (
    _compact_history,
    _llm_model_context,
    _llm_prompt_token_budget,
    _message_token_count,
    _pack_json_items,
    _shrink_to_message_budget,
)

PLAN_DIRECTIONS = {"continuation", "reversal", "nonlinear", "unknown"}
PLAN_RELATIONS = {
    "novel",
    "conditional_refinement",
    "residual_refinement",
    "orthogonal_refinement",
}
PLAN_HORIZONS = {"intraday", "1d", "2-5d", "5-20d"}
DETAILED_HISTORY_ROUNDS = 6


class PlannerModule:
    def __init__(self, *, complementary: bool = True, semantic_memory: bool = True):
        self.complementary = complementary
        self.semantic_memory = semantic_memory

    def build_request(
        self,
        specification: dict[str, Any],
        history: list[dict[str, Any]],
        *,
        plans_per_round: int,
        current_round: int,
        max_rounds: int,
        model_sota: dict[str, Any],
        model_retained_factors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        feature_capabilities = available_capabilities(
            specification.get("available_features", [])
        )
        construction_capabilities: set[str] = set()
        for family in operator_capability_summary():
            construction_capabilities.update(family["capabilities"])
        ontology = feasible_mechanisms(feature_capabilities, construction_capabilities)
        if len(ontology) < plans_per_round:
            raise ValueError(
                f"Only {len(ontology)} feasible mechanisms remain for {plans_per_round} requested plans"
            )
        previous_mechanisms = _latest_mechanisms(history)
        local_planner = resolve_chat_backend("agent.planner").backend == "local"
        minimum_new = (
            0 if local_planner or not self.complementary else min(2, plans_per_round)
        )
        suggested_new = min(1 if local_planner else 2, plans_per_round)
        transition_instruction = (
            "Prefer at least one mechanism different from the immediately preceding round. This is a research preference, not a validity requirement: mechanisms may repeat across rounds. "
            if local_planner
            else "At least two mechanisms must differ from the immediately preceding round. "
        )
        theme_target = min(3, plans_per_round)
        prompt = {
            "role": "research_portfolio_planner",
            "instruction": f"Design exactly {plans_per_round} distinct quantitative research plans for the next round. The plans must be jointly useful, not renamings or parameter variants. Cover at least {theme_target} ontology themes. {transition_instruction}Balance performance with mechanism coverage: strong retained mechanisms may be improved only through explicit conditioning, residualization, or orthogonalization. Use only feasible mechanism IDs and available capabilities. Do not choose concrete feature names, operators, windows, or formulas. All model evidence is inner-search evidence; final holdout results are unavailable. Output JSON only.",
            "research_contract": {
                "panel": _planner_panel_contract(
                    specification.get("panel_contract", {})
                ),
                "base_factors": specification.get("base_factors", []),
                "feature_capabilities": feature_capability_summary(
                    specification.get("available_features", [])
                ),
                "operator_capabilities": operator_capability_summary(),
                "mechanism_ontology": ontology,
                "allowed_factor_layers": specification.get("allowed_factor_layers", []),
            },
            "research_memory": build_research_memory(history),
            "mechanism_transition": {
                "previous_round_mechanisms": sorted(previous_mechanisms),
                "enforcement": "advisory" if local_planner else "required",
                "minimum_new_mechanisms": minimum_new if previous_mechanisms else 0,
                "suggested_new_mechanisms": suggested_new if previous_mechanisms else 0,
                "instruction": transition_instruction.strip(),
            },
            "model_context": _llm_model_context(model_sota, model_retained_factors),
            "recent_history": [],
            "context_note": "recent_history is token-budgeted; research_memory summarizes the full trajectory.",
            "output_schema": {
                "research_plans": [
                    {
                        "plan_id": "p1",
                        "mechanism_id": "one id from mechanism_ontology",
                        "event": "observable market event",
                        "economic_mechanism": "why the event should affect future returns",
                        "horizon": "intraday|1d|2-5d|5-20d",
                        "conditioning": "market state in which the mechanism should hold",
                        "expected_direction": "continuation|reversal|nonlinear|unknown",
                        "relation_to_base": "novel|conditional_refinement|residual_refinement|orthogonal_refinement",
                    }
                ]
            },
        }
        if not self.complementary:
            prompt["instruction"] = (
                f"Design exactly {plans_per_round} useful quantitative research plans for the next round. Select each plan for expected predictive quality. Mechanism IDs and themes may repeat; there is no within-round or across-round novelty quota. Use only feasible mechanism IDs and available capabilities. Do not choose concrete features, operators, windows or formulas. All evidence is inner-search evidence. Output JSON only."
            )
            prompt["mechanism_transition"].update(
                enforcement="disabled",
                minimum_new_mechanisms=0,
                suggested_new_mechanisms=0,
                instruction="No mechanism novelty quota.",
            )
        if not self.semantic_memory:
            prompt.pop("research_memory")
            prompt["context_note"] = (
                "Semantic trajectory memory is disabled for this controlled ablation."
            )
        token_budget = _llm_prompt_token_budget("agent.planner")
        compact_history = _compact_history(
            history if self.semantic_memory else [],
            limit=min(DETAILED_HISTORY_ROUNDS, max(0, current_round - 1)),
            model_sota=model_sota,
        )
        remaining = max(
            0, token_budget - _message_token_count(_planner_messages(prompt))
        )
        (prompt["recent_history"], _) = _pack_json_items(
            compact_history, remaining, newest_first=True
        )
        prompt = _shrink_to_message_budget(
            prompt, _planner_messages, token_budget, [("recent_history",)]
        )
        return prompt

    def plan(
        self,
        specification: dict[str, Any],
        history: list[dict[str, Any]],
        *,
        plans_per_round: int,
        current_round: int,
        max_rounds: int,
        model_sota: dict[str, Any],
        model_retained_factors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = self.build_request(
            specification,
            history,
            plans_per_round=plans_per_round,
            current_round=current_round,
            max_rounds=max_rounds,
            model_sota=model_sota,
            model_retained_factors=model_retained_factors,
        )
        transition = prompt["mechanism_transition"]
        previous_mechanisms = set(transition["previous_round_mechanisms"])
        minimum_new = transition["minimum_new_mechanisms"]
        suggested_new = transition["suggested_new_mechanisms"]
        (validated, raw) = call_validated_json(
            _planner_messages(prompt),
            call_site="agent.planner",
            metadata={
                "round": current_round,
                "max_rounds": max_rounds,
                "module": "planner",
            },
            validator=lambda payload: validate_plan_payload(
                payload,
                plans_per_round=plans_per_round,
                feasible_mechanism_ids={
                    row["id"]
                    for row in prompt["research_contract"]["mechanism_ontology"]
                },
                previous_mechanisms=previous_mechanisms,
                min_new_mechanisms=minimum_new,
                complementary=self.complementary,
            ),
        )
        mechanisms = {plan["mechanism_id"] for plan in validated["research_plans"]}
        validated["planner_transition"] = {
            "enforcement": transition["enforcement"],
            "minimum_new_mechanisms": minimum_new if previous_mechanisms else 0,
            "suggested_new_mechanisms": suggested_new if previous_mechanisms else 0,
            "actual_new_mechanisms": sorted(mechanisms - previous_mechanisms),
            "repeated_mechanisms": sorted(mechanisms & previous_mechanisms),
            "suggestion_satisfied": not previous_mechanisms
            or len(mechanisms - previous_mechanisms) >= suggested_new,
        }
        validated["raw_response"] = raw
        return validated


def validate_plan_payload(
    payload: dict[str, Any],
    *,
    plans_per_round: int,
    feasible_mechanism_ids: set[str] | None = None,
    previous_mechanisms: set[str] | None = None,
    min_new_mechanisms: int = 2,
    complementary: bool = True,
) -> dict[str, Any]:
    if type(min_new_mechanisms) is not int or min_new_mechanisms < 0:
        raise ValueError("min_new_mechanisms must be a nonnegative integer")
    plans = payload.get("research_plans")
    if not isinstance(plans, list) or len(plans) != plans_per_round:
        raise ValueError(f"research_plans must contain exactly {plans_per_round} items")
    validated = []
    plan_ids: set[str] = set()
    mechanisms: set[str] = set()
    themes: set[str] = set()
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict):
            raise ValueError(f"research_plans[{index}] must be an object")
        plan_id = str(plan.get("plan_id") or "")
        expected_id = f"p{index + 1}"
        if plan_id != expected_id or plan_id in plan_ids:
            raise ValueError(f"research_plans[{index}].plan_id must be {expected_id!r}")
        mechanism_id = str(plan.get("mechanism_id") or "")
        mechanism = MECHANISM_REGISTRY.get(mechanism_id)
        if mechanism is None or (complementary and mechanism_id in mechanisms):
            raise ValueError(
                f"research_plans[{index}].mechanism_id is invalid or duplicated"
            )
        if (
            feasible_mechanism_ids is not None
            and mechanism_id not in feasible_mechanism_ids
        ):
            raise ValueError(
                f"research_plans[{index}].mechanism_id is infeasible for the current feature state"
            )
        theme = mechanism.theme
        for field in ("event", "economic_mechanism", "conditioning"):
            if not isinstance(plan.get(field), str) or not str(plan[field]).strip():
                raise ValueError(f"research_plans[{index}].{field} must be non-empty")
        if plan.get("horizon") not in PLAN_HORIZONS:
            raise ValueError(f"research_plans[{index}].horizon is invalid")
        if plan.get("expected_direction") not in PLAN_DIRECTIONS:
            raise ValueError(f"research_plans[{index}].expected_direction is invalid")
        if plan.get("relation_to_base") not in PLAN_RELATIONS:
            raise ValueError(f"research_plans[{index}].relation_to_base is invalid")
        plan_ids.add(plan_id)
        mechanisms.add(mechanism_id)
        themes.add(theme)
        validated.append(
            {
                "plan_id": plan_id,
                "mechanism_id": mechanism_id,
                "theme": theme,
                "event": str(plan["event"]).strip(),
                "economic_mechanism": str(plan["economic_mechanism"]).strip(),
                "horizon": plan["horizon"],
                "conditioning": str(plan["conditioning"]).strip(),
                "expected_direction": plan["expected_direction"],
                "relation_to_base": plan["relation_to_base"],
            }
        )
    if complementary and len(themes) < min(3, plans_per_round):
        raise ValueError("research_plans must cover at least three ontology themes")
    if previous_mechanisms and len(mechanisms - previous_mechanisms) < min(
        min_new_mechanisms, plans_per_round
    ):
        required = min(min_new_mechanisms, plans_per_round)
        new = mechanisms - previous_mechanisms
        alternatives = (
            (
                feasible_mechanism_ids
                if feasible_mechanism_ids is not None
                else set(MECHANISM_REGISTRY)
            )
            - previous_mechanisms
            - mechanisms
        )
        raise ValueError(
            f"research_plans must introduce at least {required} mechanisms beyond the previous round. Previous IDs: {sorted(previous_mechanisms)}. Repeated IDs: {sorted(mechanisms & previous_mechanisms)}. New IDs: {sorted(new)} ({len(new)}/{required}). Replace at least {required - len(new)} repeated IDs and rewrite their event, economic_mechanism, conditioning and direction consistently. Available new IDs by theme: {json.dumps({name: MECHANISM_REGISTRY[name].theme for name in sorted(alternatives)})}. Return the complete {plans_per_round}-plan set with distinct IDs and at least {min(3, plans_per_round)} themes. Changing prose, order or plan_id without changing mechanism_id does not satisfy this requirement."
        )
    return {"research_plans": validated}


def _latest_mechanisms(history: list[dict[str, Any]]) -> set[str]:
    if not history:
        return set()
    return {
        str(plan["mechanism_id"])
        for plan in history[-1].get("research_plans", [])
        if isinstance(plan, dict) and plan.get("mechanism_id")
    }


def _planner_panel_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    fields = (
        "label",
        "label_meaning",
        "label_horizon_days",
        "active_markets",
        "research_universe",
        "date_range",
        "data_scope",
        "layers",
    )
    return {field: value[field] for field in fields if field in value}


def _planner_messages(prompt: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are the Planner in a quantitative research portfolio workflow. Output JSON only.",
        },
        {
            "role": "user",
            "content": json.dumps(prompt, ensure_ascii=False, default=str),
        },
    ]
