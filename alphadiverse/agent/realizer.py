from __future__ import annotations
from typing import Any
import json
import re
from ..operators import transform_prompt_catalog
from .implementation import ImplementationModule
from .llm_json import call_validated_json

REALIZER_CONTRACT_VERSION = "system_factor_ids_v2"
REALIZER_SPEC_FIELDS = frozenset(
    {
        "plan_id",
        "operator",
        "input_features",
        "input_transforms",
        "weights",
        "window",
        "quantile",
        "expected_direction",
        "rationale",
    }
)


def allocate_factor_names(
    plans, *, current_round, factor_specs_per_plan, existing_names
):
    if (
        isinstance(current_round, bool)
        or not isinstance(current_round, int)
        or current_round < 1
    ):
        raise ValueError("current_round must be a positive integer")
    if (
        isinstance(factor_specs_per_plan, bool)
        or not isinstance(factor_specs_per_plan, int)
        or factor_specs_per_plan < 1
    ):
        raise ValueError("factor_specs_per_plan must be a positive integer")
    ids = [plan["plan_id"] for plan in plans]
    if len(ids) != len(set(ids)) or any(
        (not re.fullmatch("p[1-9][0-9]*", p) for p in ids)
    ):
        raise ValueError("Factor allocation requires unique pN plan IDs")
    allocated = {
        p: [
            f"llm_r{current_round:04d}_{p}_s{i:02d}"
            for i in range(1, factor_specs_per_plan + 1)
        ]
        for p in ids
    }
    if any((name in existing_names for names in allocated.values() for name in names)):
        raise ValueError("System factor ID collision: round slots are already occupied")
    return allocated


def bind_factor_names(raw_specs, assigned_names):
    if not isinstance(raw_specs, list) or len(raw_specs) != len(assigned_names):
        raise ValueError("factor_specs must match the number of allocated slots")
    result = []
    for spec, name in zip(raw_specs, assigned_names, strict=True):
        if not isinstance(spec, dict):
            raise ValueError("factor spec must be an object")
        unknown = set(spec) - REALIZER_SPEC_FIELDS
        if unknown:
            raise ValueError(
                f"Model must omit system-owned or unknown fields: {sorted(unknown)}"
            )
        result.append({**spec, "name": name})
    return result


class RealizerModule:
    def realize(
        self,
        plans: list[dict[str, Any]],
        retrieval: list[dict[str, Any]],
        specification: dict[str, Any],
        *,
        factor_specs_per_plan: int,
        current_round: int,
        max_rounds: int,
        existing_factor_names: set[str],
    ) -> dict[str, Any]:
        allocated = allocate_factor_names(
            plans,
            current_round=current_round,
            factor_specs_per_plan=factor_specs_per_plan,
            existing_names=existing_factor_names,
        )
        prompt = _realizer_prompt(
            plans, retrieval, specification, factor_specs_per_plan
        )
        (structured, raw) = call_validated_json(
            _realizer_messages(prompt),
            call_site="agent.realizer",
            metadata={
                "round": current_round,
                "max_rounds": max_rounds,
                "module": "realizer",
                "realizer_contract_version": REALIZER_CONTRACT_VERSION,
            },
            validator=lambda payload: _validate_realizer_structure(
                payload, plans=plans, factor_specs_per_plan=factor_specs_per_plan
            ),
        )
        implementation = ImplementationModule(specification)
        retrieval_by_plan = {row["plan_id"]: row for row in retrieval}
        used_names = set(existing_factor_names)
        used_formulas: set[str] = set()
        realizations = []
        repairs = []
        for plan, realization in zip(plans, structured["realizations"], strict=True):
            plan_id = plan["plan_id"]
            retrieval_row = retrieval_by_plan[plan_id]
            try:
                specs = _validate_plan_specs(
                    realization["factor_specs"],
                    plan=plan,
                    retrieval=retrieval_row,
                    factor_specs_per_plan=factor_specs_per_plan,
                    implementation=implementation,
                    used_names=used_names,
                    used_formulas=used_formulas,
                    assigned_names=allocated[plan_id],
                )
            except ValueError as exc:
                violation = str(exc)
                specs = self._repair_plan(
                    plan=plan,
                    violation=violation,
                    retrieval=retrieval_row,
                    specification=specification,
                    factor_specs_per_plan=factor_specs_per_plan,
                    implementation=implementation,
                    used_names=used_names,
                    used_formulas=used_formulas,
                    current_round=current_round,
                    max_rounds=max_rounds,
                    assigned_names=allocated[plan_id],
                    invalid_specs=realization["factor_specs"],
                )
                repairs.append({"plan_id": plan_id, "violation": violation})
            used_names.update((spec["name"] for spec in specs))
            used_formulas.update((spec["formula"] for spec in specs))
            realizations.append({"plan_id": plan_id, "factor_specs": specs})
        return {
            "realizations": realizations,
            "plan_repairs": repairs,
            "raw_response": raw,
            "realizer_contract_version": REALIZER_CONTRACT_VERSION,
            "factor_id_allocation": allocated,
        }

    def _repair_plan(
        self,
        *,
        plan: dict[str, Any],
        violation: str,
        retrieval: dict[str, Any],
        specification: dict[str, Any],
        factor_specs_per_plan: int,
        implementation: ImplementationModule,
        used_names: set[str],
        used_formulas: set[str],
        current_round: int,
        max_rounds: int,
        assigned_names: list[str],
        invalid_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prompt = realizer_repair_prompt(
            plan,
            retrieval,
            specification,
            factor_specs_per_plan,
            violation=violation,
            forbidden_formulas=used_formulas,
            invalid_specs=invalid_specs,
        )

        def validate(payload: dict[str, Any]) -> dict[str, Any]:
            structured = _validate_realizer_structure(
                payload, plans=[plan], factor_specs_per_plan=factor_specs_per_plan
            )
            specs = _validate_plan_specs(
                structured["realizations"][0]["factor_specs"],
                plan=plan,
                retrieval=retrieval,
                factor_specs_per_plan=factor_specs_per_plan,
                implementation=implementation,
                used_names=used_names,
                used_formulas=used_formulas,
                assigned_names=assigned_names,
            )
            return {"factor_specs": specs}

        (validated, _) = call_validated_json(
            _realizer_messages(prompt),
            call_site="agent.realizer",
            metadata={
                "round": current_round,
                "max_rounds": max_rounds,
                "module": "realizer_plan_repair",
                "repair_plan_id": plan["plan_id"],
                "realizer_contract_version": REALIZER_CONTRACT_VERSION,
            },
            validator=validate,
        )
        return validated["factor_specs"]


def _validate_realizer_structure(
    payload: dict[str, Any], *, plans: list[dict[str, Any]], factor_specs_per_plan: int
) -> dict[str, Any]:
    realizations = payload.get("realizations")
    if not isinstance(realizations, list) or len(realizations) != len(plans):
        raise ValueError(f"realizations must contain exactly {len(plans)} items")
    structured = []
    for index, (plan, realization) in enumerate(zip(plans, realizations, strict=True)):
        if not isinstance(realization, dict):
            raise ValueError(f"realizations[{index}] must be an object")
        plan_id = str(realization.get("plan_id") or "")
        if plan_id != plan["plan_id"]:
            raise ValueError(
                f"realizations[{index}].plan_id must be {plan['plan_id']!r}"
            )
        specs = realization.get("factor_specs")
        if not isinstance(specs, list) or len(specs) != factor_specs_per_plan:
            raise ValueError(
                f"realizations[{index}].factor_specs must contain exactly {factor_specs_per_plan} items"
            )
        if not all(
            (
                isinstance(spec, dict) and spec.get("plan_id") == plan_id
                for spec in specs
            )
        ):
            raise ValueError(
                f"all factor specs in realizations[{index}] must use plan_id {plan_id!r}"
            )
        structured.append({"plan_id": plan_id, "factor_specs": specs})
    return {"realizations": structured}


def _validate_plan_specs(
    raw_specs: Any,
    *,
    plan: dict[str, Any],
    retrieval: dict[str, Any],
    factor_specs_per_plan: int,
    implementation: ImplementationModule,
    used_names: set[str],
    used_formulas: set[str],
    assigned_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(raw_specs, list):
        raise ValueError("factor_specs must be a list")
    if assigned_names is not None:
        raw_specs = bind_factor_names(raw_specs, assigned_names)
    allowed_features = {row["name"] for row in retrieval.get("features", [])}
    allowed_operators = {row["name"] for row in retrieval.get("operators", [])}
    for index, spec in enumerate(raw_specs):
        if not isinstance(spec, dict) or spec.get("plan_id") != plan["plan_id"]:
            raise ValueError(
                f"factor_specs[{index}].plan_id must be {plan['plan_id']!r}"
            )
        requested_features = spec.get("input_features", [])
        if (
            not isinstance(requested_features, list)
            or not all((isinstance(feature, str) for feature in requested_features))
            or (not set(requested_features).issubset(allowed_features))
        ):
            raise ValueError(
                f"factor_specs[{index}] uses an unretrieved feature; allowed: {sorted(allowed_features)}"
            )
        operator = spec.get("operator")
        if not isinstance(operator, str) or operator not in allowed_operators:
            raise ValueError(
                f"factor_specs[{index}] uses an unretrieved operator; allowed: {sorted(allowed_operators)}"
            )
    canonical = implementation.validate_specs(
        raw_specs, existing_names=used_names, expected_count=factor_specs_per_plan
    )
    enriched = [
        {**spec, "mechanism_id": plan["mechanism_id"], "theme": plan["theme"]}
        for spec in canonical
    ]
    signatures = {
        (
            spec["operator"],
            tuple(spec["input_features"]),
            tuple(spec["input_transforms"]),
            tuple(spec["weights"]),
            spec["quantile"],
        )
        for spec in enriched
    }
    if len(signatures) != len(enriched):
        raise ValueError("factor specs differ only by name or window")
    formulas = [spec["formula"] for spec in enriched]
    if len(set(formulas)) != len(formulas) or set(formulas) & used_formulas:
        raise ValueError(
            "factor specs duplicate an executable formula in the current portfolio"
        )
    return enriched


def _realizer_prompt(
    plans: list[dict[str, Any]],
    retrieval: list[dict[str, Any]],
    specification: dict[str, Any],
    factor_specs_per_plan: int,
) -> dict[str, Any]:
    return {
        "role": "factor_spec_realizer",
        "contract_version": REALIZER_CONTRACT_VERSION,
        "instruction": f"Realize each research plan as exactly {factor_specs_per_plan} executable factor specs. Preserve every plan's economic mechanism; specs within a plan must differ in feature, operator, transform, or non-zero weights, not only in name or window. Use only the feature and operator cards listed inside that plan's retrieved_capabilities entry. The system assigns factor identity by round, plan and list position; do not output name, factor_id, layer or formula. Input transforms must contain exactly one allowed transform per input. Include weights only for linear_combo, quantile only for ts_quantile, and window only when the operator or a windowed input transform requires it. Omit unused conditional fields. Omit layer and formula because the compiler derives both. The same feature may appear twice only when its transforms differ, for example a stock return and its same-date cross-sectional mean. Do not replace or fall back to unrequested features. Output JSON only.",
        "research_plans": plans,
        "retrieved_capabilities": retrieval,
        "allowed_input_transforms": transform_prompt_catalog(),
        "factor_contract": _factor_contract(specification),
        "output_schema": {
            "realizations": [
                {
                    "plan_id": plan["plan_id"],
                    "factor_specs": [_factor_schema(plan["plan_id"])],
                }
                for plan in plans
            ]
        },
        "conditional_operator_fields": _conditional_operator_fields(),
    }


def _factor_contract(specification: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_normalization": "Every implemented factor is cross-sectionally z-scored by date.",
        "factor_name_semantics": specification.get("feature_name_semantics", {}),
        "causality": specification.get("causality_contract", {}),
    }


def realizer_repair_prompt(
    plan,
    retrieval,
    specification,
    factor_specs_per_plan,
    *,
    violation,
    forbidden_formulas,
    invalid_specs,
):
    prompt = _realizer_prompt([plan], [retrieval], specification, factor_specs_per_plan)
    prompt["instruction"] += (
        f" This is a repair of one plan. Return exactly one realizations entry for {plan['plan_id']}, preserving its economic mechanism and fixing the violation. Use full specs with actual feature/operator names. System factor slots remain unchanged."
    )
    prompt["violations"] = violation
    prompt["forbidden_formulas"] = sorted(forbidden_formulas)
    prompt["invalid_factor_specs"] = invalid_specs
    return prompt


def _factor_schema(plan_id: str) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "operator": "one operator retrieved for this plan",
        "input_features": ["features retrieved for this plan"],
        "input_transforms": ["one allowed transform per input"],
        "expected_direction": "positive|negative|nonlinear|unknown",
        "rationale": "why this implementation realizes the plan",
    }


def _conditional_operator_fields() -> dict[str, str]:
    return {
        "linear_combo": "include weights with one non-zero finite number per input",
        "windowed_operator_or_transform": "include integer window in the allowed range",
        "ts_quantile": "include quantile in [0.01, 0.99]",
    }


def _realizer_messages(prompt: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are the Realizer in a quantitative research portfolio workflow. Output JSON only.",
        },
        {
            "role": "user",
            "content": json.dumps(prompt, ensure_ascii=False, default=str),
        },
    ]
