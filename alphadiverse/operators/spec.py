from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import re
from .registry import OPERATOR_REGISTRY, TRANSFORM_REGISTRY

FORBIDDEN_FEATURE_MARKERS = (
    "label_",
    "target_",
    "future_",
    "forward_",
    "lead_",
    "next_",
    "_fwd_",
)


@dataclass(frozen=True)
class FactorSpecError(ValueError):
    violations: tuple[str, ...]

    def __str__(self) -> str:
        return "; ".join(self.violations)


def validate_factor_specs(
    raw_specs: Any,
    *,
    available_features: dict[str, dict[str, Any]],
    allowed_layers: tuple[str, ...],
    existing_names: set[str] | None = None,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(raw_specs, list):
        raise FactorSpecError(("factor_specs must be a list",))
    if expected_count is not None and len(raw_specs) != expected_count:
        raise FactorSpecError(
            (f"factor_specs must contain exactly {expected_count} items",)
        )
    validated: list[dict[str, Any]] = []
    used_names = set(existing_names or ())
    violations: list[str] = []
    for index, raw_spec in enumerate(raw_specs):
        try:
            spec = validate_factor_spec(
                raw_spec,
                available_features=available_features,
                allowed_layers=allowed_layers,
                used_names=used_names,
            )
            used_names.add(spec["name"])
            validated.append(spec)
        except FactorSpecError as exc:
            violations.extend(
                (f"factor_specs[{index}].{item}" for item in exc.violations)
            )
    if violations:
        raise FactorSpecError(tuple(violations))
    return validated


def validate_factor_spec(
    raw_spec: Any,
    *,
    available_features: dict[str, dict[str, Any]],
    allowed_layers: tuple[str, ...],
    used_names: set[str],
) -> dict[str, Any]:
    if not isinstance(raw_spec, dict):
        raise FactorSpecError(("must be an object",))
    violations: list[str] = []
    plan_id = _required_string(raw_spec.get("plan_id"), "plan_id", violations)
    name = _required_string(raw_spec.get("name"), "name", violations)
    if name and (not re.fullmatch("llm_[a-z0-9_]+", name) or len(name) > 96):
        violations.append("name must match llm_[a-z0-9_]+ and be at most 96 characters")
    if name in used_names:
        violations.append("name is duplicated within the action or existing panel")
    operator = _required_string(raw_spec.get("operator"), "operator", violations)
    definition = OPERATOR_REGISTRY.get(operator)
    if definition is None:
        violations.append(f"operator {operator!r} is not allowed")
    input_features = raw_spec.get("input_features")
    if not isinstance(input_features, list) or not all(
        (isinstance(value, str) and value for value in input_features)
    ):
        violations.append("input_features must be a list of non-empty names")
        input_features = []
    missing = [
        feature for feature in input_features if feature not in available_features
    ]
    if missing:
        violations.append(f"input_features are unavailable: {missing}")
    unsafe = [
        feature for feature in input_features if not is_causal_feature_name(feature)
    ]
    if unsafe:
        violations.append(f"input_features fail the future-data guard: {unsafe}")
    if definition is not None and (
        not definition.min_inputs <= len(input_features) <= definition.max_inputs
    ):
        violations.append(
            f"operator {operator} requires {definition.min_inputs}..{definition.max_inputs} inputs"
        )
    input_transforms = raw_spec.get("input_transforms")
    if not isinstance(input_transforms, list) or len(input_transforms) != len(
        input_features
    ):
        violations.append(
            "input_transforms must contain exactly one transform per input"
        )
        input_transforms = []
    else:
        invalid_transforms = [
            value
            for value in input_transforms
            if not isinstance(value, str) or value not in TRANSFORM_REGISTRY
        ]
        if invalid_transforms:
            violations.append(f"input_transforms are not allowed: {invalid_transforms}")
        elif len(set(zip(input_features, input_transforms))) != len(input_features):
            violations.append(
                "exact input feature/transform pairs must not be duplicated"
            )
    source_layers = {
        str(available_features[feature].get("layer"))
        for feature in input_features
        if feature in available_features
    }
    inferred_layer = (
        "cross_layer" if len(source_layers) > 1 else next(iter(source_layers), "")
    )
    if not inferred_layer:
        violations.append("layer cannot be derived from the selected input features")
    elif inferred_layer not in allowed_layers:
        violations.append(f"derived layer {inferred_layer!r} is not allowed")
    window = raw_spec.get("window")
    transform_needs_window = any(
        (
            isinstance(transform, str)
            and transform in TRANSFORM_REGISTRY
            and TRANSFORM_REGISTRY[transform].uses_window
            for transform in input_transforms
        )
    )
    needs_window = bool(definition and definition.uses_window) or transform_needs_window
    minimum_window = max(
        [definition.min_window if definition and definition.uses_window else 1]
        + [
            TRANSFORM_REGISTRY[value].min_window
            for value in input_transforms
            if isinstance(value, str)
            and value in TRANSFORM_REGISTRY
            and TRANSFORM_REGISTRY[value].uses_window
        ]
    )
    if needs_window:
        if (
            isinstance(window, bool)
            or not isinstance(window, int)
            or (not minimum_window <= window <= 120)
        ):
            violations.append(f"window must be an integer in [{minimum_window}, 120]")
    elif window is not None:
        violations.append(
            "window must be null when the operator and transforms do not use it"
        )
    quantile = raw_spec.get("quantile")
    if definition and definition.uses_quantile:
        if (
            isinstance(quantile, bool)
            or not isinstance(quantile, (int, float))
            or (not math.isfinite(float(quantile)))
            or (not 0.01 <= float(quantile) <= 0.99)
        ):
            violations.append("quantile must be finite and in [0.01, 0.99]")
    elif quantile is not None:
        violations.append("quantile must be null for this operator")
    weights = raw_spec.get("weights")
    if definition and definition.uses_weights:
        if (
            not isinstance(weights, list)
            or len(weights) != len(input_features)
            or any(
                (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or (not math.isfinite(float(value)))
                    for value in weights
                )
            )
        ):
            violations.append("weights must contain one finite number per input")
        elif any((abs(float(value)) == 0.0 for value in weights)):
            violations.append("weights must be non-zero; omit zero-weight inputs")
    elif weights not in (None, []):
        violations.append("weights must be omitted for this operator")
    expected_direction = raw_spec.get("expected_direction")
    if not isinstance(expected_direction, str) or expected_direction not in {
        "positive",
        "negative",
        "nonlinear",
        "unknown",
    }:
        violations.append("expected_direction is invalid")
    rationale = _required_string(raw_spec.get("rationale"), "rationale", violations)
    if violations:
        raise FactorSpecError(tuple(violations))
    canonical = {
        "plan_id": plan_id,
        "name": name,
        "layer": inferred_layer,
        "operator": operator,
        "input_features": list(input_features),
        "input_transforms": list(input_transforms),
        "weights": [float(value) for value in weights] if weights is not None else [],
        "window": int(window) if window is not None else None,
        "quantile": float(quantile) if quantile is not None else None,
        "expected_direction": expected_direction,
        "rationale": rationale,
    }
    canonical["formula"] = render_formula(canonical)
    return canonical


def render_formula(spec: dict[str, Any]) -> str:
    window = spec.get("window")
    components = [
        _render_component(feature, transform, window)
        for (feature, transform) in zip(
            spec["input_features"], spec["input_transforms"], strict=True
        )
    ]
    operator = spec["operator"]
    if operator == "linear_combo":
        body = " + ".join(
            (
                f"{weight:g}*cs_zscore({component})"
                for (weight, component) in zip(spec["weights"], components, strict=True)
            )
        )
    elif operator == "ts_quantile":
        body = f"ts_quantile({components[0]}, window={window}, quantile={spec['quantile']:g})"
    elif operator.startswith("ts_"):
        body = f"{operator}({', '.join(components)}, window={window})"
    else:
        body = f"{operator}({', '.join(components)})"
    return f"cs_zscore({body})"


def _render_component(feature: str, transform: str, window: int | None) -> str:
    if transform == "identity":
        return feature
    if transform in TRANSFORM_REGISTRY and TRANSFORM_REGISTRY[transform].uses_window:
        return f"{transform}({feature}, window={window})"
    return f"{transform}({feature})"


def _required_string(value: Any, name: str, violations: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        violations.append(f"{name} must be a non-empty string")
        return ""
    return value.strip()


def is_causal_feature_name(name: str) -> bool:
    normalized = str(name).strip().lower()
    return bool(normalized) and (
        not any((marker in normalized for marker in FORBIDDEN_FEATURE_MARKERS))
    )
