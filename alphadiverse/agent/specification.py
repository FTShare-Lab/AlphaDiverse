from __future__ import annotations
import hashlib
import math
from collections.abc import Collection
from typing import Any
import pandas as pd
from ..config import AppConfig
from ..experiment import (
    RunRequest,
    active_market_names,
    label_mode_market,
    market_benchmark_map,
)
from ..factors.daily import base_factor_formula, feature_columns
from ..operators import (
    is_causal_feature_name,
    operator_capability_summary,
    operator_prompt_catalog,
    transform_prompt_catalog,
)

RAW_DAILY_INPUT_FEATURES = ("open", "high", "low", "close", "volume", "turnover")
FEATURE_NAME_SEMANTICS = {
    "volume": "traded share quantity (shares)",
    "turnover": "the token 'turnover' in any feature name means traded amount/value in CNY (成交额), never turnover rate; '*_turnover_share' means a share of traded amount",
}


class SpecificationModule:
    def build(
        self,
        cfg: AppConfig,
        request: RunRequest,
        panel: pd.DataFrame,
        layer_columns: dict[str, list[str]],
        label: str,
        base_model_features: Collection[str] = (),
    ) -> dict[str, Any]:
        columns = []
        base_factor_names = list(
            dict.fromkeys((str(name) for name in base_model_features))
        )
        available_input_columns = list(
            dict.fromkeys([*RAW_DAILY_INPUT_FEATURES, *feature_columns(panel)])
        )
        for col in available_input_columns:
            if (
                col not in panel
                or not pd.api.types.is_numeric_dtype(panel[col])
                or (not is_causal_feature_name(col))
            ):
                continue
            layer = _infer_layer(col, layer_columns)
            valid = panel[col].notna()
            feature = {
                "name": col,
                "layer": layer,
                "dtype": str(panel[col].dtype),
                "coverage": float(valid.mean()) if len(panel) else 0.0,
                "missing_rate": float(1.0 - valid.mean()) if len(panel) else 1.0,
                "nunique": int(panel[col].nunique(dropna=True)),
            }
            description = _available_feature_description(col)
            if description:
                feature["description"] = description
            columns.append(feature)
        columns = sorted(columns, key=lambda row: (row["layer"], row["name"]))
        (columns, feature_policy) = _mask_available_features(
            columns,
            mask_pct=cfg.agent.available_feature_mask_pct,
            seed=cfg.agent.available_feature_mask_seed,
        )
        panel_contract = _panel_contract(cfg, request, label)
        fold_count = int(cfg.search_validation.fold_count)
        required_folds = int(cfg.search_validation.min_improved_folds)
        if fold_count == 1:
            validation_mode = "single_time_fold"
            model_retain = "Improve the sole inner time-fold SOTA versus the previous SOTA; the configured 1/1 fold must improve."
        else:
            validation_mode = "rolling_time_multifold"
            model_retain = f"Improve the trading-day-weighted inner-fold aggregate SOTA versus the previous SOTA and improve at least {required_folds}/{fold_count} corresponding folds."
        return {
            "panel_contract": panel_contract,
            "feature_name_semantics": FEATURE_NAME_SEMANTICS,
            "base_factors": [
                {"name": name, "formula": base_factor_formula(name)}
                for name in base_factor_names
            ],
            "operator_catalog": operator_prompt_catalog(),
            "operator_capabilities": operator_capability_summary(),
            "input_transforms": transform_prompt_catalog(),
            "allowed_factor_layers": feature_policy["allowed_factor_layers"],
            "validation_contract": {
                "target": label,
                "evidence_scope": "inner_only",
                "validation_mode": validation_mode,
                "fold_count": fold_count,
                "required_improved_folds": required_folds,
                "dedup": "if abs(mean daily factor correlation) > threshold, keep the factor with better abs RankIC",
                "model_retain": model_retain,
            },
            "causality_contract": {
                "feature_policy": "Only close-t observables from the feature cache may enter a factor.",
                "operator_policy": "The registry exposes only contemporaneous or non-negative-lag operations.",
                "execution_policy": "Model output is a declarative spec; arbitrary generated code is never executed.",
            },
            "available_features": columns,
            "available_feature_policy": feature_policy,
        }


def _mask_available_features(
    features: list[dict[str, Any]], *, mask_pct: float, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pct = float(mask_pct)
    if not math.isfinite(pct) or not 0.0 <= pct <= 100.0:
        raise ValueError("available_feature_mask_pct must be between 0 and 100")
    protected_names = set(RAW_DAILY_INPUT_FEATURES)
    kept: list[dict[str, Any]] = []
    counts_by_layer: dict[str, dict[str, int]] = {}
    layers = sorted(
        {str(row.get("layer") or "daily") for row in features}, key=_layer_sort_key
    )
    for layer in layers:
        rows = [row for row in features if str(row.get("layer") or "daily") == layer]
        protected = [row for row in rows if row.get("name") in protected_names]
        optional = [row for row in rows if row.get("name") not in protected_names]
        mask_count = (
            len(optional) if pct == 100.0 else math.floor(len(optional) * pct / 100.0)
        )
        ranked_optional = sorted(
            optional,
            key=lambda row: (
                _feature_mask_rank(seed, layer, str(row.get("name") or "")),
                str(row.get("name") or ""),
            ),
        )
        available = [*protected, *ranked_optional[mask_count:]]
        kept.extend(available)
        counts_by_layer[layer] = {
            "before": len(rows),
            "protected": len(protected),
            "masked": mask_count,
            "available": len(available),
        }
    kept = sorted(
        kept, key=lambda row: (_layer_sort_key(str(row["layer"])), row["name"])
    )
    available_source_layers = [
        layer
        for layer in ("daily", "minute")
        if any((row.get("layer") == layer for row in kept))
    ]
    allowed_factor_layers = list(available_source_layers)
    if len(available_source_layers) >= 2:
        allowed_factor_layers.append("cross_layer")
    protected_features = [
        name
        for name in RAW_DAILY_INPUT_FEATURES
        if any((row.get("name") == name for row in kept))
    ]
    policy = {
        "mask_pct": pct,
        "seed": int(seed),
        "method": "stable_sha256_stratified_by_layer",
        "scope": "LLM-visible and Implementation-allowed features only; feature cache and fixed base model are unchanged",
        "protected_features": protected_features,
        "available_source_layers": available_source_layers,
        "allowed_factor_layers": allowed_factor_layers,
        "before_count": len(features),
        "masked_count": len(features) - len(kept),
        "available_count": len(kept),
        "counts_by_layer": counts_by_layer,
    }
    return (kept, policy)


def _feature_mask_rank(seed: int, layer: str, name: str) -> str:
    return hashlib.sha256(
        f"{int(seed)}\x00{layer}\x00{name}".encode("utf-8")
    ).hexdigest()


def _available_feature_description(name: str) -> str:
    raw_descriptions = {
        "open": "daily open price (CNY per share)",
        "high": "daily high price (CNY per share)",
        "low": "daily low price (CNY per share)",
        "close": "daily close price (CNY per share)",
        "volume": "daily traded share quantity (shares)",
        "turnover": "daily traded amount/value (CNY, 成交额), not turnover rate",
    }
    return raw_descriptions.get(name, "")


def _layer_sort_key(layer: str) -> tuple[int, str]:
    order = {"daily": 0, "minute": 1}
    return (order.get(layer, 3), layer)


def _panel_contract(cfg: AppConfig, request: RunRequest, label: str) -> dict[str, Any]:
    common = {
        "grain": "daily stock-date panel",
        "required_keys": ["symbol", "date"],
        "label": label,
        "label_meaning": _label_meaning(cfg),
        "label_mode": cfg.evaluation.label_mode,
        "label_return_mode": cfg.evaluation.label_return_mode,
    }
    search_is_inner = cfg.search_validation.protocol == "inner_multifold"
    visible_date_range = (
        {
            "start": cfg.search_validation.inner_start,
            "end": cfg.search_validation.inner_end,
        }
        if search_is_inner
        else {"start": request.start, "end": request.end}
    )
    return {
        **common,
        "label_horizon_days": cfg.evaluation.label_horizon_days,
        "active_markets": active_market_names(cfg),
        "research_universe": "union of the active market membership columns; the same markets are used in model-SOTA aggregation",
        "market_benchmarks": market_benchmark_map(cfg),
        "comparison_rule": "Compare model/factor scores only within the same label horizon, label mode, and active-market scenario; multi-day forward labels overlap across adjacent factor dates.",
        "date_range": visible_date_range,
        "data_scope": "inner_search_only" if search_is_inner else "legacy_full_request",
        "layers": list(request.layers),
    }


def _label_meaning(cfg: AppConfig) -> str:
    horizon = int(cfg.evaluation.label_horizon_days)
    if cfg.evaluation.label_return_mode == "o2o":
        base = f"At factor date t, use data available after close t and predict the signed stock return from open(t+1) to open(t+{horizon + 1}) over {horizon} trading day(s)"
        if cfg.evaluation.label_mode == "raw":
            return f"{base}; this is raw stock return, not mathematical abs(return)."
        market = label_mode_market(cfg.evaluation.label_mode)
        return f"{base}, minus the {market} benchmark or equal-weight market return for the same O2O horizon."
    if cfg.evaluation.label_mode == "raw":
        return f"Signed raw future stock return over {horizon} trading day(s)."
    market = label_mode_market(cfg.evaluation.label_mode)
    return f"Future stock return minus the {market} benchmark or market return over {horizon} trading day(s)."


def _infer_layer(col: str, layer_columns: dict[str, list[str]]) -> str:
    for layer, cols in layer_columns.items():
        if col in cols:
            return layer
    if col.startswith(
        (
            "minute_",
            "intraday_",
            "first30_",
            "last30_",
            "first_half_",
            "second_half_",
            "first_quarter_",
            "last_quarter_",
            "high_time_",
            "low_time_",
            "close_vwap_",
            "volume_u_shape",
            "ret_second_",
            "ret_last_",
        )
    ):
        return "minute"
    return "daily"
