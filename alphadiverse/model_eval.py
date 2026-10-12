from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import hashlib
import shutil
import json
import math
import os
import warnings
import numpy as np
import pandas as pd
from .config import AppConfig
from .evaluation import daily_ic, summarize_ic
from .experiment import active_market_names, market_label_column, market_member_column
from .perf import model_eval_workers, model_threads
from .portfolio import (
    attach_portfolio_equity,
    daily_portfolio_summary,
    one_day_label,
    resolve_holding_period_days,
    staggered_bucket_returns,
)

SOTA_MARKET_AVG = "multi_market_avg"
SOTA_FALLBACK_MARKET = "ashare"
SOTA_SCORE_FIELDS = (
    "rank_ic_mean",
    "rank_icir",
    "ic_mean",
    "icir",
    "long_short_annual",
)
SOTA_SCORE_SCALES = {
    "rank_ic_mean": 0.05,
    "rank_icir": 0.35,
    "ic_mean": 0.03,
    "icir": 0.25,
    "long_short_annual": 0.3,
}


@dataclass(frozen=True)
class ModelValidationConfig:
    enabled: bool = True
    models: tuple[str, ...] = ("linear", "lightgbm")
    ridge_alphas: tuple[float, ...] = (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0)
    lightgbm_estimators: int = 500
    lightgbm_early_stopping_rounds: int = 50


def evaluation_panel(
    panel: pd.DataFrame, features: list[str], label: str
) -> pd.DataFrame:
    needed = {"date", "symbol", label, *features}
    try:
        needed.add(one_day_label(label))
    except ValueError:
        pass
    keep = [
        name in needed or (isinstance(name, str) and _forbidden_model_feature(name))
        for name in panel.columns
    ]
    return panel if all(keep) else panel.loc[:, keep]


def run_model_validation(
    panel: pd.DataFrame,
    label: str,
    features: list[str],
    cfg: AppConfig,
    output_dir: Path,
    run_name: str,
    new_factors: list[str] | None = None,
    previous_sota: dict[str, Any] | None = None,
    model_cfg: ModelValidationConfig = ModelValidationConfig(),
) -> dict[str, Any]:
    panel = evaluation_panel(panel, features, label)
    if cfg.search_validation.protocol == "inner_multifold":
        return _run_inner_multifold_validation(
            panel=panel,
            label=label,
            features=features,
            cfg=cfg,
            output_dir=output_dir,
            run_name=run_name,
            new_factors=new_factors,
            previous_sota=previous_sota,
            model_cfg=model_cfg,
        )
    return _run_single_split_model_validation(
        panel=panel,
        label=label,
        features=features,
        cfg=cfg,
        output_dir=output_dir,
        run_name=run_name,
        new_factors=new_factors,
        previous_sota=previous_sota,
        model_cfg=model_cfg,
    )


def _run_single_split_model_validation(
    panel: pd.DataFrame,
    label: str,
    features: list[str],
    cfg: AppConfig,
    output_dir: Path,
    run_name: str,
    new_factors: list[str] | None = None,
    previous_sota: dict[str, Any] | None = None,
    model_cfg: ModelValidationConfig = ModelValidationConfig(),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    features = _usable_features(panel, features, label)
    payload: dict[str, Any] = {
        "run_name": run_name,
        "model_config": asdict(model_cfg),
        "split_config": asdict(cfg.dates),
        "label": label,
        "feature_count": len(features),
        "features": features,
        "new_factors": new_factors or [],
        "previous_sota": _compact_sota(previous_sota),
        "status": "started",
    }
    (output_dir / "feature_set.json").write_text(_json_dumps(payload), encoding="utf-8")
    if not model_cfg.enabled:
        payload["status"] = "disabled"
        _write_model_outputs(output_dir, payload, [], [], [], [])
        return payload
    if not features:
        payload["status"] = "skipped"
        payload["reason"] = "no_usable_features"
        _write_model_outputs(output_dir, payload, [], [], [], [])
        return payload
    if label not in panel:
        payload["status"] = "skipped"
        payload["reason"] = f"missing_label:{label}"
        _write_model_outputs(output_dir, payload, [], [], [], [])
        return payload
    prepared = _prepare_model_frame(panel, features, label, cfg)
    split_counts = prepared["split"].value_counts().to_dict()
    payload["split_counts"] = {str(k): int(v) for (k, v) in split_counts.items()}
    if (
        split_counts.get("train", 0) < 100
        or split_counts.get("valid", 0) < 20
        or split_counts.get("test", 0) < 20
    ):
        payload["status"] = "skipped"
        payload["reason"] = "insufficient_split_rows"
        _write_model_outputs(output_dir, payload, [], [], [], [])
        return payload
    train = prepared[prepared["split"] == "train"]
    valid = prepared[prepared["split"] == "valid"]
    test = prepared[prepared["split"] == "test"]
    train_stats = _fit_preprocessor(train, features)
    artifact_dir = output_dir / "artifacts"
    matrices = {
        "train": _matrix(train, features, train_stats, standardize=True),
        "valid": _matrix(valid, features, train_stats, standardize=True),
        "test": _matrix(test, features, train_stats, standardize=True),
    }
    raw_matrices = {
        "train": _matrix(train, features, train_stats, standardize=False),
        "valid": _matrix(valid, features, train_stats, standardize=False),
        "test": _matrix(test, features, train_stats, standardize=False),
    }
    y = {
        "train": train[label].to_numpy(dtype="float64"),
        "valid": valid[label].to_numpy(dtype="float64"),
        "test": test[label].to_numpy(dtype="float64"),
    }
    frames = {"train": train, "valid": valid, "test": test}
    summary_rows: list[dict[str, Any]] = []
    daily_ic_rows: list[pd.DataFrame] = []
    daily_return_rows: list[pd.DataFrame] = []
    group_return_rows: list[pd.DataFrame] = []
    artifact_rows: list[dict[str, Any]] = []
    preprocessor_path = _write_preprocessor_artifact(
        train_stats, features, artifact_dir
    )
    artifact_rows.append(
        {"model": "all", "type": "preprocessor", "path": str(preprocessor_path)}
    )
    for model_name in model_cfg.models:
        if model_name == "linear":
            result = _fit_predict_linear(
                matrices, y, features, train_stats, model_cfg, artifact_dir
            )
        elif model_name == "lightgbm":
            result = _fit_predict_lightgbm(
                raw_matrices, y, features, train_stats, model_cfg, artifact_dir
            )
        else:
            result = {
                "model": model_name,
                "status": "skipped",
                "reason": "unsupported_model",
                "predictions": {},
            }
        artifact_rows.extend(result.get("artifacts", []))
        if result.get("status") != "ok":
            summary_rows.append(
                {
                    "run_name": run_name,
                    "model": model_name,
                    "split": "all",
                    "status": result.get("status", "skipped"),
                    "reason": result.get("reason", ""),
                }
            )
            continue
        (model_summaries, model_ic, model_returns, model_groups) = (
            _evaluate_model_result_predictions(
                frames=frames,
                predictions=result["predictions"],
                label=label,
                cfg=cfg,
                run_name=run_name,
                model_name=model_name,
                feature_count=len(features),
                training_detail=result.get("detail", {}),
                daily_return_panel=panel,
            )
        )
        summary_rows.extend(model_summaries)
        daily_ic_rows.extend(model_ic)
        daily_return_rows.extend(model_returns)
        group_return_rows.extend(model_groups)
    best_test = _best_sota_summary_row(summary_rows, "test", cfg)
    best_valid = _matching_split_row(summary_rows, best_test, "valid")
    improved = _is_model_improvement(best_test, previous_sota)
    sota_markets = _sota_markets(cfg)
    payload.update(
        {
            "status": "ok"
            if any((row.get("status") == "ok" for row in summary_rows))
            else "skipped",
            "best_valid": best_valid,
            "best_test": best_test,
            "selection_key": f"test multi-market average across {sota_markets}; score=mean(metric/scale) for {SOTA_SCORE_SCALES}, ranked by market_count, normalized score, averaged rank_ic, rank_icir, ic, icir, long_short_annual, sharpe, max_drawdown",
            "sota_markets": sota_markets,
            "sota_score_fields": list(SOTA_SCORE_FIELDS),
            "sota_score_scales": dict(SOTA_SCORE_SCALES),
            "sota_improved": improved,
            "sota": _sota_payload(best_valid, best_test, run_name, features),
            "artifacts": artifact_rows,
            "best_test_artifacts": _select_best_test_artifacts(
                best_test, artifact_rows
            ),
        }
    )
    _write_model_outputs(
        output_dir,
        payload,
        summary_rows,
        daily_ic_rows,
        daily_return_rows,
        group_return_rows,
    )
    return payload


def inner_search_panel(panel: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    protocol = cfg.search_validation
    if protocol.protocol != "inner_multifold":
        return panel
    if not protocol.inner_start or not protocol.inner_end:
        raise ValueError(
            "inner_multifold requires search_validation.inner_start and inner_end"
        )
    dates = pd.to_datetime(panel["date"].astype(str), errors="coerce")
    mask = (dates >= pd.Timestamp(protocol.inner_start)) & (
        dates <= pd.Timestamp(protocol.inner_end)
    )
    out = panel.loc[mask]
    lookahead = _label_lookahead_days(cfg)
    if lookahead > 0 and (not out.empty):
        eligible_dates = sorted(out["date"].astype(str).unique().tolist())
        if len(eligible_dates) <= lookahead:
            return out.iloc[0:0]
        cutoff = eligible_dates[-lookahead]
        out = out.loc[out["date"].astype(str) < cutoff]
    return out


def _label_lookahead_days(cfg: AppConfig) -> int:
    if not cfg.search_validation.purge_label_overlap:
        return 0
    horizon = max(0, int(cfg.evaluation.label_horizon_days))
    return horizon + 1 if cfg.evaluation.label_return_mode == "o2o" else horizon


def _run_inner_multifold_validation(
    panel: pd.DataFrame,
    label: str,
    features: list[str],
    cfg: AppConfig,
    output_dir: Path,
    run_name: str,
    new_factors: list[str] | None,
    previous_sota: dict[str, Any] | None,
    model_cfg: ModelValidationConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    protocol = cfg.search_validation
    _validate_search_protocol(cfg)
    inner = inner_search_panel(panel, cfg)
    features = _usable_features(inner, features, label)
    payload: dict[str, Any] = {
        "run_name": run_name,
        "protocol": protocol.protocol,
        "model_config": asdict(model_cfg),
        "search_config": asdict(protocol),
        "label": label,
        "feature_count": len(features),
        "features": features,
        "new_factors": new_factors or [],
        "previous_sota": _compact_sota(previous_sota),
        "status": "started",
        "agent_evidence_scope": "inner_only",
    }
    (output_dir / "feature_set.json").write_text(_json_dumps(payload), encoding="utf-8")
    if not model_cfg.enabled:
        payload["status"] = "disabled"
        _write_model_outputs(output_dir, payload, [], [], [], [])
        return payload
    if not features:
        payload.update({"status": "skipped", "reason": "no_usable_features"})
        _write_model_outputs(output_dir, payload, [], [], [], [])
        return payload
    if label not in inner:
        payload.update({"status": "skipped", "reason": f"missing_label:{label}"})
        _write_model_outputs(output_dir, payload, [], [], [], [])
        return payload
    eligible_dates = sorted(
        inner.loc[pd.to_numeric(inner[label], errors="coerce").notna(), "date"]
        .astype(str)
        .unique()
        .tolist()
    )
    try:
        folds = _rolling_inner_folds(eligible_dates, cfg)
    except ValueError as exc:
        payload.update({"status": "skipped", "reason": f"invalid_inner_folds:{exc}"})
        _write_model_outputs(output_dir, payload, [], [], [], [])
        return payload
    payload["fold_plan"] = folds
    all_summary_rows: list[dict[str, Any]] = []
    all_daily_ic: list[pd.DataFrame] = []
    all_daily_returns: list[pd.DataFrame] = []
    all_group_returns: list[pd.DataFrame] = []
    for fold in folds:
        fold_id = str(fold["fold_id"])
        fold_cfg = replace(
            cfg,
            dates=replace(
                cfg.dates,
                train_start=fold["train"]["start"],
                train_end=fold["train"]["end"],
                valid_start=fold["valid"]["start"],
                valid_end=fold["valid"]["end"],
                test_start=fold["test"]["start"],
                test_end=fold["test"]["end"],
            ),
        )
        fold_dir = output_dir / "folds" / fold_id
        _run_single_split_model_validation(
            panel=inner,
            label=label,
            features=features,
            cfg=fold_cfg,
            output_dir=fold_dir,
            run_name=f"{run_name}_{fold_id}",
            new_factors=new_factors,
            previous_sota=None,
            model_cfg=model_cfg,
        )
        all_summary_rows.extend(
            _read_fold_rows(fold_dir / "model_summary.csv", fold_id)
        )
        all_daily_ic.extend(
            _read_fold_frame(fold_dir / "daily_ic.csv", fold_id, split="test")
        )
        all_daily_returns.extend(
            _read_fold_frame(fold_dir / "daily_long_short.csv", fold_id, split="test")
        )
        all_group_returns.extend(
            _read_fold_frame(fold_dir / "group_returns.csv", fold_id, split="test")
        )
    aggregates = _aggregate_inner_models(all_summary_rows, folds, cfg)
    best_test = max(
        (row["test"] for row in aggregates if row.get("test")),
        key=_model_score_tuple,
        default={},
    )
    selected = next(
        (
            row
            for row in aggregates
            if row.get("test", {}).get("model") == best_test.get("model")
        ),
        {},
    )
    best_valid = selected.get("valid", {})
    if best_test:
        best_test["run_name"] = run_name
    if best_valid:
        best_valid["run_name"] = run_name
    fold_metrics = selected.get("fold_metrics", [])
    decision = _multifold_improvement_decision(
        best_test, fold_metrics, previous_sota, cfg
    )
    status = "ok" if best_test else "skipped"
    payload.update(
        {
            "status": status,
            "reason": "" if best_test else "no_successful_fold_models",
            "best_valid": best_valid,
            "best_test": best_test,
            "search_aggregate": best_test,
            "fold_metrics": fold_metrics,
            "model_aggregates": aggregates,
            "selection_key": f"Select one model family by trading-day-weighted aggregate across inner rolling time-fold test windows; retain only when aggregate SOTA improves and at least {protocol.min_improved_folds}/{protocol.fold_count} corresponding folds improve versus previous SOTA",
            "sota_markets": _sota_markets(cfg),
            "sota_score_fields": list(SOTA_SCORE_FIELDS),
            "sota_score_scales": dict(SOTA_SCORE_SCALES),
            "retain_decision": decision,
            "sota_improved": decision["retain"],
            "sota": _multifold_sota_payload(
                best_valid, best_test, fold_metrics, run_name, features
            ),
        }
    )
    _write_model_outputs(
        output_dir,
        payload,
        all_summary_rows,
        all_daily_ic,
        all_daily_returns,
        all_group_returns,
    )
    _prune_inner_validation_outputs(output_dir)
    return payload


def _prune_inner_validation_outputs(output_dir: Path) -> None:
    keep = output_dir / "model_validation.json"
    for path in output_dir.iterdir():
        if path == keep:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _validate_search_protocol(cfg: AppConfig) -> None:
    protocol = cfg.search_validation
    if protocol.protocol not in {"fixed_split", "inner_multifold"}:
        raise ValueError(f"Unsupported search validation protocol: {protocol.protocol}")
    if protocol.final_refit_scope not in {
        "train_valid",
        "all_inner",
        "latest_train_window",
    }:
        raise ValueError(f"Unsupported final refit scope: {protocol.final_refit_scope}")
    if protocol.protocol != "inner_multifold":
        return
    if (
        not protocol.inner_start
        or not protocol.inner_end
        or (not protocol.outer_start)
        or (not protocol.outer_end)
    ):
        raise ValueError("inner_multifold requires inner and outer start/end dates")
    inner_start = pd.Timestamp(protocol.inner_start)
    inner_end = pd.Timestamp(protocol.inner_end)
    outer_start = pd.Timestamp(protocol.outer_start)
    outer_end = pd.Timestamp(protocol.outer_end)
    if inner_start > inner_end or outer_start > outer_end:
        raise ValueError("validation date ranges must be ordered")
    if inner_end >= outer_start:
        raise ValueError(
            "inner and outer ranges must be disjoint and chronologically ordered"
        )
    ratios = (
        protocol.fold_train_ratio,
        protocol.fold_valid_ratio,
        protocol.fold_test_ratio,
    )
    if any((value <= 0 for value in ratios)) or not math.isclose(
        sum(ratios), 1.0, abs_tol=1e-09
    ):
        raise ValueError(
            "inner fold train/valid/test ratios must be positive and sum to 1"
        )
    if protocol.fold_count < 1:
        raise ValueError("fold_count must be positive")
    if not 1 <= protocol.min_improved_folds <= protocol.fold_count:
        raise ValueError("min_improved_folds must be between 1 and fold_count")
    if not 0.0 < protocol.final_valid_ratio < 1.0:
        raise ValueError("final_valid_ratio must be between 0 and 1")


def _rolling_inner_folds(dates: list[str], cfg: AppConfig) -> list[dict[str, Any]]:
    protocol = cfg.search_validation
    count = protocol.fold_count
    if len(dates) < max(30, count * 6):
        raise ValueError(f"too few eligible inner dates: {len(dates)}")
    test_ratio = protocol.fold_test_ratio
    test_days = max(
        1, int(math.floor(len(dates) * test_ratio / (1.0 + (count - 1) * test_ratio)))
    )
    window_days = len(dates) - (count - 1) * test_days
    non_test_days = window_days - test_days
    train_share = protocol.fold_train_ratio / (
        protocol.fold_train_ratio + protocol.fold_valid_ratio
    )
    train_days = max(1, int(math.floor(non_test_days * train_share)))
    valid_days = non_test_days - train_days
    embargo_days = _label_lookahead_days(cfg)
    if min(train_days - embargo_days, valid_days - embargo_days, test_days) < 1:
        raise ValueError("fold geometry creates an empty split")
    folds: list[dict[str, Any]] = []
    for index in range(count):
        start = index * test_days
        train_end = start + train_days
        valid_end = train_end + valid_days
        test_end = valid_end + test_days
        fold_dates = dates[start:test_end]
        if len(fold_dates) != window_days:
            raise ValueError("fold geometry does not fit available inner dates")
        raw_train = dates[start:train_end]
        raw_valid = dates[train_end:valid_end]
        train = raw_train[:-embargo_days] if embargo_days else raw_train
        valid = raw_valid[:-embargo_days] if embargo_days else raw_valid
        train_embargo = raw_train[-embargo_days:] if embargo_days else []
        valid_embargo = raw_valid[-embargo_days:] if embargo_days else []
        folds.append(
            {
                "fold_id": f"fold_{index + 1:02d}",
                "strategy": "rolling_window",
                "weight_basis": "test_trading_days",
                "label_lookahead_trading_days": embargo_days,
                "train": _date_slice(train),
                "train_valid_embargo": _date_slice(train_embargo)
                if train_embargo
                else {},
                "valid": _date_slice(valid),
                "valid_test_embargo": _date_slice(valid_embargo)
                if valid_embargo
                else {},
                "test": _date_slice(dates[valid_end:test_end]),
            }
        )
    return folds


def _date_slice(dates: list[str]) -> dict[str, Any]:
    return {"start": dates[0], "end": dates[-1], "trading_days": len(dates)}


def _final_refit_date_ranges(inner_dates: list[str], cfg: AppConfig) -> dict[str, Any]:
    if not inner_dates:
        raise ValueError("insufficient_inner_dates_for_final_refit")
    scope = cfg.search_validation.final_refit_scope
    refit_dates = inner_dates
    if scope == "latest_train_window":
        folds = _rolling_inner_folds(inner_dates, cfg)
        refit_days = int(folds[-1]["train"]["trading_days"])
        refit_dates = inner_dates[-refit_days:]
    valid_days = max(
        1, int(math.ceil(len(refit_dates) * cfg.search_validation.final_valid_ratio))
    )
    embargo_days = _label_lookahead_days(cfg)
    train_stop = len(refit_dates) - valid_days - embargo_days
    if train_stop < 2:
        raise ValueError("insufficient_inner_dates_for_final_refit")
    return {
        "refit_scope": scope,
        "train": _date_slice(refit_dates[:train_stop]),
        "train_valid_embargo": _date_slice(
            refit_dates[train_stop : train_stop + embargo_days]
        )
        if embargo_days
        else {},
        "valid": _date_slice(refit_dates[-valid_days:]),
        "refit": _date_slice(
            refit_dates[:train_stop] if scope == "train_valid" else refit_dates
        ),
        "all_inner": _date_slice(inner_dates),
        "label_lookahead_trading_days": embargo_days,
    }


def _read_fold_rows(path: Path, fold_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    frame.insert(0, "fold_id", fold_id)
    return frame.replace({np.nan: None}).to_dict(orient="records")


def _read_fold_frame(path: Path, fold_id: str, split: str) -> list[pd.DataFrame]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    if "split" in frame:
        frame = frame[frame["split"].eq(split)].copy()
    if frame.empty:
        return []
    frame.insert(0, "fold_id", fold_id)
    return [frame]


def _aggregate_inner_models(
    rows: list[dict[str, Any]], folds: list[dict[str, Any]], cfg: AppConfig
) -> list[dict[str, Any]]:
    models = sorted(
        {str(row.get("model")) for row in rows if row.get("status") == "ok"}
    )
    markets = _sota_markets(cfg)
    out: list[dict[str, Any]] = []
    for model in models:
        fold_metrics: list[dict[str, Any]] = []
        test_rows: list[dict[str, Any]] = []
        valid_rows: list[dict[str, Any]] = []
        weights: list[float] = []
        for fold in folds:
            fold_id = fold["fold_id"]
            fold_rows = [row for row in rows if row.get("fold_id") == fold_id]
            test = _model_market_average_row(fold_rows, "test", model, markets)
            valid = _model_market_average_row(fold_rows, "valid", model, markets)
            if not test:
                continue
            weight = float(fold["test"]["trading_days"])
            test_rows.append(test)
            valid_rows.append(valid)
            weights.append(weight)
            fold_metrics.append(
                {
                    "fold_id": fold_id,
                    "weight": weight,
                    "date_ranges": {
                        key: fold[key] for key in ("train", "valid", "test")
                    },
                    "valid": _checkpoint_metric_summary(valid),
                    "test": _checkpoint_metric_summary(test),
                }
            )
        if len(test_rows) != len(folds):
            continue
        out.append(
            {
                "model": model,
                "valid": _weighted_fold_row(
                    valid_rows, weights, "inner_valid_aggregate"
                ),
                "test": _weighted_fold_row(test_rows, weights, "inner_test_aggregate"),
                "fold_metrics": fold_metrics,
            }
        )
    return out


def _weighted_fold_row(
    rows: list[dict[str, Any]], weights: list[float], split: str
) -> dict[str, Any]:
    if not rows:
        return {}
    normalized = np.asarray(weights, dtype="float64")
    normalized = normalized / normalized.sum()
    first = rows[0]
    out: dict[str, Any] = {
        "run_name": first.get("run_name"),
        "model": first.get("model"),
        "split": split,
        "market": first.get("market"),
        "markets": first.get("markets", []),
        "market_count": min(
            (int(_finite(row.get("market_count"), default=0.0)) for row in rows)
        ),
        "expected_market_count": max(
            (
                int(_finite(row.get("expected_market_count"), default=0.0))
                for row in rows
            )
        ),
        "status": "ok",
        "fold_count": len(rows),
        "fold_weights": normalized.tolist(),
        "feature_count": max(
            (int(_finite(row.get("feature_count"), default=0.0)) for row in rows)
        ),
    }
    numeric_fields = set(SOTA_SCORE_FIELDS) | {
        "rank_ic_std",
        "rank_ic_positive_ratio",
        "rank_ic_t_stat",
        "ic_std",
        "ic_positive_ratio",
        "ic_t_stat",
        "long_excess_mean",
        "short_excess_mean",
        "long_short_mean",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "turnover_mean",
        "return_days",
        "days",
        "rows",
        "group_monotonic_spearman",
        "group_top_minus_bottom",
        "group_top_minus_bottom_annual",
        "group_top_minus_bottom_icir",
        "group_top_minus_bottom_sharpe",
        "group_top_minus_bottom_win_rate",
        "group_daily_monotonic_mean",
        "group_daily_monotonic_positive_ratio",
        "group_middle_adjacent_mean",
    }
    for field in numeric_fields:
        values = np.asarray(
            [_finite(row.get(field), default=math.nan) for row in rows], dtype="float64"
        )
        valid = np.isfinite(values)
        out[field] = (
            float(np.average(values[valid], weights=normalized[valid]))
            if valid.any()
            else math.nan
        )
    market_names = list(
        dict.fromkeys(
            (str(name) for row in rows for name in row.get("markets", []) if name)
        )
    )
    market_metrics = []
    for market in market_names:
        market_rows = []
        market_weights = []
        for row, weight in zip(rows, weights):
            match = next(
                (
                    item
                    for item in row.get("market_metrics", [])
                    if item.get("market") == market
                ),
                None,
            )
            if match:
                market_rows.append(match)
                market_weights.append(weight)
        if market_rows:
            aggregate = _weighted_fold_row(market_rows, market_weights, split)
            aggregate["market"] = market
            aggregate.pop("market_metrics", None)
            market_metrics.append(aggregate)
    out["market_metrics"] = market_metrics
    _attach_sota_scores(out)
    return out


def _multifold_improvement_decision(
    aggregate: dict[str, Any],
    fold_metrics: list[dict[str, Any]],
    previous_sota: dict[str, Any] | None,
    cfg: AppConfig,
) -> dict[str, Any]:
    previous = previous_sota or {}
    previous_aggregate = (
        previous.get("search_aggregate") or previous.get("best_test") or {}
    )
    previous_by_fold = {
        row.get("fold_id"): row.get("test", {})
        for row in previous.get("fold_metrics", [])
        if isinstance(row, dict) and row.get("fold_id")
    }
    delta = cfg.search_validation.min_sota_delta
    if not previous_aggregate:
        return {
            "retain": bool(aggregate),
            "comparison": "initialize_baseline",
            "aggregate_improved": bool(aggregate),
            "improved_fold_count": len(fold_metrics) if aggregate else 0,
            "required_improved_folds": cfg.search_validation.min_improved_folds,
            "fold_comparisons": [],
        }
    candidate_score = _finite(aggregate.get("sota_score"), default=-math.inf)
    previous_score = _finite(previous_aggregate.get("sota_score"), default=-math.inf)
    comparisons = []
    for row in fold_metrics:
        fold_id = row.get("fold_id")
        candidate_fold_score = _finite(
            row.get("test", {}).get("sota_score"), default=-math.inf
        )
        previous_fold_score = _finite(
            previous_by_fold.get(fold_id, {}).get("sota_score"), default=-math.inf
        )
        improved = candidate_fold_score > previous_fold_score + delta
        comparisons.append(
            {
                "fold_id": fold_id,
                "candidate_sota_score": candidate_fold_score,
                "previous_sota_score": previous_fold_score,
                "delta": candidate_fold_score - previous_fold_score,
                "improved": improved,
            }
        )
    improved_count = sum((bool(row["improved"]) for row in comparisons))
    aggregate_improved = candidate_score > previous_score + delta
    enough_folds = improved_count >= cfg.search_validation.min_improved_folds
    return {
        "retain": aggregate_improved and enough_folds,
        "comparison": "previous_inner_multifold_sota",
        "candidate_sota_score": candidate_score,
        "previous_sota_score": previous_score,
        "aggregate_delta": candidate_score - previous_score,
        "aggregate_improved": aggregate_improved,
        "improved_fold_count": improved_count,
        "required_improved_folds": cfg.search_validation.min_improved_folds,
        "fold_majority_passed": enough_folds,
        "fold_comparisons": comparisons,
    }


def _multifold_sota_payload(
    best_valid: dict[str, Any],
    best_test: dict[str, Any],
    fold_metrics: list[dict[str, Any]],
    run_name: str,
    features: list[str],
) -> dict[str, Any]:
    return {
        "protocol": "inner_multifold",
        "run_name": run_name,
        "best_valid": best_valid,
        "best_test": best_test,
        "search_aggregate": best_test,
        "fold_metrics": fold_metrics,
        "feature_count": len(features),
        "features": features,
    }


def run_final_heldout_evaluation(
    panel: pd.DataFrame,
    label: str,
    base_features: list[str],
    final_features: list[str],
    cfg: AppConfig,
    output_dir: Path,
    search_sota: dict[str, Any],
    baseline_sota: dict[str, Any] | None = None,
    model_cfg: ModelValidationConfig = ModelValidationConfig(),
) -> dict[str, Any]:
    _validate_search_protocol(cfg)
    if cfg.search_validation.protocol != "inner_multifold":
        return {"status": "skipped", "reason": "protocol_has_no_outer_holdout"}
    output_dir.mkdir(parents=True, exist_ok=True)
    certification_path = output_dir / "outer_certification.json"
    event_path = output_dir / "outer_evaluation_event.json"
    if certification_path.exists():
        existing = json.loads(certification_path.read_text(encoding="utf-8"))
        previous_manifest = output_dir / "freeze_manifest.json"
        if previous_manifest.exists():
            previous_scope = json.loads(
                previous_manifest.read_text(encoding="utf-8")
            ).get("final_refit_scope")
            if (
                previous_scope
                and previous_scope != cfg.search_validation.final_refit_scope
            ):
                return {
                    "status": "incomplete",
                    "reason": "terminal_protocol_changed; use a new certification directory",
                }
        existing["reused_existing_certification"] = True
        return existing
    if event_path.exists():
        event = json.loads(event_path.read_text(encoding="utf-8"))
        return {
            "status": "incomplete",
            "reason": "outer_evaluation_event_already_started; automatic re-evaluation is forbidden",
            "outer_evaluation_event_count": 1,
            "event": event,
        }
    panel = evaluation_panel(panel, [*base_features, *final_features], label)
    inner = inner_search_panel(panel, cfg)
    inner_dates = sorted(
        inner.loc[pd.to_numeric(inner[label], errors="coerce").notna(), "date"]
        .astype(str)
        .unique()
        .tolist()
    )
    try:
        final_dates = _final_refit_date_ranges(inner_dates, cfg)
    except ValueError:
        return {
            "status": "skipped",
            "reason": "insufficient_inner_dates_for_final_refit",
        }
    tune_cfg = replace(
        cfg,
        dates=replace(
            cfg.dates,
            train_start=final_dates["train"]["start"],
            train_end=final_dates["train"]["end"],
            valid_start=final_dates["valid"]["start"],
            valid_end=final_dates["valid"]["end"],
            test_start=cfg.search_validation.outer_start,
            test_end=cfg.search_validation.outer_end,
        ),
    )
    candidate_defs = [
        (
            "baseline",
            base_features,
            _selected_model_name(baseline_sota or search_sota, model_cfg),
        ),
        ("final", final_features, _selected_model_name(search_sota, model_cfg)),
    ]
    refit_all_inner = cfg.search_validation.final_refit_scope != "train_valid"
    if not refit_all_inner:
        candidate_defs = [
            (name, features, "lightgbm") for (name, features, _) in candidate_defs
        ]
    frozen: list[dict[str, Any]] = []
    fitted_candidates: dict[str, dict[str, Any]] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    for candidate_name, requested_features, model_name in candidate_defs:
        features = _usable_features(inner, requested_features, label)
        if not features:
            continue
        prepared = _prepare_model_frame(inner, features, label, tune_cfg)
        train = prepared[prepared["split"] == "train"]
        valid = prepared[prepared["split"] == "valid"]
        if len(train) < 100 or len(valid) < 20:
            continue
        artifact_dir = output_dir / candidate_name / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        refit_cfg = replace(
            cfg,
            dates=replace(
                cfg.dates,
                train_start=final_dates["refit"]["start"],
                train_end=final_dates["refit"]["end"],
                valid_start=cfg.search_validation.outer_start,
                valid_end=cfg.search_validation.outer_end,
                test_start=cfg.search_validation.outer_start,
                test_end=cfg.search_validation.outer_end,
            ),
        )
        refit_prepared = _prepare_model_frame(inner, features, label, refit_cfg)
        refit_rows = refit_prepared[refit_prepared["split"] == "train"]
        fit = _fit_frozen_model(
            model_name=model_name,
            label=label,
            train=train,
            valid=valid,
            all_inner=refit_rows,
            features=features,
            model_cfg=model_cfg,
            artifact_dir=artifact_dir,
            refit_all_inner=refit_all_inner,
        )
        if fit.get("status") != "ok":
            frozen.append(
                {
                    "candidate": candidate_name,
                    "model": model_name,
                    "status": fit.get("status", "skipped"),
                    "reason": fit.get("reason", ""),
                }
            )
            continue
        fitted_candidates[candidate_name] = fit
        predictions_path = output_dir / candidate_name / "outer_predictions.csv"
        artifacts = fit.get("artifacts", [])
        frozen.append(
            {
                "candidate": candidate_name,
                "model": model_name,
                "status": "frozen",
                "features": features,
                "feature_count": len(features),
                "inner_date_ranges": final_dates,
                "selection_source": "inner_only",
                "training_detail": fit.get("detail", {}),
                "artifacts": [_artifact_with_sha256(row) for row in artifacts],
                "outer_predictions_output": str(predictions_path),
            }
        )
    freeze_manifest = {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "inner_multifold",
        "final_refit_scope": cfg.search_validation.final_refit_scope,
        "outer_metrics_read": False,
        "candidates": frozen,
    }
    freeze_path = output_dir / "freeze_manifest.json"
    freeze_manifest["path"] = str(freeze_path)
    _write_json_atomic(freeze_path, freeze_manifest)
    event = {
        "status": "started",
        "event_count": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": _sha256_file(freeze_path),
    }
    _write_json_atomic(event_path, event)
    outer_mask = _date_range_mask(
        panel, cfg.search_validation.outer_start, cfg.search_validation.outer_end
    )
    outer_indices = panel.index[outer_mask]
    prediction_errors: list[dict[str, str]] = []
    for frozen_candidate in frozen:
        candidate_name = str(frozen_candidate.get("candidate") or "")
        fit = fitted_candidates.get(candidate_name)
        if frozen_candidate.get("status") != "frozen" or fit is None:
            continue
        try:
            features = [str(value) for value in frozen_candidate.get("features", [])]
            outer_feature_frame = panel.loc[outer_indices, features].apply(
                pd.to_numeric, errors="coerce"
            )
            predictions = _predict_frozen_model(fit, outer_feature_frame, features)
            prediction_frame = panel.loc[outer_indices, ["date", "symbol"]].copy()
            prediction_frame["prediction"] = predictions
            prediction_frame.to_csv(
                frozen_candidate["outer_predictions_output"], index=False
            )
            prediction_frames[candidate_name] = prediction_frame
        except Exception as exc:
            prediction_errors.append(
                {
                    "candidate": candidate_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    summary_rows: list[dict[str, Any]] = []
    daily_ic_rows: list[pd.DataFrame] = []
    daily_return_rows: list[pd.DataFrame] = []
    group_return_rows: list[pd.DataFrame] = []
    outer_results: dict[str, Any] = {}
    outer_labels = pd.to_numeric(panel.loc[outer_indices, label], errors="coerce")
    outer_label_dates = panel.loc[
        outer_labels[outer_labels.notna()].index, "date"
    ].astype(str)
    outer_observation = {
        "rows": int(outer_labels.notna().sum()),
        "trading_days": int(outer_label_dates.nunique()),
        "first_date": outer_label_dates.min() if not outer_label_dates.empty else "",
        "last_date": outer_label_dates.max() if not outer_label_dates.empty else "",
    }
    for frozen_candidate in frozen:
        candidate_name = frozen_candidate.get("candidate")
        prediction_frame = prediction_frames.get(str(candidate_name))
        if frozen_candidate.get("status") != "frozen" or prediction_frame is None:
            continue
        eval_frame = panel.loc[outer_indices].copy()
        (model_rows, model_ic, model_returns, model_groups) = (
            _evaluate_model_result_predictions(
                frames={"outer": eval_frame},
                predictions={
                    "outer": prediction_frame["prediction"].to_numpy(dtype="float64")
                },
                label=label,
                cfg=cfg,
                run_name=f"outer_{candidate_name}",
                model_name=str(candidate_name),
                feature_count=int(frozen_candidate.get("feature_count", 0)),
                training_detail={
                    "frozen_model_family": frozen_candidate.get("model"),
                    "selection_source": "inner_only",
                },
                daily_return_panel=panel,
            )
        )
        if candidate_name == "final":
            # Plot diagnostics use deciles; the research score keeps its own settings.
            plot_cfg = replace(
                cfg,
                evaluation=replace(
                    cfg.evaluation, group_count=10, long_short_quantile=0.1
                ),
            )
            _, _, plot_returns, plot_groups = _evaluate_model_result_predictions(
                frames={"outer": eval_frame},
                predictions={
                    "outer": prediction_frame["prediction"].to_numpy(dtype="float64")
                },
                label=label,
                cfg=plot_cfg,
                run_name="outer_final",
                model_name="final",
                feature_count=int(frozen_candidate.get("feature_count", 0)),
                training_detail={},
                daily_return_panel=panel,
            )
            _concat_frames(plot_groups).to_csv(
                output_dir / "final_groups.csv", index=False
            )
            _concat_frames(plot_returns).to_csv(
                output_dir / "final_returns.csv", index=False
            )
        summary_rows.extend(model_rows)
        daily_ic_rows.extend(model_ic)
        daily_return_rows.extend(model_returns)
        group_return_rows.extend(model_groups)
        outer_results[str(candidate_name)] = _best_sota_summary_row(
            model_rows, "outer", cfg
        )
    baseline_score = _finite(
        outer_results.get("baseline", {}).get("sota_score"), default=math.nan
    )
    final_score = _finite(
        outer_results.get("final", {}).get("sota_score"), default=math.nan
    )
    certification = {
        "status": "ok" if outer_results.get("final") else "skipped",
        "protocol": "inner_multifold_outer_once",
        "outer_evaluation_event_count": 1,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "agent_visible": False,
        "freeze_manifest": str(freeze_path),
        "outer_date_range": {
            "start": cfg.search_validation.outer_start,
            "end": cfg.search_validation.outer_end,
        },
        "outer_label_observation": outer_observation,
        "results": outer_results,
        "prediction_errors": prediction_errors,
        "final_minus_baseline_sota_score": final_score - baseline_score
        if not math.isnan(final_score) and (not math.isnan(baseline_score))
        else math.nan,
    }
    summary_df = pd.DataFrame(summary_rows)
    daily_ic_df = _concat_frames(daily_ic_rows)
    daily_return_df = _concat_frames(daily_return_rows)
    group_return_df = _concat_frames(group_return_rows)
    group_cumulative_df = _group_cumulative_returns(group_return_df)
    group_diagnostics_df = _group_diagnostics_frame(group_return_df)
    summary_path = output_dir / "outer_model_summary.csv"
    daily_ic_path = output_dir / "outer_daily_ic.csv"
    daily_long_short_path = output_dir / "outer_daily_long_short.csv"
    group_returns_path = output_dir / "outer_group_returns.csv"
    group_cumulative_path = output_dir / "outer_group_cumulative_returns.csv"
    group_diagnostics_path = output_dir / "outer_group_diagnostics.csv"
    summary_df.to_csv(summary_path, index=False)
    daily_ic_df.to_csv(daily_ic_path, index=False)
    daily_return_df.to_csv(daily_long_short_path, index=False)
    group_return_df.to_csv(group_returns_path, index=False)
    group_cumulative_df.to_csv(group_cumulative_path, index=False)
    group_diagnostics_df.to_csv(group_diagnostics_path, index=False)
    certification["reports"] = {
        "summary_csv": str(summary_path),
        "daily_ic_csv": str(daily_ic_path),
        "daily_long_short_csv": str(daily_long_short_path),
        "group_returns_csv": str(group_returns_path),
        "group_cumulative_returns_csv": str(group_cumulative_path),
        "group_diagnostics_csv": str(group_diagnostics_path),
    }
    certification["plots"] = _write_plots(
        output_dir, daily_ic_df, daily_return_df, group_cumulative_df, split="outer"
    )
    _write_json_atomic(certification_path, certification)
    event.update(
        {
            "status": "completed",
            "completed_at": certification["executed_at"],
            "certification": str(certification_path),
            "certification_sha256": _sha256_file(certification_path),
        }
    )
    _write_json_atomic(event_path, event)
    return certification


def _selected_model_name(
    sota: dict[str, Any] | None, model_cfg: ModelValidationConfig
) -> str:
    sota = sota or {}
    aggregate = sota.get("search_aggregate") or sota.get("best_test") or {}
    model = str(aggregate.get("model") or "")
    if model:
        return model
    return model_cfg.models[0] if model_cfg.models else "linear"


def _date_range_mask(panel: pd.DataFrame, start: str, end: str) -> pd.Series:
    dates = pd.to_datetime(panel["date"].astype(str), errors="coerce")
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def _fit_frozen_model(
    model_name: str,
    label: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    all_inner: pd.DataFrame,
    features: list[str],
    model_cfg: ModelValidationConfig,
    artifact_dir: Path,
    refit_all_inner: bool = True,
) -> dict[str, Any]:
    if model_name == "linear":
        return _fit_frozen_linear(
            train, valid, all_inner, features, label, model_cfg, artifact_dir
        )
    if model_name == "lightgbm":
        return _fit_frozen_lightgbm(
            train,
            valid,
            all_inner,
            features,
            label,
            model_cfg,
            artifact_dir,
            refit_all_inner=refit_all_inner,
        )
    return {"status": "skipped", "reason": f"unsupported_frozen_model:{model_name}"}


def _fit_frozen_linear(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    all_inner: pd.DataFrame,
    features: list[str],
    label: str,
    model_cfg: ModelValidationConfig,
    artifact_dir: Path,
) -> dict[str, Any]:
    tune_stats = _fit_preprocessor(train, features)
    x_train = _matrix(train, features, tune_stats, standardize=True).astype("float64")
    x_valid = _matrix(valid, features, tune_stats, standardize=True).astype("float64")
    y_train = train[label].to_numpy(dtype="float64")
    y_valid = valid[label].to_numpy(dtype="float64")
    (alpha, valid_score) = _select_ridge_alpha(
        x_train, y_train, x_valid, y_valid, model_cfg.ridge_alphas
    )
    final_stats = _fit_preprocessor(all_inner, features)
    x_inner = _matrix(all_inner, features, final_stats, standardize=True).astype(
        "float64"
    )
    y_inner = all_inner[label].to_numpy(dtype="float64")
    (coef, intercept) = _ridge_solution(x_inner, y_inner, alpha)
    coef_path = artifact_dir / "linear_coefficients.csv"
    pd.DataFrame(
        {"feature": features, "coefficient": coef, "abs_coefficient": np.abs(coef)}
    ).sort_values("abs_coefficient", ascending=False).to_csv(coef_path, index=False)
    checkpoint_path = artifact_dir / "linear_model.npz"
    np.savez_compressed(
        checkpoint_path,
        features=np.asarray(features, dtype=str),
        coefficients=coef,
        intercept=np.asarray(intercept),
        alpha=np.asarray(alpha),
        fillna_median=final_stats["median"].reindex(features).to_numpy(dtype="float64"),
        standardize_mean=final_stats["mean"]
        .reindex(features)
        .to_numpy(dtype="float64"),
        standardize_std=final_stats["std"].reindex(features).to_numpy(dtype="float64"),
    )
    preprocessor = _write_preprocessor_artifact(final_stats, features, artifact_dir)
    return {
        "status": "ok",
        "predictor": {
            "model": "linear",
            "coef": coef,
            "intercept": intercept,
            "stats": final_stats,
        },
        "detail": {
            "alpha": alpha,
            "inner_valid_pearson": valid_score,
            "refit_rows": len(all_inner),
        },
        "artifacts": [
            {"model": "linear", "type": "checkpoint", "path": str(checkpoint_path)},
            {"model": "linear", "type": "coefficients", "path": str(coef_path)},
            {"model": "all", "type": "preprocessor", "path": str(preprocessor)},
        ],
    }


def _select_ridge_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[float, float]:
    best_alpha = float(alphas[0])
    best_score = -math.inf
    for alpha in alphas:
        (coef, intercept) = _ridge_solution(x_train, y_train, float(alpha))
        score = _safe_corr(x_valid @ coef + intercept, y_valid)
        finite_score = score if not math.isnan(score) else -math.inf
        if finite_score > best_score:
            (best_alpha, best_score) = (float(alpha), finite_score)
    return (best_alpha, best_score)


def _ridge_solution(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[np.ndarray, float]:
    intercept = float(np.mean(y))
    xtx = x.T @ x
    xty = x.T @ (y - intercept)
    reg = np.eye(xtx.shape[0]) * alpha
    try:
        coef = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xtx + reg) @ xty
    return (coef, intercept)


def _fit_frozen_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    all_inner: pd.DataFrame,
    features: list[str],
    label: str,
    model_cfg: ModelValidationConfig,
    artifact_dir: Path,
    refit_all_inner: bool = True,
) -> dict[str, Any]:
    try:
        import lightgbm as lgb
    except Exception as exc:
        return {"status": "skipped", "reason": f"lightgbm_unavailable:{exc}"}
    tune_stats = _fit_preprocessor(train, features)
    x_train = _matrix(train, features, tune_stats, standardize=False)
    x_valid = _matrix(valid, features, tune_stats, standardize=False)
    params = _lightgbm_params(model_cfg.lightgbm_estimators)
    tune_model = lgb.LGBMRegressor(**params)
    try:
        tune_model.fit(
            x_train,
            train[label].to_numpy(dtype="float64"),
            eval_set=[(x_valid, valid[label].to_numpy(dtype="float64"))],
            eval_metric="l2",
            callbacks=[
                lgb.log_evaluation(period=0),
                lgb.early_stopping(
                    model_cfg.lightgbm_early_stopping_rounds, verbose=False
                ),
            ],
        )
        best_iteration = int(
            getattr(tune_model, "best_iteration_", None)
            or model_cfg.lightgbm_estimators
        )
        if refit_all_inner:
            final_stats = _fit_preprocessor(all_inner, features)
            final_model = lgb.LGBMRegressor(**_lightgbm_params(best_iteration))
            final_model.fit(
                _matrix(all_inner, features, final_stats, standardize=False),
                all_inner[label].to_numpy(dtype="float64"),
                callbacks=[lgb.log_evaluation(period=0)],
            )
        else:
            (final_stats, final_model) = (tune_stats, tune_model)
    except Exception as exc:
        return {"status": "skipped", "reason": f"lightgbm_final_fit_failed:{exc}"}
    checkpoint_path = artifact_dir / "lightgbm_model.txt"
    final_model.booster_.save_model(str(checkpoint_path), num_iteration=best_iteration)
    importance_path = artifact_dir / "lightgbm_feature_importance.csv"
    pd.DataFrame(
        {
            "feature": features,
            "importance_gain": final_model.booster_.feature_importance(
                importance_type="gain"
            ),
            "importance_split": final_model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("importance_gain", ascending=False).to_csv(
        importance_path, index=False
    )
    preprocessor = _write_preprocessor_artifact(final_stats, features, artifact_dir)
    return {
        "status": "ok",
        "predictor": {
            "model": "lightgbm",
            "estimator": final_model,
            "stats": final_stats,
            "best_iteration": best_iteration,
        },
        "detail": {
            "best_iteration": best_iteration,
            "refit_rows": len(all_inner) if refit_all_inner else len(train),
            "refit_all_inner": refit_all_inner,
            "selection_source": "inner_valid",
        },
        "artifacts": [
            {"model": "lightgbm", "type": "checkpoint", "path": str(checkpoint_path)},
            {
                "model": "lightgbm",
                "type": "feature_importance",
                "path": str(importance_path),
            },
            {"model": "all", "type": "preprocessor", "path": str(preprocessor)},
        ],
    }


def _lightgbm_params(n_estimators: int) -> dict[str, Any]:
    return {
        "objective": "regression",
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 20260608,
        "n_jobs": max(1, model_threads()),
        "verbosity": -1,
    }


def _predict_frozen_model(
    fit: dict[str, Any], outer_features: pd.DataFrame, features: list[str]
) -> np.ndarray:
    predictor = fit.get("predictor", {})
    model = predictor.get("model")
    stats = predictor.get("stats")
    if not isinstance(stats, dict):
        raise ValueError("frozen predictor is missing preprocessing statistics")
    if model == "linear":
        matrix = _matrix(outer_features, features, stats, standardize=True).astype(
            "float64"
        )
        return np.asarray(
            matrix @ predictor["coef"] + predictor["intercept"], dtype="float64"
        )
    if model == "lightgbm":
        matrix = _matrix(outer_features, features, stats, standardize=False)
        return np.asarray(
            predictor["estimator"].predict(
                matrix, num_iteration=predictor["best_iteration"]
            ),
            dtype="float64",
        )
    raise ValueError(f"unsupported frozen predictor: {model}")


def _artifact_with_sha256(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    path = Path(str(row.get("path", "")))
    if path.exists() and path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        out["sha256"] = digest.hexdigest()
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(_json_dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _evaluate_model_result_predictions(
    frames: dict[str, pd.DataFrame],
    predictions: dict[str, np.ndarray],
    label: str,
    cfg: AppConfig,
    run_name: str,
    model_name: str,
    feature_count: int,
    training_detail: dict[str, Any],
    daily_return_panel: pd.DataFrame | None = None,
) -> tuple[
    list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]
]:
    tasks: list[tuple[str, str, str, pd.DataFrame, pd.DataFrame | None]] = []
    daily_sources: dict[str, pd.DataFrame] = {}
    for split, pred in predictions.items():
        pred_frame = _prediction_frame(frames[split], label, cfg)
        pred_frame["prediction"] = np.asarray(pred, dtype="float64")
        for market, eval_label, market_frame in _market_prediction_frames(
            pred_frame, label, cfg
        ):
            market_frame = market_frame.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["prediction", eval_label]
            )
            if not market_frame.empty:
                daily_source: pd.DataFrame | None = None
                daily_label: str | None = None
                if cfg.evaluation.label_horizon_days > 1:
                    daily_label = one_day_label(eval_label)
                    source = (
                        daily_return_panel
                        if daily_return_panel is not None
                        else pred_frame
                    )
                    if daily_label not in source:
                        raise ValueError(
                            f"Multi-day model evaluation requires {daily_label} in the daily return panel"
                        )
                    if daily_label not in daily_sources:
                        daily_sources[daily_label] = source[
                            ["date", "symbol", daily_label]
                        ]
                    daily_source = daily_sources[daily_label]
                tasks.append((split, market, eval_label, market_frame, daily_source))
    if not tasks:
        return ([], [], [], [])
    workers = (
        1
        if cfg.evaluation.label_horizon_days > 1
        else min(model_eval_workers(), max(1, len(tasks)))
    )
    args = [
        (
            task,
            model_name,
            cfg.evaluation.long_short_quantile,
            cfg.evaluation.group_count,
            cfg.evaluation.min_assets,
        )
        for task in tasks
    ]
    if workers <= 1:
        evaluated = [_evaluate_model_task(arg) for arg in args]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            evaluated = list(executor.map(_evaluate_model_task, args))
    summary_rows: list[dict[str, Any]] = []
    daily_ic_rows: list[pd.DataFrame] = []
    daily_return_rows: list[pd.DataFrame] = []
    group_return_rows: list[pd.DataFrame] = []
    for split, market, eval_label, metrics, ic_df, returns_df, group_df in evaluated:
        metrics.update(
            {
                "run_name": run_name,
                "model": model_name,
                "split": split,
                "market": market,
                "label": eval_label,
                "status": "ok",
                "feature_count": feature_count,
                "training_detail": training_detail,
                "eval_workers": workers,
            }
        )
        summary_rows.append(metrics)
        daily_ic_rows.append(ic_df)
        daily_return_rows.append(returns_df)
        group_return_rows.append(group_df)
    return (summary_rows, daily_ic_rows, daily_return_rows, group_return_rows)


def _evaluate_model_task(
    args: tuple[
        tuple[str, str, str, pd.DataFrame, pd.DataFrame | None], str, float, int, int
    ],
) -> tuple[str, str, str, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (
        (split, market, eval_label, market_frame, daily_return_frame),
        model_name,
        quantile,
        group_count,
        min_assets,
    ) = args
    (metrics, ic_df, returns_df, group_df) = evaluate_model_predictions(
        market_frame,
        label=eval_label,
        model=model_name,
        split=split,
        market=market,
        quantile=quantile,
        group_count=group_count,
        min_assets=min_assets,
        daily_return_frame=daily_return_frame,
    )
    return (split, market, eval_label, metrics, ic_df, returns_df, group_df)


def evaluate_model_predictions(
    pred_frame: pd.DataFrame,
    label: str,
    model: str,
    split: str,
    market: str,
    quantile: float,
    group_count: int,
    min_assets: int,
    daily_return_frame: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holding_period_days = resolve_holding_period_days(label)
    ic_df = daily_ic(pred_frame, "prediction", label, min_assets=min_assets)
    ic_df.insert(0, "model", model)
    ic_df.insert(1, "split", split)
    ic_df.insert(2, "market", market)
    returns_df = _daily_long_short(
        pred_frame,
        label,
        quantile=quantile,
        min_assets=min_assets,
        holding_period_days=holding_period_days,
        daily_return_frame=daily_return_frame,
    )
    returns_df.insert(0, "model", model)
    returns_df.insert(1, "split", split)
    returns_df.insert(2, "market", market)
    group_df = _daily_group_returns(
        pred_frame,
        label,
        group_count=group_count,
        min_assets=min_assets,
        holding_period_days=holding_period_days,
        daily_return_frame=daily_return_frame,
    )
    group_df.insert(0, "model", model)
    group_df.insert(1, "split", split)
    group_df.insert(2, "market", market)
    ic_summary = summarize_ic(ic_df)
    ls_summary = _summarize_model_returns(
        returns_df, holding_period_days=holding_period_days
    )
    group_summary = _summarize_group_returns(group_df, group_count)
    metrics = {
        "rows": int(len(pred_frame)),
        "days": int(pred_frame["date"].nunique()),
        "ic_mean": ic_summary["ic_mean"],
        "ic_std": ic_summary["ic_std"],
        "icir": ic_summary["icir"],
        "ic_positive_ratio": ic_summary["ic_positive_ratio"],
        "ic_t_stat": ic_summary["ic_t_stat"],
        "rank_ic_mean": ic_summary["rank_ic_mean"],
        "rank_ic_std": ic_summary["rank_ic_std"],
        "rank_icir": ic_summary["rank_icir"],
        "rank_ic_positive_ratio": ic_summary["rank_ic_positive_ratio"],
        "rank_ic_t_stat": ic_summary["rank_ic_t_stat"],
        **ls_summary,
        **group_summary,
    }
    _attach_sota_scores(metrics)
    return (metrics, ic_df, returns_df, group_df)


def _usable_features(panel: pd.DataFrame, features: list[str], label: str) -> list[str]:
    out = []
    seen = set()
    for feature in features:
        if (
            feature == label
            or feature in seen
            or feature not in panel
            or _forbidden_model_feature(feature)
        ):
            continue
        if not pd.api.types.is_numeric_dtype(panel[feature]):
            continue
        valid = (
            pd.to_numeric(panel[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .sum()
        )
        if valid < 10:
            continue
        seen.add(feature)
        out.append(feature)
    return out


def _forbidden_model_feature(feature: str) -> bool:
    if feature.startswith(("label_", "benchmark_", "in_")):
        return True
    return feature in {
        "st",
        "is_st",
        "listing_age_days",
        "is_new_listing",
        "prev_close",
        "limit_move_threshold",
        "limit_move_today",
        "entry_open_limit_move",
        "exit_open_limit_move",
        "universe_eligible",
    }


def _prepare_model_frame(
    panel: pd.DataFrame, features: list[str], label: str, cfg: AppConfig
) -> pd.DataFrame:
    extra_cols = [
        col
        for col in panel.columns
        if col.startswith("label_")
        and col.endswith(f"_{cfg.evaluation.label_horizon_days}d")
    ]
    member_cols = [col for col in panel.columns if col.startswith("in_")]
    cols = _unique_preserve(
        ["date", "symbol", label, *extra_cols, *member_cols, *features]
    )
    meta_cols = [col for col in cols if col not in features]
    out = panel[meta_cols].copy(deep=False)
    out["date"] = out["date"].astype(str)
    out[label] = pd.to_numeric(out[label], errors="coerce")
    for col in extra_cols:
        if col != label and col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in member_cols:
        out[col] = out[col].astype(bool)
    feature_frame = panel[features].apply(pd.to_numeric, errors="coerce")
    if np.isinf(feature_frame.to_numpy(dtype="float64", na_value=np.nan)).any():
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    out = pd.concat([out, feature_frame], axis=1)
    labeled = out[label].notna()
    if not labeled.all():
        out = out.loc[labeled]
    dates = pd.to_datetime(out["date"])
    split_values = pd.Series("", index=out.index, dtype="object")
    split_defs = {
        "train": (cfg.dates.train_start, cfg.dates.train_end),
        "valid": (cfg.dates.valid_start, cfg.dates.valid_end),
        "test": (cfg.dates.test_start, cfg.dates.test_end),
    }
    for split, (start, end) in split_defs.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        split_values.loc[mask] = split
    out["split"] = split_values
    selected = split_values.isin(split_defs)
    if not selected.all():
        out = out.loc[selected]
    out.reset_index(drop=True, inplace=True)
    return out


def _fit_preprocessor(train: pd.DataFrame, features: list[str]) -> dict[str, pd.Series]:
    med = train[features].median(numeric_only=True).reindex(features).fillna(0.0)
    filled = train[features].fillna(med)
    mean = filled.mean(numeric_only=True).reindex(features).fillna(0.0)
    std = (
        filled.std(ddof=0, numeric_only=True)
        .reindex(features)
        .replace(0, np.nan)
        .fillna(1.0)
    )
    return {
        "median": med.astype("float64"),
        "mean": mean.astype("float64"),
        "std": std.astype("float64"),
    }


def _prediction_frame(frame: pd.DataFrame, label: str, cfg: AppConfig) -> pd.DataFrame:
    horizon = cfg.evaluation.label_horizon_days
    label_cols = [
        col
        for col in frame.columns
        if col.startswith("label_") and col.endswith(f"_{horizon}d")
    ]
    member_cols = [col for col in frame.columns if col.startswith("in_")]
    cols = _unique_preserve(["date", "symbol", label, *label_cols, *member_cols])
    return frame[cols].copy()


def _market_prediction_frames(
    pred_frame: pd.DataFrame, fallback_label: str, cfg: AppConfig
) -> list[tuple[str, str, pd.DataFrame]]:
    horizon = cfg.evaluation.label_horizon_days
    markets = _market_names(cfg)
    out: list[tuple[str, str, pd.DataFrame]] = []
    for market in markets:
        member_col = market_member_column(market)
        label = market_label_column(market, horizon)
        if label not in pred_frame:
            if market == SOTA_FALLBACK_MARKET and fallback_label in pred_frame:
                label = fallback_label
            else:
                continue
        frame = pred_frame
        if member_col in frame:
            frame = frame[frame[member_col].astype(bool)]
        if frame.empty:
            continue
        out.append((market, label, frame.copy()))
    if not out and fallback_label in pred_frame:
        out.append(("all", fallback_label, pred_frame.copy()))
    return out


def _market_names(cfg: AppConfig) -> list[str]:
    return active_market_names(cfg)


def _unique_preserve(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _matrix(
    frame: pd.DataFrame,
    features: list[str],
    stats: dict[str, pd.Series],
    standardize: bool,
) -> np.ndarray:
    x = frame[features].fillna(stats["median"]).to_numpy(dtype="float32", copy=True)
    if standardize:
        x = (x - stats["mean"].to_numpy(dtype="float32")) / stats["std"].to_numpy(
            dtype="float32"
        )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _write_preprocessor_artifact(
    stats: dict[str, pd.Series], features: list[str], artifact_dir: Path
) -> Path:
    path = artifact_dir / "preprocessor.json"
    payload = {
        "features": features,
        "fillna_median": stats["median"].reindex(features).astype(float).to_dict(),
        "standardize_mean": stats["mean"].reindex(features).astype(float).to_dict(),
        "standardize_std": stats["std"].reindex(features).astype(float).to_dict(),
        "linear_input": "fill missing values with fillna_median, then standardize by mean/std",
        "lightgbm_input": "fill missing values with fillna_median; no standardization",
    }
    path.write_text(_json_dumps(payload), encoding="utf-8")
    return path


def _fit_predict_linear(
    matrices: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    features: list[str],
    train_stats: dict[str, pd.Series],
    model_cfg: ModelValidationConfig,
    artifact_dir: Path,
) -> dict[str, Any]:
    x_train = matrices["train"].astype("float64", copy=False)
    y_train = y["train"].astype("float64", copy=False)
    y_mean = float(np.mean(y_train))
    yc = y_train - y_mean
    xtx = x_train.T @ x_train
    xty = x_train.T @ yc
    best: dict[str, Any] | None = None
    for alpha in model_cfg.ridge_alphas:
        reg = np.eye(xtx.shape[0]) * float(alpha)
        try:
            coef = np.linalg.solve(xtx + reg, xty)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(xtx + reg) @ xty
        valid_pred = matrices["valid"].astype("float64", copy=False) @ coef + y_mean
        corr = _safe_corr(valid_pred, y["valid"])
        score = corr if not math.isnan(corr) else -math.inf
        if best is None or score > best["score"]:
            best = {
                "alpha": float(alpha),
                "coef": coef,
                "intercept": y_mean,
                "score": score,
            }
    if best is None:
        return {
            "model": "linear",
            "status": "skipped",
            "reason": "linear_fit_failed",
            "predictions": {},
        }
    coef_path = artifact_dir / "linear_coefficients.csv"
    pd.DataFrame(
        {
            "feature_index": np.arange(len(best["coef"])),
            "feature": features,
            "coefficient": best["coef"],
            "abs_coefficient": np.abs(best["coef"]),
        }
    ).sort_values("abs_coefficient", ascending=False).to_csv(coef_path, index=False)
    checkpoint_path = artifact_dir / "linear_model.npz"
    np.savez_compressed(
        checkpoint_path,
        features=np.asarray(features, dtype=str),
        coefficients=np.asarray(best["coef"], dtype="float64"),
        intercept=np.asarray(best["intercept"], dtype="float64"),
        alpha=np.asarray(best["alpha"], dtype="float64"),
        fillna_median=train_stats["median"].reindex(features).to_numpy(dtype="float64"),
        standardize_mean=train_stats["mean"]
        .reindex(features)
        .to_numpy(dtype="float64"),
        standardize_std=train_stats["std"].reindex(features).to_numpy(dtype="float64"),
    )
    return {
        "model": "linear",
        "status": "ok",
        "detail": {"alpha": best["alpha"], "valid_pearson": best["score"]},
        "artifacts": [
            {"model": "linear", "type": "coefficients", "path": str(coef_path)},
            {"model": "linear", "type": "checkpoint", "path": str(checkpoint_path)},
        ],
        "predictions": {
            split: matrices[split].astype("float64", copy=False) @ best["coef"]
            + best["intercept"]
            for split in ("train", "valid", "test")
        },
    }


def _fit_predict_lightgbm(
    matrices: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    features: list[str],
    train_stats: dict[str, pd.Series],
    model_cfg: ModelValidationConfig,
    artifact_dir: Path,
) -> dict[str, Any]:
    try:
        import lightgbm as lgb
    except Exception as exc:
        return {
            "model": "lightgbm",
            "status": "skipped",
            "reason": f"lightgbm_unavailable:{exc}",
            "predictions": {},
        }
    n_jobs = model_threads()
    try:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=model_cfg.lightgbm_estimators,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=20260608,
            n_jobs=max(1, n_jobs),
            verbosity=-1,
        )
    except Exception as exc:
        return {
            "model": "lightgbm",
            "status": "skipped",
            "reason": f"lightgbm_init_failed:{exc}",
            "predictions": {},
        }
    callbacks = [lgb.log_evaluation(period=0)]
    if len(y["valid"]) > 0:
        callbacks.append(
            lgb.early_stopping(model_cfg.lightgbm_early_stopping_rounds, verbose=False)
        )
    fit_kwargs: dict[str, Any] = {"callbacks": callbacks}
    if len(y["valid"]) > 0:
        fit_kwargs["eval_set"] = [(matrices["valid"], y["valid"])]
        fit_kwargs["eval_metric"] = "l2"
    try:
        model.fit(matrices["train"], y["train"], **fit_kwargs)
    except Exception as exc:
        return {
            "model": "lightgbm",
            "status": "skipped",
            "reason": f"lightgbm_fit_failed:{exc}",
            "predictions": {},
        }
    importance_path = artifact_dir / "lightgbm_feature_importance.csv"
    pd.DataFrame(
        {
            "feature": features,
            "importance_gain": model.booster_.feature_importance(
                importance_type="gain"
            ),
            "importance_split": model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("importance_gain", ascending=False).to_csv(
        importance_path, index=False
    )
    best_iter = getattr(model, "best_iteration_", None) or model.n_estimators
    checkpoint_path: Path | None = artifact_dir / "lightgbm_model.txt"
    try:
        model.booster_.save_model(str(checkpoint_path), num_iteration=best_iter)
    except Exception as exc:
        checkpoint_path = None
        checkpoint_error = str(exc)
    else:
        checkpoint_error = ""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names.*",
            category=UserWarning,
        )
        predictions = {
            split: model.predict(matrices[split], num_iteration=best_iter)
            for split in ("train", "valid", "test")
        }
    artifacts = [
        {
            "model": "lightgbm",
            "type": "feature_importance",
            "path": str(importance_path),
        }
    ]
    if checkpoint_path is not None:
        artifacts.append(
            {"model": "lightgbm", "type": "checkpoint", "path": str(checkpoint_path)}
        )
    return {
        "model": "lightgbm",
        "status": "ok",
        "detail": {
            "best_iteration": int(best_iter),
            "n_jobs": n_jobs,
            "checkpoint_error": checkpoint_error,
            "fillna_median_features": len(train_stats["median"]),
        },
        "artifacts": artifacts,
        "predictions": predictions,
    }


def _daily_long_short(
    df: pd.DataFrame,
    label: str,
    quantile: float,
    min_assets: int,
    holding_period_days: int = 1,
    daily_return_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    if holding_period_days > 1:
        return _staggered_daily_long_short(
            df,
            label,
            quantile=quantile,
            min_assets=min_assets,
            holding_period_days=holding_period_days,
            daily_return_frame=daily_return_frame,
        )
    prev_long: set[str] | None = None
    prev_short: set[str] | None = None
    for date, group in df.dropna(subset=["prediction", label]).groupby(
        "date", sort=True
    ):
        if len(group) < min_assets or group["prediction"].nunique() < 2:
            continue
        date_str = str(date)
        low_cut = group["prediction"].quantile(quantile)
        high_cut = group["prediction"].quantile(1.0 - quantile)
        long = group[group["prediction"] >= high_cut]
        short = group[group["prediction"] <= low_cut]
        long_members = set(long["symbol"].astype(str))
        short_members = set(short["symbol"].astype(str))
        long_turnover = _turnover(prev_long, long_members)
        short_turnover = _turnover(prev_short, short_members)
        turnover_values = [
            value for value in [long_turnover, short_turnover] if not pd.isna(value)
        ]
        long_ret = float(long[label].mean())
        short_ret = float(short[label].mean())
        rows.append(
            {
                "date": date_str,
                "holding_period_days": holding_period_days,
                "periods_per_year": 252.0,
                "portfolio_mode": "daily_rebalance",
                "n": int(len(group)),
                "long_count": int(len(long)),
                "short_count": int(len(short)),
                "long_excess": long_ret,
                "short_excess": short_ret,
                "long_short": long_ret - short_ret,
                "long_turnover": long_turnover,
                "short_turnover": short_turnover,
                "turnover": float(np.mean(turnover_values))
                if turnover_values
                else math.nan,
            }
        )
        prev_long = long_members
        prev_short = short_members
    return attach_portfolio_equity(pd.DataFrame(rows))


def _staggered_daily_long_short(
    df: pd.DataFrame,
    label: str,
    *,
    quantile: float,
    min_assets: int,
    holding_period_days: int,
    daily_return_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    daily_label = one_day_label(label)
    source = daily_return_frame if daily_return_frame is not None else df
    if daily_label not in source:
        raise ValueError(
            f"{holding_period_days}-day staggered backtest requires {daily_label}; pass the one-day return panel explicitly"
        )
    formations: dict[str, dict[str, set[str]]] = {}
    for date, group in df.dropna(subset=["prediction", label]).groupby(
        "date", sort=True
    ):
        if len(group) < min_assets or group["prediction"].nunique() < 2:
            continue
        low_cut = group["prediction"].quantile(quantile)
        high_cut = group["prediction"].quantile(1.0 - quantile)
        formations[str(date)] = {
            "long": set(
                group.loc[group["prediction"] >= high_cut, "symbol"].astype(str)
            ),
            "short": set(
                group.loc[group["prediction"] <= low_cut, "symbol"].astype(str)
            ),
        }
    staggered = staggered_bucket_returns(
        formations,
        source,
        daily_label=daily_label,
        holding_period_days=holding_period_days,
    )
    if staggered.empty:
        return staggered
    returns = staggered.pivot(index="date", columns="bucket", values="return")
    turnover = staggered.pivot(index="date", columns="bucket", values="turnover")
    meta = staggered.groupby("date", as_index=True).agg(
        active_sleeves=("active_sleeves", "max"),
        capital_fraction=("capital_fraction", "max"),
        member_observations=("member_observations", "sum"),
        member_observations_expected=("member_observations_expected", "sum"),
        return_coverage=("return_coverage", "mean"),
    )
    out = meta.join(returns).reset_index()
    out["long_excess"] = out.get("long", 0.0)
    out["short_excess"] = out.get("short", 0.0)
    out["long_short"] = out["long_excess"] - out["short_excess"]
    out["long_turnover"] = (
        turnover.get("long", pd.Series(dtype=float)).reindex(out["date"]).to_numpy()
    )
    out["short_turnover"] = (
        turnover.get("short", pd.Series(dtype=float)).reindex(out["date"]).to_numpy()
    )
    out["turnover"] = out[["long_turnover", "short_turnover"]].mean(axis=1)
    out["holding_period_days"] = holding_period_days
    out["periods_per_year"] = 252.0
    out["portfolio_mode"] = "staggered_daily_mark_to_market"
    out["daily_return_label"] = daily_label
    return attach_portfolio_equity(out)


def _daily_group_returns(
    df: pd.DataFrame,
    label: str,
    group_count: int,
    min_assets: int,
    holding_period_days: int = 1,
    daily_return_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    if holding_period_days > 1:
        return _staggered_daily_group_returns(
            df,
            label,
            group_count=group_count,
            min_assets=min_assets,
            holding_period_days=holding_period_days,
            daily_return_frame=daily_return_frame,
        )
    for date, group in df.dropna(subset=["prediction", label]).groupby(
        "date", sort=True
    ):
        if (
            len(group) < max(min_assets, group_count)
            or group["prediction"].nunique() < 2
        ):
            continue
        date_str = str(date)
        ranked = group.copy()
        try:
            ranked["group"] = (
                pd.qcut(
                    ranked["prediction"].rank(method="first"), group_count, labels=False
                )
                + 1
            )
        except ValueError:
            continue
        for group_id, sub in ranked.groupby("group"):
            rows.append(
                {
                    "date": date_str,
                    "holding_period_days": holding_period_days,
                    "periods_per_year": 252.0,
                    "portfolio_mode": "daily_rebalance",
                    "group": int(group_id),
                    "return": float(sub[label].mean()),
                    "count": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def _staggered_daily_group_returns(
    df: pd.DataFrame,
    label: str,
    *,
    group_count: int,
    min_assets: int,
    holding_period_days: int,
    daily_return_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    daily_label = one_day_label(label)
    source = daily_return_frame if daily_return_frame is not None else df
    if daily_label not in source:
        raise ValueError(
            f"{holding_period_days}-day staggered group backtest requires {daily_label}; pass the one-day return panel explicitly"
        )
    formations: dict[str, dict[int, set[str]]] = {}
    for date, group in df.dropna(subset=["prediction", label]).groupby(
        "date", sort=True
    ):
        if (
            len(group) < max(min_assets, group_count)
            or group["prediction"].nunique() < 2
        ):
            continue
        ranked = group.copy()
        try:
            ranked["group"] = (
                pd.qcut(
                    ranked["prediction"].rank(method="first"), group_count, labels=False
                )
                + 1
            )
        except ValueError:
            continue
        formations[str(date)] = {
            int(group_id): set(sub["symbol"].astype(str))
            for (group_id, sub) in ranked.groupby("group")
        }
    staggered = staggered_bucket_returns(
        formations,
        source,
        daily_label=daily_label,
        holding_period_days=holding_period_days,
    )
    if staggered.empty:
        return staggered
    return staggered.rename(columns={"bucket": "group"}).assign(
        group=lambda frame: frame["group"].astype(int),
        count=lambda frame: frame["target_name_count"].astype(int),
        daily_return_label=daily_label,
    )


def _summarize_model_returns(
    returns: pd.DataFrame, holding_period_days: int = 1
) -> dict[str, Any]:
    if returns.empty:
        return {
            "long_excess_mean": math.nan,
            "short_excess_mean": math.nan,
            "long_short_mean": math.nan,
            "long_short_annual": math.nan,
            "sharpe": math.nan,
            "max_drawdown": math.nan,
            "win_rate": math.nan,
            "turnover_mean": math.nan,
            "return_days": 0.0,
            "holding_period_days": float(holding_period_days),
            "annualization_periods": 252.0,
        }
    series = returns["long_short"].astype(float).fillna(0.0)
    summary = daily_portfolio_summary(series)
    result: dict[str, Any] = {
        "long_excess_mean": float(returns["long_excess"].mean()),
        "short_excess_mean": float(returns["short_excess"].mean()),
        "long_short_mean": float(series.mean()),
        "long_short_annual": summary["annual"],
        "sharpe": summary["sharpe"],
        "max_drawdown": summary["max_drawdown"],
        "win_rate": float((series > 0).mean()),
        "turnover_mean": float(returns["turnover"].mean()),
        "return_days": float(len(series)),
        "holding_period_days": float(holding_period_days),
        "annualization_periods": 252.0,
        "portfolio_mode": str(returns["portfolio_mode"].iloc[0]),
    }
    if holding_period_days > 1:
        result.update(
            {
                "staggered_sleeve_count": float(holding_period_days),
                "max_active_sleeves": float(returns["active_sleeves"].max()),
                "mean_capital_fraction": float(returns["capital_fraction"].mean()),
                "return_coverage": float(returns["return_coverage"].mean()),
            }
        )
    return result


def _summarize_group_returns(
    group_returns: pd.DataFrame, group_count: int
) -> dict[str, float]:
    empty = _empty_group_summary(group_count)
    if group_returns.empty:
        return empty
    annualization_periods = 252.0
    pivot = group_returns.pivot_table(
        index="date", columns="group", values="return", aggfunc="mean"
    ).sort_index()
    mean_by_group = pivot.mean()
    if len(mean_by_group) < 2:
        return empty
    corr = (
        pd.Series(mean_by_group.index, index=mean_by_group.index)
        .rank()
        .corr(mean_by_group.rank(), method="pearson")
    )
    spread_series = pivot.get(group_count, pd.Series(dtype=float)) - pivot.get(
        1, pd.Series(dtype=float)
    )
    spread = float(spread_series.mean()) if not spread_series.empty else math.nan
    daily_monotonic = pivot.apply(_row_group_spearman, axis=1)
    out = {
        "group_monotonic_spearman": float(corr) if not pd.isna(corr) else math.nan,
        "group_top_minus_bottom": spread,
        "group_top_minus_bottom_annual": spread * annualization_periods
        if not math.isnan(spread)
        else math.nan,
        "group_top_minus_bottom_icir": _daily_ir(spread_series),
        "group_top_minus_bottom_sharpe": _annualized_ir(spread_series),
        "group_top_minus_bottom_win_rate": float((spread_series > 0).mean())
        if not spread_series.empty
        else math.nan,
        "group_top_mean": float(mean_by_group.get(group_count, math.nan)),
        "group_bottom_mean": float(mean_by_group.get(1, math.nan)),
        "group_daily_monotonic_mean": float(daily_monotonic.mean())
        if daily_monotonic.notna().any()
        else math.nan,
        "group_daily_monotonic_positive_ratio": float((daily_monotonic > 0).mean())
        if daily_monotonic.notna().any()
        else math.nan,
    }
    middle_spreads = []
    for group_id in range(1, group_count + 1):
        series = (
            pivot[group_id].dropna() if group_id in pivot else pd.Series(dtype=float)
        )
        out[f"group_{group_id}_mean"] = (
            float(series.mean()) if not series.empty else math.nan
        )
        out[f"group_{group_id}_annual"] = (
            float(series.mean() * annualization_periods)
            if not series.empty
            else math.nan
        )
        out[f"group_{group_id}_icir"] = _daily_ir(series)
        out[f"group_{group_id}_sharpe"] = _annualized_ir(series)
    for group_id in range(1, group_count):
        if group_id in pivot and group_id + 1 in pivot:
            adjacent = pivot[group_id + 1] - pivot[group_id]
        else:
            adjacent = pd.Series(dtype=float)
        if 1 < group_id < group_count - 1 and (not adjacent.empty):
            middle_spreads.append(float(adjacent.mean()))
        out[f"group_{group_id + 1}_minus_{group_id}_mean"] = (
            float(adjacent.mean()) if not adjacent.empty else math.nan
        )
        out[f"group_{group_id + 1}_minus_{group_id}_annual"] = (
            float(adjacent.mean() * annualization_periods)
            if not adjacent.empty
            else math.nan
        )
        out[f"group_{group_id + 1}_minus_{group_id}_icir"] = _daily_ir(adjacent)
        out[f"group_{group_id + 1}_minus_{group_id}_sharpe"] = _annualized_ir(adjacent)
    out["group_middle_adjacent_mean"] = (
        float(np.mean(middle_spreads)) if middle_spreads else math.nan
    )
    return out


def _empty_group_summary(group_count: int) -> dict[str, float]:
    out = {
        "group_monotonic_spearman": math.nan,
        "group_top_minus_bottom": math.nan,
        "group_top_minus_bottom_annual": math.nan,
        "group_top_minus_bottom_icir": math.nan,
        "group_top_minus_bottom_sharpe": math.nan,
        "group_top_minus_bottom_win_rate": math.nan,
        "group_top_mean": math.nan,
        "group_bottom_mean": math.nan,
        "group_daily_monotonic_mean": math.nan,
        "group_daily_monotonic_positive_ratio": math.nan,
        "group_middle_adjacent_mean": math.nan,
    }
    for group_id in range(1, group_count + 1):
        out[f"group_{group_id}_mean"] = math.nan
        out[f"group_{group_id}_annual"] = math.nan
        out[f"group_{group_id}_icir"] = math.nan
        out[f"group_{group_id}_sharpe"] = math.nan
    for group_id in range(1, group_count):
        out[f"group_{group_id + 1}_minus_{group_id}_mean"] = math.nan
        out[f"group_{group_id + 1}_minus_{group_id}_annual"] = math.nan
        out[f"group_{group_id + 1}_minus_{group_id}_icir"] = math.nan
        out[f"group_{group_id + 1}_minus_{group_id}_sharpe"] = math.nan
    return out


def _daily_ir(series: pd.Series) -> float:
    s = series.dropna().astype(float)
    if s.empty:
        return math.nan
    std = s.std(ddof=1)
    return float(s.mean() / std) if std and (not np.isnan(std)) else math.nan


def _annualized_ir(series: pd.Series) -> float:
    value = _daily_ir(series)
    return float(value * math.sqrt(252)) if not math.isnan(value) else math.nan


def _row_group_spearman(row: pd.Series) -> float:
    valid = row.dropna()
    if len(valid) < 2:
        return math.nan
    left = pd.Series(valid.index, index=valid.index, dtype=float).rank()
    return float(left.corr(valid.rank(), method="pearson"))


def _checkpoint_metric_summary(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict) or not row:
        return {}
    fields = (
        "run_name",
        "model",
        "split",
        "market",
        "markets",
        "market_count",
        "expected_market_count",
        "feature_count",
        "sota_score",
        "sota_score_raw",
        "rank_ic_mean",
        "rank_icir",
        "ic_mean",
        "icir",
        "long_short_annual",
        "sharpe",
        "max_drawdown",
        "turnover_mean",
        "win_rate",
        "group_top_minus_bottom_icir",
    )
    return {field: row.get(field) for field in fields if field in row}


def _select_best_test_artifacts(
    best_test: dict[str, Any], artifacts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    model = best_test.get("model") if best_test else None
    if not model:
        return []
    return [
        row
        for row in artifacts
        if row.get("model") in {model, "all"}
        and row.get("type") in {"checkpoint", "preprocessor"}
    ]


def _group_cumulative_returns(group_returns: pd.DataFrame) -> pd.DataFrame:
    if group_returns.empty:
        return pd.DataFrame()
    out = group_returns.copy()
    out["date"] = out["date"].astype(str)
    sort_cols = [
        col for col in ["model", "split", "market", "group", "date"] if col in out
    ]
    out = out.sort_values(sort_cols)
    group_cols = [col for col in ["model", "split", "market", "group"] if col in out]
    out["equity"] = out.groupby(group_cols)["return"].transform(
        lambda s: (1.0 + s.fillna(0.0)).cumprod()
    )
    out["drawdown"] = out.groupby(group_cols)["equity"].transform(
        lambda s: s / s.cummax() - 1.0
    )
    return out


def _group_diagnostics_frame(group_returns: pd.DataFrame) -> pd.DataFrame:
    if group_returns.empty:
        return pd.DataFrame()
    group_cols = [col for col in ["model", "split", "market"] if col in group_returns]
    rows: list[dict[str, Any]] = []
    grouped = group_returns.groupby(group_cols) if group_cols else [((), group_returns)]
    for keys, frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        pivot = frame.pivot_table(
            index="date", columns="group", values="return", aggfunc="mean"
        ).sort_index()
        for group_id in sorted(pivot.columns):
            series = pivot[group_id].dropna()
            rows.append(
                {
                    **meta,
                    "diagnostic": f"group_{int(group_id)}",
                    "mean": float(series.mean()) if not series.empty else math.nan,
                    "annual": float(series.mean() * 252.0)
                    if not series.empty
                    else math.nan,
                    "icir": _daily_ir(series),
                    "sharpe": _annualized_ir(series),
                    "days": int(series.notna().sum()),
                }
            )
        for left, right in zip(sorted(pivot.columns), sorted(pivot.columns)[1:]):
            spread = pivot[right] - pivot[left]
            rows.append(
                {
                    **meta,
                    "diagnostic": f"group_{int(right)}_minus_{int(left)}",
                    "mean": float(spread.mean()) if not spread.empty else math.nan,
                    "annual": float(spread.mean() * 252.0)
                    if not spread.empty
                    else math.nan,
                    "icir": _daily_ir(spread),
                    "sharpe": _annualized_ir(spread),
                    "days": int(spread.notna().sum()),
                }
            )
        if len(pivot.columns) >= 2:
            spread = pivot[pivot.columns[-1]] - pivot[pivot.columns[0]]
            rows.append(
                {
                    **meta,
                    "diagnostic": f"group_{int(pivot.columns[-1])}_minus_{int(pivot.columns[0])}",
                    "mean": float(spread.mean()) if not spread.empty else math.nan,
                    "annual": float(spread.mean() * 252.0)
                    if not spread.empty
                    else math.nan,
                    "icir": _daily_ir(spread),
                    "sharpe": _annualized_ir(spread),
                    "days": int(spread.notna().sum()),
                }
            )
        daily_monotonic = pivot.apply(_row_group_spearman, axis=1)
        rows.append(
            {
                **meta,
                "diagnostic": "daily_group_monotonic",
                "mean": float(daily_monotonic.mean())
                if daily_monotonic.notna().any()
                else math.nan,
                "annual": math.nan,
                "icir": _daily_ir(daily_monotonic),
                "sharpe": math.nan,
                "days": int(daily_monotonic.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def _best_summary_row(
    rows: list[dict[str, Any]], split: str, preferred_market: str = ""
) -> dict[str, Any]:
    candidates = [
        row for row in rows if row.get("status") == "ok" and row.get("split") == split
    ]
    if not candidates:
        return {}
    market_candidates = [
        row for row in candidates if row.get("market") == preferred_market
    ]
    if market_candidates:
        candidates = market_candidates
    return max(candidates, key=_model_score_tuple)


def _best_sota_summary_row(
    rows: list[dict[str, Any]], split: str, cfg: AppConfig
) -> dict[str, Any]:
    markets = _sota_markets(cfg)
    candidates = _model_market_average_rows(rows, split, markets)
    if candidates:
        return max(candidates, key=_model_score_tuple)
    return _best_summary_row(rows, split, preferred_market=SOTA_FALLBACK_MARKET)


def _matching_split_row(
    rows: list[dict[str, Any]], reference: dict[str, Any], split: str
) -> dict[str, Any]:
    if not reference:
        return {}
    model = reference.get("model")
    market = reference.get("market")
    if market == SOTA_MARKET_AVG:
        markets = [str(x) for x in reference.get("markets", []) if x]
        if markets:
            return _model_market_average_row(rows, split, model, markets)
    for row in rows:
        if (
            row.get("status") == "ok"
            and row.get("model") == model
            and (row.get("market") == market)
            and (row.get("split") == split)
        ):
            return row
    return {}


def _model_market_average_rows(
    rows: list[dict[str, Any]], split: str, markets: list[str]
) -> list[dict[str, Any]]:
    models = sorted(
        {
            str(row.get("model"))
            for row in rows
            if row.get("status") == "ok" and row.get("split") == split
        }
    )
    return [
        row
        for model in models
        if (row := _model_market_average_row(rows, split, model, markets))
    ]


def _model_market_average_row(
    rows: list[dict[str, Any]], split: str, model: str, markets: list[str]
) -> dict[str, Any]:
    rows_by_market = {
        str(row.get("market")): row
        for row in rows
        if row.get("status") == "ok"
        and row.get("split") == split
        and (row.get("model") == model)
    }
    market_rows = [
        rows_by_market[market] for market in markets if market in rows_by_market
    ]
    if not market_rows:
        return {}
    missing = [market for market in markets if market not in rows_by_market]
    out: dict[str, Any] = {
        "run_name": market_rows[0].get("run_name"),
        "model": model,
        "split": split,
        "market": SOTA_MARKET_AVG,
        "markets": [str(row.get("market")) for row in market_rows],
        "missing_markets": missing,
        "market_count": len(market_rows),
        "expected_market_count": len(markets),
        "label": SOTA_MARKET_AVG,
        "status": "ok",
        "feature_count": int(
            max((_finite(row.get("feature_count"), default=0.0) for row in market_rows))
        ),
        "training_detail": market_rows[0].get("training_detail", {}),
        "market_metrics": [_compact_market_row(row) for row in market_rows],
    }
    out["rows"] = int(
        sum((_finite(row.get("rows"), default=0.0) for row in market_rows))
    )
    out["days"] = float(
        np.mean([_finite(row.get("days"), default=math.nan) for row in market_rows])
    )
    numeric_fields = [
        "ic_mean",
        "ic_std",
        "icir",
        "ic_positive_ratio",
        "ic_t_stat",
        "rank_ic_mean",
        "rank_ic_std",
        "rank_icir",
        "rank_ic_positive_ratio",
        "rank_ic_t_stat",
        "long_excess_mean",
        "short_excess_mean",
        "long_short_mean",
        "long_short_annual",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "turnover_mean",
        "return_days",
        "group_monotonic_spearman",
        "group_top_minus_bottom",
        "group_top_minus_bottom_annual",
        "group_top_minus_bottom_icir",
        "group_top_minus_bottom_sharpe",
        "group_top_minus_bottom_win_rate",
        "group_top_mean",
        "group_bottom_mean",
        "group_daily_monotonic_mean",
        "group_daily_monotonic_positive_ratio",
        "group_middle_adjacent_mean",
    ]
    group_count = 0
    while any((f"group_{group_count + 1}_mean" in row for row in market_rows)):
        group_count += 1
    for group_id in range(1, group_count + 1):
        numeric_fields.extend(
            [
                f"group_{group_id}_mean",
                f"group_{group_id}_annual",
                f"group_{group_id}_icir",
                f"group_{group_id}_sharpe",
            ]
        )
    for group_id in range(1, group_count):
        numeric_fields.extend(
            [
                f"group_{group_id + 1}_minus_{group_id}_mean",
                f"group_{group_id + 1}_minus_{group_id}_annual",
                f"group_{group_id + 1}_minus_{group_id}_icir",
                f"group_{group_id + 1}_minus_{group_id}_sharpe",
            ]
        )
    for field in numeric_fields:
        values = [_finite(row.get(field), default=math.nan) for row in market_rows]
        values = [value for value in values if not math.isnan(value)]
        out[field] = float(np.mean(values)) if values else math.nan
    _attach_sota_scores(out)
    return out


def _attach_sota_scores(row: dict[str, Any]) -> None:
    raw_values = []
    normalized_values = []
    components: dict[str, float] = {}
    for field in SOTA_SCORE_FIELDS:
        value = _finite(row.get(field), default=math.nan)
        if math.isnan(value):
            continue
        raw_values.append(value)
        scale = SOTA_SCORE_SCALES[field]
        normalized = value / scale
        normalized_values.append(normalized)
        components[field] = normalized
    row["sota_score_raw"] = float(np.mean(raw_values)) if raw_values else -math.inf
    row["sota_score"] = (
        float(np.mean(normalized_values)) if normalized_values else -math.inf
    )
    row["sota_score_fields"] = list(SOTA_SCORE_FIELDS)
    row["sota_score_scales"] = dict(SOTA_SCORE_SCALES)
    row["sota_score_components"] = components


def _compact_market_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": row.get("model"),
        "split": row.get("split"),
        "market": row.get("market"),
        "rows": row.get("rows"),
        "days": row.get("days"),
        "rank_ic_mean": row.get("rank_ic_mean"),
        "rank_icir": row.get("rank_icir"),
        "ic_mean": row.get("ic_mean"),
        "icir": row.get("icir"),
        "long_short_annual": row.get("long_short_annual"),
        "sharpe": row.get("sharpe"),
        "max_drawdown": row.get("max_drawdown"),
        "turnover_mean": row.get("turnover_mean"),
        "win_rate": row.get("win_rate"),
        "group_top_minus_bottom_icir": row.get("group_top_minus_bottom_icir"),
    }


def _sota_markets(cfg: AppConfig) -> list[str]:
    configured = _market_names(cfg)
    preferred = ["ashare", "hs300", "csi500", "csi1000"]
    out = [market for market in preferred if market in configured]
    out.extend((market for market in configured if market not in out))
    return out


def _model_score_tuple(
    row: dict[str, Any] | None,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    if not row:
        return (-math.inf,) * 9
    return (
        _finite(row.get("market_count"), default=1.0),
        _finite(row.get("sota_score")),
        _finite(row.get("rank_ic_mean")),
        _finite(row.get("rank_icir")),
        _finite(row.get("ic_mean")),
        _finite(row.get("icir")),
        _finite(row.get("long_short_annual")),
        _finite(row.get("sharpe")),
        -abs(_finite(row.get("max_drawdown"), default=0.0)),
    )


def _is_model_improvement(
    best_test: dict[str, Any], previous_sota: dict[str, Any] | None
) -> bool:
    if not best_test:
        return False
    if not previous_sota or not previous_sota.get("best_test"):
        return True
    return _model_score_tuple(best_test) > _model_score_tuple(
        previous_sota.get("best_test")
    )


def _sota_payload(
    best_valid: dict[str, Any],
    best_test: dict[str, Any],
    run_name: str,
    features: list[str],
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "best_valid": best_valid,
        "best_test": best_test,
        "feature_count": len(features),
        "features": features,
    }


def _compact_sota(previous_sota: dict[str, Any] | None) -> dict[str, Any]:
    if not previous_sota:
        return {}
    best_valid = previous_sota.get("best_valid", {})
    best_test = previous_sota.get("best_test", {})
    return {
        "run_name": previous_sota.get("run_name"),
        "feature_count": previous_sota.get("feature_count"),
        "valid_model": best_valid.get("model"),
        "valid_market": best_valid.get("market"),
        "valid_markets": best_valid.get("markets", []),
        "valid_sota_score": best_valid.get("sota_score"),
        "valid_sota_score_raw": best_valid.get("sota_score_raw"),
        "valid_rank_ic_mean": best_valid.get("rank_ic_mean"),
        "valid_rank_icir": best_valid.get("rank_icir"),
        "valid_ic_mean": best_valid.get("ic_mean"),
        "valid_icir": best_valid.get("icir"),
        "valid_long_short_annual": best_valid.get("long_short_annual"),
        "valid_sharpe": best_valid.get("sharpe"),
        "test_model": best_test.get("model"),
        "test_market": best_test.get("market"),
        "test_markets": best_test.get("markets", []),
        "test_sota_score": best_test.get("sota_score"),
        "test_sota_score_raw": best_test.get("sota_score_raw"),
        "test_rank_ic_mean": best_test.get("rank_ic_mean"),
        "test_rank_icir": best_test.get("rank_icir"),
        "test_ic_mean": best_test.get("ic_mean"),
        "test_icir": best_test.get("icir"),
        "test_long_short_annual": best_test.get("long_short_annual"),
        "test_sharpe": best_test.get("sharpe"),
    }


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return math.nan
    if np.nanstd(left) == 0 or np.nanstd(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _turnover(prev: set[str] | None, curr: set[str]) -> float:
    if prev is None:
        return math.nan
    union = len(prev | curr)
    if union == 0:
        return math.nan
    return float(1.0 - len(prev & curr) / union)


def _finite(value: Any, default: float = -math.inf) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if not math.isnan(out) else default


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for (k, v) in value.items()}
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_sanitize(value), indent=2, ensure_ascii=False, default=str)


def _write_plots(*args, **kwargs):
    return {}


def _write_model_outputs(
    output_dir,
    payload,
    summary_rows,
    daily_ic_rows,
    daily_return_rows,
    group_return_rows,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / "model_summary.csv", index=False)
    _concat_frames(daily_ic_rows).to_csv(output_dir / "daily_ic.csv", index=False)
    _concat_frames(daily_return_rows).to_csv(
        output_dir / "daily_long_short.csv", index=False
    )
    _concat_frames(group_return_rows).to_csv(
        output_dir / "group_returns.csv", index=False
    )
    (output_dir / "model_validation.json").write_text(
        _json_dumps(payload), encoding="utf-8"
    )
