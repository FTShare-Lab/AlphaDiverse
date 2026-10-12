from __future__ import annotations
from typing import Any
import json
from ..config import SearchValidationConfig
from ..llm import call_chat_messages, parse_json_response
from .utils import (
    _compact_factor_validation,
    _compact_history,
    _compact_portfolio_for_analysis,
    _llm_model_context,
    _llm_prompt_token_budget,
    _message_token_count,
    _model_feedback_sentence,
    _pack_json_items,
    _shrink_to_message_budget,
)

DETAILED_HISTORY_ROUNDS = 6


class AnalysisModule:
    def analyze(
        self,
        portfolio: dict[str, Any],
        validation: dict[str, Any],
        model_validation: dict[str, Any],
        history: list[dict[str, Any]],
        use_llm: bool,
        current_round: int,
        max_rounds: int,
        model_sota: dict[str, Any] | None = None,
        model_retained_factors: list[str] | None = None,
    ) -> dict[str, Any]:
        accepted = validation.get("accepted", [])
        best = accepted[0] if accepted else None
        rule_feedback = _rule_feedback(portfolio, validation, best)
        if model_validation:
            rule_feedback = f"{rule_feedback} Model validation: {_model_feedback_sentence(model_validation)}"
        payload: dict[str, Any] = {
            "portfolio_supported": bool(accepted),
            "best_factor": best,
            "feedback": rule_feedback,
            "used_llm": False,
        }
        if use_llm:
            llm_payload = self._llm_analyze(
                portfolio,
                validation,
                model_validation,
                history,
                current_round=current_round,
                max_rounds=max_rounds,
                model_sota=model_sota or {},
                model_retained_factors=model_retained_factors or [],
            )
            payload["llm_feedback"] = llm_payload
            payload["used_llm"] = bool(llm_payload.get("used_llm"))
        return payload

    def _llm_analyze(
        self,
        portfolio: dict[str, Any],
        validation: dict[str, Any],
        model_validation: dict[str, Any],
        history: list[dict[str, Any]],
        current_round: int,
        max_rounds: int,
        model_sota: dict[str, Any],
        model_retained_factors: list[str],
    ) -> dict[str, Any]:
        compact_validation = _compact_factor_validation(validation)
        fold_count = _configured_fold_count(model_validation, model_sota)
        if fold_count == 1:
            evidence_instruction = "Use only the sole inner time-fold search result and inner-only factor evidence; there is no cross-fold consistency evidence, and final holdout outcomes are unavailable."
        else:
            evidence_instruction = "Use only the inner-fold aggregate, fold consistency, and inner-only factor evidence; final holdout outcomes are unavailable."
        prompt = {
            "role": "analysis",
            "instruction": f"Judge which research plans and realized factors are supported, whether the portfolio adds model value, and what the next Planner step should try. {evidence_instruction} Output JSON only.",
            "research_portfolio": _compact_portfolio_for_analysis(portfolio),
            "validation": compact_validation,
            "model_context": _llm_model_context(
                model_sota, model_retained_factors, round_candidate=model_validation
            ),
            "history": [],
            "context_note": "history may be truncated to fit the model context window.",
            "output_schema": {
                "portfolio_supported": True,
                "sota_decision": "model_retain|factor_only_retain|reject|unclear",
                "feedback": "short feedback for the next round",
                "next_steps": ["one concrete next step"],
            },
        }
        token_budget = _llm_prompt_token_budget("agent.analysis")
        remaining = max(
            0, token_budget - _message_token_count(_analysis_messages(prompt))
        )
        compact_history = _compact_history(
            history,
            limit=min(DETAILED_HISTORY_ROUNDS, max(0, current_round - 1)),
            model_sota=model_sota,
        )
        (packed_history, _) = _pack_json_items(
            compact_history, remaining, newest_first=True
        )
        prompt["history"] = packed_history
        prompt = _shrink_to_message_budget(
            prompt, _analysis_messages, token_budget, [("history",)]
        )
        prompt["context_note"] = _context_note(
            history_included=len(prompt.get("history", [])),
            history_total=len(compact_history),
        )
        messages = _analysis_messages(prompt)
        try:
            raw = call_chat_messages(
                messages,
                call_site="agent.analysis",
                metadata={
                    "round": current_round,
                    "max_rounds": max_rounds,
                    "module": "analysis",
                },
            )
            parsed = parse_json_response(raw)
            if not isinstance(parsed, dict):
                parsed = {"feedback": str(parsed)}
            parsed["raw_response"] = raw
            parsed["used_llm"] = True
            return parsed
        except Exception as exc:
            return {"used_llm": False, "error": str(exc)}


def _analysis_messages(prompt: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are the Analysis module in a quant agent loop. Output JSON only.",
        },
        {
            "role": "user",
            "content": json.dumps(prompt, ensure_ascii=False, default=str),
        },
    ]


def _configured_fold_count(
    model_validation: dict[str, Any], model_sota: dict[str, Any]
) -> int:
    for payload in (model_validation, model_sota):
        search_config = (
            payload.get("search_config", {}) if isinstance(payload, dict) else {}
        )
        value = (
            search_config.get("fold_count") if isinstance(search_config, dict) else None
        )
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                pass
        fold_metrics = (
            payload.get("fold_metrics", []) if isinstance(payload, dict) else []
        )
        if isinstance(fold_metrics, list) and fold_metrics:
            return len(fold_metrics)
    return SearchValidationConfig().fold_count


def _context_note(history_included: int, history_total: int) -> str:
    return (
        f"Token-budgeted context: history included {history_included}/{history_total}."
    )


def _rule_feedback(
    portfolio: dict[str, Any], validation: dict[str, Any], best: dict[str, Any] | None
) -> str:
    mechanisms = [
        str(plan.get("mechanism_id"))
        for plan in portfolio.get("research_plans", [])
        if isinstance(plan, dict) and plan.get("mechanism_id")
    ]
    accepted = validation.get("accepted_factors", [])
    if best:
        rank_ic = best.get("ic", {}).get("rank_ic_mean")
        coverage = best.get("coverage")
        return f"Research portfolio {mechanisms} produced {len(accepted)} factor-accepted factor(s). Best factor {best.get('factor')} has rank_ic_mean={rank_ic}, coverage={coverage}. Continue variants that can improve the configured inner validation SOTA beyond the current retained feature set."
    reject_reasons = {}
    for row in validation.get("decisions", []):
        if row.get("decision") == "reject":
            reject_reasons[row.get("reason", "unknown")] = (
                reject_reasons.get(row.get("reason", "unknown"), 0) + 1
            )
    return f"Research portfolio {mechanisms} did not retain factors. Reject reasons: {reject_reasons}. Replace failed plans with mechanisms that are semantically and behaviorally distinct from the current portfolio while preserving executable simplicity."
