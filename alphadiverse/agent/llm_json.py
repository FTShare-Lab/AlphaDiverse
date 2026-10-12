from __future__ import annotations
from collections.abc import Callable
from typing import Any
import os
from ..llm import MissingFormalAnswerError, call_chat_messages, parse_json_response


def call_validated_json(
    messages: list[dict[str, str]],
    *,
    call_site: str,
    metadata: dict[str, Any],
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    retries = _contract_retries()
    request_messages = messages
    last_error = ""
    formal_answer_retries = 0
    for attempt in range(retries + 1):
        try:
            raw = call_chat_messages(
                request_messages,
                call_site=call_site,
                metadata={
                    **metadata,
                    **({"contract_retry": attempt} if attempt else {}),
                    **(
                        {"formal_answer_retry": formal_answer_retries}
                        if formal_answer_retries
                        else {}
                    ),
                },
            )
        except MissingFormalAnswerError as exc:
            last_error = str(exc)
            if attempt < retries:
                formal_answer_retries += 1
                request_messages = [
                    *request_messages,
                    {
                        "role": "user",
                        "content": "Your previous generation ended without a formal answer. Complete your reasoning and emit </think> before the final answer. Then provide one complete JSON object matching the original contract and all previously reported violations. JSON inside the thinking block is not a final answer. Do not output only reasoning or a code fence.",
                    },
                ]
            continue
        try:
            parsed = parse_json_response(raw)
            if not isinstance(parsed, dict):
                raise ValueError("response must be a JSON object")
            return (validator(parsed), raw)
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
        if attempt < retries:
            request_messages = contract_retry_messages(messages, raw, last_error)
    raise ValueError(
        f"{call_site} response failed after {retries} contract retries: {last_error}"
    )


def contract_retry_messages(messages, raw, error):
    return [
        *messages,
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": f"The response violated the strict contract. Return one corrected JSON object only, with no prose or code fence. Violations: {error}",
        },
    ]


def _contract_retries() -> int:
    raw = (os.getenv("ALPHADIVERSE_LLM_CONTRACT_RETRIES") or "2").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "ALPHADIVERSE_LLM_CONTRACT_RETRIES must be an integer"
        ) from exc
    if not 0 <= value <= 5:
        raise ValueError("ALPHADIVERSE_LLM_CONTRACT_RETRIES must be between 0 and 5")
    return value
