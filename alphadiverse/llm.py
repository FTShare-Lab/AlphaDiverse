from dataclasses import dataclass, field
from urllib.parse import urlsplit
import json
import math
import os
import re
import time
import requests


class MissingFormalAnswerError(ValueError):
    pass


class ModelEndpointError(RuntimeError):
    def __init__(self, message, *, call_site, reason, status_code=None):
        super().__init__(message)
        self.diagnostic = {
            "call_site": call_site,
            "reason": reason,
            "status_code": status_code,
        }


@dataclass(frozen=True)
class ChatBackendConfig:
    backend: str
    model: str
    api_base: str
    api_key: str = field(repr=False)
    temperature: float = 0.1
    context_window_tokens: int = 131072
    max_tokens: int = 8192
    qwen_enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    timeout_seconds: float = 300.0


def resolve_chat_backend(call_site="llm", temperature=None):
    role = call_site.rsplit(".", 1)[-1].upper()

    def get(name, default=""):
        return os.getenv(f"ALPHADIVERSE_{role}_{name}") or os.getenv(
            f"ALPHADIVERSE_{name}", default
        )

    (base, model) = (get("BASE_URL"), get("MODEL"))
    parsed = urlsplit(base)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
    ):
        raise ValueError(
            "Set ALPHADIVERSE_BASE_URL to an API base URL without embedded credentials"
        )
    if not model:
        raise ValueError("Set ALPHADIVERSE_MODEL")
    backend = get("BACKEND", "api")
    if backend not in {"api", "local"}:
        raise ValueError("Backend must be api or local")
    key = get("API_KEY")
    if backend == "api" and (not key):
        raise ValueError("Set ALPHADIVERSE_API_KEY in the environment")
    thinking = get("THINKING")
    if thinking not in {"", "0", "1"}:
        raise ValueError("THINKING must be 0 or 1")
    (context, maximum) = (
        int(get("CONTEXT_TOKENS", "131072")),
        int(get("MAX_TOKENS", "8192")),
    )
    if not 128 <= maximum < context - 1024:
        raise ValueError("Output budget must leave space for the prompt")
    timeout = float(get("TIMEOUT_SECONDS", "300"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("TIMEOUT_SECONDS must be finite and positive")
    return ChatBackendConfig(
        backend,
        model,
        base.rstrip("/"),
        key,
        float(get("TEMPERATURE", "0.1")) if temperature is None else temperature,
        context,
        maximum,
        None if thinking == "" else thinking == "1",
        get("REASONING_EFFORT") or None,
        timeout,
    )


def _formal_content(content):
    if not isinstance(content, str):
        raise MissingFormalAnswerError("No formal text answer")
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    elif "<think>" in content:
        raise MissingFormalAnswerError("Reasoning ended without a formal answer")
    if not content.strip():
        raise MissingFormalAnswerError("No formal text answer")
    return content.strip()


def call_chat_messages(messages, temperature=None, call_site="llm", metadata=None):
    cfg = resolve_chat_backend(call_site, temperature)
    budget = cfg.max_tokens
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    prompt_bound = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 1024
    if prompt_bound + budget > cfg.context_window_tokens:
        raise ValueError("Request exceeds the configured conservative context budget")
    with requests.Session() as session:
        if cfg.backend == "local":
            session.trust_env = False
        for attempt in range(3):
            body = dict(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=budget,
            )
            if cfg.qwen_enable_thinking is not None:
                body["chat_template_kwargs"] = {
                    "enable_thinking": cfg.qwen_enable_thinking
                }
            if cfg.reasoning_effort:
                body["reasoning_effort"] = cfg.reasoning_effort
            try:
                response = session.post(
                    cfg.api_base + "/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=(15, cfg.timeout_seconds),
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if attempt == 2:
                    raise ModelEndpointError(
                        "Model endpoint connection failed after three attempts",
                        call_site=call_site,
                        reason=type(exc).__name__,
                    ) from None
                time.sleep(2**attempt)
                continue
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(2**attempt)
                continue
            if response.status_code != 200:
                raise ModelEndpointError(
                    f"Model endpoint returned HTTP {response.status_code}",
                    call_site=call_site,
                    reason="http_status",
                    status_code=response.status_code,
                )
            try:
                choice = response.json()["choices"][0]
            except (ValueError, KeyError, IndexError, TypeError):
                raise ModelEndpointError(
                    "Model endpoint returned an invalid response envelope",
                    call_site=call_site,
                    reason="invalid_response_envelope",
                ) from None
            if choice.get("finish_reason") == "length":
                extended = min(budget * 2, cfg.context_window_tokens - prompt_bound)
                if attempt < 2 and extended > budget:
                    budget = extended
                    continue
                raise MissingFormalAnswerError(
                    "Output was truncated; increase the configured context/output budgets"
                )
            return _formal_content(choice.get("message", {}).get("content"))
    raise RuntimeError("Model call did not return an answer")


def parse_json_response(text):
    text = _formal_content(text)
    if text.startswith("```"):
        text = re.sub("^```(?:json)?\\s*", "", text, flags=re.IGNORECASE)
        text = re.sub("\\s*```$", "", text)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Expected one JSON object")
    return result
