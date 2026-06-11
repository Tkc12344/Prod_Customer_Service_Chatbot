"""
llm_clients.py
==============
LLM clients for the PROD Support Chatbot.

Roles
-----
  Mistral  — PRIMARY answer writer for all in-scope and OOS responses.
  Gemini   — Contextual analyser and OOS classifier (via langchain-google-genai).

High-traffic resilience
-----------------------
  Mistral: if the primary model returns 429 / 503 / overloaded, automatically
           retries once with MISTRAL_FALLBACK_MODEL (default: open-mistral-7b).

  Gemini:  oos_router.py and app.py initialise Gemini via LangChain.
           The model is read from GEMINI_MODEL (default: gemini-1.5-flash).
           If that is overloaded, set GEMINI_MODEL=gemini-2.0-flash in .env.

Interfaces exposed by MistralLLM
---------------------------------
  generate_content(prompt)       — Gemini-compatible single-string call, returns .text
  chat(messages, ...)            — OpenAI-style multi-turn chat
  invoke(langchain_messages)     — LangChain-compatible invoke()
"""

import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS",  "80"))
_DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# HTTP status codes that indicate the model is overloaded / rate-limited
_OVERLOAD_CODES = {429, 503, 529}

# Retry config
_MAX_RETRIES   = 2          # attempts per model before switching to fallback
_RETRY_DELAY   = 1.0        # seconds between retries on the same model


# ── Response wrapper ──────────────────────────────────────────────────────────

class LLMResponse:
    """Minimal wrapper — exposes .text and .content for cross-API compatibility."""

    def __init__(self, text: str):
        self.text    = text
        self.content = text

    def __str__(self) -> str:
        return self.text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_overload_error(exc: Exception) -> bool:
    """Return True if the exception looks like a rate-limit or overload error."""
    msg = str(exc).lower()
    # openai SDK wraps HTTP errors; check status code attribute first
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in _OVERLOAD_CODES:
        return True
    return any(kw in msg for kw in (
        "429", "503", "529",
        "rate limit", "rate_limit",
        "overloaded", "overload",
        "too many requests",
        "capacity", "quota",
        "service unavailable",
        "high volume", "high traffic",
    ))


def _call_with_retry(client, model: str, fallback_model: str,
                     messages: list, temperature: float, max_tokens: int) -> str:
    """
    Try `model` up to _MAX_RETRIES times.
    If all attempts fail with an overload error, try `fallback_model` once.
    Returns the response text, or raises the last exception.
    """
    last_exc = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_exc = e
            if _is_overload_error(e):
                logger.warning(
                    f"[Mistral] model={model} overloaded "
                    f"(attempt {attempt+1}/{_MAX_RETRIES}): {e}"
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY)
            else:
                raise  # non-overload error — propagate immediately

    # Primary model exhausted — try fallback
    if fallback_model and fallback_model != model:
        logger.warning(
            f"[Mistral] switching to fallback model: {fallback_model}"
        )
        try:
            resp = client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[Mistral] fallback model {fallback_model} also failed: {e}")
            raise e

    raise last_exc


# ── Mistral client ────────────────────────────────────────────────────────────

class MistralLLM:
    """
    OpenAI-compatible client for the Mistral API with automatic model fallback.

    Env vars
    --------
      MISTRAL_API_KEY        — required
      MISTRAL_MODEL          — primary model   (default: mistral-small-latest)
      MISTRAL_FALLBACK_MODEL — fallback model  (default: open-mistral-7b)
      MISTRAL_BASE_URL       — API base URL    (default: https://api.mistral.ai/v1)
    """

    def __init__(self):
        self._name           = "Mistral"
        self._client         = None
        self._model          = os.getenv("MISTRAL_MODEL",          "mistral-small-latest")
        self._fallback_model = os.getenv("MISTRAL_FALLBACK_MODEL", "open-mistral-7b")
        self._enabled        = False
        self._init()

    def _init(self):
        api_key  = os.getenv("MISTRAL_API_KEY", "").strip()
        base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
        if not api_key:
            logger.warning("Mistral: MISTRAL_API_KEY not set — LLM disabled")
            return
        try:
            from openai import OpenAI
            self._client  = OpenAI(api_key=api_key, base_url=base_url)
            self._enabled = True
            logger.info(
                f"Mistral: initialised — "
                f"primary={self._model}  fallback={self._fallback_model}"
            )
        except Exception as e:
            logger.error(f"Mistral: init failed — {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Public interfaces ─────────────────────────────────────────────────────

    def generate_content(self, prompt: str) -> LLMResponse:
        """Gemini-compatible single-string call — returns LLMResponse with .text."""
        if not self._enabled:
            raise RuntimeError("Mistral not initialised")
        text = _call_with_retry(
            self._client, self._model, self._fallback_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=_DEFAULT_TEMPERATURE,
            max_tokens=_DEFAULT_MAX_TOKENS,
        )
        return LLMResponse(text)

    def chat(
        self,
        messages: list,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int    = _DEFAULT_MAX_TOKENS,
    ) -> Optional[str]:
        """OpenAI-style multi-turn chat. Returns plain string or None on failure."""
        if not self._enabled:
            return None
        try:
            return _call_with_retry(
                self._client, self._model, self._fallback_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Mistral.chat failed (all models): {e}")
            return None

    def invoke(self, langchain_messages: list) -> LLMResponse:
        """
        LangChain-compatible invoke() — accepts SystemMessage / HumanMessage objects.
        Returns LLMResponse with .text and .content.
        """
        if not self._enabled:
            raise RuntimeError("Mistral not initialised")
        oai_messages = [
            {
                "role":    "system" if m.__class__.__name__ == "SystemMessage" else "user",
                "content": m.content,
            }
            for m in langchain_messages
        ]
        text = _call_with_retry(
            self._client, self._model, self._fallback_model,
            messages=oai_messages,
            temperature=0.1,
            max_tokens=min(_DEFAULT_MAX_TOKENS, 200),
        )
        return LLMResponse(text)


# ── Singleton ─────────────────────────────────────────────────────────────────

_mistral_instance: Optional[MistralLLM] = None


def get_mistral() -> MistralLLM:
    global _mistral_instance
    if _mistral_instance is None:
        _mistral_instance = MistralLLM()
    return _mistral_instance
