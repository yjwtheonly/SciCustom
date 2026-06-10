"""Thin wrappers around the OpenAI / Anthropic / Google LLM SDKs.

We only need a chat-completion interface, so each backend exposes the same
``chat(system, user, **kwargs) -> str`` method.  The factory below routes a
short model identifier to the right backend.

The mapping mirrors the set of models used in the paper (Appendix C):

* ``gpt-5`` / ``gpt-5-chat`` / ``gpt-4o``  -> OpenAI
* ``claude-opus-4-5`` / ``claude-haiku-...`` -> Anthropic
* ``gemini-3-pro-preview`` / ``gemini-2.5-flash`` -> Google

External-only models such as Qwen, DeepSeek, Grok, Kimi and Mistral are
expected to be served via OpenAI-compatible endpoints; pass ``base_url`` and
``api_key`` overrides through the factory to use them.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class ChatModel(Protocol):
    name: str

    def chat(self, system: str, user: str, **kwargs: Any) -> str: ...


@dataclass
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible
# ---------------------------------------------------------------------------

class OpenAIChat:
    """OpenAI-compatible chat backend.

    Used both for native OpenAI models (GPT-5, GPT-4o, ...) and for hosted
    third-party endpoints that expose an OpenAI-style API (DeepSeek, Qwen,
    Kimi, Mistral).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 4,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Please `pip install openai>=1.30`") from exc

        self.name = model
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            timeout=timeout,
        )
        self._max_retries = max_retries

    def chat(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int | None = None, **kwargs: Any) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - intentional broad retry
                last_err = exc
                wait = min(2 ** attempt, 30)
                logger.warning("OpenAIChat[%s] attempt %d/%d failed (%s); sleeping %.1fs",
                               self.name, attempt + 1, self._max_retries, exc, wait)
                time.sleep(wait)
        assert last_err is not None
        raise last_err


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicChat:
    def __init__(
        self,
        model: str = "claude-opus-4-5",
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 4,
    ):
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Please `pip install anthropic`") from exc

        self.name = model
        self._client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
        )
        self._max_retries = max_retries

    def chat(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 4096, **kwargs: Any) -> str:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.name,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                # The Anthropic SDK returns a list of content blocks; we
                # only consume the first text block since we never request
                # tool use here.
                parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
                return "".join(parts)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wait = min(2 ** attempt, 30)
                logger.warning("AnthropicChat[%s] attempt %d/%d failed (%s); sleeping %.1fs",
                               self.name, attempt + 1, self._max_retries, exc, wait)
                time.sleep(wait)
        assert last_err is not None
        raise last_err


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiChat:
    def __init__(
        self,
        model: str = "gemini-3-pro-preview",
        *,
        api_key: str | None = None,
        max_retries: int = 4,
    ):
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Please `pip install google-genai`") from exc

        self.name = model
        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self._max_retries = max_retries

    def chat(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 4096, **kwargs: Any) -> str:
        # The Gemini SDK exposes the system instruction via the
        # ``GenerateContentConfig`` block.
        from google.genai import types  # type: ignore

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs,
        )
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.name,
                    contents=user,
                    config=config,
                )
                return resp.text or ""
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wait = min(2 ** attempt, 30)
                logger.warning("GeminiChat[%s] attempt %d/%d failed (%s); sleeping %.1fs",
                               self.name, attempt + 1, self._max_retries, exc, wait)
                time.sleep(wait)
        assert last_err is not None
        raise last_err


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_KNOWN_OPENAI_PREFIXES = (
    "gpt-",
    "o1",
    "o3",
    "o4",
    "openai/",
)
_KNOWN_ANTHROPIC_PREFIXES = ("claude-",)
_KNOWN_GEMINI_PREFIXES = ("gemini-",)


def get_chat_model(spec: str | dict, **overrides) -> ChatModel:
    """Build a chat model from a short identifier.

    ``spec`` can be either a model id string (e.g., ``"gpt-5"``) or a dict
    with explicit ``provider``, ``model``, and optional ``base_url`` /
    ``api_key`` keys.  The latter form is what the YAML configs use.
    """
    if isinstance(spec, dict):
        provider = spec.get("provider", "").lower()
        model = spec["model"]
        kwargs = {k: v for k, v in spec.items() if k not in ("provider", "model")}
        kwargs.update(overrides)
    else:
        provider = _guess_provider(spec)
        model = spec
        kwargs = overrides

    if provider == "openai":
        return OpenAIChat(model, **kwargs)
    if provider == "anthropic":
        return AnthropicChat(model, **kwargs)
    if provider in ("google", "gemini"):
        return GeminiChat(model, **kwargs)

    # Fall through to OpenAI-compatible.  Most "open" models we evaluated
    # (Qwen, DeepSeek, Kimi, Mistral, Grok) are reachable this way.
    logger.info("Provider for %r not explicitly known - using OpenAI-compatible client", model)
    return OpenAIChat(model, **kwargs)


def _guess_provider(model: str) -> str:
    name = model.lower()
    if name.startswith(_KNOWN_OPENAI_PREFIXES):
        return "openai"
    if name.startswith(_KNOWN_ANTHROPIC_PREFIXES):
        return "anthropic"
    if name.startswith(_KNOWN_GEMINI_PREFIXES):
        return "google"
    return "openai-compatible"
