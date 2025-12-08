# llm_world/utils/openrouter_helper.py
from __future__ import annotations

import os
from typing import Iterable, Mapping, Any

from openai import OpenAI


class OpenRouterClient:
    """
    Thin OO wrapper around an OpenAI-compatible client configured for OpenRouter.

    Example:
        client = OpenRouterClient()          # reads OPENROUTER_API_KEY
        text = client.chat("qwen_next_80b", "Say hello.")
    """

    # Base URL for OpenRouter
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    # Canonical model identifiers
    QWEN_NEXT_80B = "qwen/qwen3-next-80b-a3b-thinking"
    GPT_5 = "openai/gpt-5.1"
    GEMINI_3_PRO = "google/gemini-3-pro-preview"
    CLAUDE_OPUS_45 = "anthropic/claude-opus-4.5"
    GROK_4_1_FAST = "x-ai/grok-4.1-fast"

    # Optional short-name mapping
    MODELS: Mapping[str, str] = {
        "qwen_next_80b": QWEN_NEXT_80B,
        "gpt_5": GPT_5,
        "gemini_3_pro": GEMINI_3_PRO,
        "claude_opus_45": CLAUDE_OPUS_45,
        "grok_4_1_fast": GROK_4_1_FAST,
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        """
        Initialize an OpenRouterClient.

        Args:
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
            base_url: Override the default OpenRouter base URL if needed.
            default_headers: Optional default headers (e.g. HTTP-Referer, X-Title).
        """
        if api_key is None:
            # TODO: add an api_key
            api_key = "your_key"
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY not set and no api_key provided "
                    "to OpenRouterClient()."
                )

        if base_url is None:
            base_url = self.DEFAULT_BASE_URL

        client_kwargs: dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
        }
        if default_headers:
            client_kwargs["default_headers"] = dict(default_headers)

        self._client = OpenAI(**client_kwargs)

    @property
    def client(self) -> OpenAI:
        """Underlying OpenAI-compatible client, if you need full control."""
        return self._client

    def resolve_model(self, model: str) -> str:
        """
        Accept either a short name ('gpt_5') or a full model string and
        return the full OpenRouter model identifier.
        """
        return self.MODELS.get(model, model)

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        extra_messages: Iterable[dict[str, Any]] | None = None,
    ) -> str:
        """
        Simple chat helper: send a single user prompt and return the text reply.

        Args:
            model: Short name ('gpt_5') or full model id
                   ('openai/gpt-5.1', etc.).
            prompt: User message content.
            system: Optional system message.
            temperature: Optional sampling temperature.
            extra_messages: Optional additional messages to include
                            before the final user message.

        Returns:
            Assistant message content as a stripped string.
        """
        full_model = self.resolve_model(model)

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": full_model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    

if __name__ == "__main__":
    # from utils.openrouter_helper import OpenRouterClient

    # Initialize once (e.g. at module import or in main)
    orc = OpenRouterClient()  # uses OPENROUTER_API_KEY from env

    prompt = "Explain the meaning of life in 3 words."
    llm_output = orc.chat("qwen_next_80b", prompt)
    print(llm_output)

# or: orc.chat(OpenRouterClient.QWEN_NEXT_80B, prompt)

