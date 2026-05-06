"""DeepSeek / OpenAI-compatible API client with timeout."""

from __future__ import annotations

import time
from openai import OpenAI

from .base import BaseLLMClient, LLMResponse


class DeepSeekClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str, base_url: str | None = None, timeout: float = 120.0):
        super().__init__(api_key, model_name, timeout)
        self._base_url = base_url or "https://api.deepseek.com"

    @staticmethod
    def provider_name() -> str:
        return "deepseek"

    def generate(
        self,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        client = OpenAI(api_key=self.api_key, base_url=self._base_url, timeout=self.timeout)

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        start = time.monotonic()
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        choice = completion.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            tokens_used=completion.usage.total_tokens if completion.usage else None,
            latency_ms=latency_ms,
        )
