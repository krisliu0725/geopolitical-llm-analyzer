"""Google Gemini API client with timeout."""

from __future__ import annotations

import time

from google import genai

from .base import BaseLLMClient, LLMResponse


class GeminiClient(BaseLLMClient):
    @staticmethod
    def provider_name() -> str:
        return "gemini"

    def generate(
        self,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        client = genai.Client(api_key=self.api_key)

        parts = []
        if system_prompt:
            parts.append(f"[System]\n{system_prompt}\n")
        parts.append(user_prompt)

        start = time.monotonic()
        response = client.models.generate_content(
            model=self.model_name,
            contents="\n".join(parts),
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        text = response.text or ""
        try:
            tokens = response.usage_metadata.total_token_count
        except (AttributeError, TypeError):
            tokens = None

        return LLMResponse(
            text=text,
            tokens_used=tokens,
            latency_ms=latency_ms,
        )
