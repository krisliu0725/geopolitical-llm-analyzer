"""LLM client abstraction — factory + concrete implementations."""

from __future__ import annotations

from .base import BaseLLMClient, LLMResponse
from .deepseek_client import DeepSeekClient
from .gemini_client import GeminiClient
from .rate_limiter import RateLimiter

_CLIENT_REGISTRY = {
    "deepseek": DeepSeekClient,
    "gemini": GeminiClient,
    "custom": DeepSeekClient,
}


def get_client(
    provider: str,
    api_key: str,
    model_name: str,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> BaseLLMClient:
    """Factory: return the correct LLM client for the given provider."""
    provider_lower = provider.lower()
    client_cls = _CLIENT_REGISTRY.get(provider_lower)
    if client_cls is None:
        raise ValueError(
            f"Unknown provider: '{provider}'. Supported: {list(_CLIENT_REGISTRY.keys())}"
        )

    if provider_lower == "custom":
        if not base_url:
            raise ValueError("Custom provider requires a base_url")
        return DeepSeekClient(api_key=api_key, model_name=model_name, base_url=base_url, timeout=timeout)

    return client_cls(api_key=api_key, model_name=model_name, timeout=timeout)


__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "DeepSeekClient",
    "GeminiClient",
    "RateLimiter",
    "get_client",
]
