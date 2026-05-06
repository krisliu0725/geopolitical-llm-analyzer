"""Abstract base class and dataclass for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    tokens_used: int | None
    latency_ms: int


class BaseLLMClient(ABC):
    """Unified interface for all LLM providers."""

    def __init__(self, api_key: str, model_name: str, timeout: float = 120.0):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    @abstractmethod
    def generate(
        self,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a prompt to the LLM and return the response."""
        ...

    @staticmethod
    @abstractmethod
    def provider_name() -> str:
        """Return the provider identifier string."""
        ...
