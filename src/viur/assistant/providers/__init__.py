from __future__ import annotations

from viur.assistant.config import CONFIG
from .anthropic import AnthropicProvider
from .base import BaseProvider, CompletionResult, ModelInfo
from .gemini import GeminiProvider
from .openai import OpenAIProvider


def get_provider(provider_name: str) -> BaseProvider:
    match provider_name:
        case "anthropic":
            return AnthropicProvider(api_key=CONFIG.api_anthropic_key)
        case "gemini":
            return GeminiProvider(api_key=CONFIG.api_gemini_key)
        case "openai":
            return OpenAIProvider(api_key=CONFIG.api_openai_key)
        case _:
            raise ValueError(f"Unknown provider: {provider_name!r}")


__all__ = ["BaseProvider", "CompletionResult", "ModelInfo", "AnthropicProvider", "GeminiProvider", "OpenAIProvider",
           "get_provider"]
