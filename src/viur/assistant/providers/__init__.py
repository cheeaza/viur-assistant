"""
Provider abstraction for viur-assistant.

This package defines a unified interface for LLM providers and ships concrete
implementations for Anthropic, Google Gemini, and OpenAI.

**Public API**

* :class:`BaseProvider` – Abstract base class all providers must implement.
* :class:`CompletionResult` – Dataclass returned by
  :meth:`~BaseProvider.complete_detailed` with text, finish reason, and token counts.
* :class:`ModelInfo` – TypedDict describing a model entry as returned by
  :meth:`~BaseProvider.list_models`.
* :func:`get_provider` – Factory that instantiates the correct provider from a
  name string, reading the API key from
  :data:`~viur.assistant.config.CONFIG`.

**Adding a new provider**

1. Create ``providers/<name>.py`` with a class that extends :class:`BaseProvider`
   and implements :meth:`~BaseProvider.complete_detailed`.
2. Import it here and add a ``case`` branch in :func:`get_provider`.
3. Add ``api_<name>_key`` to :class:`~viur.assistant.config.AssistantConfig` and
   wire it up in the project's ``main.py``.
"""

from __future__ import annotations

from viur.assistant.config import CONFIG
from .anthropic import AnthropicProvider
from .base import BaseProvider, CompletionResult, ModelInfo
from .gemini import GeminiProvider
from .openai import OpenAIProvider


def get_provider(provider_name: str) -> BaseProvider:
    """Instantiate a provider by name, injecting the API key from config.

    :param provider_name: One of ``"anthropic"``, ``"gemini"``, or ``"openai"``.
    :returns: A ready-to-use :class:`BaseProvider` instance.
    :raises ValueError: If *provider_name* is not a known provider.
    """
    match provider_name:
        case "anthropic":
            return AnthropicProvider(api_key=CONFIG.api_anthropic_key)
        case "gemini":
            return GeminiProvider(api_key=CONFIG.api_gemini_key)
        case "openai":
            return OpenAIProvider(api_key=CONFIG.api_openai_key)
        case _:
            raise ValueError(f"Unknown provider: {provider_name!r}")


__all__ = [
    "BaseProvider",
    "CompletionResult",
    "ModelInfo",
    "AnthropicProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "get_provider",
]
