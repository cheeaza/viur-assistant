from __future__ import annotations
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ModelInfo(t.TypedDict):
    model_id: str
    name: str
    description: str | None
    input_token_limit: int | None
    output_token_limit: int | None


@dataclass
class CompletionResult:
    text: str
    finish_reason: str | None = None  # normalized: "stop" | "length" | "content_filter" | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class BaseProvider(ABC):
    """Common interface for all LLM providers.

    Messages follow the OpenAI format. Multimodal content (images) uses:
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<data>"}}
    Each provider implementation translates this to its own API format.

    ``finish_reason`` values are normalized across all providers:
        - ``"stop"``           – natural end
        - ``"length"``         – token limit hit (truncated)
        - ``"content_filter"`` – filtered by safety system
        - ``None``             – unknown / not reported
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _require_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(f"{self.__class__.__name__}: no API key configured.")

    @abstractmethod
    def complete_detailed(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        max_thinking_tokens: int = 0,
        enable_caching: bool = False,
        response_format: str | None = None,
    ) -> CompletionResult:
        """Send a completion request and return a structured result.

        :param response_format: ``"json"`` to request JSON output, ``None`` for plain text.
        """
        ...

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        max_thinking_tokens: int = 0,
        enable_caching: bool = False,
        response_format: str | None = None,
    ) -> str:
        """Convenience wrapper – returns only the response text."""
        return self.complete_detailed(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            max_thinking_tokens=max_thinking_tokens,
            enable_caching=enable_caching,
            response_format=response_format,
        ).text

    def list_models(self) -> list[ModelInfo]:
        """Return available models for this provider. Override to implement."""
        return []

    def supports_vision(self) -> bool:
        return False

    def supports_thinking(self) -> bool:
        return False
