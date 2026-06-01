from __future__ import annotations
import typing as t
from abc import ABC, abstractmethod


class ModelInfo(t.TypedDict):
    model_id: str
    name: str
    description: str | None
    input_token_limit: int | None
    output_token_limit: int | None


class BaseProvider(ABC):
    """Common interface for all LLM providers.

    Messages follow the OpenAI format. Multimodal content (images) uses:
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<data>"}}
    Each provider implementation translates this to its own API format.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _require_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(f"{self.__class__.__name__}: no API key configured.")

    @abstractmethod
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
    ) -> str:
        """Send a completion request and return the response as plain text."""
        ...

    def list_models(self) -> list[ModelInfo]:
        """Return available models for this provider. Override to implement."""
        return []

    def supports_vision(self) -> bool:
        return False

    def supports_thinking(self) -> bool:
        return False
