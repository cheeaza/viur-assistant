import json
import typing as t
from dataclasses import dataclass, field

from viur.core import errors


@dataclass
class LLMMessage:
    role: t.Literal["system", "user", "assistant"]
    content: str | list[dict[str, t.Any]]


@dataclass
class LLMRequest:
    prompt: str | None = None
    messages: list[LLMMessage] = field(default_factory=list)
    system_prompt: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stop_sequences: list[str] | None = None
    extra_params: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    raw_response: t.Any = None


_PROVIDER_REGISTRY: dict[str, type["LLMProviderInterface"]] = {}


def register_provider(provider_cls: type["LLMProviderInterface"]) -> type["LLMProviderInterface"]:
    if not getattr(provider_cls, "provider_type", None):
        raise ValueError("provider_type must be defined on provider class")
    _PROVIDER_REGISTRY[provider_cls.provider_type] = provider_cls
    return provider_cls


class LLMProviderInterface:
    provider_type: str

    @classmethod
    def from_config(cls, config: t.Any) -> "LLMProviderInterface":
        raise NotImplementedError

    @classmethod
    def create(cls, provider_type: str, config: t.Any) -> "LLMProviderInterface":
        provider_cls = _PROVIDER_REGISTRY.get(provider_type)
        if not provider_cls:
            raise errors.InternalServerError(descr=f"Unsupported LLM provider: {provider_type}")
        return provider_cls.from_config(config)

    def generate_response(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def stream_response(self, request: LLMRequest) -> t.Iterable[LLMResponse]:
        raise NotImplementedError

    def prepare_translation_request(self, request: LLMRequest) -> None:
        pass

    def parse_translation_response(self, response: LLMResponse) -> str:
        return response.content

    def build_image_description_content(
        self,
        *,
        text: str,
        base64_image: str
    ) -> list[dict[str, t.Any]]:
        raise NotImplementedError

    def prepare_image_description_request(self, request: LLMRequest) -> None:
        pass

    def parse_image_description_response(self, response: LLMResponse) -> str:
        return response.content

    def prepare_script_request(
        self,
        request: LLMRequest,
        *,
        skel: dict[str, t.Any],
        max_thinking_tokens: int,
        enable_caching: bool
    ) -> None:
        pass

    def parse_script_response(self, response: LLMResponse) -> str:
        return json.dumps({"code": response.content})
