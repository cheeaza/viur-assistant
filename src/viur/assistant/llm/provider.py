import typing as t
from dataclasses import dataclass, field


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


class LLMProviderInterface:
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def stream_response(self, request: LLMRequest) -> t.Iterable[LLMResponse]:
        raise NotImplementedError
