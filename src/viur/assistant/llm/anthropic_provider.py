import typing as t
import anthropic
from viur.core import errors
from .provider import LLMProviderInterface, LLMRequest, LLMResponse, LLMMessage


class AnthropicProvider(LLMProviderInterface):
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, t.Any]]:
        converted = []
        for msg in messages:
            converted.append({
                "role": msg.role,
                "content": msg.content
            })
        return converted

    def generate_response(self, request: LLMRequest) -> LLMResponse:
        messages = self._convert_messages(request.messages)

        params = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": request.system_prompt,
            "stop_sequences": request.stop_sequences,
            **request.extra_params
        }
        # filter None values
        params = {k: v for k, v in params.items() if v is not None}

        try:
            message = self.client.messages.create(**params)
        except Exception as e:
            raise errors.InternalServerError(descr=f"Anthropic error: {str(e)}") from e

        content = ""
        for block in message.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            usage={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
            stop_reason=message.stop_reason,
            raw_response=message
        )
