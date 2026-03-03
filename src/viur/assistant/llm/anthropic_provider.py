import typing as t

import anthropic
from viur.core import errors

from .provider import LLMProviderInterface, LLMRequest, LLMResponse, LLMMessage


class AnthropicProvider(LLMProviderInterface):
    provider_type = "anthropic"

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    @classmethod
    def from_config(cls, config: t.Any) -> "AnthropicProvider":
        if not config.api_anthropic_key:
            raise errors.InternalServerError(descr="Anthropic API Key is missing")
        return cls(api_key=config.api_anthropic_key)

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, t.Any]]:
        converted = []
        for msg in messages:
            converted.append({
                "role": msg.role,
                "content": msg.content
            })
        return converted

    def prepare_script_request(
        self,
        request: LLMRequest,
        *,
        skel: dict[str, t.Any],
        max_thinking_tokens: int,
        enable_caching: bool
    ) -> None:
        request.max_tokens = skel["anthropic_max_tokens"] + max_thinking_tokens
        if max_thinking_tokens > 0:
            request.extra_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": max_thinking_tokens,
            }
        if enable_caching:
            # Placeholder für spätere, spezifische Cache-Optionen
            pass

    def parse_script_response(self, response: LLMResponse) -> str:
        if hasattr(response.raw_response, "model_dump_json"):
            return response.raw_response.model_dump_json()
        return super().parse_script_response(response)

    def build_image_description_content(
        self,
        *,
        text: str,
        base64_image: str
    ) -> list[dict[str, t.Any]]:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image,
                },
            },
            {
                "type": "text",
                "text": text,
            }
        ]

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
