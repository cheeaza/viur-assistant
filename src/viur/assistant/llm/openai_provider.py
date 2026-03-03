import typing as t
import json
import openai
from viur.core import errors
from .provider import LLMProviderInterface, LLMRequest, LLMResponse, LLMMessage


class OpenAIProvider(LLMProviderInterface):
    provider_type = "openai"

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    @classmethod
    def from_config(cls, config: t.Any) -> "OpenAIProvider":
        if not config.api_openai_key:
            raise errors.InternalServerError(descr="OpenAI API Key is missing")
        return cls(api_key=config.api_openai_key)

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
        request.max_tokens = skel.get("openai_max_tokens", 1024)

    def prepare_translation_request(self, request: LLMRequest) -> None:
        request.extra_params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "viur-assistant-translation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"}
                    },
                    "required": ["answer"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

    def parse_translation_response(self, response: LLMResponse) -> str:
        try:
            data = json.loads(response.content)
            return data["answer"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return response.content

    def build_image_description_content(
        self,
        *,
        text: str,
        base64_image: str
    ) -> list[dict[str, t.Any]]:
        return [
            {
                "type": "text",
                "text": text,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low",
                },
            },
        ]

    def prepare_image_description_request(self, request: LLMRequest) -> None:
        request.extra_params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "viur-assistant-image-desc",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"}
                    },
                    "required": ["answer"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

    def parse_image_description_response(self, response: LLMResponse) -> str:
        try:
            data = json.loads(response.content)
            return data["answer"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return response.content
