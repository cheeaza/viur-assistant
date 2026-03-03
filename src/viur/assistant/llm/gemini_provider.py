import typing as t

import google.generativeai as genai
from viur.core import errors

from .provider import LLMProviderInterface, LLMRequest, LLMMessage


class GeminiProvider(LLMProviderInterface):
    provider_type = "gemini"

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    @classmethod
    def from_config(cls, config: t.Any) -> "GeminiProvider":
        if not config.api_gemini_key:
            raise errors.InternalServerError(descr="Gemini API Key is missing")
        return cls(api_key=config.api_gemini_key)

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, t.Any]]:
        converted: list[dict[str, t.Any]] = []
        for msg in messages:
            if msg.role == "system":
                # System wird über system_instruction gesetzt
                continue

            role = "model" if msg.role == "assistant" else msg.role
            parts: list[dict[str, t.Any]] = []

            if isinstance(msg.content, str):
                parts.append({"text": msg.content})
            else:
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
                    elif isinstance(item, dict) and "text" in item:
                        parts.append({"text": item["text"]})
                    else:
                        parts.append(item)

            converted.append({
                "role": role,
                "parts": parts
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

    def build_image_description_content(
        self,
        *,
        text: str,
        base64_image: str
    ) -> list[dict[str, t.Any]]:
        raise errors.InternalServerError("Image description not supported for Gemini")
