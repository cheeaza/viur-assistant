from __future__ import annotations
import json
from json import JSONDecodeError

from .base import BaseProvider, ModelInfo

# Only chat-completion model prefixes; extend as OpenAI releases new series.
_CHAT_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


class OpenAIProvider(BaseProvider):
    def supports_vision(self) -> bool:
        return True

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
        import openai
        client = openai.OpenAI(api_key=self._api_key)

        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "viur-assistant",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )

        try:
            return json.loads(response.choices[0].message.content)["answer"]
        except (JSONDecodeError, KeyError):
            return response.choices[0].message.content or ""

    def list_models(self) -> list[ModelInfo]:
        self._require_api_key()
        import openai
        client = openai.OpenAI(api_key=self._api_key)
        result = []
        for model in client.models.list():
            model_id = model.id or ""
            if not model_id or not any(model_id.startswith(p) for p in _CHAT_PREFIXES):
                continue
            result.append(ModelInfo(
                model_id=model_id,
                name=model_id,
                description=None,
                input_token_limit=None,
                output_token_limit=None,
            ))
        return result
