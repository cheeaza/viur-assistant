from __future__ import annotations
import base64

from .base import BaseProvider, ModelInfo


class GeminiProvider(BaseProvider):
    def supports_vision(self) -> bool:
        return True

    def supports_thinking(self) -> bool:
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
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            if isinstance(content, str):
                parts = [types.Part(text=content)]
            else:
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(types.Part(text=part["text"]))
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        media_type, b64data = url.split(";base64,")
                        media_type = media_type.split(":")[1]
                        parts.append(types.Part(
                            inline_data=types.Blob(
                                mime_type=media_type,
                                data=base64.b64decode(b64data),
                            )
                        ))
            contents.append(types.Content(role=role, parts=parts))

        config_kwargs: dict = {}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if max_thinking_tokens > 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=max_thinking_tokens,
            )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        )
        return (response.text or "").strip()

    def list_models(self) -> list[ModelInfo]:
        self._require_api_key()
        from google import genai
        client = genai.Client(api_key=self._api_key)
        result = []
        for model in client.models.list():
            if "generateContent" not in (model.supported_actions or []):
                continue
            model_id = (model.name or "").removeprefix("models/").strip()
            if not model_id:
                continue
            result.append(ModelInfo(
                model_id=model_id,
                name=(model.display_name or model_id).strip(),
                description=model.description,
                input_token_limit=model.input_token_limit,
                output_token_limit=model.output_token_limit,
            ))
        return result
