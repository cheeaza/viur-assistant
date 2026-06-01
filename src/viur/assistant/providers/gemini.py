from __future__ import annotations
import base64

from .base import BaseProvider, CompletionResult, ModelInfo


_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
}


class GeminiProvider(BaseProvider):
    def supports_vision(self) -> bool:
        return True

    def supports_thinking(self) -> bool:
        return True

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
        if response_format == "json":
            config_kwargs["response_mime_type"] = "application/json"

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        )

        text = (response.text or "").strip()
        candidates = getattr(response, "candidates", None) or []
        raw_finish = str(getattr(candidates[0], "finish_reason", "") or "") if candidates else ""
        finish_reason = _FINISH_REASON_MAP.get(raw_finish.upper(), "stop" if raw_finish else None)

        usage_meta = getattr(response, "usage_metadata", None)
        return CompletionResult(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=getattr(usage_meta, "prompt_token_count", None) if usage_meta else None,
            completion_tokens=getattr(usage_meta, "candidates_token_count", None) if usage_meta else None,
            total_tokens=getattr(usage_meta, "total_token_count", None) if usage_meta else None,
        )

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
