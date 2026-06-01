from __future__ import annotations

from .base import BaseProvider, ModelInfo


def _to_anthropic_message(msg: dict) -> dict:
    """Translate an OpenAI-format message to Anthropic format."""
    role = msg.get("role", "user")
    content = msg.get("content")

    if isinstance(content, str):
        return {"role": role, "content": content}

    parts = []
    for part in content:
        if part.get("type") == "text":
            parts.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            # data:image/jpeg;base64,<data>
            url = part["image_url"]["url"]
            media_type, b64data = url.split(";base64,")
            media_type = media_type.split(":")[1]
            parts.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64data},
            })
    return {"role": role, "content": parts}


class AnthropicProvider(BaseProvider):
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
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)

        params: dict = {
            "model": model,
            "max_tokens": (max_tokens or 1024) + max_thinking_tokens,
            "messages": [_to_anthropic_message(m) for m in messages],
        }
        if temperature is not None:
            params["temperature"] = temperature
        if system_prompt:
            system_block: dict = {"type": "text", "text": system_prompt}
            if enable_caching:
                system_block["cache_control"] = {"type": "ephemeral"}
            params["system"] = [system_block]
        if max_thinking_tokens > 0:
            params["thinking"] = {"type": "enabled", "budget_tokens": max_thinking_tokens}

        response = client.messages.create(**params)
        return next((block.text for block in response.content if block.type == "text"), "")

    def list_models(self) -> list[ModelInfo]:
        self._require_api_key()
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)
        result = []
        for model in client.models.list():
            result.append(ModelInfo(
                model_id=model.id,
                name=model.display_name,
                description=None,
                input_token_limit=None,
                output_token_limit=None,
            ))
        return result
