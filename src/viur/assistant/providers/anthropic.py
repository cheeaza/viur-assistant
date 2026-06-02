"""
Anthropic provider implementation for viur-assistant.

Translates the OpenAI-compatible message format used throughout the provider
abstraction into Anthropic's ``messages`` API format and maps Anthropic-specific
response fields back to the normalised :class:`~base.CompletionResult`.

Supported capabilities:

* Text completions via ``claude-*`` models.
* Vision – images are converted from the ``image_url`` data-URI format to
  Anthropic's ``image`` block with a base64 source.
* Extended thinking – enabled when ``max_thinking_tokens > 0``.
* Prompt caching – the system-prompt block gets ``cache_control`` when
  ``enable_caching=True``.
"""

from __future__ import annotations

from .base import BaseProvider, CompletionResult, ModelInfo


def _to_anthropic_message(msg: dict) -> dict:
    """Translate an OpenAI-format message dict to Anthropic's message format.

    Handles both plain-text content (``str``) and multimodal content lists.
    ``image_url`` parts with data-URI values are converted to Anthropic's
    ``image`` block with a ``base64`` source.

    :param msg: Message dict with ``role`` and ``content`` keys in OpenAI format.
    :returns: Equivalent message dict in Anthropic API format.
    """
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


# Maps Anthropic stop_reason values to the normalised finish_reason vocabulary.
_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
}


class AnthropicProvider(BaseProvider):
    """LLM provider backed by the Anthropic Messages API.

    Supports text, vision, extended thinking, and ephemeral prompt caching.
    API key is read from :attr:`~viur.assistant.config.AssistantConfig.api_anthropic_key`.
    """

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
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)

        system_text = system_prompt or ""
        if response_format == "json":
            system_text = (system_text + "\nRespond with valid JSON only.").strip()

        params: dict = {
            "model": model,
            "max_tokens": (max_tokens or 1024) + max_thinking_tokens,
            "messages": [_to_anthropic_message(m) for m in messages],
        }
        if temperature is not None:
            params["temperature"] = temperature
        if system_text:
            system_block: dict = {"type": "text", "text": system_text}
            if enable_caching:
                system_block["cache_control"] = {"type": "ephemeral"}
            params["system"] = [system_block]
        if max_thinking_tokens > 0:
            params["thinking"] = {"type": "enabled", "budget_tokens": max_thinking_tokens}

        response = client.messages.create(**params)

        text = next((block.text for block in response.content if block.type == "text"), "")
        finish_reason = _FINISH_REASON_MAP.get(response.stop_reason or "", response.stop_reason)
        usage = response.usage

        return CompletionResult(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            total_tokens=(
                (usage.input_tokens or 0) + (usage.output_tokens or 0)
                if usage else None
            ),
        )

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
