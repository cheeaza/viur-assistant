"""
OpenAI provider implementation for viur-assistant.

Wraps the OpenAI Chat Completions API and maps its response fields to the
normalised :class:`~base.CompletionResult`.

Supported capabilities:

* Text completions via ``gpt-*``, ``o1``, ``o3``, ``o4``, and ``chatgpt-*`` models.
* Vision – ``image_url`` content parts are passed through unchanged (OpenAI accepts
  the same data-URI format used in the canonical message schema).
* JSON output – ``response_format={"type": "json_object"}`` is set when
  ``response_format="json"``.

.. note::
   OpenAI does not provide a public "thinking" mode comparable to Anthropic's
   extended thinking or Gemini's ``ThinkingConfig``.  Reasoning models (o1, o3,
   o4) perform reasoning internally; ``max_thinking_tokens`` is ignored.
"""

from __future__ import annotations

from .base import BaseProvider, CompletionResult, ModelInfo

# Prefixes of model IDs that support the Chat Completions API.
# Extend this tuple as OpenAI releases new model series.
_CHAT_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


class OpenAIProvider(BaseProvider):
    """LLM provider backed by the OpenAI Chat Completions API.

    Supports text and vision.  API key is read from
    :attr:`~viur.assistant.config.AssistantConfig.api_openai_key`.
    """

    def supports_vision(self) -> bool:
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
        import openai
        client = openai.OpenAI(api_key=self._api_key)

        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        raw_finish = choice.finish_reason or ""
        finish_reason = "length" if raw_finish == "length" else ("stop" if raw_finish == "stop" else raw_finish or None)

        usage = response.usage
        return CompletionResult(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
        )

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
