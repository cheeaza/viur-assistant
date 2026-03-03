import typing as t
import openai
from viur.core import errors
from .provider import LLMProviderInterface, LLMRequest, LLMResponse, LLMMessage


class OpenAIProvider(LLMProviderInterface):
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

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
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        params = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stop": request.stop_sequences,
            **request.extra_params
        }
        # filter None values
        params = {k: v for k, v in params.items() if v is not None}

        try:
            response = self.client.chat.completions.create(**params)
        except openai.APIConnectionError as e:
            raise errors.ServiceUnavailable(descr=f"OpenAI connection error: {e}") from e
        except openai.RateLimitError as e:
            raise errors.HTTPException(status=429, name="RateLimitError", descr=f"OpenAI rate limit: {e}") from e
        except openai.APIStatusError as e:
            raise errors.HTTPException(status=e.status_code, name="APIStatusError", descr=f"OpenAI error: {e}") from e

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            stop_reason=choice.finish_reason,
            raw_response=response
        )
