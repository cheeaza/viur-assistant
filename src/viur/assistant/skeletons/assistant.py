import typing as t

from viur.core.bones import *
from viur.core.skeleton import Skeleton


class AssistantSkel(Skeleton):
    kindName: t.Final[str] = "viur-assistant"

    # --- Common LLM settings ---
    provider = SelectBone(
        descr="LLM Provider",
        values={
            "anthropic": "Anthropic",
            "openai": "OpenAI",
            "gemini": "Gemini (Google)",
        },
        defaultValue="anthropic",
    )

    model = StringBone(
        descr="Model",
        defaultValue="claude-3-7-sonnet-20250219",
    )

    max_tokens = NumericBone(
        descr="Maximum Tokens",
        min=1,
        max=128000,
        defaultValue=1024,
    )

    temperature = NumericBone(
        descr="Temperature",
        params={"category": "Anthropic"},
        precision=1,
        min=0,
        max=2,
        defaultValue=1.0,
    )

    system_prompt = TextBone(
        descr="Systemprompt",
        params={"category": "Anthropic"},
        defaultValue="You are a coding-assistant that helps develop python-code for accessing a viur-backend. You only output json-strings containing a single key named \"code\".",
    )

    # --- Provider selection ---
