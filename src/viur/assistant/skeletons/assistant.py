import typing as t

from viur.core.bones import *
from viur.core.skeleton import Skeleton


class AssistantSkel(Skeleton):
    kindName: t.Final[str] = "viur-assistant"

    provider = SelectBone(
        descr="LLM Provider",
        values={
            "openai": "OpenAI",
            "anthropic": "Anthropic",
        },
        defaultValue="openai",
    )

    anthropic_model = StringBone(
        descr="Model",
        params={"category": "Anthropic"},
        defaultValue="claude-3-7-sonnet-20250219",
    )

    anthropic_max_tokens = NumericBone(
        descr="Maximum Tokens",
        params={"category": "Anthropic"},
        min=512,
        max=4096,
        defaultValue=1024,
    )

    anthropic_max_thinking_tokens = NumericBone(
        descr="Maximum Thinking Tokens",
        params={"category": "Anthropic"},
        min=0,
        max=64000,
        defaultValue=0,
    )

    anthropic_temperature = NumericBone(
        descr="Temperature",
        params={"category": "Anthropic"},
        precision=1,
        min=0,
        max=1,
        defaultValue=1.0,
    )
    anthropic_system_prompt = TextBone(
        descr="Systemprompt",
        params={"category": "Anthropic"},
        defaultValue="You are a coding-assistant that helps develop python-code for accessing a viur-backend. You only output json-strings containing a single key named \"code\".",
    )

    openai_model = StringBone(
        descr="Model",
        params={"category": "OpenAi"},
        defaultValue="gpt-4o-mini",
    )

    openai_max_tokens = NumericBone(
        descr="Maximum Tokens",
        params={"category": "OpenAi"},
        min=512,
        max=4096,
        defaultValue=1024,
    )

    openai_temperature = NumericBone(
        descr="Temperature",
        params={"category": "OpenAi"},
        precision=1,
        min=0,
        max=1,
        defaultValue=1.0,
    )

    openai_system_prompt = TextBone(
        descr="Systemprompt",
        params={"category": "OpenAi"},
        defaultValue="You are a coding-assistant that helps develop python-code for accessing a viur-backend. You only output json-strings containing a single key named \"code\".",
    )
