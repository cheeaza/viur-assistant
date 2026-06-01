import typing as t

from viur.core.bones import *
from viur.core.skeleton import Skeleton


class AssistantSkel(Skeleton):
    kindName: t.Final[str] = "viur-assistant"

    ai_model = RelationalBone(
        descr="AI Model",
        kind="aimodel",
        module="aimodel",
        format="$(dest.name)",
        refKeys=["key", "name", "model_id", "provider"],
        required=True,
    )

    temperature = NumericBone(
        descr="Temperature",
        precision=2,
        min=0,
        max=2,
        defaultValue=1.0,
    )

    max_tokens = NumericBone(
        descr="Maximum Tokens",
        min=512,
        max=65536,
        defaultValue=4096,
    )

    max_thinking_tokens = NumericBone(
        descr="Maximum Thinking Tokens",
        min=0,
        max=64000,
        defaultValue=0,
    )

    system_prompt = TextBone(
        descr="System Prompt",
    )
