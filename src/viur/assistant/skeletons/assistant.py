"""
AssistantSkel – singleton skeleton for the viur-assistant configuration.

Stores the runtime settings for the :class:`~viur.assistant.modules.assistant.Assistant`
singleton: which AI model to use (via the ``ai_model`` relation) and the generation
parameters that apply to all endpoints (temperature, token limits, system prompt).

**After changing** ``ai_model`` **in the admin, re-save the entry** so that the
``refKeys`` (``model_id``, ``provider``) are written into the relational data.
Without a re-save the assistant cannot resolve the provider at request time.
"""

import typing as t

from viur.core.bones import *
from viur.core.skeleton import Skeleton


class AssistantSkel(Skeleton):
    """Singleton skeleton holding the viur-assistant runtime configuration.

    All bones here are read at request time by
    :meth:`~viur.assistant.modules.assistant.Assistant._get_provider_and_model`
    and the individual endpoint methods.  Change them in the admin to switch
    models or tune generation parameters without redeploying.
    """

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
