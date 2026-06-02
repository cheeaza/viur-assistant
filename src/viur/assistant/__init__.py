"""
viur-assistant – AI-based assistance plugin for ViUR.

This package provides drop-in modules, skeletons, and bones that add LLM-powered
features to any ViUR project:

* :class:`~viur.assistant.modules.assistant.Assistant` – Singleton with endpoints
  for script generation, image description, and text translation.
* :class:`~viur.assistant.modules.aimodel.Aimodel` – Admin module for managing
  available AI models; can sync model lists from Gemini, OpenAI, and Anthropic.
* :class:`~viur.assistant.bones.image.ImageBone` – A :class:`~viur.core.bones.FileBone`
  variant that bundles an AI-powered alt-text action for the admin UI.
* :class:`~viur.assistant.bones.actions.BoneAction` – String constants for
  admin-triggerable AI actions on individual bones.

**Quick setup** (in your project's ``main.py``)::

    from viur.assistant import CONFIG as ASSISTANT_CONFIG

    ASSISTANT_CONFIG.api_openai_key   = ...
    ASSISTANT_CONFIG.api_anthropic_key = ...
    ASSISTANT_CONFIG.api_gemini_key   = ...

All AI calls are routed through the provider abstraction in
:mod:`viur.assistant.providers`.  The active provider is selected at request time
from the ``ai_model`` relation in the
:class:`~viur.assistant.skeletons.assistant.AssistantSkel` singleton – switching
models requires no code changes.
"""

from .bones.actions import BONE_ACTION_KEY, BoneAction
from .bones.image import ImageBone, ImageBoneRelSkel
from .config import CONFIG
from .modules.assistant import Assistant
from .modules.aimodel import Aimodel
from .skeletons.assistant import AssistantSkel
from .skeletons.aimodel import AimodelSkel


__all__ = [
    "Assistant",
    "Aimodel",
    "BONE_ACTION_KEY",
    "BoneAction",
    "CONFIG",
    "ImageBone",
    "ImageBoneRelSkel",
]
