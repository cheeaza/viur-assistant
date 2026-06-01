"""
viur-assistant

AI-based assistance module plugin for ViUR
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
