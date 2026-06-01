from pathlib import Path

from viur.core import conf

# Before we can import any skeleton we must allow this dir in the viur-core
_dir = str(Path(__file__).parent)
if _dir not in conf.skeleton_search_path:
    conf.skeleton_search_path.append(_dir)
    conf.skeleton_search_path.append(
        _dir
        .replace(str(conf.instance.project_base_path), "")
        .replace(str(conf.instance.core_base_path), "")
    )

from .assistant import AssistantSkel
from .aimodel import AimodelSkel

__all__ = [
    "AssistantSkel", "AimodelSkel"
]
