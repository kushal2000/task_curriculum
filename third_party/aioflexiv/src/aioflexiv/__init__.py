from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._loader import load_rt
from .controller import FlexivController
from .mujoco_model import (
    DEFAULT_MUJOCO_SITE_NAME,
    MujocoModelBackend,
    default_mujoco_model_path,
    default_mujoco_scene_path,
)
from .robot import DEFAULT_MUJOCO_TIMESTEP, FlexivRobotInterface, MujocoRobotInterface
from .tools import ToolPayload

try:
    __version__ = version("aioflexiv")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "FlexivController",
    "DEFAULT_MUJOCO_TIMESTEP",
    "DEFAULT_MUJOCO_SITE_NAME",
    "FlexivRobotInterface",
    "MujocoModelBackend",
    "MujocoRobotInterface",
    "ToolPayload",
    "default_mujoco_model_path",
    "default_mujoco_scene_path",
    "load_rt",
]
