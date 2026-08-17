"""Cloth folding task registration.

Registers ``Isaacsimenvs-Cloth-Direct-v0``: the Play task with a VBD cloth sheet in place of the
rigid procedural tool, on a coupled MJWarp + VBD solve. Fold the far half of the sheet onto the
near half; state and goal are keypoints on the moving half.

Subclasses ``PlayNewtonEnv``, so the observation stays 140-dim and the existing harness, player and
renderer work unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from .cloth_env import ClothEnv
    from .cloth_env_cfg import ClothEnvCfg

__all__ = ["ClothEnv", "ClothEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"


def __getattr__(name: str):
    """Deferred class import; see the note in ``isaacsimenvs/tasks/play/__init__.py``."""
    if name == "ClothEnv":
        from .cloth_env import ClothEnv

        return ClothEnv
    if name == "ClothEnvCfg":
        from .cloth_env_cfg import ClothEnvCfg

        return ClothEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


gym.register(
    id="Isaacsimenvs-Cloth-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.cloth_env:ClothEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.cloth_env_cfg:ClothEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "Cloth.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PlaySAPG.yaml"),
    },
)
