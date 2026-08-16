"""Bottle-flip task registration.

Registers ``Isaacsimenvs-BottleFlip-Direct-v0``. The env subclasses ``PlayEnv``, so its
observation and action layout match play2perfect's and its checkpoints load directly.

Entry points:
- ``env_cfg_entry_point``           → BottleFlipEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task/BottleFlip.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/BottleFlipPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/BottleFlipSAPG.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from .bottle_flip_env import BottleFlipEnv
    from .bottle_flip_env_cfg import BottleFlipEnvCfg

__all__ = ["BottleFlipEnv", "BottleFlipEnvCfg"]


def __getattr__(name: str):
    """Deferred class import; see the note in ``isaacsimenvs/tasks/play/__init__.py``."""
    if name == "BottleFlipEnv":
        from .bottle_flip_env import BottleFlipEnv

        return BottleFlipEnv
    if name == "BottleFlipEnvCfg":
        from .bottle_flip_env_cfg import BottleFlipEnvCfg

        return BottleFlipEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-BottleFlip-Direct-v0",
    entry_point="isaacsimenvs.tasks.bottle_flip.bottle_flip_env:BottleFlipEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaacsimenvs.tasks.bottle_flip.bottle_flip_env_cfg:BottleFlipEnvCfg"
        ),
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "BottleFlip.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "BottleFlipPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "BottleFlipSAPG.yaml"),
    },
)
