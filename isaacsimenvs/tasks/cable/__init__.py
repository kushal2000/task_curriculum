"""Cable goal-reaching task registration.

Registers ``Isaacsimenvs-Cable-Direct-v0``: the Play goal-reaching task with a deformable cable
in place of the rigid procedural tool, on a coupled MJWarp + VBD solve.

The env subclasses ``PlayNewtonEnv``, so the observation stays 140-dim and the pretrained
checkpoint loads unchanged. Requires ``.venv_isaaclab3`` and a populated USD cache.

Entry points:
- ``env_cfg_entry_point``           → CableEnvCfg
- ``env_cfg_yaml_entry_point``      → cfg/task/Cable.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/PlayPPO.yaml   (shared with Play)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/PlaySAPG.yaml  (shared with Play)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from .cable_env import CableEnv
    from .cable_env_cfg import CableEnvCfg

__all__ = ["CableEnv", "CableEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"


def __getattr__(name: str):
    """Deferred class import; see the note in ``isaacsimenvs/tasks/play/__init__.py``."""
    if name == "CableEnv":
        from .cable_env import CableEnv

        return CableEnv
    if name == "CableEnvCfg":
        from .cable_env_cfg import CableEnvCfg

        return CableEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


gym.register(
    id="Isaacsimenvs-Cable-Direct-v0",
    entry_point="isaacsimenvs.tasks.cable.cable_env:CableEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cable.cable_env_cfg:CableEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "Cable.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PlaySAPG.yaml"),
    },
)
