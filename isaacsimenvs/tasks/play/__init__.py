"""Play task registration.

Registers ``Isaacsimenvs-Play-Direct-v0`` with the gymnasium registry
for the DirectRLEnv training path.

Entry points:
- ``env_cfg_entry_point``           → PlayEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task/Play.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/PlayPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/PlaySAPG.yaml (default)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from .play_env import PlayEnv
    from .play_env_cfg import PlayEnvCfg

__all__ = ["PlayEnv", "PlayEnvCfg"]


def __getattr__(name: str):
    """Resolve ``PlayEnv`` / ``PlayEnvCfg`` on first access rather than at import.

    ``gym.register`` below takes *entry-point strings*, so registration never needed the classes
    imported -- but importing them here pulled ``isaaclab.envs`` into every ``import
    isaacsimenvs``, and through it the Kit-backed PhysX stack. The Newton backend runs kit-less,
    where that import chain is not merely slow but unavailable, so a process that only wants the
    task ids must be able to get them without it. Deferring also keeps ``check_registration.py``
    and the Kit-free tests from booting Isaac.
    """
    if name == "PlayEnv":
        from .play_env import PlayEnv

        return PlayEnv
    if name == "PlayEnvCfg":
        from .play_env_cfg import PlayEnvCfg

        return PlayEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-Play-Direct-v0",
    entry_point="isaacsimenvs.tasks.play.play_env:PlayEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.play.play_env_cfg:PlayEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "Play.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PlaySAPG.yaml"),
    },
)
