"""Play task registration: rigid-object pretraining (play2perfect / SimToolReal).

One env class (``PlayEnv``); the cfg class picks the robot and self-collision:

    Isaacsimenvs-Play-Direct-v0                       Kuka iiwa14 + Sharpa (SimToolReal's robot)

Legacy, kept for their checkpoints (the left-hand setups; the right-hand robot trains Play on
another branch):

    Isaacsimenvs-PlaySelfCollision-Direct-v0          the Kuka with robot self-collision on
    Isaacsimenvs-PlayFlexivWuji-Direct-v0             Flexiv Rizon 4s + left Wuji Hand 2
    Isaacsimenvs-PlayFlexivWujiHumanLike-Direct-v0    that robot with human-like joint limits
    Isaacsimenvs-PlayFlexivSharpa-Direct-v0           the Rizon carrying the Kuka's Sharpa

Entry points:
- ``env_cfg_entry_point``           → PlayEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → configs/task/Play.yaml overlay
- ``rl_games_cfg_entry_point``      → configs/train/PlayPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → configs/train/PlaySAPG.yaml (default)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym

from isaacsimenvs import CONFIG_DIR as _CFG_DIR

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


def _register(task_id: str, cfg_class: str, task_yaml: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaacsimenvs.tasks.play.play_env:PlayEnv",
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"isaacsimenvs.tasks.play.play_env_cfg:{cfg_class}",
            "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / task_yaml),
            "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
            "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PlaySAPG.yaml"),
        },
    )


_register("Isaacsimenvs-Play-Direct-v0", "PlayEnvCfg", "Play.yaml")

# --- legacy
_register("Isaacsimenvs-PlaySelfCollision-Direct-v0", "PlaySelfCollisionEnvCfg", "Play.yaml")
_register("Isaacsimenvs-PlayFlexivWuji-Direct-v0", "PlayFlexivWujiEnvCfg", "PlayFlexivWuji.yaml")
_register("Isaacsimenvs-PlayFlexivWujiHumanLike-Direct-v0", "PlayFlexivWujiHumanLikeEnvCfg", "PlayFlexivWujiHumanLike.yaml")
_register("Isaacsimenvs-PlayFlexivSharpa-Direct-v0", "PlayFlexivSharpaEnvCfg", "PlayFlexivSharpa.yaml")
