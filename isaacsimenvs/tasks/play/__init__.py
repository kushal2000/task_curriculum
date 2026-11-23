"""Play task registration: rigid-object goal reaching (play2perfect / SimToolReal), the pretraining
task of every cloth policy's prior.

One task, ``Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0``: one env class (``PlayEnv``) and
config (``PlayEnvCfg``), on the Flexiv Rizon 4s + right Wuji Hand 2 (HAND2_DRIVE hand, human-like
joint limits) in the real setup's table and goal geometry.

Entry points:
- ``env_cfg_entry_point``           → PlayEnvCfg (typed defaults in code: the task)
- ``rl_games_cfg_entry_point``      → config/PlayPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → config/PlaySAPG.yaml (default)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym

from isaacsimenvs import PLAY_CONFIG_DIR

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


def _register(task_id: str, cfg_class: str, cfg_module: str = "play_env_cfg", env: str = "play_env:PlayEnv") -> None:
    gym.register(
        id=task_id,
        entry_point=f"isaacsimenvs.tasks.play.{env}",
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"isaacsimenvs.tasks.play.{cfg_module}:{cfg_class}",
            "rl_games_cfg_entry_point": str(PLAY_CONFIG_DIR / "PlayPPO.yaml"),
            "rl_games_sapg_cfg_entry_point": str(PLAY_CONFIG_DIR / "PlaySAPG.yaml"),
        },
    )


_register("Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0", "PlayEnvCfg")
# The same task observed as point clouds: camera cloud + goal cloud for the actor (play_pc_env).
_register("Isaacsimenvs-PlayFlexivWujiRightHumanLikePC-Direct-v0", "PlayPCEnvCfg",
          cfg_module="play_pc_env_cfg", env="play_pc_env:PlayPCEnv")
# ... on the cloth tasks' robot and table: real joint limits, 0.6 x 0.4 m table 0.33 m above the base.
_register("Isaacsimenvs-PlayFlexivWujiRightRealLimitsPC-Direct-v0", "PlayPCRealEnvCfg",
          cfg_module="play_pc_env_cfg", env="play_pc_env:PlayPCEnv")
