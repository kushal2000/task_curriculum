"""Play-on-Newton task registration.

Registers ``Isaacsimenvs-PlayNewton-Direct-v0``: the same task as ``Isaacsimenvs-Play-Direct-v0``,
constructed on Isaac Lab 3.0 + Newton/MJWarp instead of Isaac Lab 2.3 + PhysX.

A separate task id rather than a backend flag, so which physics a result came from is visible in
the command that produced it. The env subclasses ``PlayEnv`` and overrides only construction, so
the 140-dim observation stays byte-identical and the pretrained checkpoint loads unchanged.

Requires ``.venv_isaaclab3`` (see ``scripts/install_isaaclab3.sh``) and a populated USD cache
(``python -m isaacsimenvs.newton.usd_cache --populate``, run under ``.venv_isaacsim``).

Entry points:
- ``env_cfg_entry_point``           → PlayNewtonEnvCfg
- ``env_cfg_yaml_entry_point``      → cfg/task/PlayNewton.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/PlayPPO.yaml   (shared with Play)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/PlaySAPG.yaml  (shared with Play)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from .play_newton_env import PlayNewtonEnv
    from .play_newton_env_cfg import PlayNewtonEnvCfg

__all__ = ["PlayNewtonEnv", "PlayNewtonEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"


def __getattr__(name: str):
    """Deferred class import; see the note in ``isaacsimenvs/tasks/play/__init__.py``.

    It matters more here than elsewhere: importing this env pulls the Isaac Lab 3.0 compat shim
    and the Newton patch layer, which cannot even import under ``.venv_isaacsim``. Deferring keeps
    ``import isaacsimenvs`` working from *both* venvs, so one registry serves both.
    """
    if name == "PlayNewtonEnv":
        from .play_newton_env import PlayNewtonEnv

        return PlayNewtonEnv
    if name == "PlayNewtonEnvCfg":
        from .play_newton_env_cfg import PlayNewtonEnvCfg

        return PlayNewtonEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


gym.register(
    id="Isaacsimenvs-PlayNewton-Direct-v0",
    entry_point="isaacsimenvs.tasks.play_newton.play_newton_env:PlayNewtonEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaacsimenvs.tasks.play_newton.play_newton_env_cfg:PlayNewtonEnvCfg"
        ),
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "PlayNewton.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PlaySAPG.yaml"),
    },
)
