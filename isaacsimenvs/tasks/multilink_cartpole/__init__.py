"""Multi-link cartpole task registration.

Registers ``Isaacsimenvs-MultiLinkCartpole-Direct-v0`` for the DirectRLEnv training path.

Entry points:
- ``env_cfg_entry_point``           → MultiLinkCartpoleEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task/MultiLinkCartpole.yaml overlay
- ``rl_games_cfg_entry_point``            → cfg/train/MultiLinkCartpolePPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point``       → cfg/train/MultiLinkCartpoleSAPG.yaml
- ``rl_games_sapg_small_cfg_entry_point`` → cfg/train/MultiLinkCartpoleSAPGSmall.yaml
  (same SAPG settings, MLP [256,128,64] and no LSTM — the architecture ablation)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from .multilink_cartpole_env import MultiLinkCartpoleEnv
    from .multilink_cartpole_env_cfg import MultiLinkCartpoleEnvCfg

__all__ = ["MultiLinkCartpoleEnv", "MultiLinkCartpoleEnvCfg"]


def __getattr__(name: str):
    """Deferred class import; see the note in ``isaacsimenvs/tasks/play/__init__.py``."""
    if name == "MultiLinkCartpoleEnv":
        from .multilink_cartpole_env import MultiLinkCartpoleEnv

        return MultiLinkCartpoleEnv
    if name == "MultiLinkCartpoleEnvCfg":
        from .multilink_cartpole_env_cfg import MultiLinkCartpoleEnvCfg

        return MultiLinkCartpoleEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-MultiLinkCartpole-Direct-v0",
    entry_point="isaacsimenvs.tasks.multilink_cartpole.multilink_cartpole_env:MultiLinkCartpoleEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaacsimenvs.tasks.multilink_cartpole.multilink_cartpole_env_cfg:"
            "MultiLinkCartpoleEnvCfg"
        ),
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "MultiLinkCartpole.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "MultiLinkCartpolePPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "MultiLinkCartpoleSAPG.yaml"),
        "rl_games_sapg_small_cfg_entry_point": str(
            _CFG_DIR / "train" / "MultiLinkCartpoleSAPGSmall.yaml"
        ),
    },
)
