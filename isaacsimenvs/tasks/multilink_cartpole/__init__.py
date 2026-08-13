"""Multi-link cartpole task registration.

Registers ``Isaacsimenvs-MultiLinkCartpole-Direct-v0`` for the DirectRLEnv training path.

Entry points:
- ``env_cfg_entry_point``           → MultiLinkCartpoleEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task/MultiLinkCartpole.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/MultiLinkCartpolePPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/MultiLinkCartpoleSAPG.yaml
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .multilink_cartpole_env import MultiLinkCartpoleEnv
from .multilink_cartpole_env_cfg import MultiLinkCartpoleEnvCfg

__all__ = ["MultiLinkCartpoleEnv", "MultiLinkCartpoleEnvCfg"]

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
    },
)
