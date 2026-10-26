"""Joint-reach deployment test task on the Flexiv Rizon 4s + Wuji Hand 2 (human-like limits).

Registers ``Isaacsimenvs-JointReachFlexivWujiHumanLike-Direct-v0``: reach a commanded 27-joint
configuration through the Play action pipeline. See joint_reach_env_cfg.py.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-JointReachFlexivWujiHumanLike-Direct-v0",
    entry_point="isaacsimenvs.tasks.joint_reach.joint_reach_env:JointReachEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.joint_reach.joint_reach_env_cfg:JointReachEnvCfg",
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "JointReachSAPG.yaml"),
    },
)
