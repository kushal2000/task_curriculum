"""Cloth folding: the rigid-body sheet (train, PhysX) and the deformable VBD sheet (evaluate, Newton).

    rigid_cloth_env.py / rigid_cloth_env_cfg.py   Isaacsimenvs-RigidBodyCloth-Direct-v0   (.venv/isaacsim)
    deformable_cloth_env.py / deformable_cloth_env_cfg.py   Isaacsimenvs-Cloth-Direct-v0            (.venv/isaaclab3)
    utils/                                      geometry, backend helpers, and shared task functions

Robot: Flexiv Rizon 4s + right Wuji Hand 2 with the identified hand (utils/mdp/flexiv_wuji.py), for both.

Both share ``ClothSAPG.yaml``: comparing the two manipulands is only meaningful with the learner held fixed.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"


# The deformable VBD sheet on Newton (coupled MJWarp + VBD), the same robot built in.
gym.register(
    id="Isaacsimenvs-Cloth-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.deformable_cloth_env:DeformableClothEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.deformable_cloth_env_cfg:DeformableClothEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "Cloth.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
    },
)


# Rigid2Deform: the fold task on a rigid single-hinge sheet, one size design per env, on PhysX. One
# robot (Flexiv Rizon 4s + right Wuji Hand 2, identified hand: utils/mdp/flexiv_wuji.py, built into the cfg).
# Not `Cloth*`-prefixed: experiments/_common.sh sends `Isaacsimenvs-Cloth*` to Newton.
gym.register(
    id="Isaacsimenvs-RigidBodyCloth-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.rigid_cloth_env:RigidClothEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.rigid_cloth_env_cfg:RigidClothEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "RigidBodyCloth.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
    },
)
