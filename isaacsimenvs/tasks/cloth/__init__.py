"""Cloth folding: the rigid-body sheet (train, PhysX) and the deformable VBD sheet (evaluate, Newton).

    rigid_body_env.py / rigid_body_env_cfg.py   Isaacsimenvs-RigidBodyCloth-Direct-v0   (.venv/isaacsim)
    deformable_env.py / deformable_env_cfg.py   Isaacsimenvs-Cloth*-Direct-v0           (.venv/isaaclab3)
    fold_task.py                                the fold task both run (simulator-free)

Both share ``ClothSAPG.yaml``: comparing the two manipulands is only meaningful with the learner held fixed.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"


# The deformable sheet on Newton: the Play task with a VBD cloth in place of the rigid tool, on a
# coupled MJWarp + VBD solve.
gym.register(
    id="Isaacsimenvs-Cloth-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.deformable_env:ClothEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.deformable_env_cfg:ClothEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "Cloth.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
    },
)

# The same on the Flexiv Rizon 4s + right Wuji Hand 2, e.g.
# `Isaacsimenvs-ClothFlexivWujiRightReal-Direct-v0`: the task's cfg and YAML, then a robot overlay
# from `cfg/robot/` (robot spec, URDF, self-collision, base pose).
_ROBOT_VARIANTS = {
    "FlexivWujiRightHumanLike": "flexiv_wuji_right_human_like",
    "FlexivWujiRightReal": "flexiv_wuji_right_real",
    "FlexivWujiRightHLRL": "flexiv_wuji_right_hl_reallimits",
}
_base = gym.spec("Isaacsimenvs-Cloth-Direct-v0")
for _suffix, _robot in _ROBOT_VARIANTS.items():
    gym.register(
        id=f"Isaacsimenvs-Cloth{_suffix}-Direct-v0",
        entry_point=_base.entry_point,
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            **_base.kwargs,
            "env_cfg_robot_yaml_entry_point": str(_CFG_DIR / "robot" / f"{_robot}.yaml"),
        },
    )
del _base, _suffix, _robot


# Rigid2Deform: the fold task on a rigid single-hinge sheet, one size design per env, on PhysX. One
# robot (Flexiv Rizon 4s + right Wuji Hand 2, identified hand: utils/robot.py, built into the cfg).
# Not `Cloth*`-prefixed: experiments/_common.sh sends `Isaacsimenvs-Cloth*` to Newton.
gym.register(
    id="Isaacsimenvs-RigidBodyCloth-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.rigid_body_env:RigidBodyClothEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.rigid_body_env_cfg:RigidBodyClothEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "RigidBodyCloth.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
    },
)
