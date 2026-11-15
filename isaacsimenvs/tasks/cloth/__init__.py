"""Cloth folding task registration.

Registers ``Isaacsimenvs-Cloth-Direct-v0``: the Play task with a VBD cloth sheet in place of the
rigid procedural tool, on a coupled MJWarp + VBD solve. Fold the far half of the sheet onto the
near half; state and goal are keypoints on the moving half.

Subclasses ``PlayNewtonEnv``, so the observation stays 140-dim and the existing harness, player and
renderer work unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from .cloth_env import ClothEnv
    from .cloth_env_cfg import ClothEnvCfg

__all__ = [
    "ClothEnv",
    "ClothEnvCfg",
]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"


def __getattr__(name: str):
    """Deferred class import; see the note in ``isaacsimenvs/tasks/play/__init__.py``."""
    if name == "ClothEnv":
        from .cloth_env import ClothEnv

        return ClothEnv
    if name == "ClothEnvCfg":
        from .cloth_env_cfg import ClothEnvCfg

        return ClothEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


gym.register(
    id="Isaacsimenvs-Cloth-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.cloth_env:ClothEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.cloth_env_cfg:ClothEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "Cloth.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
    },
)


# The rigid-plate approximation of the cloth, on PhysX (Isaac Sim 5.1, `.venv_isaacsim`): one hinged
# chain (`RigidClothPhysx`, the chain picked by `env.rigid_cloth.variant`) and the fold-shape pool
# (`FoldPoolPhysx`, a shape drawn per episode). Same task code, same observation, same agent configs
# as the VBD task -- see `physx_envs.py`. Sharing `ClothSAPG.yaml` with the VBD task is deliberate:
# comparing the two manipulands is only meaningful if the learner is held fixed.
gym.register(
    id="Isaacsimenvs-RigidClothPhysx-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.physx_envs:RigidClothPhysxEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.physx_env_cfg:RigidClothPhysxEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "RigidClothPhysx.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
    },
)


gym.register(
    id="Isaacsimenvs-FoldPoolPhysx-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.physx_envs:FoldPoolPhysxEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.physx_env_cfg:FoldPoolPhysxEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "FoldPoolPhysx.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
    },
)


# The same three tasks on the Flexiv Rizon 4s + right Wuji Hand 2, e.g.
# `Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0`: the task's own cfg and YAML, then a
# robot overlay from `cfg/robot/` (robot spec, URDF, self-collision, base pose).
_ROBOT_VARIANTS = {
    "FlexivWujiRightHumanLike": "flexiv_wuji_right_human_like",
    "FlexivWujiRightReal": "flexiv_wuji_right_real",
    "FlexivWujiRightHLRL": "flexiv_wuji_right_hl_reallimits",
}
for _task in ("Cloth", "RigidClothPhysx", "FoldPoolPhysx"):
    _base = gym.spec(f"Isaacsimenvs-{_task}-Direct-v0")
    for _suffix, _robot in _ROBOT_VARIANTS.items():
        gym.register(
            id=f"Isaacsimenvs-{_task}{_suffix}-Direct-v0",
            entry_point=_base.entry_point,
            order_enforce=False,
            disable_env_checker=True,
            kwargs={
                **_base.kwargs,
                "env_cfg_robot_yaml_entry_point": str(_CFG_DIR / "robot" / f"{_robot}.yaml"),
            },
        )
del _task, _base, _suffix, _robot
