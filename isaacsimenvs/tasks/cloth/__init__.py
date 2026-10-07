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
    from .fold_pool_env import FoldPoolEnv
    from .fold_pool_env_cfg import FoldPoolEnvCfg
    from .rigid_cloth_env import RigidClothEnv
    from .rigid_cloth_env_cfg import RigidClothEnvCfg

__all__ = [
    "ClothEnv",
    "ClothEnvCfg",
    "FoldPoolEnv",
    "FoldPoolEnvCfg",
    "RigidClothEnv",
    "RigidClothEnvCfg",
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
    if name == "RigidClothEnv":
        from .rigid_cloth_env import RigidClothEnv

        return RigidClothEnv
    if name == "RigidClothEnvCfg":
        from .rigid_cloth_env_cfg import RigidClothEnvCfg

        return RigidClothEnvCfg
    if name == "FoldPoolEnv":
        from .fold_pool_env import FoldPoolEnv

        return FoldPoolEnv
    if name == "FoldPoolEnvCfg":
        from .fold_pool_env_cfg import FoldPoolEnvCfg

        return FoldPoolEnvCfg
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


# ONE id, with the approximation chosen by config rather than by a family of ids:
#
#     --task Isaacsimenvs-RigidCloth-Direct-v0 env.rigid_cloth.variant=box3-mid
#
# Five ids would need five cfg classes (the gym kwargs carry a cfg *entry point string*, so a
# per-variant default cannot be expressed without one), which is five places for the five variants
# to drift apart. The variant is a physical parameter of one manipuland, like its density -- and
# density does not get its own task id either.
#
# Shares `ClothSAPG.yaml` with the VBD task deliberately: comparing the two manipulands is only
# meaningful if the learner is held fixed.
gym.register(
    id="Isaacsimenvs-RigidCloth-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.rigid_cloth_env:RigidClothEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.rigid_cloth_env_cfg:RigidClothEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "RigidCloth.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
    },
)


# Stage 2: the same half-fold task, with the manipuland drawn per episode from the fold-shape pool
# (`utils/fold_shapes.POOL`). Same learner config as the chain, for the same reason.
gym.register(
    id="Isaacsimenvs-FoldPool-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.fold_pool_env:FoldPoolEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.fold_pool_env_cfg:FoldPoolEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "FoldPool.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
    },
)


# The two rigid-manipuland tasks on PhysX (Isaac Sim 5.1, `.venv_isaacsim`): same task code, same
# observation, same agent configs -- see `physx_envs.py`.
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
