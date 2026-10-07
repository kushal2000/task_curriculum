"""The chain and pool adapters, for Isaac Lab 2.3 on PhysX.

The adapters in ``rigid_cloth_adapter`` / ``fold_pool_adapter`` speak Isaac Lab 3.0 on Newton: root
poses go out as (x, y, z, w), and joint state goes through ``write_joint_state_to_sim`` with
``env_ids=None`` for a full write (an explicit full ``arange`` was measured to do nothing there).
Isaac Lab 2.3 on PhysX is wxyz on both reads and writes and takes ``env_ids`` normally. Everything
the adapters compute -- the particle grid, the Kabsch orientation, placement, parking -- is
inherited unchanged; only the two sim writes and the initial quaternion layout differ.

The layout is still MEASURED at startup (``detect_quat_layout`` in the env's placement proof); the
default here only matters before that runs.

Pure torch, no Isaac imports.
"""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.cloth.utils.fold_pool_adapter import PoolAsClothMesh, PoolAsRigidObject
from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import ChainAsClothMesh, ChainAsRigidObject

__all__ = [
    "ChainAsClothMeshPhysx",
    "ChainAsRigidObjectPhysx",
    "PoolAsClothMeshPhysx",
    "PoolAsRigidObjectPhysx",
    "xyzw_pose_to_wxyz",
]


def xyzw_pose_to_wxyz(pose_xyzw: torch.Tensor) -> torch.Tensor:
    """``(n, 7)`` position + (x, y, z, w) quaternion -> position + (w, x, y, z)."""
    out = pose_xyzw.clone()
    out[:, 3] = pose_xyzw[:, 6]
    out[:, 4:7] = pose_xyzw[:, 3:6]
    return out


class ChainAsClothMeshPhysx(ChainAsClothMesh):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quat_layout = "wxyz"


class ChainAsRigidObjectPhysx(ChainAsRigidObject):
    def _write_root_pose(self, pose_xyzw: torch.Tensor, env_ids: torch.Tensor) -> None:
        self._art.write_root_pose_to_sim(xyzw_pose_to_wxyz(pose_xyzw), env_ids)

    def _write_joints(self, position, velocity, env_ids: torch.Tensor) -> None:
        self._art.write_joint_state_to_sim(position, velocity, env_ids=env_ids)


class PoolAsClothMeshPhysx(PoolAsClothMesh):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quat_layout = "wxyz"


class PoolAsRigidObjectPhysx(PoolAsRigidObject):
    def _write_root_pose(self, art, pose_xyzw: torch.Tensor, env_ids: torch.Tensor) -> None:
        art.write_root_pose_to_sim(xyzw_pose_to_wxyz(pose_xyzw), env_ids)

    def _write_joints(self, art, position, velocity, env_ids: torch.Tensor) -> None:
        art.write_joint_state_to_sim(position, velocity, env_ids=env_ids)
