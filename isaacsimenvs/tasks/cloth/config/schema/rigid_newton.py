"""The fold-shape pool on Newton: the PhysX pool's configuration plus the Newton solver block."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from .rigid_cloth import RigidClothEnvCfg
from .vbd import NewtonSolverCfg


@configclass
class RigidNewtonClothEnvCfg(RigidClothEnvCfg):
    """The fold task on the fold-shape pool, robot and plates on MJWarp (Isaac Lab 3.0 + Newton)."""

    newton: NewtonSolverCfg = NewtonSolverCfg()
    use_gravcomp: bool = MISSING
    """The robot's per-body disableGravity as MuJoCo gravity compensation (as on VBD); the plates fall."""
