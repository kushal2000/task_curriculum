"""Config of the rigid-body cloth env (``rigid_body_env.RigidBodyClothEnv``), PhysX / Isaac Lab 2.3.

The shared sections (``utils/cfg.ClothTaskEnvCfg``: robot, action, obs, reward, reset, termination,
DR, the fold) + PhysX's ``sim`` + ``sheet``: the rigid sheets -- the design grid, plate / hinge
physics and their per-reset randomisation.
"""

from __future__ import annotations

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaacsimenvs.tasks.cloth.utils.cfg import AssetsCfg, ClothTaskEnvCfg

__all__ = ["RigidBodyClothEnvCfg", "SheetCfg", "SheetRandomizationCfg"]


@configclass
class SheetRandomizationCfg:
    """Material differences, re-drawn per env at EVERY reset (``utils/dr_utils.py``)."""

    enabled: bool = True
    hinge_damping_scale: list[float] = [0.5, 3.0]
    """Log-uniform multiplier on ``sheet.hinge_damping``."""
    hinge_stiffness: list[float] = [0.0, 5.0e-4]
    """Uniform hinge stiffness [N m / rad]: how strongly the sheet springs back toward flat."""
    sheet_friction_scale: list[float] = [0.7, 1.5]
    robot_friction_scale: list[float] = [0.7, 1.3]
    """On every robot shape's base friction (links and fingertips alike, so their ratio stays)."""
    table_friction_scale: list[float] = [0.7, 1.3]
    friction_buckets: int = 16


@configclass
class SheetCfg:
    """The sheet designs and their rigid-body physics."""

    design_levels: int = 16
    """Levels per dimension: ``design_levels**3`` designs, env ``i`` gets design ``perm[i % that]``.
    16 -> 4096 = one SAPG block at 24576 envs."""
    length_range: list[float] = [0.10, 0.30]
    """Along the fold axis (x) [m]."""
    width_range: list[float] = [0.10, 0.30]
    """Along the crease (y) [m]."""
    thickness_range: list[float] = [0.002, 0.020]
    """Plate thickness [m]; also the folded plies' gap (the fold-target lift)."""
    point_resolution: int = 7
    """Grid of tracked points per side carried by the two plates (fold error, footprint, frame)."""
    density: float = 2.0
    """Areal density [kg/m^2]: a sheet weighs ``density * area`` whatever its thickness."""
    friction: float = 1.0
    contact_offset: float = 0.002
    rest_offset: float = 0.0
    max_depenetration_velocity: float = 1.0
    """[m/s]. At Play's 1000 a plate spawned under a fingertip was shot ~1 m off the table."""
    hinge_damping: float = 0.05
    hinge_stiffness: float = 0.0
    hinge_armature: float = 1.0e-3
    """1e-5 let a squeeze on one plate leak into the other."""
    hinge_limits_deg: list[float] = [-90.0, 180.0]
    solver_position_iterations: int = 8
    solver_velocity_iterations: int = 0
    spawn_clearance: float = 0.002
    """Plates' underside above the table at reset [m]."""
    place_x: list[float] = [-0.29, 0.29]
    place_y: list[float] = [-0.18, 0.19]
    """Reset region (table frame) the yawed sheet's bounding box is kept inside, its centre uniform
    over every position that allows; centred where it does not fit, and everywhere when both
    ``reset.reset_position_noise_x/y`` are 0 (the eval protocol)."""
    randomization: SheetRandomizationCfg = SheetRandomizationCfg()


def _sim() -> SimulationCfg:
    return SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,  # TGS
            min_position_iteration_count=8, max_position_iteration_count=8,
            min_velocity_iteration_count=0, max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2, friction_offset_threshold=0.04,
            friction_correlation_distance=0.025, gpu_max_rigid_patch_count=2**21,
        ),
    )


@configclass
class RigidBodyClothEnvCfg(ClothTaskEnvCfg):
    """The fold task on a single-hinge rigid sheet, one design per env."""

    sim: SimulationCfg = _sim()
    assets: AssetsCfg = AssetsCfg(table_urdf="assets/urdf/table_fold_pool.urdf")
    sheet: SheetCfg = SheetCfg()
