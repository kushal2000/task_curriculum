"""Config of the rigid-body cloth env (``rigid_body_env.RigidBodyClothEnv``), PhysX / Isaac Lab 2.3.

``PlayEnvCfg`` (robot, table, obs, reward, DR) + the shared fold task block ``cloth`` (``ClothCfg``,
the same class the VBD env uses, so a hydra override means the same on both) + ``sheet``: the
rigid sheets themselves -- the design grid, plate/hinge physics and their per-reset randomisation.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaacsimenvs.tasks.cloth.fold_task import ClothCfg
from isaacsimenvs.tasks.play.play_env_cfg import PlayEnvCfg

__all__ = ["RigidBodyClothEnvCfg", "SheetCfg", "SheetRandomizationCfg"]


@configclass
class SheetRandomizationCfg:
    """Material differences, re-drawn per env at EVERY reset."""

    enabled: bool = True

    hinge_damping_scale: list[float] = [0.5, 3.0]
    """Log-uniform multiplier on ``sheet.hinge_damping``."""

    hinge_stiffness: list[float] = [0.0, 5.0e-4]
    """Uniform hinge stiffness [N m / rad]: how strongly the sheet springs back toward flat."""

    sheet_friction_scale: list[float] = [0.7, 1.5]
    """Multiplier on ``sheet.friction`` (both plates)."""

    robot_friction_scale: list[float] = [0.7, 1.3]
    """Multiplier on every robot shape's base friction (links and fingertips alike, so their ratio
    stays the configured ``assets.robot_friction`` : ``assets.finger_tip_friction``)."""

    table_friction_scale: list[float] = [0.7, 1.3]
    """Multiplier on ``assets.table_friction``."""

    friction_buckets: int = 16
    """Distinct values per friction multiplier: PhysX caps a scene at 64K materials, and every
    distinct (static, dynamic, restitution) triple a tensor write creates is one."""


@configclass
class SheetCfg:
    """The sheet designs and their rigid-body physics."""

    design_levels: int = 16
    """Levels per dimension: ``design_levels**3`` designs, env ``i`` gets design ``i % that``
    (``utils/sheet_designs.py``). 16 -> 4096 = one SAPG block at 24576 envs."""

    length_range: list[float] = [0.10, 0.30]
    """Sheet extent along the fold axis (x) [m]."""

    width_range: list[float] = [0.10, 0.30]
    """Sheet extent along the crease (y) [m]."""

    thickness_range: list[float] = [0.002, 0.020]
    """Plate thickness [m]; also the folded plies' gap (the fold target lift)."""

    density: float = 2.0
    """Areal density [kg/m^2], as ``ClothCfg.density``: a sheet weighs ``density * area``."""

    friction: float = 1.0
    """Plate friction (static = dynamic), before DR. PhysX averages the two sides of a contact."""

    contact_offset: float = 0.002
    """PhysX contact offset of the plate colliders [m]: the Play env's (``scene_utils._CONTACT_OFFSET``)."""

    rest_offset: float = 0.0
    """PhysX rest offset of the plate colliders [m]: the Play env's (``scene_utils._REST_OFFSET``)."""

    max_depenetration_velocity: float = 1.0
    """Cap on how fast PhysX pushes a plate out of a penetration [m/s]. Play's tool uses 1000; a
    plate spawned under a fingertip that rests ~1.5 cm above the table (Play's random reset arm
    pose) was then shot off the table at ~1 m above it. A cloth is never launched like that."""

    hinge_damping: float = 0.05
    """Hinge damping [N m s / rad] before DR."""

    hinge_stiffness: float = 0.0
    """Hinge stiffness [N m / rad] when DR is off."""

    hinge_armature: float = 1.0e-3
    """Fictitious hinge inertia [kg m^2]. 1e-5 let a squeeze on one plate leak into the other
    (the PhysX pool's finding, ``physx_env_cfg.PhysxChainCfg.hinge_armature``)."""

    hinge_limits_deg: list[float] = [-90.0, 180.0]
    """Hinge range [deg]: up to 180 (the + plate folded flat onto the - plate, one thickness above);
    down to -90 so a lifted sheet can droop over an edge."""

    solver_position_iterations: int = 8
    solver_velocity_iterations: int = 0

    spawn_clearance: float = 0.002
    """Height of the plates' underside above the table top at reset [m]: a short settle, no drop."""

    place_x: list[float] = [-0.29, 0.29]
    place_y: list[float] = [-0.18, 0.19]
    """Reset placement region, in the table frame (centre ``reset.table_reset_xy``) [m]: the yawed
    sheet's bounding box is kept inside it, its centre uniform over every position that allows
    (replacing Play's ``reset_position_noise``). A sheet wider than the region along an axis is
    centred on it there. Defaults: the 0.6 x 0.4 m ``table_fold_pool`` top less 1 cm, the far side
    (-y) cut at -0.18 m, the end of the arm's reach. With zero ``reset.reset_position_noise_{x,y}``
    (the eval protocol) every sheet is centred on the region."""

    randomization: SheetRandomizationCfg = SheetRandomizationCfg()


@configclass
class RigidBodyClothEnvCfg(PlayEnvCfg):
    """The fold task on a single-hinge rigid sheet, one design per env."""

    cloth: ClothCfg = ClothCfg()
    sheet: SheetCfg = SheetCfg()
