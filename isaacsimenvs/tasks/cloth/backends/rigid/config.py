"""PhysX plate/hinge geometry, materials, and solver settings for ClothEnv."""

from __future__ import annotations

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg
from isaacsimenvs.tasks.cloth.mdp.config import AssetsCfg

__all__ = ["RigidClothEnvCfg", "RigidClothCfg", "RigidClothRandomizationCfg"]


@configclass
class RigidClothEpisodeChecksCfg:
    """Physics-rate resets; recipes enable these explicitly for new training."""

    outside_table: bool = False
    press_driven_fold: bool = False
    table_margin: float = 0.00001  # m, floating-point tolerance only
    minimum_lifting_work_fraction: float = 0.1
    minimum_lift_angle_deg: float = 30.0
    maximum_base_tilt_deg: float = 20.0
    minimum_opposed_load: float = 0.001  # N, contact detection, not a force cap
    contact_points_per_env: int = 128
    max_robot_penetration_fraction: float = 0.0
    """Reset the episode (failure ``robot_penetration``) when any robot link is deeper in the cloth than
    this fraction of the panel thickness (PhysX contact separation, every physics substep). 0 = off.
    A rigid panel this light cannot stop a hand in PhysX (see experiments/nov18/press_test), so the
    check keeps the policy from folding by pushing through it."""



@configclass
class RigidClothRandomizationCfg:
    """Material differences, re-drawn per env at EVERY reset (``randomization.py``)."""

    enabled: bool = True
    profile: str = "material"
    """material: per-reset sheet material draws; yibo: historical startup friction
    buckets and per-reset contact shells, retaining configured nominal masses/gains."""
    yibo_cloth_friction: list[float] = [0.7, 1.5]
    yibo_robot_friction: list[float] = [0.35, 0.65]
    yibo_table_friction: list[float] = [0.7, 1.3]
    """Absolute startup friction bucket bounds; static equals dynamic in yibo."""
    yibo_rest_offset: list[float] = [0.004, 0.010]
    """Cloth collision shell [m], uniform each reset in yibo; contact_offset stays fixed."""
    yibo_friction_seed: int = 0
    density: list[float] = [0.02, 0.5]
    """Log-uniform absolute areal density [kg/m²], or 20–500 g/m², per cloth reset."""
    hinge_armature_scale: list[float] = [0.0, 0.0]
    """Optional uniform fraction of each hinge's physical locked-child-subtree inertia.
    Disabled by default: physical cloth has no joint rotor.
    One fraction per cloth preserves symmetry; actual joint armatures [kg m²]
    depend on topology, dimensions and the sampled density. This is numerical
    stabilization, not a measured cloth material parameter."""
    bending_rigidity: list[float] = [1.0e-6, 2.5e-4]
    """Log-uniform sheet rigidity B [N m], one draw per cloth.
    Provisional calibration bounds; not empirically validated fabric measurements."""
    bending_viscosity: list[float] = [1.0e-6, 1.5e-4]
    """Log-uniform sheet bending viscosity C [N m s], one draw per cloth.
    Provisional calibration bounds; not empirically validated fabric measurements."""
    table_static_friction: list[float] = [0.1, 0.8]
    """Uniform effective cloth/table static contact coefficient."""
    hand_static_friction: list[float] = [0.3, 1.2]
    """Uniform effective cloth/robot static contact coefficient."""
    dynamic_friction_ratio: list[float] = [0.6, 1.0]
    """Uniform dynamic/static contact-friction ratio, shared by both contact pairs."""
    friction_buckets: int = 16


@configclass
class RigidClothCfg:
    """The cloth design and its rigid-body physics."""

    designs: list[str] = [
        "single_hinge.yaml", "strip_4.yaml", "strip_8.yaml", "hinge_band.yaml", "flexible_corners.yaml",
    ]
    """YAML paths, or filenames under scripts/configs/rigid_cloth_designs.
    Environments cycle through this list; dimensions vary independently. Different
    topologies share one RL batch through separate real PhysX articulations."""

    num_geometries: int = 256
    """Distinct sheet sizes [length, width], drawn uniformly from the ranges below with
    ``geometry_seed``. Every topology sees every geometry (``designs.geometry_of``)."""
    geometry_seed: int = 0
    length_range: list[float] = [0.10, 0.30]
    """Along the sheet's x direction, across the crease [m]."""
    width_range: list[float] = [0.10, 0.30]
    """Along the crease (y) [m]."""
    geometries: list = []
    """Optional explicit sizes [[length, width], ...] [m]; when set, used instead of sampling
    (num_geometries and the ranges are then ignored). For tests and targeted evaluation."""
    thickness: float = 0.002
    """Panel collision thickness [m], the same for every env; also the folded plies' target gap.
    Independent of areal density and the sheet bending coefficients."""
    point_resolution: int = 7
    """Grid of tracked points per side carried by the plates (fold error, footprint, frame)."""
    density: float = 0.1
    """Nominal areal density [kg/m²] (100 g/m²); mass is density × area, independent
    of the collision plate thickness. Reset randomization draws absolute density."""
    table_static_friction: float = 0.4
    """Effective cloth/table static contact coefficient."""
    hand_static_friction: float = 0.8
    """Effective cloth/robot static contact coefficient."""
    dynamic_friction_ratio: float = 0.8
    """Dynamic friction divided by static friction for both contact pairs."""
    contact_offset: float = 0.002
    rest_offset: float = 0.0
    max_depenetration_velocity: float = 1.0
    """[m/s]. At Play's 1000 a plate spawned under a fingertip was shot ~1 m off the table."""
    bending_rigidity: float = 5.0e-5
    """Sheet rigidity B [N m]; hinge stiffness is B × crease_length / COM_span.
    Nominal value is provisional and requires calibration to the chosen fabric."""
    bending_viscosity: float = 5.0e-5
    """Sheet bending viscosity C [N m s]; hinge damping is C × length / span.
    Nominal value is provisional and requires calibration to the chosen fabric."""
    hinge_stiffness_override: float = -1.0
    """Optional fixed stiffness [N m/rad] on every hinge, bypassing material B and geometry.
    Used to replay historical policies; -1 retains the sheet material model.
    A numeric sentinel permits Isaac Lab's strict YAML scalar type checking."""
    hinge_damping_override: float = -1.0
    """Optional fixed damping [N m s/rad] on every hinge, bypassing material C and geometry.
    -1 retains the sheet material model. Fixed overrides also remain fixed
    when other material properties are randomized."""
    hinge_armature: float = 0.0
    """Nominal added joint inertia [kg m²], used when randomization is disabled.
    Physical cloth has no joint rotor; the optional randomization fraction is also
    zero by default. A nonzero override is numerical stabilization only."""
    hinge_limits_deg: list[float] = [-90.0, 180.0]
    solver_position_iterations: int = 8
    solver_velocity_iterations: int = 0
    spawn_clearance: float = 0.002
    """Plates' underside above the table at reset [m]."""
    place_x: list[float] = [-0.20, 0.20]
    place_y: list[float] = [-0.17, 0.17]
    """Placement region in the table frame, intersected with the actual tabletop.
    Headings are conditioned on the whole sheet fitting; impossible ranges raise.
    Centers are uniform within valid bounds, or centered when position noise is off."""
    randomization: RigidClothRandomizationCfg = RigidClothRandomizationCfg()
    episode_checks: RigidClothEpisodeChecksCfg = RigidClothEpisodeChecksCfg()


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
class RigidClothEnvCfg(ClothCfg):
    """The fold task on articulated plate cloths with mixed topologies and varied sizes."""

    sim: SimulationCfg = _sim()
    assets: AssetsCfg = AssetsCfg(table_urdf="assets/urdf/table_fold_pool.urdf")
    rigid_cloth: RigidClothCfg = RigidClothCfg()
