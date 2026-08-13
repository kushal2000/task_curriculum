"""Scene construction for the multi-link cartpole: URDF → USD → Articulation.

Same conversion pipeline play2perfect uses for its procedural objects
(`tasks/play/utils/scene_utils.py`), minus the mesh-alias handling — this URDF is
primitives only, so there are no mesh filenames to sanitise.

Only *one* USD is ever built. Every env spawns it, which keeps `replicate_physics=True`
available (the fast clone path) and guarantees the uniform DOF count that Isaac Lab's
single `Articulation` view requires. All per-env variation is applied at runtime through
joint locking; see `reset_utils`.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane

from .generate_cartpole import CART_JOINT_NAME, POLE_JOINT_PREFIX, generate_cartpole_urdf

__all__ = [
    "POLE_JOINT_REGEX",
    "build_cartpole_articulation_cfg",
    "convert_cartpole_urdf_to_usd",
    "setup_scene",
]

POLE_JOINT_REGEX = f"{POLE_JOINT_PREFIX}.*"

_PRIM_PATH = "/World/envs/env_.*/Cartpole"


def _log(t0: float, message: str) -> None:
    print(f"[cartpole/scene_utils][+{time.perf_counter() - t0:.2f}s] {message}", flush=True)


def convert_cartpole_urdf_to_usd(urdf_path: str | Path, usd_work_dir: Path) -> str:
    """Convert the generated URDF to USD with drive APIs on every joint.

    `target_type="position"` with zero gains matters: the USD DriveAPI prims must already
    exist for `Articulation.write_joint_stiffness_to_sim` to have anything to write into
    at runtime, and runtime gain writes are how a pole joint gets locked. This is the
    same reason play2perfect's `_robot_joint_drive_cfg` sets zero-gain position drives.
    """
    urdf_path = Path(urdf_path)
    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(usd_work_dir / urdf_path.stem),
        force_usd_conversion=True,
        fix_base=True,
        merge_fixed_joints=True,
        make_instanceable=False,
        self_collision=False,
        # Capsule replacement would change the effective link length, and link length is
        # the quantity the whole curriculum is defined in terms of.
        replace_cylinders_with_capsules=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    return UrdfConverter(cfg).usd_path


def build_cartpole_articulation_cfg(usd_path: str, geom) -> ArticulationCfg:
    """`ArticulationCfg` for the converted cartpole.

    Both actuator groups start at zero stiffness: the cart is effort-controlled by the
    policy, and the pole joints are passive until `reset_utils` locks a subset of them.
    """
    return ArticulationCfg(
        prim_path=_PRIM_PATH,
        spawn=UsdFileCfg(
            usd_path=usd_path,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=geom.sim_position_iterations,
                solver_velocity_iteration_count=geom.sim_velocity_iterations,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "cart": ImplicitActuatorCfg(
                joint_names_expr=[CART_JOINT_NAME],
                stiffness=0.0,
                damping=0.0,
            ),
            "poles": ImplicitActuatorCfg(
                joint_names_expr=[POLE_JOINT_REGEX],
                stiffness=0.0,
                damping=geom.pole_joint_damping,
            ),
        },
    )


def setup_scene(env) -> None:
    """Generate the articulation, spawn it, and register it with the scene."""
    geom = env.cfg.geometry
    t0 = time.perf_counter()

    if len(geom.link_lengths) != geom.n_max:
        raise ValueError(
            f"geometry.link_lengths has {len(geom.link_lengths)} entries but n_max is "
            f"{geom.n_max}. They must match — n_max sets the observation width, so a "
            "mismatch would silently change the policy's input size."
        )

    env._tmp_asset_dir = tempfile.mkdtemp(prefix="multilink_cartpole_")
    urdf_path = generate_cartpole_urdf(
        Path(env._tmp_asset_dir) / "multilink_cartpole.urdf",
        link_lengths=geom.link_lengths,
        rail_length=geom.rail_length,
        cart_size=tuple(geom.cart_size),
        cart_mass=geom.cart_mass,
        pole_radius=geom.pole_radius,
        pole_density=geom.pole_density,
        pole_joint_damping=geom.pole_joint_damping,
    )
    _log(t0, f"generated URDF with {geom.n_max} pole links (total {sum(geom.link_lengths):.3f} m)")

    usd_path = convert_cartpole_urdf_to_usd(urdf_path, Path(env._tmp_asset_dir) / "usd")
    _log(t0, f"converted to USD: {usd_path}")

    env.cartpole = Articulation(build_cartpole_articulation_cfg(usd_path, geom))

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    env.scene.clone_environments(copy_from_source=False)
    if env.device == "cpu":
        # PhysX CPU pipeline does not auto-filter inter-env collisions.
        env.scene.filter_collisions(global_prim_paths=[])

    env.scene.articulations["cartpole"] = env.cartpole
    _log(t0, "scene ready")
