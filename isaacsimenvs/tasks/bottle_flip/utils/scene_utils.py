"""Bottle-flip scene setup.

Mirrors `tasks/play/utils/scene_utils.py:setup_scene` and reuses its helpers verbatim —
the robot, table, goal-viz and camera paths are all identical. The only substitution is
the object: a pool of procedural bottles instead of the handle/head primitives.

Keeping the robot and table construction byte-identical to play's is what preserves
observation and action compatibility with play2perfect checkpoints.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import find_matching_prim_paths

from isaacsimenvs.tasks.play.utils.scene_utils import (
    _bake_usd,
    _convert_urdf_to_usd,
    _log_scene_step,
    _materialize_env_prims,
    _robot_joint_drive_cfg,
    build_rigid_object_cfg,
    build_robot_articulation_usd_cfg,
    hide_goal_viz_for_student_camera,
    setup_student_camera,
)

from .generate_bottle import generate_bottle_urdfs

__all__ = ["setup_scene"]

REPO_ROOT = Path(__file__).resolve().parents[4]


def _asset_path(path: str | Path) -> str:
    asset_path = Path(path)
    if not asset_path.is_absolute():
        asset_path = REPO_ROOT / asset_path
    return str(asset_path)


def _build_bottle_shape_tensors(env, shapes: list[tuple[float, float]], num_usds: int) -> None:
    """Record which bottle each env got, and derive play's object-scale tensor from it.

    `MultiUsdFileCfg` assigns variants round-robin over the spawned prim order, so the
    mapping is recoverable from the prim paths — the same trick play uses in
    `_build_object_scale_tensor`. We need it because the bottle's height and radius feed
    the landing-height check, the keypoint sizing, and the runtime fill-level model.
    """
    num_envs = env.num_envs
    prim_paths = find_matching_prim_paths("/World/envs/env_.*/Object")
    if len(prim_paths) != num_envs:
        raise RuntimeError(
            f"Expected {num_envs} Object prims after MultiUsdFileCfg spawn, got "
            f"{len(prim_paths)}. Check scene.replicate_physics / clone_in_fabric are false."
        )

    base_size = env.cfg.reward.object_base_size
    env._bottle_height = torch.zeros(num_envs, device=env.device)
    env._bottle_radius = torch.zeros(num_envs, device=env.device)
    env._object_scale_per_env = torch.zeros(num_envs, 3, device=env.device)
    env._object_asset_index_per_env = torch.zeros(num_envs, dtype=torch.long, device=env.device)

    for source_idx, prim_path in enumerate(prim_paths):
        env_id = int(prim_path.rsplit("/", 2)[-2].removeprefix("env_"))
        asset_index = source_idx % num_usds
        height, radius = shapes[asset_index]
        env._bottle_height[env_id] = height
        env._bottle_radius[env_id] = radius
        # Play sizes its reward keypoints from a normalised bounding box; a bottle's is
        # (2r, 2r, h).
        env._object_scale_per_env[env_id] = torch.tensor(
            [2.0 * radius / base_size, 2.0 * radius / base_size, height / base_size],
            device=env.device,
        )
        env._object_asset_index_per_env[env_id] = asset_index


def setup_scene(env) -> None:
    """Build robot, table, bottle pool, goal viz, ground, and light."""
    assets_cfg = env.cfg.assets
    t0 = time.perf_counter()
    _log_scene_step(t0, f"bottle-flip setup start num_envs={env.num_envs}")

    env._tmp_asset_dir = tempfile.mkdtemp(prefix="bottle_flip_assets_")
    urdf_paths, shapes = generate_bottle_urdfs(
        Path(env._tmp_asset_dir) / "bottles",
        num_variants=assets_cfg.num_bottle_variants,
        height_range=tuple(assets_cfg.bottle_height_range),
        aspect_ratio_range=tuple(assets_cfg.bottle_aspect_ratio_range),
        wall_thickness=assets_cfg.bottle_wall_thickness,
        nominal_fill_fraction=assets_cfg.nominal_fill_fraction,
        shell_density=assets_cfg.bottle_shell_density,
        liquid_density=assets_cfg.bottle_liquid_density,
    )
    env._object_urdf_paths = [str(p) for p in urdf_paths]
    env._bottle_shapes = shapes
    _log_scene_step(t0, f"generated {len(urdf_paths)} bottle URDFs")

    usd_work_dir = Path(env._tmp_asset_dir) / "usd"
    bake_root = Path(env._tmp_asset_dir) / "baked_usd"
    usd_work_dir.mkdir(parents=True, exist_ok=True)

    bottle_raw_usds = [
        _convert_urdf_to_usd(
            str(p), usd_work_dir, fix_base=False, replace_cylinders_with_capsules=False
        )
        for p in urdf_paths
    ]
    object_usd_paths = [
        _bake_usd(
            usd, bake_root, "object",
            props=dict(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1000.0,
                articulation_enabled=False,
            ),
        )
        for usd in bottle_raw_usds
    ]
    goalviz_usd_paths = [
        _bake_usd(
            usd, bake_root, "goalviz",
            props=dict(kinematic_enabled=True, disable_gravity=True, articulation_enabled=False),
            collision_enabled=False,
        )
        for usd in bottle_raw_usds
    ]

    robot_usd_path = _bake_usd(
        _convert_urdf_to_usd(
            _asset_path(assets_cfg.robot_urdf), usd_work_dir,
            fix_base=True, self_collision=False, joint_drive=_robot_joint_drive_cfg(),
        ),
        bake_root, "robot",
        props=dict(
            disable_gravity=True, max_depenetration_velocity=1000.0,
            enabled_self_collisions=False,
            solver_position_iterations=8, solver_velocity_iterations=0,
        ),
        apply_physx_articulation=True,
    )
    table_usd_path = _bake_usd(
        _convert_urdf_to_usd(_asset_path(assets_cfg.table_urdf), usd_work_dir, fix_base=False),
        bake_root, "table",
        props=dict(kinematic_enabled=True, disable_gravity=True, articulation_enabled=False),
    )
    env._table_variant_scales = [(1.0, 1.0)]
    _log_scene_step(t0, "baked USDs")

    _materialize_env_prims(env)

    env.robot = Articulation(build_robot_articulation_usd_cfg(robot_usd_path))
    env.table = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Table", [table_usd_path]))
    env.object = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Object", object_usd_paths))
    env.goal_viz = RigidObject(
        build_rigid_object_cfg("/World/envs/env_.*/GoalViz", goalviz_usd_paths)
    )

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    _build_bottle_shape_tensors(env, shapes, len(object_usd_paths))

    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    hide_goal_viz_for_student_camera(env)
    setup_student_camera(env)
    _log_scene_step(t0, "scene ready")
