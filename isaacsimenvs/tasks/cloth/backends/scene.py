"""The cloth task's scene, shared by both backends: robot, table, goal marker, ground and light.

The sheet itself is the backend's: :func:`setup_scene` ends with ``env._install_cloth()``. On Newton
that is also the last moment an asset can be added, since the scene is replicated right after
(``backends/vbd/newton_support/patches.install_cloning`` wraps this function).

The robot and table go through ``isaacsimenvs.robots.usd`` (URDF -> USD, baked physics properties);
the goal marker is authored here: a kinematic, collision-free copy of the moving half's mesh, which
``ClothEnv._write_goal`` keeps on the fold target so viewers and renders show the goal.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import get_current_stage

from isaacsimenvs.robots import usd as robot_usd
from isaacsimenvs.tasks.cloth.mdp.geometry.grid import half_mesh

__all__ = ["setup_scene"]


def setup_scene(env) -> None:
    """Spawn every asset and register it with the scene."""
    cfg = env.cfg
    work_dir = Path(tempfile.mkdtemp(prefix="cloth_assets_"))
    usd_dir, bake_dir = work_dir / "usd", work_dir / "baked_usd"
    usd_dir.mkdir(parents=True)

    robot_path = _bake_robot(env, usd_dir, bake_dir)
    table_path = robot_usd.bake_usd(
        robot_usd.convert_urdf_to_usd(cfg.assets.table_urdf, usd_dir, fix_base=False),
        bake_dir, "table",
        props=dict(kinematic_enabled=True, disable_gravity=True, articulation_enabled=False),
    )
    marker_path = _write_goal_marker(env, work_dir / "goal_marker.usda")

    # Env roots first, so the regex prim paths below resolve to every env.
    stage = get_current_stage()
    for env_path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(env_path).IsValid():
            stage.DefinePrim(env_path, "Xform")

    env.robot = Articulation(
        robot_usd.build_robot_articulation_usd_cfg(robot_path, spec=env.robot_spec, root_pos=cfg.robot_root_pos)
    )
    env.table = RigidObject(robot_usd.build_rigid_object_cfg("/World/envs/env_.*/Table", [table_path]))
    env.goal_viz = RigidObject(robot_usd.build_rigid_object_cfg("/World/envs/env_.*/GoalViz", [marker_path]))
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    # The observed sheet size per env, normalised by reward.object_base_size; the backend fills it.
    env._object_scale_per_env = torch.zeros(env.num_envs, 3, device=env.device)
    env._install_cloth()
    _pin_env_order(stage, env.scene.env_prim_paths)


def _pin_env_order(stage, env_paths: list[str]) -> None:
    """List ``/World/envs``'s children in env order, explicitly (``primOrder`` on the root layer).

    PhysX parses the stage in its composed child order, and every PhysX view (robot, table, pool)
    lists its instances in that order -- which Isaac Lab takes to be the env order. A sublayer that
    authors specs for only some envs (the rigid pool's per-shape layers: env_0, env_5, env_10, ...)
    reorders the composed children, and then each row of ``robot.data`` belongs to another env.
    """
    stage.GetPrimAtPath(env_paths[0].rsplit("/", 1)[0]).SetChildrenReorder([p.rsplit("/", 1)[1] for p in env_paths])


def _bake_robot(env, usd_dir: Path, bake_dir: Path) -> str:
    """The robot's USD: converted with zero-gain drives (the actuators supply the gains), its
    self-collision pairs filtered when ``robot_self_collision`` is on, physics properties baked."""
    self_collision = bool(env.cfg.robot_self_collision)
    converted = robot_usd.convert_urdf_to_usd(
        env.cfg.assets.robot_urdf, usd_dir,
        fix_base=True, self_collision=self_collision, joint_drive=robot_usd.robot_joint_drive_cfg(),
    )
    if self_collision:
        robot_usd.apply_self_collision_filters(
            converted, env.robot_spec.self_collision_filter, keep=robot_usd.self_collision_keep(env)
        )
    return robot_usd.bake_usd(
        converted, bake_dir, "robot",
        props=dict(
            disable_gravity=True, max_depenetration_velocity=1000.0, enabled_self_collisions=self_collision,
            solver_position_iterations=8, solver_velocity_iterations=0,
        ),
        apply_physx_articulation=True,
    )


def _write_goal_marker(env, path: Path) -> str:
    """A kinematic body carrying the moving half's mesh with collision off: it shows the fold goal
    and can never touch anything. (A sheet's thickness is a collision radius the renderer never
    draws, so a flat mesh is what the folded half looks like.)"""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    c = env.cfg.cloth
    verts, indices = half_mesh(float(c.size), int(c.resolution), c.fold_axis)
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/goal_marker").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(1.0e-3)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0e-6, 1.0e-6, 1.0e-6))
    mesh = UsdGeom.Mesh.Define(stage, "/goal_marker/cloth_marker")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*(float(x) for x in v)) for v in verts]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in indices]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
    mesh.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(False)
    stage.GetRootLayer().Save()
    return str(path)
