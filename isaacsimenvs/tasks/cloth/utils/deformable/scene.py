"""The Newton scene (Isaac Lab 3.0, kit-less): robot, table, ground, light, the VBD cloths, replication.

What the PhysX scene does, with four Newton specifics:

* **no URDF converter** (it needs Kit): robot and table USDs are looked up in the content-addressed
  cache that a PhysX run filled (``newton/usd_cache.py``, same converter options, so the same
  geometry on both backends), then baked as on PhysX (``rigid_scene.bake_usd``);
* **prototypes + the cloner**: robot and table are spawned into env 0 only and
  ``isaaclab.cloner.replicate`` builds every world from the plan -- Newton's importer ignores prims
  under ``/World/envs`` the plan does not name (``newton/patches.py``, "Multi-asset");
* **before replicating**: the cloth meshes (replication is when Newton imports the stage, so a mesh
  authored later is invisible to the solver), ``CollisionAPI`` pushed onto the robot's mesh prims
  (Newton skips the URDF importer's collision Xforms) and the PD gains written into the USD drives
  (MuJoCo bakes actuation at build);
* quaternions are xyzw (the robot's base orientation).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from isaacsimenvs.tasks.cloth.utils.rigid import scene as rigid_scene

CLOTH = "Cloth"


def _log(t0: float, msg: str) -> None:
    print(f"[scene] {time.perf_counter() - t0:7.1f}s  {msg}", flush=True)


def build(env, spec, rest_points: list, triangles: list[int]) -> None:
    """Spawn everything and replicate; sets ``env.robot``, ``env.table``, ``env._cloth``,
    ``env._table_half_height``. ``rest_points``: the cloth's rest grid (local, z = 0)."""
    import isaaclab.sim as sim_utils
    from isaaclab import cloner
    from isaaclab.assets import Articulation, DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
    from isaaclab.sim.utils import get_current_stage

    from isaacsimenvs.newton import patches, physx_schema_shim, usd_cache

    physx_schema_shim.install()  # bake_usd authors PhysxSchema attributes
    t0 = time.perf_counter()
    a, n = env.cfg.assets, int(env.num_envs)
    tmp = Path(tempfile.mkdtemp(prefix="cloth_assets_"))
    env._tmp_asset_dir = str(tmp)
    work, bake = tmp / "usd", tmp / "baked_usd"
    work.mkdir(parents=True)

    # The cache keys on the converter options: exactly those the PhysX conversion used.
    sc = bool(env.cfg.robot_self_collision)
    robot_raw = usd_cache.lookup(a.robot_urdf, work, fix_base=True, self_collision=sc,
                                 joint_drive=rigid_scene.position_drive())
    if sc:
        rigid_scene.apply_self_collision_filters(robot_raw, spec.self_collision_filter)
    robot_usd = rigid_scene.bake_usd(
        robot_raw, bake, "robot",
        rigid_body={"physxRigidBody:disableGravity": True, "physxRigidBody:maxDepenetrationVelocity": 1000.0},
        articulation={"physxArticulation:enabledSelfCollisions": sc,
                      "physxArticulation:solverPositionIterationCount": 8,
                      "physxArticulation:solverVelocityIterationCount": 0},
    )
    table_usd = rigid_scene.bake_usd(
        usd_cache.lookup(a.table_urdf, work, fix_base=False), bake, "table",
        rigid_body={"physics:kinematicEnabled": True, "physxRigidBody:disableGravity": True},
        articulation={"physics:articulationEnabled": False},
    )
    env._robot_urdf_path, env._table_urdf_path = a.robot_urdf, a.table_urdf
    env._table_half_height = 0.5 * rigid_scene.urdf_box_size(a.table_urdf)[2]
    _log(t0, "robot + table USDs from the cache, baked")

    stage = get_current_stage()
    for path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(path).IsValid():
            stage.DefinePrim(path, "Xform")
    robot_cfg = rigid_scene.robot_cfg(robot_usd, spec, env.cfg.robot_root_pos, env.cfg.action.arm_effort_limit,
                                      rot=(0.0, 0.0, 0.0, 1.0))
    robot_cfg.spawn.spawn_path = "/World/envs/env_0/Robot"           # a prototype; the cloner replicates
    env.robot = Articulation(robot_cfg)
    env.table = RigidObject(RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=MultiUsdFileCfg(usd_path=[table_usd], spawn_paths=["/World/envs/env_0/Table"], random_choice=False),
    ))
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    author_cloth(env, stage, rest_points, triangles)
    env._cloth = DeformableObject(DeformableObjectCfg(
        prim_path=f"/World/envs/env_.*/{CLOTH}", spawn=None,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),  # the height is in the points
    ))
    _log(t0, f"robot, table and {n} cloths authored")

    fixed, marked = patches.push_collision_api_to_meshes(stage)
    hulls = patches.author_convex_hull_approximation(stage)
    written, missing = patches.author_drive_gains(stage, spec)
    print(f"[scene] CollisionAPI on {marked} mesh prims under {fixed} Xforms, {hulls} convex hulls, PD gains on "
          f"{written} drives" + (f" ({missing} without drive attrs)" if missing else ""), flush=True)

    queued = list(cloner.REPLICATION_QUEUE)
    groups = patches._plan_groups(queued)
    valid_set = patches._round_robin_valid_set([cloner.num_spawn_variants(c.spawn) for c in groups], n, str(env.device))
    before = [("spawn_paths", list(c.spawn.spawn_paths)) if getattr(c.spawn, "spawn_paths", None) is not None
              else ("spawn_path", c.spawn.spawn_path) for c in groups]
    plan = cloner.make_clone_plan(queued, n, env.scene.cfg.env_spacing, str(env.device),
                                  clone_strategy=cloner.sequential, valid_set=valid_set)
    patches._assert_spawn_paths_unchanged(groups, before)
    cloner.replicate(plan, stage=env.scene.stage, replicate_physics=True)
    _log(t0, f"replicated {n} envs from {len(plan.sources)} prototype row(s)")

    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.deformable_objects["cloth"] = env._cloth


def author_cloth(env, stage, rest_points: list, triangles: list[int]) -> None:
    """One ``UsdGeom.Mesh`` per env, bound to a physics material carrying the ``newton:*`` VBD
    parameters (``DeformableObject`` identifies it by ``newton:density``). The spawn height is baked
    into the POINTS: a transform on the prim composes with ``init_state.pos``, and ``init_state``
    alone left the cloth on the ground."""
    from pxr import Sdf, UsdGeom, UsdShade

    c = env.cfg.deformable_cloth
    z = float(env.cfg.reset.table_reset_z) + float(c.start_height)
    points = [(p[0], p[1], p[2] + z) for p in rest_points]
    params = (("newton:density", c.density), ("newton:particleRadius", c.particle_radius),
              ("newton:triKe", c.tri_ke), ("newton:triKa", c.tri_ka), ("newton:triKd", c.tri_kd),
              ("newton:edgeKe", c.edge_ke), ("newton:edgeKd", c.edge_kd))
    for path in env.scene.env_prim_paths:
        root = Sdf.Path(path)
        mesh = UsdGeom.Mesh.Define(stage, root.AppendChild(CLOTH))
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexIndicesAttr(list(triangles))
        mesh.CreateFaceVertexCountsAttr([3] * (len(triangles) // 3))
        mesh.CreateDisplayColorAttr([tuple(c.color)])
        mesh.CreateDoubleSidedAttr(True)  # a folded cloth is seen from both sides
        mat_path = root.AppendChild("ClothMaterial")
        mat = UsdShade.Material.Define(stage, mat_path).GetPrim()
        for name, value in params:
            mat.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(float(value))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).GetDirectBindingRel("physics").SetTargets([mat_path])


def set_base_materials(env, spec) -> None:
    """``model.shape_material_mu`` (Newton has one mu per shape; PhysX's static = dynamic collapse to
    it exactly): every shape ``assets.robot_friction``, fingertip shapes ``finger_tip_friction``, the
    table's ``table_friction``. Restitution 0. Shapes are matched through their owning body's label."""
    import re

    import warp as wp

    a = env.cfg.assets
    if not a.modify_asset_frictions:
        return
    model = env.sim.physics_manager.get_model()
    labels = [str(lab) for lab in model.body_label]
    shape_body = model.shape_body.numpy()
    mu = model.shape_material_mu.numpy().copy()
    mu[:] = float(a.robot_friction)
    tips = tuple(spec.fingertip_link_names)
    table = re.compile(r"/World/envs/env_[^/]+/Table")
    n_tip = n_table = 0
    for s, b in enumerate(shape_body):
        if b < 0:
            continue
        lab = labels[int(b)]
        if any(lab.rsplit("/", 1)[-1] == t or lab.endswith("/" + t) for t in tips):
            mu[s] = float(a.finger_tip_friction)
            n_tip += 1
        elif table.search(lab):
            mu[s] = float(a.table_friction)
            n_table += 1
    if n_tip == 0 or n_table == 0:
        raise RuntimeError(f"material pass matched {n_tip} fingertip and {n_table} table shapes; body labels look "
                           f"like {labels[:3]}")
    model.shape_material_mu.assign(wp.array(mu, dtype=model.shape_material_mu.dtype))
    rest = model.shape_material_restitution.numpy().copy()
    rest[:] = 0.0
    model.shape_material_restitution.assign(wp.array(rest, dtype=model.shape_material_restitution.dtype))
    print(f"[scene] friction: {len(mu)} shapes at {a.robot_friction}, {n_tip} fingertip shapes at "
          f"{a.finger_tip_friction}, {n_table} table shapes at {a.table_friction}", flush=True)
