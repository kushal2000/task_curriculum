"""The PhysX scene (Isaac Sim 5.1 / Isaac Lab 2.3): robot, table, ground, light, and the rigid sheets.

Robot and table go URDF -> USD (Isaac Lab's converter) -> a baked copy with the PhysX properties
authored (Play's recipe: contact offset 2 mm / rest 0 on every collider; the robot gravity-free,
8/0 solver iterations, self-collision on with the spec's filtered pairs). The sheets are authored
straight into the stage: a template in a private in-memory stage, ``Sdf.CopySpec``-ed into every env
inside one ``Sdf.ChangeBlock``, each env given its design's dimensions -- one articulation spans all
envs (every env has the same prim structure).
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from . import sheet_designs

SHEET = "Sheet"
PLATES = ("minus", "plus")      # body names; "minus" is the articulation root
HINGE = "hinge"

CONTACT_OFFSET = 0.002
REST_OFFSET = 0.0


def _log(t0: float, msg: str) -> None:
    print(f"[scene] {time.perf_counter() - t0:7.1f}s  {msg}", flush=True)


def urdf_box_size(urdf_path: str) -> tuple[float, float, float]:
    """The single collision ``<box size>`` of a table URDF."""
    boxes = ET.parse(urdf_path).getroot().findall("link/collision/geometry/box")
    if len(boxes) != 1:
        raise ValueError(f"{urdf_path}: expected one collision box, got {len(boxes)}")
    return tuple(float(v) for v in boxes[0].get("size").split())


# ------------------------------------------------------------------ URDF -> baked USD


def convert_urdf(asset_path: str, work_dir: Path, *, fix_base: bool, self_collision: bool | None = None,
                 joint_drive=None) -> str:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    kwargs = dict(asset_path=asset_path, usd_dir=str(work_dir / Path(asset_path).stem),
                  force_usd_conversion=True, fix_base=fix_base, merge_fixed_joints=True,
                  make_instanceable=False, joint_drive=joint_drive)
    if self_collision is not None:
        kwargs["self_collision"] = self_collision
    return UrdfConverter(UrdfConverterCfg(**kwargs)).usd_path


def position_drive():
    """Zero-gain position drives, so the DriveAPI prims the implicit actuators write exist."""
    from isaaclab.sim.converters import UrdfConverterCfg

    return UrdfConverterCfg.JointDriveCfg(
        drive_type="force", target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    )


def apply_self_collision_filters(usd_path: str, pairs: dict[str, tuple[str, ...]]) -> int:
    """``FilteredPairsAPI`` on the robot bodies for ``pairs``. Every name must be a body of the
    imported robot: skipping unknown names would leave self-collision on with nothing filtered."""
    from pxr import Usd, UsdPhysics

    raw = Path(usd_path)
    physics = raw.parent / "configuration" / f"{raw.stem}_physics.usd"
    path = physics if physics.exists() else raw
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    bodies = {p.GetName(): p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
              if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    unknown = sorted({n for a, bs in pairs.items() for n in (a, *bs)} - set(bodies))
    if unknown:
        raise RuntimeError(f"self-collision filter names {unknown} are not bodies of {path.name}: {sorted(bodies)}")
    n = 0
    for link, partners in pairs.items():
        rel = UsdPhysics.FilteredPairsAPI.Apply(bodies[link]).CreateFilteredPairsRel()
        have = set(rel.GetTargets())
        for p in partners:
            if bodies[p].GetPath() not in have:
                rel.AddTarget(bodies[p].GetPath())
                n += 1
    stage.GetRootLayer().Save()
    return n


def bake_usd(raw_usd: str, bake_root: Path, subdir: str, *, rigid_body: dict, articulation: dict | None = None) -> str:
    """Copy the converted USD (and its folder) and author PhysX properties on the copy:
    ``rigid_body`` / ``articulation`` map attribute names to values; every collider gets Play's
    contact / rest offsets. Instances are de-instanced so the edits land."""
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    raw = Path(raw_usd)
    baked = bake_root / subdir / raw.parent.name / raw.name
    baked.parent.mkdir(parents=True, exist_ok=True)
    for child in raw.parent.iterdir():
        if child.name.startswith(".") or child.name == "config.yaml":
            continue
        dst = baked.parent / child.name
        if child.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(child, dst)
        else:
            shutil.copy2(child, dst)
    stage = Usd.Stage.Open(str(baked))
    root = stage.GetDefaultPrim() or next(iter(stage.GetPseudoRoot().GetChildren()))
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)

    def put(prim, name, value):
        vtype = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}[type(value)]
        attr = prim.GetAttribute(name)
        if attr and not str(attr.GetTypeName()):  # the converter sometimes emits untyped attributes
            prim.RemoveProperty(name)
            attr = None
        (attr or prim.CreateAttribute(name, vtype, False)).Set(value)

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            for k, v in rigid_body.items():
                put(prim, k, v)
        if articulation is not None and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if any(k.startswith("physxArticulation:") for k in articulation):
                PhysxSchema.PhysxArticulationAPI.Apply(prim)
            for k, v in articulation.items():
                put(prim, k, v)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            px = PhysxSchema.PhysxCollisionAPI(prim) or PhysxSchema.PhysxCollisionAPI.Apply(prim)
            px.CreateContactOffsetAttr().Set(CONTACT_OFFSET)
            px.CreateRestOffsetAttr().Set(REST_OFFSET)
    stage.GetRootLayer().Save()
    return str(baked)


def robot_cfg(usd_path: str, spec, root_pos, arm_effort_limit):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.from_files import UsdFileCfg

    arm_effort = None
    if arm_effort_limit:
        if len(arm_effort_limit) != len(spec.arm_joint_names):
            raise ValueError(f"action.arm_effort_limit needs {len(spec.arm_joint_names)} values")
        arm_effort = dict(zip(spec.arm_joint_names, map(float, arm_effort_limit)))
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(v) for v in root_pos), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={**spec.arm_default_joint_pos, **{n: 0.0 for n in spec.hand_joint_names}},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=[spec.arm_joint_regex], stiffness=spec.arm_stiffness,
                                       damping=spec.arm_damping, effort_limit_sim=arm_effort),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[spec.hand_joint_regex], stiffness=spec.hand_stiffness, damping=spec.hand_damping,
                armature=spec.hand_armature, effort_limit_sim=spec.hand_effort_limit,
                velocity_limit_sim=spec.hand_velocity_limit, friction=spec.hand_friction,
                dynamic_friction=spec.hand_dynamic_friction, viscous_friction=spec.hand_viscous_friction,
            ),
        },
    )


def build_robot_and_table(env, spec) -> None:
    """Spawn robot, table, ground and light into every env; sets ``env.robot``, ``env.table``,
    ``env._table_half_height`` and registers both with the scene."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane
    from isaaclab.sim.utils import get_current_stage

    t0 = time.perf_counter()
    a = env.cfg.assets
    tmp = Path(tempfile.mkdtemp(prefix="cloth_assets_"))
    env._tmp_asset_dir = str(tmp)
    work, bake = tmp / "usd", tmp / "baked_usd"
    work.mkdir(parents=True)

    self_collision = bool(env.cfg.robot_self_collision)
    robot_raw = convert_urdf(a.robot_urdf, work, fix_base=True, self_collision=self_collision, joint_drive=position_drive())
    if self_collision:
        apply_self_collision_filters(robot_raw, spec.self_collision_filter)
    robot_usd = bake_usd(
        robot_raw, bake, "robot",
        rigid_body={"physxRigidBody:disableGravity": True, "physxRigidBody:maxDepenetrationVelocity": 1000.0},
        articulation={"physxArticulation:enabledSelfCollisions": self_collision,
                      "physxArticulation:solverPositionIterationCount": 8,
                      "physxArticulation:solverVelocityIterationCount": 0},
    )
    table_usd = bake_usd(
        convert_urdf(a.table_urdf, work, fix_base=False), bake, "table",
        rigid_body={"physics:kinematicEnabled": True, "physxRigidBody:disableGravity": True},
        articulation={"physics:articulationEnabled": False},
    )
    env._robot_urdf_path, env._table_urdf_path = a.robot_urdf, a.table_urdf
    env._table_half_height = 0.5 * urdf_box_size(a.table_urdf)[2]
    _log(t0, "robot + table USDs baked")

    stage = get_current_stage()
    for path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(path).IsValid():
            stage.DefinePrim(path, "Xform")
    env.robot = Articulation(robot_cfg(robot_usd, spec, env.cfg.robot_root_pos, env.cfg.action.arm_effort_limit))
    env.table = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Table", spawn=UsdFileCfg(usd_path=table_usd)))
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    _log(t0, "robot + table spawned")


# ------------------------------------------------------------------ the sheets


def _apply_physx_api(prim, schema: str, attrs: dict) -> None:
    if schema not in prim.GetAppliedSchemas():
        prim.AddAppliedSchema(schema)
    for name, (vtype, value) in attrs.items():
        prim.CreateAttribute(name, vtype).Set(value)


def author_sheets(num_envs: int, dims, sheet_cfg, spawn_z) -> None:
    """Every env's two-plate, one-hinge sheet (``dims (N, 3)`` numpy; ``spawn_z(i)`` its root's
    env-frame height). The ~60 schema edits of the template go into a private in-memory stage --
    written into the scene's stage under Kit they took 78 s at 6144 envs."""
    from isaaclab.sim.utils import get_current_stage
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    s = sheet_cfg
    layer = get_current_stage().GetRootLayer()
    stage = Usd.Stage.CreateInMemory()
    tpl = f"/{SHEET}"
    dbl = UsdGeom.XformOp.PrecisionDouble
    UsdGeom.Xform.Define(stage, tpl).AddTranslateOp(precision=dbl)
    colors = {"minus": (0.29, 0.56, 0.76), "plus": (0.95, 0.68, 0.22)}
    for name in PLATES:
        body = UsdGeom.Xform.Define(stage, f"{tpl}/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr(1.0)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
        _apply_physx_api(body, "PhysxRigidBodyAPI", {
            "physxRigidBody:maxDepenetrationVelocity": (Sdf.ValueTypeNames.Float, float(s.max_depenetration_velocity)),
        })
        if name == PLATES[0]:
            UsdPhysics.ArticulationRootAPI.Apply(body)
            _apply_physx_api(body, "PhysxArticulationAPI", {
                "physxArticulation:enabledSelfCollisions": (Sdf.ValueTypeNames.Bool, False),
                "physxArticulation:solverPositionIterationCount": (Sdf.ValueTypeNames.Int, int(s.solver_position_iterations)),
                "physxArticulation:solverVelocityIterationCount": (Sdf.ValueTypeNames.Int, int(s.solver_velocity_iterations)),
            })
        cube = UsdGeom.Cube.Define(stage, f"{tpl}/{name}/collider")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp(precision=dbl)
        cube.AddScaleOp(precision=dbl)
        cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colors[name])]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        _apply_physx_api(cube.GetPrim(), "PhysxCollisionAPI", {
            "physxCollision:contactOffset": (Sdf.ValueTypeNames.Float, float(s.contact_offset)),
            "physxCollision:restOffset": (Sdf.ValueTypeNames.Float, float(s.rest_offset)),
        })
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{tpl}/{HINGE}")
    joint.CreateBody0Rel().SetTargets([f"{tpl}/{PLATES[0]}"])
    joint.CreateBody1Rel().SetTargets([f"{tpl}/{PLATES[1]}"])
    # Joint x along the crease pointing -y: a positive angle turns the + plate up and over.
    joint.CreateAxisAttr(UsdPhysics.Tokens.x)
    rot = Gf.Quatf(math.cos(-math.pi / 4), Gf.Vec3f(0.0, 0.0, math.sin(-math.pi / 4)))
    joint.CreateLocalRot0Attr(rot)
    joint.CreateLocalRot1Attr(rot)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    joint.CreateLowerLimitAttr(float(s.hinge_limits_deg[0]))
    joint.CreateUpperLimitAttr(float(s.hinge_limits_deg[1]))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)
    src = stage.GetRootLayer()

    def put(path: str, value) -> None:
        spec = layer.GetAttributeAtPath(Sdf.Path(path))
        if spec is None:
            raise RuntimeError(f"no attribute spec at {path}")
        spec.default = value

    with Sdf.ChangeBlock():
        for i in range(num_envs):
            root = f"/World/envs/env_{i}/{SHEET}"
            Sdf.CreatePrimInLayer(layer, f"/World/envs/env_{i}")
            if not Sdf.CopySpec(src, tpl, layer, root):
                raise RuntimeError(f"Sdf.CopySpec {tpl} -> {root} failed")
            spec = sheet_designs.single_hinge(*(float(v) for v in dims[i]))
            put(f"{root}.xformOp:translate", Gf.Vec3d(0.0, 0.0, float(spawn_z(i))))
            for box in spec.plates:
                m, com, inertia = sheet_designs.box_mass_properties(box, float(s.density))
                put(f"{root}/{box.name}.physics:mass", float(m))
                put(f"{root}/{box.name}.physics:centerOfMass", Gf.Vec3f(*map(float, com)))
                put(f"{root}/{box.name}.physics:diagonalInertia", Gf.Vec3f(*map(float, inertia)))
                put(f"{root}/{box.name}/collider.xformOp:translate", Gf.Vec3d(*box.center))
                put(f"{root}/{box.name}/collider.xformOp:scale", Gf.Vec3d(*(2.0 * h for h in box.half_extents)))
            anchor = Gf.Vec3f(*map(float, spec.hinges[0].anchor))
            put(f"{root}/{HINGE}.physics:localPos0", anchor)
            put(f"{root}/{HINGE}.physics:localPos1", anchor)


def sheet_articulation(sheet_cfg, spawn_z0: float):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    s, rd = sheet_cfg, sheet_cfg.randomization
    stiffness = float(s.hinge_stiffness)
    # The position gain must exist to be written per reset: a negligible one when DR draws it.
    if rd.enabled and float(rd.hinge_stiffness[1]) > 0.0 and stiffness == 0.0:
        stiffness = 1.0e-9
    return Articulation(ArticulationCfg(
        prim_path=f"/World/envs/env_.*/{SHEET}",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, float(spawn_z0))),
        actuators={HINGE: ImplicitActuatorCfg(
            joint_names_expr=[HINGE], stiffness=stiffness, damping=float(s.hinge_damping),
            armature=float(s.hinge_armature), effort_limit_sim=None,
        )},
    ))


# ------------------------------------------------------------------ materials


def set_base_materials(env, spec) -> None:
    """Robot shapes ``assets.robot_friction``, fingertip shapes ``finger_tip_friction``, the table
    ``table_friction`` (static = dynamic, restitution 0; PhysX averages the two sides of a contact)."""
    a = env.cfg.assets
    if not a.modify_asset_frictions:
        return
    ids = torch.arange(env.num_envs, dtype=torch.int64)
    view = env.robot.root_physx_view
    mats = view.get_material_properties()
    mats[:] = torch.tensor([a.robot_friction, a.robot_friction, 0.0])
    start = 0
    tips = set(spec.fingertip_link_names)
    for name, path in zip(view.shared_metatype.link_names, view.link_paths[0]):
        n = env.robot._physics_sim_view.create_rigid_body_view(path).max_shapes
        if name in tips:
            mats[:, start:start + n] = torch.tensor([a.finger_tip_friction, a.finger_tip_friction, 0.0])
        start += n
    if start != view.max_shapes:
        raise RuntimeError(f"robot shape count mismatch: walked {start}, view has {view.max_shapes}")
    view.set_material_properties(mats, ids)
    tview = env.table.root_physx_view
    tm = tview.get_material_properties()
    tm[:] = torch.tensor([a.table_friction, a.table_friction, 0.0])
    tview.set_material_properties(tm, ids)
