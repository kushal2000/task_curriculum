"""Robot/table USD preparation shared by the PhysX and Newton scene adapters."""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CONTACT_OFFSET = 0.002
REST_OFFSET = 0.0

def urdf_box_size(urdf_path: str) -> tuple[float, float, float]:
    """The single collision ``<box size>`` of a table URDF."""
    boxes = ET.parse(urdf_path).getroot().findall("link/collision/geometry/box")
    if len(boxes) != 1:
        raise ValueError(f"{urdf_path}: expected one collision box, got {len(boxes)}")
    return tuple(float(v) for v in boxes[0].get("size").split())


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


def robot_cfg(usd_path: str, spec, root_pos, arm_effort_limit, rot=(1.0, 0.0, 0.0, 0.0)):
    """``rot``: the base orientation in the backend's convention (identity: wxyz on Isaac Lab 2.3,
    ``(0, 0, 0, 1)`` xyzw on 3.0)."""
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
            pos=tuple(float(v) for v in root_pos), rot=tuple(rot),
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
