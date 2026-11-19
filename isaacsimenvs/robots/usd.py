"""Robot and table assets: URDF -> USD conversion, physics baking, and the Isaac Lab asset configs.

Shared by every task that spawns a robot from a ``RobotSpec`` (the cloth tasks, the Play task):

    convert_urdf_to_usd    Isaac Sim's URDF importer (needs Kit), with SDF collision markers
    bake_usd               a copy of the converted USD with physics properties authored
    apply_self_collision_filters / self_collision_keep
                           the robot's self-collision pairs (``RobotSpec.self_collision_filter``,
                           ``cfg.robot_self_collision_scope``)
    build_robot_articulation_usd_cfg / build_rigid_object_cfg
                           the ``ArticulationCfg`` / ``RigidObjectCfg`` that spawn them

Callers reach these through the module (``robot_usd.convert_urdf_to_usd(...)``), so the Newton
backend can replace them in one place (``tasks/cloth/backends/vbd/newton_support``): conversion by a
lookup in the kit-less USD cache, the configs by prototype-and-clone spawning.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg

from isaacsimenvs.robots import RobotSpec

__all__ = [
    "apply_self_collision_filters",
    "bake_usd",
    "build_rigid_object_cfg",
    "build_robot_articulation_usd_cfg",
    "convert_urdf_to_usd",
    "robot_joint_drive_cfg",
    "self_collision_keep",
]

_CONTACT_OFFSET = 0.002
_REST_OFFSET = 0.0

# group: "rb" (RigidBodyAPI) or "art" (ArticulationRootAPI).
# attr_name: USD attribute path. vtype_str: matched against pxr.Sdf.ValueTypeNames.
_PHYSICS_SPECS: dict[str, tuple[str, str, str]] = {
    "kinematic_enabled": ("rb", "physics:kinematicEnabled", "Bool"),
    "disable_gravity": ("rb", "physxRigidBody:disableGravity", "Bool"),
    "max_depenetration_velocity": ("rb", "physxRigidBody:maxDepenetrationVelocity", "Float"),
    "rb_solver_position_iterations": ("rb", "physxRigidBody:solverPositionIterationCount", "Int"),
    "rb_solver_velocity_iterations": ("rb", "physxRigidBody:solverVelocityIterationCount", "Int"),
    "articulation_enabled": ("art", "physics:articulationEnabled", "Bool"),
    "enabled_self_collisions": ("art", "physxArticulation:enabledSelfCollisions", "Bool"),
    "solver_position_iterations": ("art", "physxArticulation:solverPositionIterationCount", "Int"),
    "solver_velocity_iterations": ("art", "physxArticulation:solverVelocityIterationCount", "Int"),
}


def build_robot_articulation_usd_cfg(
    usd_path: str,
    spec: RobotSpec,
    root_pos: tuple[float, float, float] = (0.0, 0.8, 0.0),
) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(v) for v in root_pos),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                **spec.arm_default_joint_pos,
                **{name: 0.0 for name in spec.hand_joint_names},
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[spec.arm_joint_regex],
                stiffness=spec.arm_stiffness,
                damping=spec.arm_damping,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[spec.hand_joint_regex],
                stiffness=spec.hand_stiffness,  # PD only; the passive spring is a joint effort, applied by the task
                damping=spec.hand_damping,
                armature=spec.hand_armature,
                effort_limit_sim=spec.hand_effort_limit or None,
                velocity_limit_sim=spec.hand_velocity_limit,
                friction=spec.hand_friction,
                dynamic_friction=spec.hand_dynamic_friction,
                viscous_friction=spec.hand_viscous_friction,
            ),
        },
    )


def build_rigid_object_cfg(prim_path: str, usd_paths: list[str]) -> RigidObjectCfg:
    """Spawn a RigidObject from one or more pre-baked USDs (round-robin)."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=MultiUsdFileCfg(usd_path=list(usd_paths), random_choice=False),
    )


def _set_usd_attr(prim, name: str, value, value_type) -> None:
    # The URDF converter occasionally emits attributes with malformed type
    # names; in that case remove and recreate so the typed Set lands.
    attr = prim.GetAttribute(name)
    if attr and (not attr.GetTypeName() or not str(attr.GetTypeName())):
        prim.RemoveProperty(name)
        attr = None
    (attr or prim.CreateAttribute(name, value_type, False)).Set(value)


@dataclass(frozen=True)
class _UrdfSdfCollisionMarker:
    mesh_stem: str
    mesh_filename: str
    resolution: int | None = None
    margin: float | None = None
    narrow_band_thickness: float | None = None
    subgrid_resolution: int | None = None


def _usd_safe_identifier(name: str) -> str:
    """Mirror the conservative subset of USD identifier rules we need here."""
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"mesh_{safe}"
    return safe


def _resolve_urdf_mesh_path(urdf_path: Path, mesh_filename: str) -> Path:
    mesh_path = Path(mesh_filename)
    if mesh_path.is_absolute():
        return mesh_path
    return (urdf_path.parent / mesh_path).resolve()


def _prepare_urdf_for_isaacsim(asset_path: str, usd_work_dir: Path) -> str:
    """Return a URDF path whose mesh stems are valid USD prim identifiers.

    Isaac's URDF importer names USD prims from mesh stems.  Meshes such as
    ``6_hole_patch.obj`` therefore fail conversion because USD identifiers
    cannot start with a digit.  When needed, write a temporary URDF with safe
    mesh aliases while leaving the source asset untouched.
    """
    urdf_path = Path(asset_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    changed = False
    alias_dir = usd_work_dir / "_mesh_aliases" / urdf_path.stem
    alias_by_source: dict[Path, Path] = {}
    used_alias_names: set[str] = set()

    for mesh_tag in root.findall(".//mesh"):
        filename = mesh_tag.get("filename")
        if not filename:
            continue
        source_mesh = _resolve_urdf_mesh_path(urdf_path, filename)
        original_stem = source_mesh.stem
        safe_stem = _usd_safe_identifier(original_stem)
        if safe_stem == original_stem:
            continue

        changed = True
        alias = alias_by_source.get(source_mesh)
        if alias is None:
            alias_dir.mkdir(parents=True, exist_ok=True)
            alias_name = f"{safe_stem}{source_mesh.suffix}"
            if alias_name in used_alias_names:
                index = 1
                while f"{safe_stem}_{index}{source_mesh.suffix}" in used_alias_names:
                    index += 1
                alias_name = f"{safe_stem}_{index}{source_mesh.suffix}"
            used_alias_names.add(alias_name)
            alias = alias_dir / alias_name
            shutil.copy2(source_mesh, alias)
            alias_by_source[source_mesh] = alias

        mesh_tag.set("filename", str(alias))

    if not changed:
        return asset_path

    # The copied URDF lives in the converter work dir, so make every remaining
    # mesh path absolute to preserve source-relative references.
    for mesh_tag in root.findall(".//mesh"):
        filename = mesh_tag.get("filename")
        if filename:
            mesh_tag.set("filename", str(_resolve_urdf_mesh_path(urdf_path, filename)))

    out_dir = usd_work_dir / "_urdf_preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / urdf_path.name
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return str(out_path)


def _parse_optional_int(value: str | None) -> int | None:
    return None if value is None else int(value)


def _parse_optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _parse_urdf_sdf_collision_markers(asset_path: str) -> list[_UrdfSdfCollisionMarker]:
    urdf_path = Path(asset_path)
    root = ET.parse(urdf_path).getroot()
    markers: list[_UrdfSdfCollisionMarker] = []
    for collision in root.findall(".//collision"):
        sdf_tag = collision.find("sdf")
        if sdf_tag is None:
            continue
        mesh_tag = collision.find("geometry/mesh")
        if mesh_tag is None or not mesh_tag.get("filename"):
            continue
        mesh_filename = str(mesh_tag.get("filename"))
        mesh_path = _resolve_urdf_mesh_path(urdf_path, mesh_filename)
        markers.append(
            _UrdfSdfCollisionMarker(
                mesh_stem=_usd_safe_identifier(mesh_path.stem),
                mesh_filename=mesh_filename,
                resolution=_parse_optional_int(sdf_tag.get("resolution")),
                margin=_parse_optional_float(sdf_tag.get("margin")),
                narrow_band_thickness=_parse_optional_float(
                    sdf_tag.get("narrow_band_thickness")
                    or sdf_tag.get("narrowBandThickness")
                ),
                subgrid_resolution=_parse_optional_int(
                    sdf_tag.get("subgrid_resolution")
                    or sdf_tag.get("subgridResolution")
                ),
            )
        )
    return markers


def _apply_urdf_sdf_collision_markers(
    usd_path: str,
    source_asset_path: str,
    markers: list[_UrdfSdfCollisionMarker],
) -> None:
    if not markers:
        return

    from pxr import Usd, UsdPhysics

    from isaaclab.sim.schemas import SDFMeshPropertiesCfg, define_mesh_collision_properties

    raw_usd_path = Path(usd_path)
    physics_usd_path = raw_usd_path.parent / "configuration" / f"{raw_usd_path.stem}_physics.usd"
    edit_usd_path = physics_usd_path if physics_usd_path.exists() else raw_usd_path

    stage = Usd.Stage.Open(str(edit_usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open USD while applying URDF SDF markers: {edit_usd_path}")
    stage.Load()

    marker_by_stem = {marker.mesh_stem: marker for marker in markers}
    matched: dict[str, int] = {marker.mesh_stem: 0 for marker in markers}

    fallback_matches = []
    collider_matches = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        path = prim.GetPath().pathString
        path_parts = [part for part in path.split("/") if part]
        marker = next((marker_by_stem[part] for part in path_parts if part in marker_by_stem), None)
        if marker is None:
            continue
        if path.startswith("/colliders/"):
            collider_matches.append((prim, marker))
        else:
            fallback_matches.append((prim, marker))

    for prim, marker in collider_matches or fallback_matches:
        define_mesh_collision_properties(
            str(prim.GetPath()),
            SDFMeshPropertiesCfg(
                sdf_margin=marker.margin,
                sdf_narrow_band_thickness=marker.narrow_band_thickness,
                sdf_resolution=marker.resolution,
                sdf_subgrid_resolution=marker.subgrid_resolution,
            ),
            stage=stage,
        )
        matched[marker.mesh_stem] += 1

    stage.GetRootLayer().Save()

    matched_count = sum(matched.values())
    missing = [stem for stem, count in matched.items() if count == 0]
    if missing:
        print(
            f"[robot_usd] warning: URDF SDF markers in {source_asset_path!r} did not match "
            f"USD collision prims for mesh stems {missing}",
            flush=True,
        )
    if matched_count:
        details = ", ".join(f"{stem}:{count}" for stem, count in matched.items() if count)
        print(
            f"[robot_usd] applied URDF SDF collision markers to {matched_count} prims "
            f"in {edit_usd_path.name} ({details})",
            flush=True,
        )


def convert_urdf_to_usd(
    asset_path: str,
    usd_work_dir: Path,
    *,
    fix_base: bool,
    self_collision: bool | None = None,
    replace_cylinders_with_capsules: bool = False,
    joint_drive=None,
) -> str:
    converter_asset_path = _prepare_urdf_for_isaacsim(asset_path, usd_work_dir)
    cfg_kwargs = dict(
        asset_path=converter_asset_path,
        usd_dir=str(usd_work_dir / Path(asset_path).stem),
        force_usd_conversion=True,
        fix_base=fix_base,
        merge_fixed_joints=True,
        make_instanceable=False,
        replace_cylinders_with_capsules=replace_cylinders_with_capsules,
        joint_drive=joint_drive,
    )
    if self_collision is not None:
        cfg_kwargs["self_collision"] = self_collision
    usd_path = UrdfConverter(UrdfConverterCfg(**cfg_kwargs)).usd_path
    _apply_urdf_sdf_collision_markers(
        usd_path,
        converter_asset_path,
        _parse_urdf_sdf_collision_markers(converter_asset_path),
    )
    return usd_path



def self_collision_keep(env):
    """``cfg.robot_self_collision_scope`` as a predicate over two body names (None = keep all).

    "all": everything collides except the spec's filter (SimToolReal). "hand": only pairs of hand
    bodies (palm + finger segments); the arm's links do not collide with anything of the robot.
    "tips": only the distal / middle segments of different fingers, and the thumb's distal / middle
    against the palm -- the pairs a policy trained without self-collision pushes through each other.
    """
    import re

    scope = str(getattr(env.cfg, "robot_self_collision_scope", "all"))
    if scope == "all":
        return None
    spec = env.robot_spec
    seg = {k: re.compile(v) for k, v in spec.finger_segment_regex.items()}
    palm = {spec.palm_body_name, *spec.palm_side_link_names}

    def is_hand(b):
        return b in palm or any(r.fullmatch(b) for r in seg.values())

    def tip_finger(b):  # finger name of a distal / middle segment, else None
        for k in ("distal", "middle"):
            m = seg[k].fullmatch(b)
            if m:
                return m.group(1) if m.groups() else b
        return None

    if scope == "hand":
        return lambda a, b: is_hand(a) and is_hand(b)
    if scope == "tips":
        def keep(a, b):
            fa, fb = tip_finger(a), tip_finger(b)
            if fa is not None and fb is not None:
                return fa != fb
            return (fa is not None and "thumb" in fa and b in palm) or (fb is not None and "thumb" in fb and a in palm)
        return keep
    raise ValueError(f"robot_self_collision_scope {scope!r}: expected all / hand / tips")


def apply_self_collision_filters(usd_path: str, pairs: dict[str, tuple[str, ...]], keep=None) -> int:
    """Author ``FilteredPairsAPI`` on the robot's bodies so ``pairs`` (``RobotSpec.
    self_collision_filter``) do not collide with each other; returns how many were authored.

    With the articulation's self-collisions on, this is Isaac Gym's behaviour in SimToolReal:
    everything collides except the listed pairs. Ported from simtoolreal 84058661, with one
    change: every name must be a body of the imported robot. Upstream skipped unknown names,
    which is how a filter list for the wrong robot (or pre-merge_fixed_joints names) would leave
    self-collision on with nothing filtered, and the hand exploding at reset.
    """
    from pxr import Usd, UsdPhysics

    raw_usd_path = Path(usd_path)
    physics_usd_path = raw_usd_path.parent / "configuration" / f"{raw_usd_path.stem}_physics.usd"
    edit_usd_path = physics_usd_path if physics_usd_path.exists() else raw_usd_path
    stage = Usd.Stage.Open(str(edit_usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open USD while applying self-collision filters: {edit_usd_path}")

    body_by_name = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    unknown = sorted({n for a, bs in pairs.items() for n in (a, *bs)} - set(body_by_name))
    if unknown:
        raise RuntimeError(
            f"self-collision filter names {unknown} are not bodies of {edit_usd_path.name} "
            f"(bodies: {sorted(body_by_name)}); names must be post-merge_fixed_joints"
        )

    authored = 0
    for link, partners in pairs.items():
        rel = UsdPhysics.FilteredPairsAPI.Apply(body_by_name[link]).CreateFilteredPairsRel()
        existing = set(rel.GetTargets())
        for partner in partners:
            path = body_by_name[partner].GetPath()
            if path not in existing:
                rel.AddTarget(path)
                existing.add(path)
                authored += 1
    kept = []
    if keep is not None:
        # Filter every remaining pair the scope does not keep.
        names = sorted(body_by_name)
        for i, a in enumerate(names):
            rel = UsdPhysics.FilteredPairsAPI.Apply(body_by_name[a]).CreateFilteredPairsRel()
            existing = set(rel.GetTargets())
            for b in names[i + 1:]:
                if keep(a, b):
                    if body_by_name[b].GetPath() not in existing and b not in pairs.get(a, ()) and a not in pairs.get(b, ()):
                        kept.append((a, b))
                    continue
                path = body_by_name[b].GetPath()
                if path not in existing:
                    rel.AddTarget(path)
                    existing.add(path)
                    authored += 1
    stage.GetRootLayer().Save()
    print(
        f"[robot_usd] self-collision: filtered {authored} link pairs "
        f"across {len(body_by_name)} robot bodies in {edit_usd_path.name}"
        + (f"; scope keeps {len(kept)} body pairs, e.g. {kept[:6]}" if keep is not None else ""),
        flush=True,
    )
    return authored


def robot_joint_drive_cfg():
    # DriveAPI prims must exist for ImplicitActuator runtime gains to land.
    return UrdfConverterCfg.JointDriveCfg(
        drive_type="force", target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    )


def bake_usd(
    raw_usd_path: str,
    bake_root: Path,
    baked_subdir: str,
    *,
    props: dict | None = None,
    apply_physx_articulation: bool = False,
    collision_enabled: bool | None = None,
) -> str:
    """Copy raw USD into bake_root/baked_subdir and pre-author physics props.

    ``props`` keys come from ``_PHYSICS_SPECS``; ``None`` values are skipped,
    and keys whose group doesn't match a prim's APIs are skipped per-prim.
    """
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    vtypes = {
        "Bool": Sdf.ValueTypeNames.Bool,
        "Float": Sdf.ValueTypeNames.Float,
        "Int": Sdf.ValueTypeNames.Int,
    }
    props = props or {}

    raw = Path(raw_usd_path)
    baked_usd_path = bake_root / baked_subdir / raw.parent.name / raw.name
    baked_usd_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw, baked_usd_path)
    for child in raw.parent.iterdir():
        if child.name.startswith(".") or child.name in (raw.name, "config.yaml"):
            continue
        dst = baked_usd_path.parent / child.name
        if child.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(child, dst)
        elif child.is_file():
            shutil.copy2(child, dst)

    stage = Usd.Stage.Open(str(baked_usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open baked USD: {baked_usd_path}")
    root = stage.GetDefaultPrim()
    if not (root and root.IsValid()):
        root = next((p for p in stage.GetPseudoRoot().GetChildren() if p.IsValid()), None)
    if root is None:
        raise RuntimeError(f"No root prim in USD: {baked_usd_path}")

    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)

    for prim in Usd.PrimRange(root):
        is_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        is_art = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        if is_rb:
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if is_art and apply_physx_articulation:
            PhysxSchema.PhysxArticulationAPI.Apply(prim)
        for key, val in props.items():
            if val is None:
                continue
            group, attr_name, vtype_str = _PHYSICS_SPECS[key]
            if group == "rb" and not is_rb:
                continue
            if group == "art" and not is_art:
                continue
            _set_usd_attr(prim, attr_name, val, vtypes[vtype_str])
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            px = PhysxSchema.PhysxCollisionAPI(prim) or PhysxSchema.PhysxCollisionAPI.Apply(prim)
            px.CreateContactOffsetAttr().Set(_CONTACT_OFFSET)
            px.CreateRestOffsetAttr().Set(_REST_OFFSET)
            if collision_enabled is not None:
                ce = UsdPhysics.CollisionAPI(prim)
                (ce.GetCollisionEnabledAttr() or ce.CreateCollisionEnabledAttr()).Set(
                    collision_enabled
                )

    stage.GetRootLayer().Save()
    return str(baked_usd_path)
