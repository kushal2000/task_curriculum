"""Author Play's procedural tools straight into the stage, without the URDF importer.

The default path (``scene_utils.setup_scene``) converts every tool URDF to a layered USD (base,
physics, sensor and robot layers; ~70 s for the 1200-tool pool), bakes a second copy for the goal
marker, spawns all 2400 as prototypes and copies one of each into every env (``MultiUsdFileCfg``):
~9 minutes at 24576 envs. A tool is one rigid body of 1-2 boxes / capsules, so here its prims are
written with the Sdf API -- one prototype per tool, in the root layer, then ``Sdf.CopySpec`` into the
envs inside one ``Sdf.ChangeBlock`` -- with what the importer + ``robots.usd.bake_usd`` produce
(checked against a flattened baked USD, 2026-11-23):

    <env>/Object                 Xform
      object_root                Xform, RigidBodyAPI + MassAPI + PhysxRigidBodyAPI:
                                 mass, centerOfMass, diagonalInertia, principalAxes (URDF <inertial>),
                                 kinematicEnabled, disableGravity, maxDepenetrationVelocity
        collisions/mesh_<k>      Xform at the URDF <collision> origin (a box: scaled to its size)
          box | capsule          Cube(size 1) | Capsule(axis Z, radius, height = URDF length),
                                 CollisionAPI + PhysxCollisionAPI: contactOffset 2 mm, restOffset 0

The goal marker (``GoalViz``) is the same tool, kinematic, without gravity, its shapes not colliding.
The shapes are left renderable (the importer's separate visual copies are not authored). Materials
are not authored: ``apply_physx_material_properties`` sets the frictions through the tensor API.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

__all__ = ["author_tool_prims"]

CONTACT_OFFSET = 0.002
REST_OFFSET = 0.0


def _quat_rpy(rpy) -> tuple[float, float, float, float]:
    """wxyz of URDF roll-pitch-yaw (R = Rz(y) Ry(p) Rx(r))."""
    r, p, y = (0.5 * float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def _origin(el):
    o = el.find("origin")
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
    return xyz, rpy


def parse_tool(urdf_path) -> dict:
    """The tool's rigid body: inertial and collision shapes, from its single-link URDF."""
    link = ET.parse(urdf_path).getroot().find("link")
    inertial = link.find("inertial")
    com, com_rpy = _origin(inertial)
    i = inertial.find("inertia")
    shapes = []
    for col in link.findall("collision"):
        xyz, rpy = _origin(col)
        geom = col.find("geometry")
        box, cyl = geom.find("box"), geom.find("cylinder")
        if box is not None:
            shapes.append({"kind": "box", "xyz": xyz, "rpy": rpy, "size": [float(v) for v in box.get("size").split()]})
        elif cyl is not None:
            shapes.append({"kind": "capsule", "xyz": xyz, "rpy": rpy,
                           "radius": float(cyl.get("radius")), "length": float(cyl.get("length"))})
        else:
            raise ValueError(f"{urdf_path}: unsupported collision geometry {[c.tag for c in geom]}")
    return {
        "mass": float(inertial.find("mass").get("value")), "com": com, "com_rpy": com_rpy,
        "inertia": [float(i.get("ixx")), float(i.get("iyy")), float(i.get("izz"))],
        "shapes": shapes,
    }


def _write_tool(layer, path: str, tool: dict, goal: bool) -> None:
    from pxr import Gf, Sdf

    def prim(p, type_name, schemas=None):
        spec = Sdf.CreatePrimInLayer(layer, p)
        spec.specifier = Sdf.SpecifierDef
        spec.typeName = type_name
        if schemas:
            spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas))
        return spec

    def attr(spec, name, tname, value, uniform=False):
        a = Sdf.AttributeSpec(spec, name, tname, Sdf.VariabilityUniform if uniform else Sdf.VariabilityVarying)
        a.default = value

    def xform(spec, xyz=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        attr(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*xyz))
        attr(spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd, Gf.Quatd(quat[0], Gf.Vec3d(*quat[1:])))
        attr(spec, "xformOp:scale", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*scale))
        attr(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray,
             ["xformOp:translate", "xformOp:orient", "xformOp:scale"], uniform=True)

    xform(prim(path, "Xform"))
    body = prim(f"{path}/object_root", "Xform", ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"])
    xform(body)
    attr(body, "physics:mass", Sdf.ValueTypeNames.Float, tool["mass"])
    attr(body, "physics:centerOfMass", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(*tool["com"]))
    attr(body, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*tool["inertia"]))
    q = _quat_rpy(tool["com_rpy"])
    attr(body, "physics:principalAxes", Sdf.ValueTypeNames.Quatf, Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    attr(body, "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, bool(goal))
    attr(body, "physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, bool(goal))
    if not goal:
        attr(body, "physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float, 1000.0)
    xform(prim(f"{path}/object_root/collisions", "Xform"))
    for k, s in enumerate(tool["shapes"]):
        holder = prim(f"{path}/object_root/collisions/mesh_{k}", "Xform")
        quat = _quat_rpy(s["rpy"])
        if s["kind"] == "box":
            xform(holder, s["xyz"], quat, s["size"])
            shape = prim(f"{path}/object_root/collisions/mesh_{k}/box", "Cube", ["PhysicsCollisionAPI", "PhysxCollisionAPI"])
            attr(shape, "size", Sdf.ValueTypeNames.Double, 1.0)
            h = [0.5 * v for v in s["size"]]
            attr(shape, "extent", Sdf.ValueTypeNames.Float3Array, [Gf.Vec3f(-h[0], -h[1], -h[2]), Gf.Vec3f(*h)])
        else:
            xform(holder, s["xyz"], quat)
            shape = prim(f"{path}/object_root/collisions/mesh_{k}/capsule", "Capsule", ["PhysicsCollisionAPI", "PhysxCollisionAPI"])
            r, hh = s["radius"], 0.5 * s["length"]
            attr(shape, "axis", Sdf.ValueTypeNames.Token, "Z", uniform=True)
            attr(shape, "radius", Sdf.ValueTypeNames.Double, r)
            attr(shape, "height", Sdf.ValueTypeNames.Double, s["length"])
            attr(shape, "extent", Sdf.ValueTypeNames.Float3Array, [Gf.Vec3f(-r, -r, -hh - r), Gf.Vec3f(r, r, hh + r)])
        # Render colours (a camera / video run; PhysX ignores them): the tool brown, the goal translucent green.
        attr(shape, "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray,
             [Gf.Vec3f(0.2, 0.8, 0.3) if goal else Gf.Vec3f(0.6, 0.35, 0.15)])
        if goal:
            attr(shape, "primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray, [0.35])
        attr(shape, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, CONTACT_OFFSET)
        attr(shape, "physxCollision:restOffset", Sdf.ValueTypeNames.Float, REST_OFFSET)
        if goal:
            attr(shape, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, False)


def author_tool_prims(stage, env_paths: list[str], urdf_paths: list[str], asset_index: list[int]) -> None:
    """``<env>/Object`` and ``<env>/GoalViz`` for every env: tool ``urdf_paths[asset_index[i]]``."""
    from pxr import Sdf

    layer = stage.GetRootLayer()
    tools = [parse_tool(p) for p in urdf_paths]
    proto_root = "/World/ToolPrototypes"
    with Sdf.ChangeBlock():
        for a, tool in enumerate(tools):
            _write_tool(layer, f"{proto_root}/tool_{a:04d}", tool, goal=False)
            _write_tool(layer, f"{proto_root}/goal_{a:04d}", tool, goal=True)
        for env_path, a in zip(env_paths, asset_index):
            for name, proto in (("Object", f"{proto_root}/tool_{a:04d}"), ("GoalViz", f"{proto_root}/goal_{a:04d}")):
                dst = f"{env_path}/{name}"
                Sdf.CreatePrimInLayer(layer, dst)
                Sdf.CopySpec(layer, Sdf.Path(proto), layer, Sdf.Path(dst))
        del layer.GetPrimAtPath(proto_root.rsplit("/", 1)[0]).nameChildren[proto_root.rsplit("/", 1)[1]]
