"""Write a ``fold_shapes.FoldShape`` as a USD articulation, directly -- no URDF, no Kit, no bake.

The stage-1 chains go URDF -> Isaac Sim's converter -> a content-addressed USD cache, because their
URDF importer is Kit's. The pool cannot take that road: URDF has no way to say "this joint closes a
loop", and three of the five shapes need exactly that (``box2x8``, ``cross4``, ``corners8``). USD
does -- ``physics:excludeFromArticulation`` on a joint -- and Newton's USD importer turns such a
revolute joint into a loop joint that ``SolverMuJoCo`` simulates as two ``CONNECT`` equality rows.
So the pool authors USD itself, with plain ``pxr`` (``usd-core``), which the kit-less Isaac Lab 3
venv already has. The file is a pure function of the shape, so it is written fresh per run.

Layout mirrors what the URDF converter produces for the stage-1 chains, so Isaac Lab and the env's
indexing see a familiar articulation::

    /<root>                          Xform (default prim)
        slat_<plate>                 RigidBodyAPI + MassAPI (+ ArticulationRootAPI on the root plate)
            collision                Mesh, CollisionAPI, MeshCollisionAPI approximation=convexHull
        joints/slat_joint_<child>    RevoluteJoint, axis X, DriveAPI:angular (gains set by Isaac Lab)
        joints/seam_<k>              RevoluteJoint, excludeFromArticulation = true, no drive

Every plate's body frame sits at the SHEET origin (identity), and its mesh is the plate's polygon in
sheet coordinates -- the convention ``fold_shapes`` uses throughout -- so the articulation's root pose
IS the sheet's pose and no root offset has to be carried. Joints are named for their child plate;
the ``slat_`` prefixes are the stage-1 ones, so ``RigidClothEnv``'s prefix-based model indexing picks
the pool's bodies and hinges up unchanged, and the ``seam_`` prefix keeps loop joints out of it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

__all__ = ["BODY_PREFIX", "JOINT_PREFIX", "SEAM_PREFIX", "root_prim_name", "write_shape_usd"]

BODY_PREFIX = "slat_"
JOINT_PREFIX = "slat_joint_"
SEAM_PREFIX = "seam_"


def root_prim_name(shape) -> str:
    return "fold_" + shape.name.replace("-", "_")


def _quat_z(angle: float):
    """Gf.Quatf rotating +x onto the in-plane direction at ``angle`` about +z."""
    from pxr import Gf

    return Gf.Quatf(math.cos(0.5 * angle), Gf.Vec3f(0.0, 0.0, math.sin(0.5 * angle)))


def _principal(inertia: np.ndarray):
    """``(diagonal, Gf.Quatf)`` -- USD stores inertia as principal moments plus their frame."""
    from pxr import Gf

    w, v = np.linalg.eigh(inertia)
    if np.linalg.det(v) < 0.0:
        v[:, 0] = -v[:, 0]
    # Rotation matrix -> quaternion (columns of v are the principal axes in the body frame).
    r = v
    tr = np.trace(r)
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2.0
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    return Gf.Vec3f(*[float(x) for x in w]), Gf.Quatf(
        float(q[0]), Gf.Vec3f(*[float(x) for x in q[1:]])
    )


def _prism(stage, path: str, polygon, thickness: float, color):
    """A closed slab mesh: the polygon at +-thickness/2, outward-facing."""
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    n = len(polygon)
    h = 0.5 * thickness
    pts = [Gf.Vec3f(x, y, -h) for x, y in polygon] + [Gf.Vec3f(x, y, h) for x, y in polygon]
    counts = [n, n] + [4] * n
    idx = list(reversed(range(n))) + [n + k for k in range(n)]  # bottom faces -z, top faces +z
    for k in range(n):
        j = (k + 1) % n
        idx += [k, j, n + j, n + k]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
        UsdPhysics.Tokens.convexHull
    )
    return mesh


def _revolute(stage, path: str, body0: str, body1: str, a, b, z: float, limits=None):
    """Revolute joint about the sheet line ``a -> b`` at height ``z``, joint axis = joint-frame X.

    Both bodies' frames coincide with the sheet frame at rest, so ``localPos0 == localPos1`` is the
    hinge point in sheet coordinates and both local rotations turn joint-X onto the hinge line.
    """
    from pxr import Gf, UsdPhysics

    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateAxisAttr(UsdPhysics.Tokens.x)
    rot = _quat_z(math.atan2(b[1] - a[1], b[0] - a[0]))
    pos = Gf.Vec3f(float(a[0]), float(a[1]), float(z))
    j.CreateLocalPos0Attr(pos)
    j.CreateLocalPos1Attr(pos)
    j.CreateLocalRot0Attr(rot)
    j.CreateLocalRot1Attr(rot)
    if limits is not None:
        j.CreateLowerLimitAttr(float(math.degrees(limits[0])))
        j.CreateUpperLimitAttr(float(math.degrees(limits[1])))
    return j


def write_shape_usd(shape, out_path: str | Path, *, density: float = 2.0) -> Path:
    """Write ``shape`` as a floating-base USD articulation at ``out_path``.

    ``density`` is per AREA [kg/m^2], as ``ClothCfg.density`` -- every pool sheet weighs 20 g.

    Hinge axes are written as the sheet line FROM ``Hinge.a`` TO ``Hinge.b``, which is not
    necessarily the fold_shapes axis (oriented so +angle folds UP). Where the two disagree the line is
    reversed here, so a positive joint angle in the simulator means the same as in ``fold_shapes``.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root = f"/{root_prim_name(shape)}"
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, root).GetPrim())
    palette = [(0.29, 0.56, 0.76), (0.31, 0.67, 0.51), (0.95, 0.68, 0.22), (0.59, 0.43, 0.78)]
    for i, plate in enumerate(shape.plates):
        path = f"{root}/{BODY_PREFIX}{plate.name}"
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        m, com, inertia = _mass(shape, plate, density)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(m))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*[float(x) for x in com]))
        diag, axes = _principal(inertia)
        mass.CreateDiagonalInertiaAttr(diag)
        mass.CreatePrincipalAxesAttr(axes)
        if plate.name == shape.root:
            UsdPhysics.ArticulationRootAPI.Apply(prim)
        color = (0.85, 0.35, 0.55) if plate.name == shape.root else palette[(i - 1) % len(palette)]
        _prism(stage, f"{path}/collision", plate.polygon, shape.thickness, color)

    UsdGeom.Scope.Define(stage, f"{root}/joints")
    lo, hi = shape.joint_range()
    for h in shape.hinges:
        _p, u = shape.axis(h.child)
        # Line direction a->b vs the fold-up axis u: reverse the line where they disagree.
        d = np.array([h.b[0] - h.a[0], h.b[1] - h.a[1]])
        a, b = (h.a, h.b) if float(d @ u[:2]) > 0.0 else (h.b, h.a)
        j = _revolute(
            stage,
            f"{root}/joints/{JOINT_PREFIX}{h.child}",
            f"{root}/{BODY_PREFIX}{h.parent}",
            f"{root}/{BODY_PREFIX}{h.child}",
            a,
            b,
            shape.hinge_z,
            limits=(lo, hi),
        )
        # Gains are Isaac Lab's (ImplicitActuatorCfg); the drive only has to EXIST so Newton builds an
        # actuator for the joint. Zero here, overwritten at spawn.
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
    for k, s in enumerate(shape.seams):
        j = _revolute(
            stage,
            f"{root}/joints/{SEAM_PREFIX}{k}",
            f"{root}/{BODY_PREFIX}{s.plate1}",
            f"{root}/{BODY_PREFIX}{s.plate2}",
            s.a,
            s.b,
            shape.hinge_z,
        )
        j.CreateExcludeFromArticulationAttr(True)
    stage.GetRootLayer().Save()
    return out_path


def _mass(shape, plate, density):
    """``fold_shapes.plate_mass_properties`` from whichever module defined ``shape``.

    Looked up rather than imported: the env imports ``fold_shapes`` as a package member and the
    tests load it by path (so the package ``__init__`` and its Kit imports never run), and the two
    are different module objects. Either way it is the module the shape came from.
    """
    import sys

    return sys.modules[type(shape).__module__].plate_mass_properties(
        plate, shape.thickness, density
    )


def scale_builder_world(
    builder, body_ids, joint_ids, shape_ids, s_xy: float, s_z: float = 1.0
) -> None:
    """Resize one world's copy of a pool shape IN A NEWTON BUILDER: ``s_xy`` in-plane, ``s_z`` in
    thickness, about the sheet origin. Called from a per-world replication hook, before finalize.

    Why here and not at reset, like the rest of the physics DR: Newton does not re-read a convex
    mesh's ``shape_scale`` after finalize (measured: a 3x thicker plate rested at exactly the same
    height), and even at build time ``shape_scale`` alone leaves mass, inertia and hinge positions at
    the 10 cm values -- a 2x plate then rested in a 2 deg V with corners 2 mm through the table. So
    every quantity that carries a length is rewritten together:

      * collision/visual meshes -- replaced by meshes with scaled VERTICES (scale stays 1), so
        nothing downstream has to honour a scale field;
      * mass ``x s_xy^2`` (density is per AREA), centre of mass, inertia ``x s_xy^4`` in-plane
        (the slab's own ``m t^2 / 12`` term is negligible at these thicknesses);
      * every joint's parent/child anchor -- hinges AND seam loop joints -- so the axes stay on
        the plate edges they belong to.

    Plate frames all sit at the sheet origin (see the module docstring), so scaling about each
    body's own origin IS scaling about the sheet's.
    """
    import newton
    import warp as wp

    S = np.array([s_xy, s_xy, s_z], dtype=float)
    for i in shape_ids:
        src = builder.shape_source[i]
        if src is None or not hasattr(src, "vertices"):
            continue
        builder.shape_source[i] = newton.Mesh(
            np.asarray(src.vertices) * S,
            np.asarray(src.indices),
            compute_inertia=False,
            is_solid=True,
        )
        builder.shape_scale[i] = (1.0, 1.0, 1.0)
    m2 = s_xy * s_xy
    for b in body_ids:
        mass = float(builder.body_mass[b]) * m2
        builder.body_mass[b] = mass
        builder.body_inv_mass[b] = 1.0 / mass if mass > 0.0 else 0.0
        builder.body_com[b] = wp.vec3(*(np.array(builder.body_com[b], dtype=float) * S))
        inertia = np.array(builder.body_inertia[b], dtype=float).reshape(3, 3) * (m2 * s_xy * s_xy)
        builder.body_inertia[b] = wp.mat33(*inertia.ravel())
        builder.body_inv_inertia[b] = wp.mat33(*np.linalg.inv(inertia).ravel())
    for j in joint_ids:
        for attr in ("joint_X_p", "joint_X_c"):
            x = getattr(builder, attr)[j]
            p = np.array(x[:3], dtype=float) * S
            getattr(builder, attr)[j] = wp.transform(
                wp.vec3(*p), wp.quat(*np.array(x[3:7], dtype=float))
            )
