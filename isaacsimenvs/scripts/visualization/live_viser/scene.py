"""What to draw for each backend: the bodies' geometry (once per drawn env) and poses (every frame).

Both classes give the same two calls:
    shapes(w)        -> [Shape]: every drawable shape of env ``w``
    poses(w, keys)   -> {key: (position, wxyz)}, world frame, for the bodies ``keys`` of env ``w``
What is drawn is the collision geometry, the shapes the solver sees.
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass

import numpy as np

ROBOT_RGB = (205, 208, 214)
TABLE_RGB = (96, 112, 138)


@dataclass
class Shape:
    key: object | None   # the body whose pose moves it (``poses`` key); None: static, world frame
    label: str
    verts: np.ndarray    # (V, 3) in the body frame (static: the world frame)
    faces: np.ndarray    # (F, 3)
    pos: np.ndarray      # the shape's offset in that frame
    wxyz: np.ndarray
    kind: str            # robot | table | object (the manipuland: translucent under a student)
    rgb: tuple


def _object_hues(names) -> dict[str, tuple]:
    """A hue sweep over the manipuland's bodies, so neighbouring plates are told apart."""
    names = sorted(set(names))
    out = {}
    for i, n in enumerate(names):
        r, g, b = colorsys.hsv_to_rgb(0.02 + 0.75 * i / max(len(names), 1), 0.8, 0.95)
        out[n] = (int(r * 255), int(g * 255), int(b * 255))
    return out


class NewtonScene:
    """The VBD task on Newton: shapes from Newton's own viewer pipeline (``ViewerBase.set_model``,
    captured by a null viewer), poses from ``body_q``. The sheet itself is ``vbd.ClothSheet``."""

    def __init__(self, inner) -> None:
        import newton
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from newton.viewer import ViewerNull

        self._manager = NewtonManager
        self.device = inner.device
        model = NewtonManager.get_model()

        class Capture(ViewerNull):
            def __init__(self):
                super().__init__(num_frames=1)
                self.meshes = {}

            def log_mesh(self, name, points, indices, *args, **kwargs):
                pts = points.numpy() if hasattr(points, "numpy") else np.asarray(points)
                idx = indices.numpy() if hasattr(indices, "numpy") else np.asarray(indices)
                self.meshes[name] = (np.asarray(pts, np.float32).reshape(-1, 3), np.asarray(idx, np.int32).reshape(-1, 3))

        v = Capture()
        v.show_collision = True  # the robot's shapes are collision-only: hidden otherwise
        v.show_static = True
        v.set_model(model)

        labels = [str(x) for x in model.body_label]
        self._all = []  # (world, Shape)
        for batch in v._shape_instances.values():
            if not v._should_show_shape(batch.flags, batch.static, batch.geo_type):
                continue
            if int(batch.geo_type) == int(newton.GeoType.PLANE) or batch.mesh not in v.meshes:
                continue
            verts, faces = v.meshes[batch.mesh]
            for parent, xf, sc, world in zip(batch.parents.numpy(), batch.xforms.numpy(),
                                             batch.scales.numpy(), batch.worlds.numpy()):
                label = labels[parent] if parent >= 0 else ""
                if "GoalViz" in label:
                    continue
                robot = "Robot" in label
                self._all.append((int(world), Shape(
                    key=int(parent) if parent >= 0 else None, label=label, verts=verts * sc[None, :], faces=faces,
                    pos=xf[:3].astype(np.float64), wxyz=np.array([xf[6], xf[3], xf[4], xf[5]]),
                    kind="robot" if robot else "table", rgb=ROBOT_RGB if robot else TABLE_RGB)))

    def shapes(self, w: int) -> list[Shape]:
        return [s for world, s in self._all if world in (-1, w)]

    def poses(self, w: int, keys) -> dict:
        import torch
        import warp as wp

        q = wp.to_torch(self._manager._state_0.body_q)[torch.as_tensor(list(keys), device=self.device)].cpu().numpy()
        return {k: (x[:3], (x[6], x[3], x[4], x[5])) for k, x in zip(keys, q)}  # xyzw -> wxyz


def _prim_tris(prim):
    """(vertices, triangles) of a UsdGeom mesh or implicit shape in the prim's own frame; None if the
    prim is not drawable geometry."""
    import trimesh
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts, counts, idx = m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
        if pts is None or counts is None or idx is None or len(pts) == 0:
            return None
        idx = np.asarray(idx, np.int64)
        tris, o = [], 0
        for c in counts:  # fan-triangulate each polygon
            tris += [(idx[o], idx[o + k], idx[o + k + 1]) for k in range(1, int(c) - 1)]
            o += int(c)
        return np.asarray(pts, np.float64), np.asarray(tris, np.int32).reshape(-1, 3)
    if prim.IsA(UsdGeom.Cube):
        e = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0)
        tm = trimesh.creation.box(extents=(e, e, e))
    elif prim.IsA(UsdGeom.Sphere):
        tm = trimesh.creation.icosphere(subdivisions=2, radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
    elif prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cylinder):
        capsule = prim.IsA(UsdGeom.Capsule)
        g = UsdGeom.Capsule(prim) if capsule else UsdGeom.Cylinder(prim)
        r, h = float(g.GetRadiusAttr().Get()), float(g.GetHeightAttr().Get())
        tm = trimesh.creation.capsule(height=h, radius=r) if capsule else trimesh.creation.cylinder(radius=r, height=h)
        axis = str(g.GetAxisAttr().Get() or "Z")
        if axis == "X":
            tm.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
        elif axis == "Y":
            tm.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
    else:
        return None
    return np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int32)


class PhysxScene:
    """The rigid sheet on PhysX: shapes from the USD stage (under each Isaac Lab asset's prim in env
    ``w``, every collision prim in the frame of its nearest rigid body), poses from the assets' data.

    An asset's instance index is read from its PhysX view's prim paths, so an articulation that only
    some envs carry still maps right."""

    SKIP = ("object", "goal_viz")

    def __init__(self, inner) -> None:
        self.assets = []  # (asset, root prim name, {env: instance}, body names)
        reg = {**dict(inner.scene.articulations), **dict(inner.scene.rigid_objects)}
        for name, asset in reg.items():
            if name in self.SKIP:
                continue
            inst = {}
            for i, path in enumerate(asset.root_physx_view.prim_paths):
                m = re.search(r"/env_(\d+)/", path)
                if m:
                    inst[int(m.group(1))] = i
            self.assets.append((asset, asset.cfg.prim_path.rsplit("/", 1)[-1], inst, list(asset.body_names)))
        self._cache: dict[int, list[Shape]] = {}
        self._hues = _object_hues(s.label.split("/", 1)[-1] for s in self._read(0) if s.kind == "object")

    def shapes(self, w: int) -> list[Shape]:
        if w not in self._cache:
            shapes = self._read(w)
            for s in shapes:
                if s.kind == "object":
                    s.rgb = self._hues.get(s.label.split("/", 1)[-1], TABLE_RGB)
            self._cache[w] = shapes
        return self._cache[w]

    def _read(self, w: int) -> list[Shape]:
        from isaaclab.sim.utils import get_current_stage
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = get_current_stage()
        xc = UsdGeom.XformCache(Usd.TimeCode.Default())
        out, missing = [], set()
        for k, (_asset, root, inst, bodies) in enumerate(self.assets):
            top = stage.GetPrimAtPath(f"/World/envs/env_{w}/{root}")
            if w not in inst or not top.IsValid():
                continue
            kind = "robot" if root.startswith("Robot") else "table" if root.startswith("Table") else "object"
            found = {"col": [], "vis": []}
            for prim in Usd.PrimRange(top, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                body, collider, p = None, False, prim
                while p.IsValid() and p != top.GetParent():  # up to the nearest rigid body
                    if p.HasAPI(UsdPhysics.CollisionAPI):
                        en = p.GetAttribute("physics:collisionEnabled")
                        collider = collider or not (en and en.HasAuthoredValue() and not en.Get())
                    if p.HasAPI(UsdPhysics.RigidBodyAPI):
                        body = p
                        break
                    p = p.GetParent()
                if body is None:
                    continue
                if body.GetName() not in bodies:
                    missing.add(body.GetName())
                    continue
                geo = _prim_tris(prim)
                if geo is None:
                    continue
                v, f = geo
                rel = np.array(xc.GetLocalToWorldTransform(prim) * xc.GetLocalToWorldTransform(body).GetInverse())
                found["col" if collider else "vis"].append(Shape(
                    key=(k, bodies.index(body.GetName())), label=f"{root}/{body.GetName()}",
                    verts=(np.c_[v, np.ones(len(v))] @ rel)[:, :3].astype(np.float32), faces=f,
                    pos=np.zeros(3), wxyz=np.array([1.0, 0, 0, 0]), kind=kind,
                    rgb=ROBOT_RGB if kind == "robot" else TABLE_RGB))
            out += found["col"] or found["vis"]  # an asset without colliders still shows its visuals
        if missing:
            print(f"[live] PhysX bodies with no tensor index (not drawn): {sorted(missing)[:8]}", flush=True)
        return out

    def poses(self, w: int, keys) -> dict:
        by_asset: dict[int, list[int]] = {}
        for k, b in keys:
            by_asset.setdefault(k, []).append(b)
        out = {}
        for k, links in by_asset.items():
            asset, _root, inst, _bodies = self.assets[k]
            i = inst[w]
            pos = asset.data.body_pos_w[i, links].detach().cpu().numpy()
            quat = asset.data.body_quat_w[i, links].detach().cpu().numpy()  # wxyz (Isaac Lab 2.3)
            for b, p, q in zip(links, pos, quat):
                out[(k, b)] = (p, q)
        return out
