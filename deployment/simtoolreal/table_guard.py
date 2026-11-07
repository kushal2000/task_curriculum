"""Whole-hand table check: the lowest point of the wrist and hand meshes against the real table.

The real table has the training table's top height and the same near edge (the side closest to the
arm base), but is larger in x and y. So the guarded region is everything from the near edge away from
the base, out to --table_x_half / --table_far_y (env frame: the base is at +y, the table towards -y).
Below the table top outside that region (between the base and the near edge, beyond --table_edge_margin)
is not checked.

Points: each wrist / hand link's visual mesh, reduced to its convex hull vertices (a few hundred per
link), thinned to one per 4 mm cell, in the link frame. The link frames come from core.RobotFK's joint
tree, so they match the policy's FK.

    guard = TableGuard(spec, table_x_half=1.0, table_far_y=-1.0)
    dz, link, point = guard.lowest(q)   # dz: m above the table top (negative = inside the table)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from deployment.simtoolreal.core import RobotFK, Spec, quat_from_rpy

HERE = Path(__file__).resolve().parent
MESH_DIR = HERE.parents[1] / "assets" / "flexiv_wuji"
# the links checked: the forearm end, the wrist and the whole hand (the arm's upper links stay high)
ARM_LINKS = ("link5", "link6", "link7")


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _thin(v: np.ndarray, voxel: float) -> np.ndarray:
    """One point per voxel-sized cell (the cell's lowest, so the thinning never raises the hand): any
    hull point is then within ~voxel of a kept one."""
    keys = np.floor(v / voxel).astype(np.int64)
    order = np.lexsort((v[:, 2], keys[:, 2], keys[:, 1], keys[:, 0]))
    k = keys[order]
    first = np.ones(len(k), bool)
    first[1:] = np.any(k[1:] != k[:-1], axis=1)
    return v[order[first]]


class TableGuard:
    def __init__(self, spec: Spec, table_x_half: float = 1.0, table_far_y: float = -1.0,
                 edge_margin: float = 0.03, voxel: float = 0.004) -> None:
        import trimesh

        self.fk = RobotFK(spec)
        self.top = float(spec["table_top_z_env"])
        tp, ts = np.asarray(spec["table_pos_env"], float), np.asarray(spec["table_size"], float)
        self.near_y = float(tp[1] + ts[1] / 2)  # the edge closest to the base (base at +y)
        self.x_half, self.far_y = float(table_x_half), float(table_far_y)
        self.edge_margin = float(edge_margin)  # points this close outside the near edge count as over it
        # the real table, as a box for the viewer: same top and near edge, larger in x and y
        self.box_size = [2 * self.x_half, self.near_y - self.far_y, float(ts[2])]
        self.box_pos = [float(tp[0]), (self.near_y + self.far_y) / 2, self.top - float(ts[2]) / 2]
        # training's table: the policy learned to press on it; the rest of the real table it never felt
        self.train_x_half, self.train_far_y = float(ts[0] / 2), float(tp[1] - ts[1] / 2)

        root = ET.parse(spec.urdf_path).getroot()
        self.points: dict[str, np.ndarray] = {}
        for link in root.iter("link"):
            name = link.get("name")
            if not (name in ARM_LINKS or name.startswith(("r_", "l_"))):
                continue
            pts = []
            for vis in link.iter("visual"):
                mesh = vis.find("geometry/mesh")
                if mesh is None:
                    continue
                path = MESH_DIR / mesh.get("filename")
                if not path.is_file():
                    raise FileNotFoundError(f"{path} missing: run scripts/flexiv_wuji/fetch_assets.sh")
                m = trimesh.load(path, force="mesh")
                scale = mesh.get("scale")
                v = np.asarray(m.convex_hull.vertices, float)
                if scale:
                    v = v * np.array([float(s) for s in scale.split()])
                o = vis.find("origin")
                xyz = np.array([float(s) for s in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
                rpy = [float(s) for s in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
                pts.append(v @ _quat_to_mat(quat_from_rpy(rpy)).T + xyz)
            if pts:
                self.points[name] = _thin(np.concatenate(pts), voxel)
        if not self.points:
            raise RuntimeError("no hand meshes found in the URDF")
        self.links = list(self.points)
        roots = {j.parent for j in self.fk.joints.values()} - set(self.fk.joints)
        self.fk_root = roots.pop() if len(roots) == 1 else "base_link"
        self._build_tree()

    def _build_tree(self) -> None:
        """Joints from the root to every checked link, parents first, as 4x4 transforms."""
        fk, order, seen = self.fk, [], set()

        def visit(link: str) -> None:
            j = fk.joints.get(link)
            if j is None or link in seen:
                return
            visit(j.parent)
            seen.add(link)
            order.append(j)

        for link in self.links:
            visit(link)
        self._order = []
        for j in order:
            o = np.eye(4)
            o[:3, :3], o[:3, 3] = _quat_to_mat(j.quat), j.xyz
            kind = j.type if j.type in ("revolute", "continuous", "prismatic") else "fixed"
            qi = fk.index.get(j.name, -1) if kind != "fixed" else -1
            self._order.append((j.parent, j.child, o, kind, qi, j.axis))
        self._root = np.eye(4)
        self._root[:3, :3], self._root[:3, 3] = _quat_to_mat(fk.root_quat), fk.root_pos

    def _link_poses(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """4x4 env-frame transform of every checked link (and its ancestors)."""
        out = {self.fk_root: self._root}
        for parent, child, o, kind, qi, axis in self._order:
            t = out[parent] @ o
            if kind in ("revolute", "continuous"):
                a = q[qi]
                k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                m = np.eye(4)
                m[:3, :3] = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)
                t = t @ m
            elif kind == "prismatic":
                m = np.eye(4)
                m[:3, 3] = axis * q[qi]
                t = t @ m
            out[child] = t
        return out

    def lowest_by_region(self, q: np.ndarray) -> dict[str, tuple[float, str, np.ndarray]]:
        """{"train": ..., "extra": ...}: the lowest point over training's table, and over the rest of the
        real table, each as (height above the top in m, link, point); +inf when no point is there."""
        poses = self._link_poses(np.asarray(q, float))
        best = {k: (np.inf, "", np.full(3, np.nan)) for k in ("train", "extra")}
        for link in self.links:
            t = poses[link]
            w = self.points[link] @ t[:3, :3].T + t[:3, 3]
            over = (np.abs(w[:, 0]) <= self.x_half) & (w[:, 1] <= self.near_y + self.edge_margin) & (w[:, 1] >= self.far_y)
            if not over.any():
                continue
            w = w[over]
            train = (np.abs(w[:, 0]) <= self.train_x_half) & (w[:, 1] >= self.train_far_y)
            for key, m in (("train", train), ("extra", ~train)):
                if m.any():
                    i = int(np.argmin(np.where(m, w[:, 2], np.inf)))
                    if w[i, 2] - self.top < best[key][0]:
                        best[key] = (float(w[i, 2] - self.top), link, w[i])
        return best

    def lowest(self, q: np.ndarray) -> tuple[float, str, np.ndarray]:
        """(height above the table top in m, link, point) of the lowest wrist/hand point over the table.

        Points not over the table (between the base and the near edge, or past the far side) count
        as +inf. Height is negative when the point is inside the table."""
        poses = self._link_poses(np.asarray(q, float))
        best = (np.inf, "", np.full(3, np.nan))
        for link in self.links:
            t = poses[link]
            w = self.points[link] @ t[:3, :3].T + t[:3, 3]
            over = (np.abs(w[:, 0]) <= self.x_half) & (w[:, 1] <= self.near_y + self.edge_margin) & (w[:, 1] >= self.far_y)
            if not over.any():
                continue
            w = w[over]
            i = int(np.argmin(w[:, 2]))
            if w[i, 2] - self.top < best[0]:
                best = (float(w[i, 2] - self.top), link, w[i])
        return best
