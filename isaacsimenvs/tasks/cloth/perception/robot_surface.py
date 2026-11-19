"""Surface samples of a robot's link meshes, for ``pointcloud.PointCloud``.

``bake(urdf)`` samples the visual meshes of every link (uniform over each mesh's area) and writes
them next to the URDF as ``<urdf stem>.surface.npz``, in the links' own frames. ``load(urdf,
body_names)`` reads that file (baking it first if it is missing) and returns the samples in the
frames of the simulator's bodies: the URDF importers merge a fixed-joint child into its parent, so
a link that is not a body is carried up its fixed joints to the first ancestor that is.

    python -m isaacsimenvs.tasks.cloth.perception.robot_surface <robot.urdf>      # (re)bake

numpy + trimesh only (no torch, no Isaac Lab). The file is baked once and shared, so the PhysX and
Newton environments (different venvs, different trimesh versions) use the same samples.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

__all__ = ["bake", "load", "fk_subset"]

#: Links that never reach the table (the base and the shoulder).
SKIP_LINKS = ("base_link", "link1", "link2")


def _rpy(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _origin(node) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` of an ``<origin>`` child (identity if absent)."""
    o = None if node is None else node.find("origin")
    if o is None:
        return np.eye(3), np.zeros(3)
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return _rpy(o.get("rpy", "0 0 0").split()), np.array(xyz)


def bake(urdf: str | Path, n_points: int = 16384, hand_boost: float = 2.0, seed: int = 0) -> Path:
    """Sample the links' visual meshes and write ``<urdf stem>.surface.npz``.

    Every link gets points in proportion to its area (hand links ``hand_boost`` times denser: thin
    fingers need more samples to occlude); ``weight`` is the area [m^2] each point stands for, so a
    draw proportional to it is uniform over the surface whatever the density.
    """
    import trimesh

    urdf = Path(urdf)
    root = ET.parse(urdf).getroot()
    meshes: dict[str, list] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name in SKIP_LINKS:
            continue
        for vis in link.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is None:
                continue
            mesh = trimesh.load(urdf.parent / m.get("filename"), force="mesh", process=False)
            scale = np.array([float(v) for v in m.get("scale", "1 1 1").split()])
            rot, t = _origin(vis)
            verts = (np.asarray(mesh.vertices) * scale) @ rot.T + t
            meshes.setdefault(name, []).append(trimesh.Trimesh(verts, np.asarray(mesh.faces), process=False))
    if not meshes:
        raise ValueError(f"{urdf} has no visual meshes")
    names = sorted(meshes)
    merged = {k: trimesh.util.concatenate(meshes[k]) for k in names}
    area = np.array([float(merged[k].area) for k in names])
    is_hand = np.array([not k.startswith(("link", "flange", "base")) for k in names])
    share = area * np.where(is_hand, hand_boost, 1.0)
    counts = np.maximum(1, np.round(share / share.sum() * n_points).astype(int))
    pts, link_id, weight = [], [], []
    for i, k in enumerate(names):
        p, _ = trimesh.sample.sample_surface(merged[k], int(counts[i]), seed=seed + i)
        pts.append(np.asarray(p))
        link_id.append(np.full(len(p), i))
        weight.append(np.full(len(p), area[i] / len(p)))
    # A link's fixed joint to its parent: {child: [parent, R (9), t (3)]}.
    fixed = {}
    for j in root.findall("joint"):
        if j.get("type") == "fixed":
            rot, t = _origin(j)
            fixed[j.find("child").get("link")] = [j.find("parent").get("link"), rot.reshape(-1).tolist(), t.tolist()]
    out = urdf.with_suffix(".surface.npz")
    np.savez_compressed(
        out, points=np.concatenate(pts).astype(np.float32), link_id=np.concatenate(link_id).astype(np.int64),
        weight=np.concatenate(weight).astype(np.float32), links=np.array(names), fixed=json.dumps(fixed),
    )
    return out


def load(urdf: str | Path, body_names: list[str]) -> dict:
    """``{body (P,), offset (P, 3), weight (P,), link (P,), links [names]}`` in the simulator's
    body frames (``body`` indexes ``body_names``)."""
    path = Path(urdf).with_suffix(".surface.npz")
    if not path.exists():
        bake(urdf)
    data = np.load(path, allow_pickle=False)
    links = [str(v) for v in data["links"]]
    fixed = json.loads(str(data["fixed"]))
    body = np.zeros(len(links), dtype=np.int64)
    rots, ts = [], []
    for i, name in enumerate(links):
        rot, t, at = np.eye(3), np.zeros(3), name
        while at not in body_names:
            if at not in fixed:
                raise ValueError(f"link {name}: {at} is not a body and has no fixed joint to one (bodies: {body_names})")
            parent, r, tt = fixed[at]
            r = np.asarray(r).reshape(3, 3)
            rot, t, at = r @ rot, r @ t + np.asarray(tt), parent
        body[i] = body_names.index(at)
        rots.append(rot)
        ts.append(t)
    lid = data["link_id"]
    rot, t = np.stack(rots)[lid], np.stack(ts)[lid]
    offset = np.einsum("pij,pj->pi", rot, data["points"].astype(np.float64)) + t
    return {"body": body[lid], "offset": offset.astype(np.float32), "weight": data["weight"], "link": lid, "links": links}


def fk_subset(surface: dict, count: int, seed: int = 0) -> np.ndarray:
    """``count`` pool indices on the forearm and hand (link5 on), uniform over their area."""
    keep = np.array([not k.startswith(("link3", "link4")) for k in surface["links"]])[surface["link"]]
    idx = np.nonzero(keep)[0]
    w = surface["weight"][idx].astype(np.float64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=int(count), replace=False, p=w / w.sum()))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        out = bake(arg)
        d = np.load(out)
        per = {str(k): int((d["link_id"] == i).sum()) for i, k in enumerate(d["links"])}
        print(f"{out}: {len(d['points'])} points, area {float(d['weight'].sum()):.3f} m^2\n  {per}")
