"""Is a finger THROUGH the VBD cloth? Geometry shared by the pierce probe and the live viewer.

The cloth feels rigid bodies only through its particles (Newton's VBD has particle-vs-shape contact
and nothing else), so a finger can end up on both sides of the sheet. A **pierce** is the axis of a
finger link (distal or middle phalanx; principal axis of its collision mesh, end to end) crossing one
of its own env's cloth triangles. Ordinary contact keeps the finger surface a particle radius off the
mid-surface and the axis is another ~7 mm inside the finger, so a crossing is never a grazing touch.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["FINGER_RE", "PierceDetector", "finger_re", "finger_segments", "quat_rotate_xyzw", "segment_hits"]


def finger_re(robot_spec) -> re.Pattern:
    """The watched finger links (distal and middle phalanges) of ``robot_spec``, by body label."""
    return re.compile(r"/Robot/" + robot_spec.finger_links_regex("distal", "middle") + "$")


FINGER_RE = re.compile(r"/Robot/left_(thumb|index|middle|ring|pinky)_(DP|MP)$")  # the Kuka + Sharpa's


def finger_segments(model, pattern: re.Pattern = FINGER_RE):
    """Per finger body: (body index, world, label, segment endpoints in the BODY frame, radius)."""
    import warp as wp  # noqa: F401

    labels = [str(x) for x in model.body_label]
    shape_body = model.shape_body.numpy()
    shape_tf = model.shape_transform.numpy()
    shape_scale = model.shape_scale.numpy()
    src = model.shape_source
    out = []
    for b, lab in enumerate(labels):
        if not pattern.search(lab):
            continue
        pts = []
        for s in np.nonzero(shape_body == b)[0]:
            mesh = src[s]
            if mesh is None or not hasattr(mesh, "vertices"):
                continue
            v = np.asarray(mesh.vertices, dtype=np.float64) * shape_scale[s]
            p, q = shape_tf[s][:3], shape_tf[s][3:]  # q is xyzw
            x, y, z, w = q
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            pts.append(v @ R.T + p)
        if not pts:
            continue
        pts = np.concatenate(pts)
        c = pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
        axis = vt[0]
        t = (pts - c) @ axis
        radial = np.linalg.norm((pts - c) - np.outer(t, axis), axis=1)
        out.append((b, int(model.body_world.numpy()[b]), lab.rsplit("/", 1)[-1],
                    np.stack([c + t.min() * axis, c + t.max() * axis]), float(np.median(radial))))
    return out


def quat_rotate_xyzw(q, v):
    """Rotate v (..., 3) by unit quaternions q (..., 4) in xyzw."""
    import torch

    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def segment_hits(a, b, tri):
    """Moller-Trumbore. a, b: (E, F, 3) segment ends; tri: (E, T, 3, 3). Returns (E, F) bool."""
    import torch

    d = (b - a)[:, :, None, :]                       # (E, F, 1, 3)
    v0, v1, v2 = tri[:, None, :, 0], tri[:, None, :, 1], tri[:, None, :, 2]  # (E, 1, T, 3)
    e1, e2 = v1 - v0, v2 - v0
    h = torch.cross(d.expand(-1, -1, e2.shape[2], -1), e2.expand(-1, d.shape[1], -1, -1), dim=-1)
    det = (e1 * h).sum(-1)
    ok = det.abs() > 1e-12
    inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
    s = a[:, :, None, :] - v0
    u = (s * h).sum(-1) * inv
    qv = torch.cross(s, e1.expand_as(s), dim=-1)
    v = (d * qv).sum(-1) * inv
    t = (e2 * qv).sum(-1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
    return hit.any(dim=-1)


class PierceDetector:
    """Per env and finger link, whether that link's axis crosses the env's cloth this step."""

    def __init__(self, model, num_envs: int, device, robot_spec=None):
        import torch

        fingers = finger_segments(model, FINGER_RE if robot_spec is None else finger_re(robot_spec))
        self.names = sorted({f[2] for f in fingers})
        bidx = np.full((num_envs, len(self.names)), -1, dtype=np.int64)
        seg = np.zeros((len(self.names), 2, 3))
        self.radius = {}
        for b, w, name, s, r in fingers:
            bidx[w, self.names.index(name)] = b
            seg[self.names.index(name)] = s
            self.radius[name] = r
        if (bidx < 0).any():
            raise RuntimeError("a finger body is missing in some world")
        owner = model.particle_world.numpy()
        tri = model.tri_indices.numpy()
        per = [tri[owner[tri[:, 0]] == w] for w in range(num_envs)]
        if len({len(t) for t in per}) != 1:
            raise RuntimeError("worlds hold different cloth meshes")
        self.n = num_envs
        self.bidx = torch.as_tensor(bidx, device=device)
        self.seg = torch.as_tensor(seg, device=device, dtype=torch.float32)
        self.seg_np = seg
        self.tri = torch.as_tensor(np.stack(per), device=device, dtype=torch.long)

    def axes(self, state):
        """World-frame finger axis ends ``(a, b)``, each ``(E, F, 3)``."""
        import warp as wp

        bq = wp.to_torch(state.body_q)[self.bidx]
        pos, quat = bq[..., :3], bq[..., 3:]
        a = pos + quat_rotate_xyzw(quat, self.seg[None, :, 0].expand(self.n, -1, -1))
        b = pos + quat_rotate_xyzw(quat, self.seg[None, :, 1].expand(self.n, -1, -1))
        return a, b

    def __call__(self, state):
        """``(hits (E, F) bool, a, b)``."""
        import warp as wp

        a, b = self.axes(state)
        return segment_hits(a, b, wp.to_torch(state.particle_q)[self.tri]), a, b
