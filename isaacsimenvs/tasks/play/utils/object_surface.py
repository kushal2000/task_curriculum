"""The Play tools' surfaces as point clouds and signed distance functions, and their symmetries.

A procedural tool (``generate_objects.py``) is one link of 1-2 collision primitives: boxes, and
cylinders the URDF importer turns into capsules (``replace_cylinders_with_capsules``; the capsule's
cylinder part is the URDF ``length``, its caps the URDF ``radius``, axis local z). This module reads
those primitives and gives, in the object's root-link frame:

* ``sample_surface``  -- points uniform by area over the tool's OUTER surface: every face of every
  primitive, minus the parts inside another primitive (the handle-head junction);
* ``sdf``             -- the exact signed distance to the tool (union of its primitives: the min),
  batched over envs whose tools differ (``ShapeBank``);
* ``symmetries``      -- the rotations that map the tool onto itself (checked numerically on the
  surface samples against the SDF), for canonicalising a keypoint teacher's goal.

The pose distance the point-cloud task scores (``surface_distance``) is two-sided and exact:

    d = max( max_k |sdf_cur(goal_pts_k)|, max_k |sdf_goal(cur_pts_k)| )

0 exactly when the two surfaces coincide, so it is invariant to every symmetry of the tool and has no
sampling floor (a point-to-point Chamfer distance has one at the sample spacing, ~1 cm here).

numpy for the per-asset preprocessing, torch for the per-step batched parts. No simulator.
"""

from __future__ import annotations

import itertools
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "Primitive", "parse_primitives", "sample_surface", "surface_area", "symmetry_rotations",
    "ShapeBank", "quat_from_matrix", "matrix_from_quat",
]

BOX, CAPSULE = 0, 1
MAX_PRIMS = 2
"""Primitives per tool: handle + head (generate_objects.py)."""


@dataclass
class Primitive:
    kind: int                 # BOX | CAPSULE
    rot: np.ndarray           # (3, 3) primitive frame -> link frame
    pos: np.ndarray           # (3,)
    half: np.ndarray          # (3,) box half extents; capsule: (radius, radius, half cylinder length)

    @property
    def radius(self) -> float:
        return float(self.half[0])

    @property
    def half_len(self) -> float:
        return float(self.half[2])


def _rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def parse_primitives(urdf_path: str | Path) -> list[Primitive]:
    """The collision primitives of a single-link procedural tool, in its link frame."""
    root = ET.parse(urdf_path).getroot()
    links = root.findall("link")
    if len(links) != 1:
        raise ValueError(f"{urdf_path}: expected a single-link object, got {len(links)} links")
    prims = []
    for col in links[0].findall("collision"):
        origin = col.find("origin")
        xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()]) if origin is not None else np.zeros(3)
        rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
        geom = col.find("geometry")
        box, cyl = geom.find("box"), geom.find("cylinder")
        if box is not None:
            half = np.array([float(v) for v in box.get("size").split()]) / 2.0
            prims.append(Primitive(BOX, _rpy_matrix(rpy), xyz, half))
        elif cyl is not None:
            r, length = float(cyl.get("radius")), float(cyl.get("length"))
            prims.append(Primitive(CAPSULE, _rpy_matrix(rpy), xyz, np.array([r, r, length / 2.0])))
        else:
            raise ValueError(f"{urdf_path}: unsupported collision geometry {[c.tag for c in geom]}")
    if not 1 <= len(prims) <= MAX_PRIMS:
        raise ValueError(f"{urdf_path}: expected 1-{MAX_PRIMS} collision primitives, got {len(prims)}")
    return prims


# ----------------------------------------------------------------------------
# numpy: per-primitive SDF and surface samples (asset preprocessing)
# ----------------------------------------------------------------------------


def _sdf_prim_np(p: Primitive, x: np.ndarray) -> np.ndarray:
    local = (x - p.pos) @ p.rot                                  # R^T (x - t), row vectors
    if p.kind == BOX:
        q = np.abs(local) - p.half
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)
    z = np.clip(local[..., 2], -p.half_len, p.half_len)
    d = local.copy()
    d[..., 2] -= z
    return np.linalg.norm(d, axis=-1) - p.radius


def sdf_np(prims: list[Primitive], x: np.ndarray) -> np.ndarray:
    return np.min(np.stack([_sdf_prim_np(p, x) for p in prims]), axis=0)


def _prim_area(p: Primitive) -> float:
    if p.kind == BOX:
        a, b, c = 2.0 * p.half
        return 2.0 * (a * b + b * c + a * c)
    r, h = p.radius, 2.0 * p.half_len
    return 2.0 * math.pi * r * h + 4.0 * math.pi * r * r


def _sample_prim(p: Primitive, n: int, rng: np.random.Generator) -> np.ndarray:
    if p.kind == BOX:
        h = p.half
        areas = np.array([h[1] * h[2], h[1] * h[2], h[0] * h[2], h[0] * h[2], h[0] * h[1], h[0] * h[1]])
        face = rng.choice(6, size=n, p=areas / areas.sum())
        local = (2.0 * rng.random((n, 3)) - 1.0) * h
        axis, sign = face // 2, np.where(face % 2 == 0, 1.0, -1.0)
        local[np.arange(n), axis] = sign * h[axis]
    else:
        r, hl = p.radius, p.half_len
        side, caps = 2.0 * math.pi * r * 2.0 * hl, 4.0 * math.pi * r * r
        on_side = rng.random(n) < side / (side + caps)
        local = np.zeros((n, 3))
        th = 2.0 * math.pi * rng.random(n)
        local[:, 0], local[:, 1] = r * np.cos(th), r * np.sin(th)
        local[:, 2] = (2.0 * rng.random(n) - 1.0) * hl
        # caps: uniform on a sphere of radius r, shifted to the end it lies on
        m = int((~on_side).sum())
        v = rng.normal(size=(m, 3))
        v = r * v / np.linalg.norm(v, axis=1, keepdims=True)
        v[:, 2] += np.where(v[:, 2] >= 0.0, hl, -hl)
        local[~on_side] = v
    return local @ p.rot.T + p.pos


def surface_area(prims: list[Primitive]) -> float:
    """Sum of the primitives' areas (an upper bound on the outer surface; the overlap is small)."""
    return float(sum(_prim_area(p) for p in prims))


def sample_surface(prims: list[Primitive], k: int, rng: np.random.Generator, tol: float = 1e-5) -> np.ndarray:
    """``(k, 3)`` points uniform by area on the tool's outer surface (link frame)."""
    areas = np.array([_prim_area(p) for p in prims])
    out, have = [], 0
    while have < k:
        m = 4 * k
        which = rng.choice(len(prims), size=m, p=areas / areas.sum())
        pts = np.concatenate([_sample_prim(p, int((which == i).sum()), rng) for i, p in enumerate(prims)])
        own = np.concatenate([np.full(int((which == i).sum()), i) for i in range(len(prims))])
        keep = np.ones(len(pts), dtype=bool)
        for i, p in enumerate(prims):          # drop what lies inside ANOTHER primitive
            other = own != i
            keep[other] &= _sdf_prim_np(p, pts[other]) > -tol
        pts = pts[keep]
        out.append(pts)
        have += len(pts)
    pts = np.concatenate(out)
    return pts[rng.permutation(len(pts))[:k]].astype(np.float32)


# ----------------------------------------------------------------------------
# symmetries
# ----------------------------------------------------------------------------


def _axis_rotations() -> list[np.ndarray]:
    """The 24 proper rotations that map the coordinate axes onto themselves."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            m = np.zeros((3, 3))
            for i, j in enumerate(perm):
                m[i, j] = signs[i]
            if np.linalg.det(m) > 0.5:
                mats.append(m)
    return mats


def _rot_about(axis: int, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    m = np.eye(3)
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


_CANDIDATES: np.ndarray | None = None


def _candidates(n_axial: int) -> np.ndarray:
    """The 24 axis-aligned rotations, each composed with rotations about the x, y and z axes by
    multiples of ``360 / n_axial`` deg, deduplicated, identity first."""
    global _CANDIDATES
    if _CANDIDATES is not None and _CANDIDATES.shape[0] > 0 and getattr(_candidates, "n", None) == n_axial:
        return _CANDIDATES
    cands = [np.eye(3)] + [_rot_about(axis, 2.0 * math.pi * k / n_axial) @ base
                           for base in _axis_rotations() for axis in range(3) for k in range(n_axial)]
    flat = np.round(np.stack(cands).reshape(len(cands), 9), 6)
    _, first = np.unique(flat, axis=0, return_index=True)
    order = np.sort(first)                                # identity (index 0) stays first
    _CANDIDATES, _candidates.n = np.stack(cands)[order], n_axial
    return _CANDIDATES


def symmetry_rotations(prims: list[Primitive], samples: np.ndarray, centre: np.ndarray | None = None,
                       tol: float = 1.5e-3, n_axial: int = 24) -> np.ndarray:
    """``(M, 3, 3)`` rotations ``g`` about ``centre`` (link frame) with ``g(tool) = tool``, identity first.

    Candidates (``_candidates``) include a capsule handle's continuous symmetry, sampled every
    ``360 / n_axial`` deg. A candidate is kept when every surface sample lands within ``tol`` of the
    surface.
    """
    c = np.zeros(3) if centre is None else np.asarray(centre, dtype=float)
    cands = _candidates(n_axial)                                                  # (C, 3, 3)
    moved = np.einsum("cij,kj->cki", cands, samples - c) + c                      # (C, K, 3)
    ok = np.abs(sdf_np(prims, moved)).max(axis=1) < tol
    return cands[ok].astype(np.float32)


# ----------------------------------------------------------------------------
# quaternions (wxyz)
# ----------------------------------------------------------------------------


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).view(*q.shape[:-1], 3, 3)


def quat_from_matrix(m: torch.Tensor) -> torch.Tensor:
    """wxyz, w >= 0 (Shepperd's method, batched)."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    cands = torch.stack((
        torch.stack((1 + tr, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]), -1),
        torch.stack((m[..., 2, 1] - m[..., 1, 2], 1 + m00 - m11 - m22, m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]), -1),
        torch.stack((m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1 - m00 + m11 - m22, m[..., 1, 2] + m[..., 2, 1]), -1),
        torch.stack((m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m00 - m11 + m22), -1),
    ), dim=-2)                                                          # (..., 4, 4)
    pick = torch.stack((tr, m00, m11, m22), dim=-1).argmax(dim=-1)
    q = torch.gather(cands, -2, pick[..., None, None].expand(*pick.shape, 1, 4)).squeeze(-2)
    q = torch.nn.functional.normalize(q, dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


# ----------------------------------------------------------------------------
# torch: a bank of tools, indexed per env
# ----------------------------------------------------------------------------


class ShapeBank:
    """Every pool tool's primitives, surface samples and symmetries as padded tensors.

    ``asset`` (N,) per env selects a tool. Surface samples are drawn once per tool (``k_fixed``,
    the scored set; and ``k_pool`` for the observed clouds, resampled from per step).
    """

    def __init__(self, urdf_paths, device, k_fixed: int = 256, k_pool: int = 4096, seed: int = 0,
                 with_symmetries: bool = True) -> None:
        rng = np.random.default_rng(seed)
        self.device = device
        a = len(urdf_paths)
        kind = np.zeros((a, MAX_PRIMS), dtype=np.int64)
        valid = np.zeros((a, MAX_PRIMS), dtype=bool)
        rot = np.tile(np.eye(3), (a, MAX_PRIMS, 1, 1))
        pos = np.zeros((a, MAX_PRIMS, 3))
        half = np.ones((a, MAX_PRIMS, 3))
        fixed = np.zeros((a, k_fixed, 3), dtype=np.float32)
        pool = np.zeros((a, k_pool, 3), dtype=np.float32)
        area = np.zeros(a, dtype=np.float32)
        syms, centre = [], np.zeros((a, 3), dtype=np.float32)
        for i, path in enumerate(urdf_paths):
            prims = parse_primitives(path)
            for j, p in enumerate(prims):
                kind[i, j], valid[i, j], rot[i, j], pos[i, j], half[i, j] = p.kind, True, p.rot, p.pos, p.half
            fixed[i] = sample_surface(prims, k_fixed, rng)
            pool[i] = sample_surface(prims, k_pool, rng)
            area[i] = surface_area(prims)
            centre[i] = pool[i].mean(axis=0)
            if with_symmetries:
                syms.append(symmetry_rotations(prims, fixed[i], centre=centre[i]))
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)  # noqa: E731
        self.kind, self.valid = t(kind, torch.long), t(valid, torch.bool)
        self.rot, self.pos, self.half = t(rot), t(pos), t(half)
        self.fixed, self.pool, self.area, self.centre = t(fixed), t(pool), t(area), t(centre)
        if with_symmetries:
            m = max(len(s) for s in syms)
            sym = np.tile(np.eye(3, dtype=np.float32), (a, m, 1, 1))
            sym_ok = np.zeros((a, m), dtype=bool)
            for i, s in enumerate(syms):
                sym[i, : len(s)], sym_ok[i, : len(s)] = s, True
            self.sym, self.sym_ok = t(sym), t(sym_ok, torch.bool)
            self.num_sym = torch.as_tensor([len(s) for s in syms], device=device)

    # -- SDF ------------------------------------------------------------------

    def sdf(self, asset: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``x`` (N, K, 3) in each env's tool link frame -> signed distance (N, K)."""
        out = None
        for j in range(MAX_PRIMS):
            r, t, h = self.rot[asset, j], self.pos[asset, j], self.half[asset, j]   # (N,3,3) (N,3) (N,3)
            local = torch.matmul(x - t.unsqueeze(1), r)                             # R^T (x - t)
            q = local.abs() - h.unsqueeze(1)
            box = torch.linalg.norm(q.clamp(min=0.0), dim=-1) + q.max(dim=-1).values.clamp(max=0.0)
            hl = h[:, 2:3]
            z = torch.maximum(torch.minimum(local[..., 2], hl), -hl)
            d = torch.stack((local[..., 0], local[..., 1], local[..., 2] - z), dim=-1)
            cap = torch.linalg.norm(d, dim=-1) - h[:, 0:1]
            dj = torch.where((self.kind[asset, j] == BOX).unsqueeze(1), box, cap)
            dj = torch.where(self.valid[asset, j].unsqueeze(1), dj, torch.full_like(dj, float("inf")))
            out = dj if out is None else torch.minimum(out, dj)
        return out

    # -- clouds ---------------------------------------------------------------

    def points_w(self, asset, pos, quat, which: str = "fixed", k: int | None = None) -> torch.Tensor:
        """Surface samples at a pose: ``pos`` (N, 3), ``quat`` (N, 4) wxyz -> (N, K, 3).

        ``which="fixed"``: the scored set, always the same points; ``"pool"``: ``k`` points drawn
        afresh from the tool's pool (no correspondence between calls)."""
        if which == "fixed":
            local = self.fixed[asset]
        else:
            n, kp = asset.shape[0], self.pool.shape[1]
            idx = torch.randint(0, kp, (n, int(k)), device=asset.device)
            local = torch.gather(self.pool[asset], 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        return torch.matmul(local, matrix_from_quat(quat).transpose(1, 2)) + pos.unsqueeze(1)

    def surface_distance(self, asset, pos_a, quat_a, pos_b, quat_b) -> tuple[torch.Tensor, torch.Tensor]:
        """Two-sided distance between the tool at pose a and at pose b: ``(max, mean)`` (N,) [m]."""
        ra, rb = matrix_from_quat(quat_a), matrix_from_quat(quat_b)
        s = self.fixed[asset]                                                       # (N, K, 3)
        # a's samples in b's frame, and b's samples in a's frame
        rel = torch.matmul(ra.transpose(1, 2), rb)                                   # R_a^T R_b
        dt_ab = torch.matmul((pos_b - pos_a).unsqueeze(1), ra).squeeze(1)            # R_a^T (t_b - t_a)
        b_in_a = torch.matmul(s, rel.transpose(1, 2)) + dt_ab.unsqueeze(1)
        dt_ba = torch.matmul((pos_a - pos_b).unsqueeze(1), rb).squeeze(1)
        a_in_b = torch.matmul(s, rel) + dt_ba.unsqueeze(1)
        da, db = self.sdf(asset, b_in_a).abs(), self.sdf(asset, a_in_b).abs()
        return torch.maximum(da.max(dim=1).values, db.max(dim=1).values), 0.5 * (da.mean(dim=1) + db.mean(dim=1))

    # -- symmetry canonicalisation -------------------------------------------

    def canonical_goal_pose(self, asset, obj_pos, obj_quat, goal_pos, goal_quat, kp_offsets):
        """The goal pose, among the tool's symmetric equivalents, whose keypoints (``kp_offsets``
        (N, P, 3), link frame) are nearest the object's: what a keypoint teacher should be asked to
        reach. A symmetry ``g`` is about the tool's centroid ``c``, so the equivalent pose of
        ``(R, t)`` is ``(R g, t + R (c - g c))``. Returns ``(pos (N, 3), quat (N, 4) wxyz)``."""
        n = asset.shape[0]
        g, c = self.sym[asset], self.centre[asset]                                   # (N,M,3,3) (N,3)
        r = matrix_from_quat(goal_quat)
        rg = torch.matmul(r.unsqueeze(1), g)                                         # (N, M, 3, 3)
        shift = c.unsqueeze(1) - torch.matmul(g, c.view(n, 1, 3, 1)).squeeze(-1)     # (N, M, 3)
        tg = goal_pos.unsqueeze(1) + torch.matmul(shift, r.transpose(1, 2))          # (N, M, 3)
        kp_goal = torch.matmul(kp_offsets.unsqueeze(1), rg.transpose(-1, -2)) + tg.unsqueeze(2)
        kp_obj = (torch.matmul(kp_offsets, matrix_from_quat(obj_quat).transpose(1, 2)) + obj_pos.unsqueeze(1)).unsqueeze(1)
        err = torch.linalg.norm(kp_goal - kp_obj, dim=-1).max(dim=-1).values         # (N, M)
        err = torch.where(self.sym_ok[asset], err, torch.full_like(err, float("inf")))
        best = err.argmin(dim=1)
        ar = torch.arange(n, device=asset.device)
        q = quat_from_matrix(rg[ar, best])
        q = torch.where((q * goal_quat).sum(dim=-1, keepdim=True) < 0.0, -q, q)
        return tg[ar, best], q
