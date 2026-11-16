"""The sheet as points, and the fold defined on them -- one implementation for both backends.

A sheet is a ``resolution x resolution`` grid of points over its ``length x width`` rectangle, in
its own rest frame: x in [-L/2, L/2] along the fold axis, y across, z = 0 the mid-plane. The crease
is the line x = 0; the ``+x`` half (``moving``) folds onto the ``-x`` half (``stationary``). The VBD
cloth's particles ARE these points; the rigid sheet carries them on its two plates.

**The fold is defined relative to the sheet, not the world.** A target pinned in the world cannot
tell a fold from a slide (pushing the flat sheet half its length puts the moving half where the
targets are). So targets live in the rest frame -- each point reflected across the crease and
lifted one ply gap -- and are carried onto the stationary half's CURRENT rigid frame, fitted from
its points every step: translating or rotating the sheet moves the target with it, only folding
the flap reduces the error.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import quat

# ------------------------------------------------------------------ the grid (index arithmetic)


def grid_points(length: float, width: float, resolution: int) -> list[tuple[float, float, float]]:
    """Row-major in x then y, centred on the origin, z = 0."""
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    sx, sy = length / (resolution - 1), width / (resolution - 1)
    return [
        (i * sx - length / 2.0, j * sy - width / 2.0, 0.0)
        for i in range(resolution)
        for j in range(resolution)
    ]


def grid_triangles(resolution: int) -> list[int]:
    """Flat index list, 2 (resolution-1)^2 triangles, +z winding, the cut alternating per cell.

    A uniform diagonal couples bending to twist (a released fold springs back with one end of the
    crease lifted); the checkerboard has no handedness.
    """
    idx: list[int] = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            a = i * resolution + j
            b, c = a + 1, a + resolution
            d = c + 1
            idx += [a, c, d, a, d, b] if (i + j) % 2 else [a, c, b, b, c, d]
    return idx


def half_indices(resolution: int, positive: bool) -> list[int]:
    """Points strictly on one side of the crease row: the crease row itself (odd resolution) is the
    hinge and belongs to neither half."""
    mid = (resolution - 1) / 2.0
    return [i * resolution + j for i in range(resolution) for j in range(resolution)
            if (i > mid if positive else i < mid)]


def corner_indices(resolution: int) -> list[int]:
    """The moving half's four corners: (near-left, near-right, far-left, far-right).

    Near = the first row strictly past the crease, not the crease row: crease points barely move in
    a fold while the target rule asks them to rise one ply gap, so a perfect fold would not score 0.
    """
    near = resolution // 2 + (resolution % 2)
    far = resolution - 1
    return [near * resolution, near * resolution + far, far * resolution, far * resolution + far]


def half_triangles(resolution: int, reflected: bool = True) -> list[int]:
    """Triangles fully inside the moving half, indexed into ``half_indices(.., True)`` order. With
    ``reflected`` the winding is reversed: a fold is a reflection, which flips every triangle, and
    a folded goal drawn with the rest winding renders as an unlit back face."""
    keep = half_indices(resolution, positive=True)
    remap = {v: k for k, v in enumerate(keep)}
    full = grid_triangles(resolution)
    out: list[int] = []
    for t in range(0, len(full), 3):
        tri = full[t:t + 3]
        if all(v in remap for v in tri):
            out += [remap[v] for v in (reversed(tri) if reflected else tri)]
    return out


# ------------------------------------------------------------------ per-env sheets


@dataclass
class Sheet:
    """Every env's rest sheet and the fold's rest-frame quantities, ``N`` envs."""

    resolution: int
    dims: torch.Tensor            # (N, 3) length, width, thickness [m]
    ply_gap: torch.Tensor         # (N,) fold-target lift [m]
    rest: torch.Tensor            # (N, P, 3) rest frame
    moving_idx: torch.Tensor      # (M,)
    stationary_idx: torch.Tensor  # (S,)
    kp_idx: torch.Tensor          # (4,) the tracked corners
    st_rest_centroid: torch.Tensor  # (N, 3)
    st_rest_centred: torch.Tensor   # (N, S, 3)

    @property
    def num_points(self) -> int:
        return self.rest.shape[1]

    def folded_rest(self, idx: torch.Tensor) -> torch.Tensor:
        """``(N, len(idx), 3)``: rest points ``idx`` reflected across the crease, lifted one ply gap."""
        f = self.rest[:, idx].clone()
        f[..., 0] = -f[..., 0]
        f[..., 2] = f[..., 2] + self.ply_gap.unsqueeze(1)
        return f


def make_sheet(dims: torch.Tensor, resolution: int, ply_gap: torch.Tensor) -> Sheet:
    """``dims (N, 3)``: one rectangle per env (equal rows = one shared size)."""
    dev = dims.device
    unit = torch.tensor(grid_points(1.0, 1.0, resolution), device=dev, dtype=torch.float32)
    rest = unit.unsqueeze(0) * torch.cat([dims[:, :2], torch.zeros_like(dims[:, :1])], dim=1).unsqueeze(1)
    st = torch.tensor(half_indices(resolution, False), device=dev, dtype=torch.long)
    st_rest = rest[:, st]
    c = st_rest.mean(dim=1)
    return Sheet(
        resolution=int(resolution),
        dims=dims.float(),
        ply_gap=ply_gap.float().expand(dims.shape[0]).clone(),
        rest=rest,
        moving_idx=torch.tensor(half_indices(resolution, True), device=dev, dtype=torch.long),
        stationary_idx=st,
        kp_idx=torch.tensor(corner_indices(resolution), device=dev, dtype=torch.long),
        st_rest_centroid=c,
        st_rest_centred=st_rest - c.unsqueeze(1),
    )


def folded_start(sheet: Sheet, angle: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
    """Rest points with the + half turned up by ``angle (n,)`` [rad] about an axis ``ply_gap / 2``
    above the crease: at 180 deg it lies reflected onto the - half one ply gap above it, at 0 it is
    the rest sheet. ``(n, P, 3)`` for envs ``ids`` (all by default)."""
    rest = sheet.rest if ids is None else sheet.rest[ids]
    gap = sheet.ply_gap if ids is None else sheet.ply_gap[ids]
    out = rest.clone()
    moving = rest[..., 0] > 1e-9
    a = rest[..., 0]
    r = (0.5 * gap).unsqueeze(1)
    c, s = torch.cos(angle).unsqueeze(1), torch.sin(angle).unsqueeze(1)
    out[..., 0] = torch.where(moving, a * c + r * s, a)
    out[..., 2] = torch.where(moving, rest[..., 2] + a * s + r * (1.0 - c), rest[..., 2])
    return out


# ------------------------------------------------------------------ live quantities


def stationary_frame(sheet: Sheet, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(R (N, 3, 3), centroid (N, 3))``: the rigid transform taking the stationary half's rest
    pose to its current one (Kabsch). Ridge + determinant correction + identity fallback: a
    crumpled half can make the fit rank deficient, and a NaN or a mirrored frame here would
    corrupt every target."""
    cur = points[:, sheet.stationary_idx]
    centroid = cur.mean(dim=1)
    h = torch.einsum("nki,nkj->nij", sheet.st_rest_centred, cur - centroid.unsqueeze(1))
    eye = torch.eye(3, device=h.device, dtype=h.dtype)
    # A diverged env's NaN would make the SVD raise (CPU) or return NaN (CUDA): fit it as identity;
    # the obs guard flags and resets it.
    bad = ~torch.isfinite(h).all(dim=-1).all(dim=-1)
    h = torch.where(bad.view(-1, 1, 1), eye, h) + 1e-8 * eye
    u, _, vh = torch.linalg.svd(h)
    v = vh.transpose(-2, -1)
    d = torch.ones((cur.shape[0], 3), device=cur.device, dtype=cur.dtype)
    d[:, 2] = torch.linalg.det(v @ u.transpose(-2, -1))
    r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)
    bad = bad | ~torch.isfinite(r).all(dim=-1).all(dim=-1)
    return torch.where(bad.view(-1, 1, 1), eye, r), centroid


def carry(sheet: Sheet, rest_pts: torch.Tensor, r: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """Rest-frame points ``(N, M, 3)`` onto the stationary half's live frame."""
    local = rest_pts - sheet.st_rest_centroid.unsqueeze(1)
    return torch.einsum("nij,nmj->nmi", r, local) + centroid.unsqueeze(1)


def keypoint_targets(sheet: Sheet, r: torch.Tensor, centroid: torch.Tensor, unfold: torch.Tensor) -> torch.Tensor:
    """``(N, 4, 3)``: the folded corners, or the flat ones where ``unfold``."""
    folded = carry(sheet, sheet.folded_rest(sheet.kp_idx), r, centroid)
    flat = carry(sheet, sheet.rest[:, sheet.kp_idx], r, centroid)
    return torch.where(unfold.view(-1, 1, 1), flat, folded)


def moving_half_targets(sheet: Sheet, r: torch.Tensor, centroid: torch.Tensor, unfold: torch.Tensor) -> torch.Tensor:
    """``(N, M, 3)``: where every moving-half point belongs (the goal drawn as a ghost sheet)."""
    folded = carry(sheet, sheet.folded_rest(sheet.moving_idx), r, centroid)
    flat = carry(sheet, sheet.rest[:, sheet.moving_idx], r, centroid)
    return torch.where(unfold.view(-1, 1, 1), flat, folded)


def fold_error(keypoints: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Max over the corners of the distance to the target: one corner still flat is no fold."""
    return (keypoints - targets).norm(dim=-1).amax(dim=-1)


def footprint_ratio(sheet: Sheet, points: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Extent along the sheet's OWN fold axis (the stationary frame's x) / its length. A fold halves
    it; a slide or a crumple need not, which keypoints alone cannot tell apart."""
    coord = torch.einsum("npi,ni->np", points, r[:, :, 0])
    return (coord.amax(dim=1) - coord.amin(dim=1)) / sheet.dims[:, 0]


def flat_on_table(points: torch.Tensor, table_top_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(spread, lift)``: height range over all points, and the lowest point above the table."""
    z = points[..., 2]
    low = z.amin(dim=1)
    return z.amax(dim=1) - low, low - table_top_z


def sheet_frame(points: torch.Tensor, points_vel: torch.Tensor, r: torch.Tensor):
    """The sheet as a rigid object: ``(pos, quat wxyz, lin vel, ang vel)``. Position and velocity of
    the points' centroid, orientation of the stationary half, angular velocity the least-squares
    rigid fit of the point velocities about the centroid."""
    pos = points.mean(dim=1)
    lin = points_vel.mean(dim=1)
    rel = points - pos.unsqueeze(1)
    dv = points_vel - lin.unsqueeze(1)
    eye = torch.eye(3, device=points.device, dtype=points.dtype)
    inertia = (rel * rel).sum(dim=(1, 2)).view(-1, 1, 1) * eye - torch.einsum("npi,npj->nij", rel, rel)
    mom = torch.cross(rel, dv, dim=-1).sum(dim=1)
    ang = torch.linalg.solve(inertia + 1e-9 * eye, mom.unsqueeze(-1)).squeeze(-1)
    return pos, quat.from_matrix(r), lin, ang
