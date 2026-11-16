"""The cloth as points: a ``resolution x resolution`` grid over its ``length x width`` rectangle.

In the cloth's own rest frame: x in [-L/2, L/2] across the crease, y along it, z = 0 the mid-plane.
The crease is the line x = 0; the ``+x`` half (``moving``) folds onto the ``-x`` half
(``stationary``). The deformable cloth's particles ARE these points; the rigid cloth carries them on
its articulated plates. Row-major indexing keeps the halves and corners simple index arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

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


# ------------------------------------------------------------------ per-env cloths


@dataclass
class ClothGrid:
    """Every env's rest grid and the fold's rest-frame quantities, ``N`` envs."""

    resolution: int
    dims: torch.Tensor            # (N, 3) length, width, thickness [m]
    ply_gap: torch.Tensor         # (N,) fold-target lift [m]
    rest: torch.Tensor            # (N, P, 3) rest frame
    moving_idx: torch.Tensor      # (M,)
    stationary_idx: torch.Tensor  # (S,)
    corner_idx: torch.Tensor          # (4,) the tracked corners
    st_rest_centroid: torch.Tensor  # (N, 3)
    st_rest_centred: torch.Tensor   # (N, S, 3)
    keypoint_rest: torch.Tensor | None = None  # (N, 4, 3) optional exact moving-half corners

    @property
    def num_points(self) -> int:
        return self.rest.shape[1]

    def folded_rest(self, idx: torch.Tensor) -> torch.Tensor:
        """``(N, len(idx), 3)``: rest points ``idx`` reflected across the crease, lifted one ply gap."""
        f = self.rest[:, idx].clone()
        f[..., 0] = -f[..., 0]
        f[..., 2] = f[..., 2] + self.ply_gap.unsqueeze(1)
        return f


def make_grid(dims: torch.Tensor, resolution: int, ply_gap: torch.Tensor) -> ClothGrid:
    """``dims (N, 3)``: one rectangle per env (equal rows = one shared size)."""
    dev = dims.device
    unit = torch.tensor(grid_points(1.0, 1.0, resolution), device=dev, dtype=torch.float32)
    rest = unit.unsqueeze(0) * torch.cat([dims[:, :2], torch.zeros_like(dims[:, :1])], dim=1).unsqueeze(1)
    st = torch.tensor(half_indices(resolution, False), device=dev, dtype=torch.long)
    st_rest = rest[:, st]
    c = st_rest.mean(dim=1)
    return ClothGrid(
        resolution=int(resolution),
        dims=dims.float(),
        ply_gap=ply_gap.float().expand(dims.shape[0]).clone(),
        rest=rest,
        moving_idx=torch.tensor(half_indices(resolution, True), device=dev, dtype=torch.long),
        stationary_idx=st,
        corner_idx=torch.tensor(corner_indices(resolution), device=dev, dtype=torch.long),
        st_rest_centroid=c,
        st_rest_centred=st_rest - c.unsqueeze(1),
    )


def folded_start(grid: ClothGrid, angle: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
    """Rest points with the + half turned up by ``angle (n,)`` [rad] about an axis ``ply_gap / 2``
    above the crease: at 180 deg it lies reflected onto the - half one ply gap above it, at 0 it is
    the rest grid. ``(n, P, 3)`` for envs ``ids`` (all by default)."""
    rest = grid.rest if ids is None else grid.rest[ids]
    gap = grid.ply_gap if ids is None else grid.ply_gap[ids]
    out = rest.clone()
    moving = rest[..., 0] > 1e-9
    a = rest[..., 0]
    r = (0.5 * gap).unsqueeze(1)
    c, s = torch.cos(angle).unsqueeze(1), torch.sin(angle).unsqueeze(1)
    out[..., 0] = torch.where(moving, a * c + r * s, a)
    out[..., 2] = torch.where(moving, rest[..., 2] + a * s + r * (1.0 - c), rest[..., 2])
    return out
