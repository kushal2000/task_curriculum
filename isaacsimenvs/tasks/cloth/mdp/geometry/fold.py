"""What a fold is, measured on the cloth's points (``grid.ClothGrid``).

**The fold is defined relative to the cloth, not the world.** A target pinned in the world cannot
tell a fold from a slide (pushing the flat grid half its length puts the moving half where the
targets are). So targets live in the rest frame -- each point reflected across the crease and
lifted one ply gap -- and are carried onto the stationary half's CURRENT rigid frame, fitted from
its points every step: translating or rotating the grid moves the target with it, only folding
the flap reduces the error.
"""

from __future__ import annotations

import torch

from . import rotations
from .grid import ClothGrid


def stationary_frame(grid: ClothGrid, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(R (N, 3, 3), centroid (N, 3))``: the rigid transform taking the stationary half's rest
    pose to its current one (Kabsch). Ridge + determinant correction + identity fallback: a
    crumpled half can make the fit rank deficient, and a NaN or a mirrored frame here would
    corrupt every target."""
    cur = points[:, grid.stationary_idx]
    centroid = cur.mean(dim=1)
    h = torch.einsum("nki,nkj->nij", grid.st_rest_centred, cur - centroid.unsqueeze(1))
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


def carry(grid: ClothGrid, rest_pts: torch.Tensor, r: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """Rest-frame points ``(N, M, 3)`` onto the stationary half's live frame."""
    local = rest_pts - grid.st_rest_centroid.unsqueeze(1)
    return torch.einsum("nij,nmj->nmi", r, local) + centroid.unsqueeze(1)


def corner_targets(grid: ClothGrid, r: torch.Tensor, centroid: torch.Tensor, unfold: torch.Tensor) -> torch.Tensor:
    """``(N, 4, 3)``: the folded corners, or the flat ones where ``unfold``."""
    folded = carry(grid, grid.folded_rest(grid.corner_idx), r, centroid)
    flat = carry(grid, grid.rest[:, grid.corner_idx], r, centroid)
    return torch.where(unfold.view(-1, 1, 1), flat, folded)


def moving_half_targets(grid: ClothGrid, r: torch.Tensor, centroid: torch.Tensor, unfold: torch.Tensor) -> torch.Tensor:
    """``(N, M, 3)``: where every moving-half point belongs (the goal drawn as a ghost grid)."""
    folded = carry(grid, grid.folded_rest(grid.moving_idx), r, centroid)
    flat = carry(grid, grid.rest[:, grid.moving_idx], r, centroid)
    return torch.where(unfold.view(-1, 1, 1), flat, folded)


def fold_error(keypoints: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Max over the corners of the distance to the target: one corner still flat is no fold."""
    return (keypoints - targets).norm(dim=-1).amax(dim=-1)


def footprint_ratio(grid: ClothGrid, points: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Extent along the grid's OWN fold axis (the stationary frame's x) / its length. A fold halves
    it; a slide or a crumple need not, which keypoints alone cannot tell apart."""
    coord = torch.einsum("npi,ni->np", points, r[:, :, 0])
    return (coord.amax(dim=1) - coord.amin(dim=1)) / grid.dims[:, 0]


def flat_on_table(points: torch.Tensor, table_top_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(spread, lift)``: height range over all points, and the lowest point above the table."""
    z = points[..., 2]
    low = z.amin(dim=1)
    return z.amax(dim=1) - low, low - table_top_z


def cloth_frame(points: torch.Tensor, points_vel: torch.Tensor, r: torch.Tensor):
    """The grid as a rigid object: ``(pos, quat wxyz, lin vel, ang vel)``. Position and velocity of
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
    return pos, rotations.from_matrix(r), lin, ang
