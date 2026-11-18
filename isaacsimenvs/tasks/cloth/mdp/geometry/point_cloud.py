"""The cloth as a point cloud: ``P`` samples on its top face, and where they belong.

**Samples live on the cloth, not in the world.** Each one is fixed to a spot on the top face
(``+thickness / 2`` in the sheet's rest frame, which is every plate's body frame), bound to the
plate that owns it, and moves with that plate. So sample ``i`` and goal ``i`` are the same piece of
cloth and a matched distance is exact; after a fold the moving half's samples sit on its
underside, facing the stationary half.

**Goals live in the world.** At reset the goal is the folded (or, for an unfold, flat) cloth lying
on the table at the pose the sheet was placed in, fixed for the episode. Unlike the corner targets,
which ride on the stationary half's live frame, a world-frame goal also penalises dragging the cloth
away while folding it.

Draws come from a fixed pool of normalized points (uniform over the unit square, so area-uniform
for every geometry), bound to plates once; each reset picks ``P`` of them per env. Everything here
is backend-free; the rigid backend owns the pool and the per-env draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import rotations


@dataclass
class SurfaceSamples:
    """Every env's current draw, ``N`` envs x ``P`` points, rewritten by the backend at reset."""

    rest: torch.Tensor          # (N, P, 3) sheet rest frame, on the top face (z = +thickness / 2)
    moving: torch.Tensor        # (N, P) bool: owned by a plate of the folding (+x) half
    placed_pos_w: torch.Tensor  # (N, 3) sheet origin, lying flat on the table where it was placed
    placed_quat_w: torch.Tensor  # (N, 4) wxyz, its yaw

    @property
    def num_points(self) -> int:
        return self.rest.shape[1]

    @classmethod
    def empty(cls, num_envs: int, num_points: int, device) -> "SurfaceSamples":
        quat = torch.zeros(num_envs, 4, device=device)
        quat[:, 0] = 1.0
        return cls(
            rest=torch.zeros(num_envs, num_points, 3, device=device),
            moving=torch.zeros(num_envs, num_points, dtype=torch.bool, device=device),
            placed_pos_w=torch.zeros(num_envs, 3, device=device),
            placed_quat_w=quat,
        )


def sample_pool(pool_size: int, seed: int) -> np.ndarray:
    """``(K, 3)`` normalized points, uniform over ``[-0.5, 0.5]^2``, ``z = 0``, seeded.

    Kept a hair inside the boundary so every point has an owner plate."""
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.5 + 1e-6, 0.5 - 1e-6, size=(pool_size, 2))
    return np.c_[xy, np.zeros(pool_size)]


def draw(num_envs: int, pool_size: int, num_points: int, device, generator=None, chunk: int = 2048) -> torch.Tensor:
    """``(num_envs, num_points)`` pool indices, distinct within each env."""
    if num_points > pool_size:
        raise ValueError(f"cannot draw {num_points} distinct points from a pool of {pool_size}")
    out = torch.empty(num_envs, num_points, dtype=torch.long, device=device)
    for s in range(0, num_envs, chunk):
        n = min(chunk, num_envs - s)
        keys = torch.rand(n, pool_size, device=device, generator=generator)
        out[s:s + n] = keys.topk(num_points, dim=1).indices
    return out


def top_face_rest(normalized: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    """Normalized ``(n, P, 3)`` -> rest-frame points on the top face of cloths ``dims (n, 3)``."""
    out = normalized * torch.cat([dims[:, :2], torch.zeros_like(dims[:, :1])], dim=1).unsqueeze(1)
    out[..., 2] = 0.5 * dims[:, 2:3]
    return out


def goal_rest(rest: torch.Tensor, moving: torch.Tensor, ply_gap: torch.Tensor, unfold: torch.Tensor) -> torch.Tensor:
    """Where each sample belongs, in the rest frame.

    Fold: the moving half turned 180 deg about the hinge axis (on the crease, ``ply_gap / 2`` up:
    the hinges sit on the top face), ``(x, z) -> (-x, ply_gap - z)``, so its top face lands face
    down on the stationary top face. Unfold, and every stationary sample: the flat rest pose."""
    folded = rest.clone()
    folded[..., 0] = -rest[..., 0]
    folded[..., 2] = ply_gap.view(-1, 1) - rest[..., 2]
    flip = moving & ~unfold.view(-1, 1)
    return torch.where(flip.unsqueeze(-1), folded, rest)


def place(rest: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    """Rest-frame points ``(n, P, 3)`` under sheet poses ``(n, 3), (n, 4)``."""
    return pos_w.unsqueeze(1) + rotations.apply(quat_w.unsqueeze(1).expand(*rest.shape[:2], 4), rest)


def on_bodies(body_pos_w: torch.Tensor, body_quat_w: torch.Tensor, body_idx: torch.Tensor,
              rest: torch.Tensor) -> torch.Tensor:
    """Samples carried by their plates: ``body_*_w (n, B, .)``, ``body_idx (n, P)``, ``rest (n, P, 3)``."""
    pos = body_pos_w.gather(1, body_idx.unsqueeze(-1).expand(-1, -1, 3))
    quat = body_quat_w.gather(1, body_idx.unsqueeze(-1).expand(-1, -1, 4))
    return pos + rotations.apply(quat, rest)


def matched_distance(points: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    """``(n,)`` mean distance between corresponding points: EMD under the known matching (an
    upper bound on the true EMD)."""
    return (points - goals).norm(dim=-1).mean(dim=-1)


def chamfer_distance(points: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    """``(n,)`` symmetric mean nearest-neighbour distance; ignores which point is which."""
    d = torch.cdist(points, goals, compute_mode="donot_use_mm_for_euclid_dist")
    return 0.5 * (d.amin(dim=2).mean(dim=1) + d.amin(dim=1).mean(dim=1))
