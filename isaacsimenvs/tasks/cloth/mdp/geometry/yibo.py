"""Observation geometry frozen to origin/yiboc_wuji (2b82faf1c73f).

This deliberately preserves the historical ridge fit, quaternion sign branch,
planar inset markers and relabelled goal quaternion. It never changes physical
cloth scoring. The old training path reconstructed markers from pose + offsets;
it did not pass raw deformed marker coordinates to the actor.
"""
from __future__ import annotations

import torch

from . import grid as grid_geometry
from . import rotations


def _mat_to_wxyz(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(N, 3, 3)`` to wxyz quaternions, via the trace formulation.

    Uses the largest-diagonal branch rather than the naive trace form, which loses precision (and
    can divide by ~0) when the rotation approaches 180 degrees -- exactly the case a completed fold
    produces.
    """
    n = r.shape[0]
    q = torch.zeros((n, 4), device=r.device, dtype=r.dtype)
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]

    big = trace > 0
    if bool(big.any()):
        sq = torch.sqrt(trace[big] + 1.0) * 2.0
        q[big, 0] = 0.25 * sq
        q[big, 1] = (r[big, 2, 1] - r[big, 1, 2]) / sq
        q[big, 2] = (r[big, 0, 2] - r[big, 2, 0]) / sq
        q[big, 3] = (r[big, 1, 0] - r[big, 0, 1]) / sq
    rest = ~big
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = rest & (r[:, i, i] >= r[:, j, j]) & (r[:, i, i] >= r[:, k, k])
        if not bool(sel.any()):
            continue
        sq = torch.sqrt(1.0 + r[sel, i, i] - r[sel, j, j] - r[sel, k, k]) * 2.0
        q[sel, 0] = (r[sel, k, j] - r[sel, j, k]) / sq
        q[sel, 1 + i] = 0.25 * sq
        q[sel, 1 + j] = (r[sel, j, i] + r[sel, i, j]) / sq
        q[sel, 1 + k] = (r[sel, k, i] + r[sel, i, k]) / sq
    # eps guards a zero-norm quaternion, which a degenerate fit can produce.
    return torch.nn.functional.normalize(q, dim=-1, eps=1e-12)


def _fit(rest, current):
    center = current.mean(dim=1)
    centered_rest = rest - rest.mean(dim=1, keepdim=True)
    h = torch.einsum("nki,nkj->nij", centered_rest, current - center[:, None])
    eye = torch.eye(3, dtype=h.dtype, device=h.device).expand_as(h)
    h = h + 1e-8 * eye
    # Preserve finite branch arithmetic exactly, while allowing the normal
    # nonfinite-state guard to handle bad simulation samples after this readout.
    finite = torch.isfinite(h).all(dim=(1, 2))
    h = torch.where(finite[:, None, None], h, eye)
    u, _, vh = torch.linalg.svd(h)
    v = vh.transpose(-2, -1)
    correction = torch.ones((len(rest), 3), device=rest.device, dtype=rest.dtype)
    correction[:, 2] = torch.linalg.det(v @ u.transpose(-2, -1))
    rotation = v @ torch.diag_embed(correction) @ u.transpose(-2, -1)
    return rotation, center


def yibo_object_features(grid, state, goal_lift=0.0):
    """Return historical pos, wxyz quat, linear/angular velocity, current/goal markers.

    The first strictly positive grid row is tracked, not the crease: at resolution
    seven, marker x coordinates are L/6 and L/2 and their center is L/3. Goal
    orientation retains the old xyzw-as-wxyz relabelling, independently of the
    modern physical fold target. goal_lift is the historical observation lift.
    """
    indices = grid_geometry.corner_indices(grid.resolution)
    rest, current = grid.rest[:, indices], state.points_w[:, indices]
    rotation, center = _fit(rest, current)
    quat = _mat_to_wxyz(rotation)
    offsets = rest - rest.mean(dim=1, keepdim=True)
    markers = center[:, None] + rotations.apply(quat[:, None], offsets)
    stationary_rotation, stationary_center = _fit(grid.rest[:, grid.stationary_idx],
                                                   state.points_w[:, grid.stationary_idx])
    goal_rest = rest.clone()
    goal_rest[..., 0] *= -1
    goal_rest[..., 2] += float(goal_lift)
    stationary_rest_center = grid.rest[:, grid.stationary_idx].mean(dim=1)
    goal_physical = torch.einsum("nij,nkj->nki", stationary_rotation,
                                 goal_rest - stationary_rest_center[:, None]) + stationary_center[:, None]
    goal_center = goal_physical.mean(dim=1)
    goal_rotation = stationary_rotation * stationary_rotation.new_tensor([-1., 1., -1.])
    goal_quat = _mat_to_wxyz(goal_rotation)[:, [1, 2, 3, 0]]
    targets = goal_center[:, None] + rotations.apply(goal_quat[:, None], offsets)
    linear = state.points_vel_w[:, indices].mean(dim=1)
    return center, quat, linear, torch.zeros_like(linear), markers, targets
