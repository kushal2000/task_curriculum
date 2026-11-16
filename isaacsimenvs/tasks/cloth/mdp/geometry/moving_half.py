"""A rigid object pose fitted to the moving half's surface samples.

For a single hinge this is the moving plate's physical orientation and mid-plane
center, including its downward normal after a 180-degree fold. When the moving
half bends across several plates or deforms, one rigid pose is a least-squares approximation.
No orientation is forced to agree with the stationary half's normal.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import fold, rotations
from .grid import ClothGrid


@dataclass(frozen=True)
class ObjectPose:
    """World-frame centers, velocities, and proper unit quaternions in wxyz order."""

    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor
    goal_pos_w: torch.Tensor
    goal_quat_w: torch.Tensor


def _fit_rotation(rest: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Proper Kabsch rotation; rank below two or nonfinite input falls back to identity.

    The planar landmarks have rank two by construction. Adding an identity ridge
    would bias large rotations, so determinant correction operates on the exact SVD.
    """
    eye = torch.eye(3, device=rest.device, dtype=rest.dtype).expand(rest.shape[0], -1, -1)
    finite = torch.isfinite(rest).all(dim=(1, 2)) & torch.isfinite(current).all(dim=(1, 2))
    a = torch.where(finite[:, None, None], rest, torch.zeros_like(rest))
    b = torch.where(finite[:, None, None], current, torch.zeros_like(current))
    covariance = torch.einsum('nki,nkj->nij', a-a.mean(dim=1, keepdim=True), b-b.mean(dim=1, keepdim=True))
    finite = finite & torch.isfinite(covariance).all(dim=(1, 2))
    covariance = torch.where(finite[:, None, None], covariance, eye)
    u, values, vh = torch.linalg.svd(covariance)
    v = vh.transpose(-2, -1)
    correction = torch.ones_like(values)
    correction[:, 2] = torch.where(torch.linalg.det(v @ u.transpose(-2, -1)) < 0, -1., 1.)
    rotation = v @ torch.diag_embed(correction) @ u.transpose(-2, -1)
    usable = finite & (values[:, 1] > 1e-6 * values[:, 0]) & (values[:, 0] > 0)
    usable = usable & torch.isfinite(rotation).all(dim=(1, 2))
    return torch.where(usable[:, None, None], rotation, eye)


def moving_half_pose(grid: ClothGrid, state, view, unfold: torch.Tensor) -> ObjectPose:
    """Fit a common moving-half object pose for rigid and deformable cloth.

    Explicit rigid landmarks are used when available; otherwise fit the positive-x
    surface samples. Both report the canonical rest center (L/4, 0, 0), carried by
    the fit, rather than the resolution-dependent sample centroid. Velocity comes
    from a rigid fit of the moving surface samples, shifted to that object center.
    A bent half has no unique rigid velocity; this is its least-squares estimate. Nonfinite current samples fall back to finite centers/zero velocity;
    the original state remains available to the environment's divergence guard.
    """
    if grid.keypoint_rest is not None and state.keypoints_w is not None:
        rest, current = grid.keypoint_rest, state.keypoints_w
    else:
        rest, current = grid.rest[:, grid.moving_idx], state.points_w[:, grid.moving_idx]
    if rest.shape != current.shape or rest.ndim != 3 or rest.shape[-1] != 3 or rest.shape[1] < 3:
        raise ValueError('moving-half samples must have matching shape (N, M>=3, 3)')
    rotation = _fit_rotation(rest, current)
    finite_current = torch.isfinite(current).all(dim=(1, 2))
    canonical_center = rest.new_zeros((rest.shape[0], 3))
    canonical_center[:, 0] = grid.dims[:, 0] / 4
    fitted_center = current.mean(dim=1) + torch.einsum(
        'nij,nj->ni', rotation, canonical_center-rest.mean(dim=1)
    )
    center = torch.where(finite_current[:, None], fitted_center, torch.zeros_like(fitted_center))
    points, velocities = state.points_w[:, grid.moving_idx], state.points_vel_w[:, grid.moving_idx]
    finite_velocity = torch.isfinite(points).all(dim=(1, 2)) & torch.isfinite(velocities).all(dim=(1, 2))
    points = torch.where(finite_velocity[:, None, None], points, torch.zeros_like(points))
    velocities = torch.where(finite_velocity[:, None, None], velocities, torch.zeros_like(velocities))
    sample_center, quat, linear, angular = fold.cloth_frame(points, velocities, rotation)
    linear = linear + torch.cross(angular, center-sample_center, dim=-1)
    valid = finite_current & finite_velocity
    linear = torch.where(valid[:, None], linear, torch.zeros_like(linear))
    angular = torch.where(valid[:, None], angular, torch.zeros_like(angular))

    goal_rest = canonical_center[:, None].clone()
    goal_rest[..., 0] = torch.where(unfold[:, None], goal_rest[..., 0], -goal_rest[..., 0])
    goal_rest[..., 2] += torch.where(unfold, torch.zeros_like(grid.ply_gap), grid.ply_gap)[:, None]
    goal_center = fold.carry(grid, goal_rest, view.frame_r, view.frame_c)[:, 0]
    # A physical half turn, not a reflection: the folded sheet's normal points down.
    folded_rotation = view.frame_r * view.frame_r.new_tensor([-1., 1., -1.])
    goal_rotation = torch.where(unfold[:, None, None], view.frame_r, folded_rotation)
    return ObjectPose(center, quat, linear, angular, goal_center, rotations.from_matrix(goal_rotation))
