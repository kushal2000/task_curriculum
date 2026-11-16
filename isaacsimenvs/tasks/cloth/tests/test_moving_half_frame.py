"""Moving-half pose tracks actual folding and the corresponding proper-rotation goal."""
from types import SimpleNamespace as NS

import pytest
import torch

from isaacsimenvs.tasks.cloth.mdp.geometry import grid, rotations
from isaacsimenvs.tasks.cloth.mdp.geometry.moving_half import moving_half_pose


def _scene(angle_deg, unfold=False):
    dtype = torch.float64
    dims = torch.tensor([[.2, .15, .004], [.32, .21, .009]], dtype=dtype)
    cloth = grid.make_grid(dims, 7, dims[:, 2])
    cloth.ply_gap = dims[:, 2]
    cloth.dims = dims
    unit = torch.tensor([[0., -.5, 0.], [0., .5, 0.], [.5, -.5, 0.], [.5, .5, 0.]], dtype=dtype)
    cloth.keypoint_rest = unit[None] * dims[:, None]
    axes = torch.nn.functional.normalize(torch.tensor([[.3, .4, .9], [-.2, .8, .5]], dtype=dtype), dim=-1)
    global_rotation = rotations.to_matrix(rotations.from_angle_axis(torch.tensor([.9, -1.2], dtype=dtype), axes))
    translation = torch.tensor([[1., 2., 3.], [-.7, .1, .6]], dtype=dtype)
    angle = torch.full((2,), angle_deg*torch.pi/180, dtype=dtype)
    fold_rotation = rotations.to_matrix(rotations.from_angle_axis(angle, torch.tensor([0., -1., 0.], dtype=dtype).expand(2, -1)))
    anchor = torch.zeros(2, 3, dtype=dtype)
    anchor[:, 2] = dims[:, 2]/2

    def world(points):
        return torch.einsum('nij,nmj->nmi', global_rotation, points)+translation[:, None]

    def folded(points):
        return torch.einsum('nij,nmj->nmi', fold_rotation, points-anchor[:, None])+anchor[:, None]

    current = world(folded(cloth.keypoint_rest))
    points = world(cloth.rest)
    points[:, cloth.moving_idx] = world(folded(cloth.rest[:, cloth.moving_idx]))
    center = current.mean(dim=1)
    omega = torch.tensor([[.2, -.7, .3], [-.4, .2, .9]], dtype=dtype)
    linear = torch.tensor([[.03, -.12, .2], [.8, .05, -.02]], dtype=dtype)
    velocities = linear[:, None]+torch.cross(omega[:, None].expand_as(points), points-center[:, None], dim=-1)
    state = NS(keypoints_w=current, points_w=points, points_vel_w=velocities)
    view = NS(frame_r=global_rotation, frame_c=world(cloth.st_rest_centroid[:, None])[:, 0])
    mask = torch.full((2,), unfold, dtype=torch.bool)
    return cloth, state, view, mask, center, global_rotation@fold_rotation, linear, omega


@pytest.mark.parametrize('angle', [0., 45., 90., 180.])
@pytest.mark.parametrize('unfold', [False, True])
def test_physical_pose_and_velocity_under_arbitrary_world_transform(angle, unfold):
    cloth, state, view, mask, center, rotation, linear, angular = _scene(angle, unfold)
    result = moving_half_pose(cloth, state, view, mask)
    torch.testing.assert_close(result.pos_w, center, atol=1e-12, rtol=0)
    torch.testing.assert_close(rotations.to_matrix(result.quat_w), rotation, atol=1e-12, rtol=0)
    torch.testing.assert_close(result.lin_vel_w, linear, atol=2e-7, rtol=0)
    torch.testing.assert_close(result.ang_vel_w, angular, atol=2e-7, rtol=0)
    expected_goal_rotation = view.frame_r if unfold else view.frame_r * view.frame_r.new_tensor([-1., 1., -1.])
    torch.testing.assert_close(rotations.to_matrix(result.goal_quat_w), expected_goal_rotation, atol=1e-12, rtol=0)
    assert torch.all(torch.linalg.det(rotations.to_matrix(result.quat_w)) > .999999)
    # This explicitly rejects incorrectly forcing the current normal to face upward.
    normal_dot = (rotations.to_matrix(result.quat_w)[:, :, 2]*view.frame_r[:, :, 2]).sum(-1)
    torch.testing.assert_close(normal_dot, torch.full_like(normal_dot, torch.cos(torch.tensor(angle*torch.pi/180)).item()),
                               atol=1e-7, rtol=0)
    at_goal = angle == (0. if unfold else 180.)
    if at_goal:
        torch.testing.assert_close(result.goal_pos_w, result.pos_w, atol=1e-12, rtol=0)
        # A nonplanar object marker set distinguishes a proper half-turn from reflection.
        offsets = torch.tensor([[1., 1., 1.], [-1., -1., 1.], [1., -1., -1.], [-1., 1., -1.]], dtype=center.dtype)
        offsets = offsets[None]*cloth.dims.to(center.dtype)[:, None]/2
        current_markers = torch.einsum('nij,nmj->nmi', rotations.to_matrix(result.quat_w), offsets)+result.pos_w[:, None]
        goal_markers = torch.einsum('nij,nmj->nmi', rotations.to_matrix(result.goal_quat_w), offsets)+result.goal_pos_w[:, None]
        torch.testing.assert_close(current_markers, goal_markers, atol=1e-12, rtol=0)


def test_mixed_goal_modes_are_applied_per_environment():
    cloth, state, view, _, *_ = _scene(90.)
    mixed = moving_half_pose(cloth, state, view, torch.tensor([False, True]))
    folded = moving_half_pose(cloth, state, view, torch.zeros(2, dtype=torch.bool))
    flat = moving_half_pose(cloth, state, view, torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(mixed.goal_pos_w[0], folded.goal_pos_w[0])
    torch.testing.assert_close(mixed.goal_quat_w[1], flat.goal_quat_w[1])


@pytest.mark.parametrize('invalid', [False, True])
def test_degenerate_or_nonfinite_current_geometry_has_finite_fallback(invalid):
    cloth, state, view, mask, *_ = _scene(90.)
    state.keypoints_w[:] = float('nan') if invalid else 2.
    state.points_w[:] = float('nan') if invalid else 2.
    state.points_vel_w[:] = float('nan') if invalid else 0.
    result = moving_half_pose(cloth, state, view, mask)
    for value in vars(result).values():
        assert torch.isfinite(value).all()
    torch.testing.assert_close(result.quat_w, torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64).expand(2, -1))
    assert torch.count_nonzero(result.ang_vel_w) == 0


@pytest.mark.parametrize('angle', [0., 45., 90., 180.])
def test_grid_only_fit_matches_explicit_pose_without_sample_centroid_bias(angle):
    cloth, state, view, mask, *_ = _scene(angle)
    explicit = moving_half_pose(cloth, state, view, mask)
    # A seven-row grid has mean positive x=L/3; the represented half center is L/4.
    assert not torch.allclose(cloth.rest[:, cloth.moving_idx].mean(dim=1)[:, 0], cloth.dims[:, 0]/4)
    cloth.keypoint_rest = None
    state.keypoints_w = None
    sampled = moving_half_pose(cloth, state, view, mask)
    for name in vars(explicit):
        if 'quat' in name:
            torch.testing.assert_close(rotations.to_matrix(getattr(sampled, name)),
                                       rotations.to_matrix(getattr(explicit, name)), atol=1e-12, rtol=0)
        else:
            torch.testing.assert_close(getattr(sampled, name), getattr(explicit, name), atol=1e-12, rtol=0)
