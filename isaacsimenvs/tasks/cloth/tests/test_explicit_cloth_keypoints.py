"""Rigid landmarks represent the exact moving half independently of surface-grid resolution."""

from types import SimpleNamespace as NS

import pytest
import torch

from isaacsimenvs.tasks.cloth.backends.base import ClothBackend
from isaacsimenvs.tasks.cloth.mdp.geometry import fold, grid
from isaacsimenvs.tasks.cloth.mdp.state import SimState
from isaacsimenvs.tasks.cloth.mdp.task import ClothTask


def _geometry(resolution):
    dims = torch.tensor([[.2, .2, .004], [.32, .12, .009]])
    cloth = grid.make_grid(dims, resolution, dims[:, 2])
    unit = torch.tensor([[0., -.5, 0.], [0., .5, 0.], [.5, -.5, 0.], [.5, .5, 0.]])
    cloth.keypoint_rest = unit[None] * dims[:, None]
    # Full 3-D rotations and different translations exercise local-to-world targets.
    angles = torch.tensor([.7, -1.1])
    c, s = angles.cos(), angles.sin()
    rotations = torch.zeros(2, 3, 3)
    rotations[:, 0, 0], rotations[:, 0, 2] = c, s
    rotations[:, 1, 1] = 1
    rotations[:, 2, 0], rotations[:, 2, 2] = -s, c
    translation = torch.tensor([[1., 2., 3.], [-4., .8, .6]])
    return cloth, rotations, translation


def _world(points, rotation, translation):
    return torch.einsum('nij,nmj->nmi', rotation, points) + translation[:, None]


@pytest.mark.parametrize('resolution', [5, 7, 10])
def test_explicit_targets_and_task_use_exact_corners_under_global_transform(resolution):
    cloth, rotation, translation = _geometry(resolution)
    unfold = torch.tensor([False, True])
    rest = cloth.keypoint_rest.clone()
    rest[0, :, 0] *= -1
    rest[0, :, 2] += cloth.ply_gap[0]
    expected = _world(rest, rotation, translation)
    centroid = _world(cloth.st_rest_centroid[:, None], rotation, translation)[:, 0]
    original = cloth.keypoint_rest.clone()
    torch.testing.assert_close(fold.corner_targets(cloth, rotation, centroid, unfold), expected)
    torch.testing.assert_close(cloth.keypoint_rest, original)

    points = _world(grid.folded_start(cloth, torch.tensor([torch.pi, 0.])), rotation, translation)
    state = SimState(
        joint_pos=torch.zeros(2, 27), joint_vel=torch.zeros(2, 27),
        palm_pos_w=torch.zeros(2, 3), palm_quat_w=torch.tensor([[1., 0., 0., 0.]]).expand(2, -1),
        palm_vel_w=torch.zeros(2, 6), fingertip_pos_w=torch.zeros(2, 5, 3),
        fingertip_quat_w=torch.tensor([1., 0., 0., 0.]).expand(2, 5, 4), hand_pos_w=torch.zeros(2, 6, 3),
        points_w=points, points_vel_w=torch.zeros_like(points), table_top_z=torch.zeros(2), keypoints_w=expected,
    )
    task = ClothTask.__new__(ClothTask)
    task._cloth_grid, task._unfold_env = cloth, unfold
    task.cfg = NS(reward=NS(fingertip_distance_from_pad=False))
    view = task.derive(state)
    assert view.corners_w is state.keypoints_w
    torch.testing.assert_close(view.targets_w, expected, atol=2e-6, rtol=0)
    assert view.fold_error.max() < 2e-6
    # Removing explicit samples restores the legacy surface-grid corner readout.
    state.keypoints_w = None
    torch.testing.assert_close(task.derive(state).corners_w, points[:, cloth.corner_idx])


def test_legacy_grid_targets_are_unchanged():
    cloth, rotation, translation = _geometry(7)
    cloth.keypoint_rest = None
    centroid = _world(cloth.st_rest_centroid[:, None], rotation, translation)[:, 0]
    expected = _world(cloth.folded_rest(cloth.corner_idx), rotation, translation)
    torch.testing.assert_close(fold.corner_targets(cloth, rotation, centroid, torch.zeros(2, dtype=torch.bool)), expected)


def test_policy_object_size_is_the_moving_half_for_both_explicit_and_legacy_grids():
    from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg

    cfg = ClothCfg()
    cfg.reward.object_base_size = .04
    cloth, _, _ = _geometry(7)
    task = ClothTask(cfg, cloth, torch.zeros(2, 3), list(range(7)), list(range(7, 27)))
    torch.testing.assert_close(task._object_scales[0], torch.tensor([.1/.04, .2/.04, .004/.04]))
    cloth.keypoint_rest = None
    legacy = ClothTask(cfg, cloth, torch.zeros(2, 3), list(range(7)), list(range(7, 27)))
    torch.testing.assert_close(legacy._object_scales, task._object_scales)


def test_backend_read_state_forwards_optional_keypoints():
    backend = ClothBackend(NS())
    bodies = torch.zeros(1, 6, 13)
    bodies[..., 3] = 1
    backend.robot = NS(data=NS(body_state_w=bodies, joint_pos=torch.zeros(1, 27), joint_vel=torch.zeros(1, 27)))
    backend._palm_id, backend._fingertip_ids, backend._hand_body_ids = 0, list(range(1, 6)), list(range(6))
    backend.scene = NS(env_origins=torch.zeros(1, 3))
    backend._table_z, backend._table_half_height = torch.tensor([.38]), .15
    backend.cloth_points = lambda: (torch.zeros(1, 49, 3), torch.zeros(1, 49, 3))
    assert backend.read_state().keypoints_w is None
    expected = torch.arange(12).reshape(1, 4, 3).float()
    backend.cloth_keypoints = lambda: expected
    assert backend.read_state().keypoints_w is expected
