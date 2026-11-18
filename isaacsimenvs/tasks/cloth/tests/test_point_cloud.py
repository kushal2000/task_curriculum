"""The top-face point cloud: samples stay on their plates, goals are the exactly folded cloth."""

import math
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from isaacsimenvs.tasks.cloth.backends.rigid import designs
from isaacsimenvs.tasks.cloth.mdp import observations
from isaacsimenvs.tasks.cloth.mdp.geometry import point_cloud, rotations
from isaacsimenvs.tasks.cloth.mdp.task import ClothTask

from .test_rigid_backend_readout import _backend, _capture_reset_state

P = 64


def _with_surface(names, pool_size=512, num_points=P):
    backend = _backend(names)
    backend.cfg.obs = NS(num_points=num_points, point_pool_size=pool_size, point_seed=3)
    backend._init_surface()
    for group in backend.groups:
        _capture_reset_state(group.articulation)
    n = backend.num_envs
    backend.scene = NS(env_origins=torch.arange(n)[:, None] * torch.tensor([[2., 3., 0.]]))
    backend._table_half_height = .15
    backend._table_z = .38 + torch.arange(n) * .01
    ids = torch.arange(n)
    yaw = torch.linspace(-1., 1., n)
    backend._write_cloth(ids, torch.zeros(n, 2), yaw, torch.zeros(n))
    return backend, yaw


def _rotate_about_hinge(rest, gap):
    """Reference fold: 180 deg about the y axis through (0, 0, gap / 2)."""
    q = rotations.from_angle_axis(torch.tensor(math.pi), torch.tensor([0., 1., 0.]))
    axis = torch.tensor([0., 0., 1.]) * gap / 2
    return axis + rotations.apply(q.expand(*rest.shape[:-1], 4), rest - axis)


def test_pool_is_seeded_and_inside_the_unit_square():
    a, b = point_cloud.sample_pool(1000, 7), point_cloud.sample_pool(1000, 7)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, point_cloud.sample_pool(1000, 8))
    assert np.all(np.abs(a[:, :2]) < .5) and np.all(a[:, 2] == 0)
    # Area-uniform: each quadrant holds about a quarter.
    quadrants = np.bincount((a[:, 0] > 0) * 2 + (a[:, 1] > 0), minlength=4)
    assert np.all(np.abs(quadrants - 250) < 60)


def test_draws_are_distinct_within_an_env():
    idx = point_cloud.draw(300, 128, P, "cpu", chunk=64)
    assert idx.shape == (300, P)
    assert all(len(set(row.tolist())) == P for row in idx)
    with pytest.raises(ValueError):
        point_cloud.draw(1, 10, 11, "cpu")


def test_goal_is_the_moving_half_turned_about_the_top_face_hinge():
    t = torch.tensor([.002, .02])
    rest = point_cloud.top_face_rest(torch.rand(2, 50, 3) - .5, torch.tensor([[.2, .15, .002], [.1, .3, .02]]))
    moving = rest[..., 0] > 0
    folded = point_cloud.goal_rest(rest, moving, t, torch.tensor([False, False]))
    expected = torch.where(moving.unsqueeze(-1), _rotate_about_hinge(rest, t.view(-1, 1, 1)), rest)
    torch.testing.assert_close(folded, expected)
    # Top face lands face down on the stationary top face: one height for every sample.
    torch.testing.assert_close(folded[..., 2], (t / 2).view(-1, 1).expand(2, 50))
    flat = point_cloud.goal_rest(rest, moving, t, torch.tensor([True, False]))
    torch.testing.assert_close(flat[0], rest[0])
    torch.testing.assert_close(flat[1], folded[1])


def test_distances_zero_at_the_goal_and_chamfer_ignores_order():
    pts = torch.rand(3, P, 3)
    assert torch.all(point_cloud.matched_distance(pts, pts) == 0)
    shuffled = pts[:, torch.randperm(P)]
    assert torch.all(point_cloud.matched_distance(pts, shuffled) > 0)
    torch.testing.assert_close(point_cloud.chamfer_distance(pts, shuffled), torch.zeros(3), atol=1e-4, rtol=0)
    shifted = pts + torch.tensor([0., 0., .01])
    torch.testing.assert_close(point_cloud.matched_distance(pts, shifted), torch.full((3,), .01))


@pytest.mark.parametrize("names", [("single_hinge",), ("single_hinge", "strip_4", "strip_8", "hinge_band")])
def test_samples_sit_on_the_top_face_of_their_owning_plates(names):
    backend, _ = _with_surface(names)
    sf = backend.surface
    dims = backend._dims
    torch.testing.assert_close(sf.rest[..., 2], (dims[:, 2:3] / 2).expand(-1, P))
    assert torch.all(sf.rest[..., 0].abs() <= dims[:, :1] / 2) and torch.all(sf.rest[..., 1].abs() <= dims[:, 1:2] / 2)
    points = backend.cloth_surface_points()
    for env_id in range(backend.num_envs):
        group = backend.groups[int(backend._env_group[env_id])]
        owners = designs.point_owners(group.design, (sf.rest[env_id] / dims[env_id]).numpy() * [1, 1, 0])
        body = torch.as_tensor(group.body_ids)[owners]
        torch.testing.assert_close(backend._surface_body[env_id], body)
        # Fixture: body b of env e sits at (10 e + b, 2 e, 0), identity orientation.
        offset = torch.stack([10. * env_id + body, torch.full((P,), 2. * env_id), torch.zeros(P)], dim=1)
        torch.testing.assert_close(points[env_id], sf.rest[env_id] + offset)
        moving = torch.as_tensor([group.design.plates[i].center[0] > 0 for i in owners])
        torch.testing.assert_close(sf.moving[env_id], moving)


def test_samples_follow_a_rotating_plate():
    backend, _ = _with_surface(("single_hinge",))
    group = backend.groups[0]
    data = group.articulation.data
    q = rotations.from_angle_axis(torch.tensor(.7), torch.tensor([0., 1., 0.]))
    data.body_link_quat_w[:] = q
    points = backend.cloth_surface_points()
    pos = data.body_link_pos_w.gather(1, backend._surface_body[group.env_ids].unsqueeze(-1).expand(-1, -1, 3))
    expected = pos + rotations.apply(q.expand(len(group.env_ids), P, 4), backend.surface.rest[group.env_ids])
    torch.testing.assert_close(points[group.env_ids], expected)


def test_placed_pose_lies_flat_on_the_table_and_goals_follow_it():
    backend, yaw = _with_surface(("single_hinge", "strip_4"))
    sf, n = backend.surface, backend.num_envs
    table_top = backend.scene.env_origins[:, 2] + backend._table_z + .15
    torch.testing.assert_close(sf.placed_pos_w[:, 2], table_top + backend._dims[:, 2] / 2)
    torch.testing.assert_close(sf.placed_quat_w, rotations.yaw(yaw))

    task = ClothTask.__new__(ClothTask)
    task._surface, task._cloth_grid = sf, NS(ply_gap=backend._dims[:, 2].clone())
    task._unfold_env = torch.arange(n) % 3 == 0
    goals = task.point_goals(torch.arange(n))
    rest = point_cloud.goal_rest(sf.rest, sf.moving, backend._dims[:, 2], task._unfold_env)
    torch.testing.assert_close(goals, point_cloud.place(rest, sf.placed_pos_w, sf.placed_quat_w))
    # Every goal sample is on the top of the (one- or two-ply) stack on the table.
    t = backend._dims[:, 2:3]
    torch.testing.assert_close(goals[..., 2], (table_top.unsqueeze(1) + t).expand(-1, P))
    # Fold goals are reflected across the crease; unfold goals are the flat sheet.
    local = rotations.apply(rotations.yaw(-yaw).unsqueeze(1).expand(-1, P, 4), goals - sf.placed_pos_w.unsqueeze(1))
    flip = sf.moving & ~task._unfold_env.unsqueeze(1)
    torch.testing.assert_close(local[..., 0], torch.where(flip, -sf.rest[..., 0], sf.rest[..., 0]))


def test_resets_redraw_only_the_selected_envs():
    backend, yaw = _with_surface(("single_hinge", "strip_8"))
    before = backend.surface.rest.clone()
    ids = torch.tensor([3, 0])
    backend._write_cloth(ids, torch.zeros(2, 2), yaw[ids], torch.zeros(2))
    changed = (backend.surface.rest != before).any(dim=-1).any(dim=-1)
    assert changed[ids].all() and not changed[[1, 2]].any()


def test_point_fields_size_with_num_points():
    base = observations.obs_dim(("joint_pos",), 10)
    assert observations.obs_dim(("joint_pos", "cloth_points", "goal_points"), 10, P) == base + 6 * P
    with pytest.raises(ValueError):
        observations.obs_dim(("cloth_points",), 10)
