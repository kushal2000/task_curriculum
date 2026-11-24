"""One-fold goal tasks (mdp/goal_tasks.py) and the fold they make FoldGeometry / ClothTask score
(mdp/fold.py, mdp/task.py): CPU."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from isaacsimenvs.tasks.cloth.mdp.fold import FoldGeometry
from isaacsimenvs.tasks.cloth.mdp.geometry import fold_shapes as fs
from isaacsimenvs.tasks.cloth.mdp.geometry.grid import corner_indices, grid_mesh
from isaacsimenvs.tasks.cloth.mdp.goal_tasks import GoalTasks
from isaacsimenvs.tasks.cloth.mdp.state import SheetLayout
from isaacsimenvs.tasks.cloth.mdp.task import ClothTask

SHAPES = ["box2-mid", "box2diag4-mid", "box2x8-mid", "cross4-mid", "corners8-mid", "box4-mid", "star8-mid"]
RES = 7
TOL = 0.015


def _shapes():
    return [fs.shape(n) for n in SHAPES]


def _tables(mode: str, **kw) -> GoalTasks:
    shapes = _shapes()
    joints = [[f"hinge_{h.child}" for h in reversed(sh.hinges)] for sh in shapes]  # a shuffled sim order
    return GoalTasks(shapes, RES, mode, joints, "hinge_", "cpu", **kw)


def _yaw_rot(yaw: torch.Tensor) -> torch.Tensor:
    rot = torch.zeros(yaw.shape[0], 3, 3)
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = yaw.cos(), -yaw.sin(), yaw.sin(), yaw.cos(), 1.0
    return rot


@pytest.mark.parametrize("mode", ["half_fold", "any_fold"])
def test_goal_error(mode):
    gt = _tables(mode)
    shapes = _shapes()
    t = gt.num_tasks
    assert t == (len(SHAPES) if mode == "half_fold" else 209)
    task = torch.arange(t)
    s3 = torch.tensor([1.5, 1.5, 1.0]).view(1, 1, 3).expand(t, 1, 3)
    rot = _yaw_rot(torch.rand(t) * 2.0 * math.pi)
    place = lambda g: torch.einsum("nij,npj->npi", rot, g * s3) + torch.tensor([0.3, -0.2, 0.5])  # noqa: E731
    assert float(gt.error(task, s3, place(gt.p_goal)).max()) < 1e-5          # at the goal, wherever it lies
    assert float(gt.error(task, s3, place(gt.p_start)).min()) > 0.02         # the start is not the goal
    # the hinge joints land in the simulator's (shuffled) order
    for i in range(t):
        n_j = len(shapes[int(gt.task_shape[i])].hinges)
        if n_j < gt.q_start.shape[1]:
            assert float(gt.q_start[i, n_j:].abs().max()) == 0.0
    has = gt.m_ok[:, 1]
    if bool(has.any()):
        t2 = task[has]
        parts = torch.einsum("nij,npj->npi", rot[has], gt.m_goal[t2, 1] * s3[has]) + 1.0
        member, ok = gt.nearest_alternative(t2, gt.m_q[t2, 1])
        assert bool(ok.all()) and bool((member == 1).all())
        assert float(gt.error(t2, s3[has], parts).min()) > 0.005             # not the designated goal
        assert float(gt.error(t2, s3[has], parts, member).max()) < 1e-5      # but a look-alike
    shape_ids = torch.arange(len(shapes)).repeat(100)
    draw = gt.draw(shape_ids)
    assert bool((gt.task_shape[draw] == shape_ids).all())
    travel = torch.rad2deg(gt.sector_yaw(draw, 90.0, 180.0, 10.0) + gt.direction[draw]) % 360.0
    assert float(travel.min()) >= 145.0 - 1e-3 and float(travel.max()) <= 215.0 + 1e-3


def test_unknown_mode():
    with pytest.raises(ValueError):
        _tables("half")


def _geometry(gt: GoalTasks, scale: torch.Tensor) -> FoldGeometry:
    """FoldGeometry on the pool's grid at per-env sizes ``scale`` (as the rigid backend lays it out)."""
    size = fs.shape(SHAPES[0]).size
    verts, _ = grid_mesh(size, RES)
    n = scale.shape[0]
    layout = SheetLayout(
        rest_local=verts, keypoint_idx=corner_indices(RES, "x"), resolution=RES, fold_axis="x", size=size,
        fold_lift=0.0, object_scale=torch.ones(n, 3), xy_scale=scale, env_size=scale * size, goal_tasks=gt,
    )
    return FoldGeometry(layout, num_envs=n, device="cpu", observed_goal_quat=lambda r: r)


def _task(geometry: FoldGeometry):
    """Just what ``ClothTask.is_folded_designated`` reads."""
    cloth = SimpleNamespace(keypoint_tolerance=TOL, max_folded_footprint=0.75)
    return SimpleNamespace(cfg=SimpleNamespace(cloth=cloth), geometry=geometry)


def test_any_fold_view():
    gt = _tables("any_fold", accept_lookalike=True)
    t = gt.num_tasks
    scale = torch.rand(t) * 0.8 + 0.7
    geo = _geometry(gt, scale)
    s3 = geo._scale3()
    rot = _yaw_rot(torch.rand(t) * 2.0 * math.pi)
    place = lambda g: torch.einsum("nij,npj->npi", rot, g * s3) + torch.tensor([0.3, -0.2, 0.5])  # noqa: E731

    # Before the backend's first draw: the stage-1 fold, no goal-task fields.
    view = geo.derive(place(gt.p_start), torch.zeros(t, RES * RES, 3))
    assert view.designated_error is None

    gt.task = torch.arange(t)
    gt.read_hinge_abs = lambda: gt.q_start[gt.task].abs()        # the sheet sits in its start state
    start = geo.derive(place(gt.p_start), torch.zeros(t, RES * RES, 3))
    # The tracked points are the task's, on the piece its fold moves; their offsets are its start state's.
    kp = gt.kp[gt.task]
    assert bool(gt.moving[gt.task].gather(1, kp).all())
    assert torch.allclose(start.keypoints_w, torch.gather(place(gt.p_start), 1, kp.unsqueeze(-1).expand(-1, -1, 3)))
    off = torch.einsum("nij,nkj->nki", rot, geo.keypoint_offsets)
    assert torch.allclose(start.keypoints_w - start.keypoints_w.mean(dim=1, keepdim=True), off, atol=1e-5)
    assert float(start.designated_error.min()) > 0.02
    assert not bool(ClothTask.is_folded_designated(_task(geo), start).any())

    gt.read_hinge_abs = lambda: gt.m_q[gt.task, 0]               # the sheet sits in its goal state
    goal = geo.derive(place(gt.p_goal), torch.zeros(t, RES * RES, 3))
    assert float(goal.fold_error.max()) < 1e-5
    assert float((goal.keypoints_w - goal.targets_w).norm(dim=-1).max()) < 1e-5
    assert bool(ClothTask.is_folded_designated(_task(geo), goal).all())
    # The goal attitude: the fold's rotation on the live frame takes the start offsets onto the goal's.
    want = goal.targets_w - goal.targets_w.mean(dim=1, keepdim=True)
    got = torch.einsum("nij,nkj->nki", geo.goal_rotation(goal.frame_r), geo.keypoint_offsets)
    assert torch.allclose(got, want, atol=1e-4)


def test_half_fold_lookalike():
    gt = _tables("half_fold", accept_lookalike=True)
    has = gt.m_ok[:, 1]
    assert bool(has.any())                                       # cross4 / box4: another fold looks the same
    t2 = torch.arange(gt.num_tasks)[has]
    n = t2.shape[0]
    geo = _geometry(gt, torch.ones(n))
    gt.task = t2
    gt.read_hinge_abs = lambda: gt.m_q[gt.task, 1]
    s3 = geo._scale3()
    other = geo.derive(gt.m_goal[t2, 1] * s3, torch.zeros(n, RES * RES, 3))
    assert float(other.designated_error.min()) > 0.02            # not the designated half fold
    assert float(other.lookalike_error.max()) < 1e-5
    assert bool(ClothTask.is_folded_designated(_task(geo), other).all())
    # the designated half fold itself: the stage-1 construction, keypoints on the +x half's corners
    gt.read_hinge_abs = lambda: gt.m_q[gt.task, 0]
    own = geo.derive(gt.p_goal[t2] * s3, torch.zeros(n, RES * RES, 3))
    assert float(own.designated_error.max()) < 1e-5
    assert bool(ClothTask.is_folded_designated(_task(geo), own).all())
