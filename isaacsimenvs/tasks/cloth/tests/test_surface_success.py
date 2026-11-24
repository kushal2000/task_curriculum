"""``ClothTask`` with ``termination.success: surface``: the frozen goal, the aligned and in-place
distances, the in-place gate and its curriculum, and the stationary half's drift (logged).

Simulator-free: a nominal 7x7 sheet on the task's own geometry, moved by hand; the designated goal
sheet (``goal_now_w``, which needs the pool's goal tables) is given directly.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from isaacsimenvs.tasks.cloth.mdp.geometry.grid import grid_mesh, keypoint_indices
from isaacsimenvs.tasks.cloth.mdp.state import SheetLayout
from isaacsimenvs.tasks.cloth.mdp.task import ClothTask

N, RES, SIZE = 3, 7, 0.10


@pytest.fixture(scope="module")
def cfg():
    from isaacsimenvs.tasks.cloth.config import vbd_cloth_env_cfg

    c = vbd_cloth_env_cfg()
    c.termination.success = "surface"
    return c


def _flat():
    verts, _ = grid_mesh(SIZE, RES)
    return torch.tensor(verts, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1) + torch.tensor([0.0, 0.0, 0.40])


def _fold(pts, sign: float):
    """The +x half over onto -x (sign -1) or the -x half over onto +x (sign +1); ply gap 0."""
    x = pts[..., 0]
    folded = torch.where(x * sign < 0, -x, x)
    return torch.stack((folded, pts[..., 1], pts[..., 2]), dim=-1)


def _rotz(p, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return torch.stack((c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1], p[..., 2]), dim=-1)


def _task(cfg, goal, fold_axis: str = "x") -> ClothTask:
    verts, _ = grid_mesh(SIZE, RES)
    layout = SheetLayout(
        rest_local=verts, keypoint_idx=keypoint_indices(RES, fold_axis, 4), resolution=RES, fold_axis=fold_axis,
        size=SIZE, fold_lift=0.002, object_scale=torch.ones(N, 3),
    )
    task = ClothTask(
        cfg, layout, SimpleNamespace(), num_envs=N, device="cpu", env_origins=torch.zeros(N, 3), extras={},
        observed_goal_quat=lambda r: torch.zeros(r.shape[0], 4),
    )
    task.goal_now_w = lambda view: goal.clone()
    return task


def _view(points):
    return SimpleNamespace(points_w=points)


def _steps(k):
    return torch.full((N,), k)


def test_defaults_are_the_old_criterion():
    from isaacsimenvs.tasks.cloth.config import vbd_cloth_env_cfg

    c = vbd_cloth_env_cfg()
    assert c.termination.success == "fold_error"
    assert c.termination.max_in_place_dist == float("inf") and c.reward.slide_penalty_scale == 0.0


def test_nothing_counts_before_the_goal_freezes(cfg):
    goal = _fold(_flat(), -1)
    task = _task(cfg, goal)
    task._update_surface(_view(goal), _steps(int(cfg.termination.surface.ref_step) - 1))
    assert not bool(task.is_folded(_view(goal)).any())
    task._update_surface(_view(goal), _steps(int(cfg.termination.surface.ref_step)))
    assert bool(task.is_folded(_view(goal)).all())


def test_mirror_fold_elsewhere_counts_and_the_flat_sheet_does_not(cfg):
    flat = _flat()
    goal = _fold(flat, -1)
    task = _task(cfg, goal)
    ref = int(cfg.termination.surface.ref_step)
    task._update_surface(_view(flat), _steps(ref))                 # the goal freezes here
    assert not bool(task.is_folded(_view(flat)).any())
    other = _rotz(_fold(flat, +1), 90.0) + torch.tensor([0.05, -0.04, 0.0])
    task._update_surface(_view(other), _steps(ref + 1))
    assert bool(task.is_folded(_view(other)).all())
    assert torch.all(task.in_place_dist > 0.04)                 # but not in place
    # The goal stays frozen: moving the live goal now changes nothing.
    task.goal_now_w = lambda view: goal + 1.0
    task._update_surface(_view(other), _steps(ref + 2))
    assert bool(task.is_folded(_view(other)).all())


def test_lifted_fold_does_not_count(cfg):
    goal = _fold(_flat(), -1)
    task = _task(cfg, goal)
    task._update_surface(_view(goal + torch.tensor([0.0, 0.0, 0.03])), _steps(int(cfg.termination.surface.ref_step)))
    assert not bool(task.is_folded(_view(goal)).any())


def test_in_place_gate(cfg):
    c = copy.deepcopy(cfg)
    c.termination.max_in_place_dist = 0.02
    flat = _flat()
    goal = _fold(flat, -1)
    task = _task(c, goal)
    ref = int(c.termination.surface.ref_step)
    here = goal + torch.tensor([0.01, 0.0, 0.0])
    there = _fold(flat, +1)
    task._update_surface(_view(here), _steps(ref))
    assert bool(task.is_folded(_view(here)).all())
    task._update_surface(_view(there), _steps(ref + 1))
    assert not bool(task.is_folded(_view(there)).any())


def test_gate_curriculum_advances_on_the_window_rate(cfg):
    c = copy.deepcopy(cfg)
    gc = c.termination.gate_curriculum
    gc.max_in_place_dist, gc.slide_penalty_scale = [0.12, 0.06, 0.02], [0.0, 500.0, 1000.0]
    gc.advance_rate, gc.window_episodes, gc.start_stage = 0.5, 6, 0
    task = _task(c, _flat())
    assert task.gate() == (0.12, 0.0)
    done = torch.ones(N, dtype=torch.bool)
    task._advance_gate(done, torch.tensor([True, False, False]))
    task._advance_gate(done, torch.tensor([True, False, False]))
    assert task.gate_stage == 0 and abs(task.gate_last_rate - 2 / 6) < 1e-9
    task._advance_gate(done, torch.tensor([True, True, False]))
    task._advance_gate(done, torch.tensor([True, True, False]))
    assert task.gate_stage == 1 and task.gate() == (0.06, 500.0)
    for _ in range(6):
        task._advance_gate(done, torch.ones(N, dtype=torch.bool))
    assert task.gate_stage == 2 and task.extras["gate_max_in_place_dist"] == 0.02


def test_config_rejects_bad_gate(cfg):
    c = copy.deepcopy(cfg)
    c.termination.gate_curriculum.max_in_place_dist = [0.1, 0.05]
    c.termination.gate_curriculum.slide_penalty_scale = [0.0]
    with pytest.raises(ValueError):
        c.termination.check()
    c = copy.deepcopy(cfg)
    c.termination.success = "fold_error"
    c.termination.max_in_place_dist = 0.05
    with pytest.raises(ValueError):
        c.termination.check()


def test_drift_translation_and_yaw(cfg):
    task = _task(cfg, _flat())
    settle = int(cfg.termination.drift_settle_steps)
    pts = _flat()
    task._update_drift(_view(pts), _steps(settle - 1))
    assert torch.all(task.drift == 0)
    task._update_drift(_view(pts), _steps(settle))                 # the latch
    task._update_drift(_view(pts + torch.tensor([0.03, 0.04, 0.0])), _steps(settle + 1))
    assert torch.allclose(task.drift, torch.full((N,), 0.05), atol=1e-6)
    assert torch.allclose(task.slide_step, torch.full((N,), 0.05), atol=1e-6)
    task._update_drift(_view(_rotz(pts, 20.0)), _steps(settle + 2))
    assert torch.allclose(task.yaw, torch.full((N,), math.radians(20.0)), atol=1e-5)
    task.reset(torch.tensor([1]))
    assert not bool(task.drift_anchored[1]) and float(task.drift_max[1]) == 0.0


def _top_ply_bunched(goal):
    """The goal with its top ply (the +x rows it folded over, ``fold_axis`` x) shrunk to half its size
    about its own centre, in place: still inside the bottom ply's outline."""
    sheet = goal.clone()
    top = (torch.arange(RES) > RES // 2).view(RES, 1).expand(RES, RES).reshape(-1)
    centre = sheet[:, top, :2].mean(dim=1, keepdim=True)
    sheet[:, top, :2] = centre + 0.5 * (sheet[:, top, :2] - centre)
    return sheet


def _lifted_fold(flat, gap=0.003):
    """The +x half over onto -x, ``gap`` above the -x half (``_fold`` has no ply gap)."""
    folded = _fold(flat, -1)
    moved = flat[..., 0] > 1e-6
    folded[..., 2] = torch.where(moved, folded[..., 2] + gap, folded[..., 2])
    return folded


def test_layered_is_the_default_and_fails_a_misaligned_top_ply(cfg):
    assert cfg.termination.surface.layered
    goal = _lifted_fold(_flat())
    task = _task(cfg, goal)
    ref = int(cfg.termination.surface.ref_step)
    task._update_surface(_view(goal), _steps(ref))
    assert bool(task.is_folded(_view(goal)).all()) and torch.all(task.layered_dist < 1e-3)
    bunched = _top_ply_bunched(goal)
    task._update_surface(_view(bunched), _steps(ref + 1))
    tolerance = task.surface_tolerance()
    assert torch.all(task.aligned_dist < tolerance)               # unlabelled: within the tolerance
    assert torch.all(task.layered_dist > tolerance)               # layered: the top ply is 1.25 cm short
    assert not bool(task.is_folded(_view(bunched)).any())


def test_unlabelled_when_layered_is_off(cfg):
    c = copy.deepcopy(cfg)
    c.termination.surface.layered = False
    goal = _lifted_fold(_flat())
    task = _task(c, goal)
    ref = int(c.termination.surface.ref_step)
    task._update_surface(_view(goal), _steps(ref))
    bunched = _top_ply_bunched(goal)
    task._update_surface(_view(bunched), _steps(ref + 1))
    assert bool(task.is_folded(_view(bunched)).all())
    assert torch.all(torch.isinf(task.layered_dist))


def test_layered_fold_axis_y(cfg):
    """A sheet folding along y: the goal's top ply is the columns right of the middle one."""
    flat = _flat()
    goal = _rotz(_lifted_fold(_rotz(flat, -90.0)), 90.0)          # the +y half over onto -y
    task = _task(cfg, goal, fold_axis="y")
    ref = int(cfg.termination.surface.ref_step)
    task._update_surface(_view(goal), _steps(ref))
    assert torch.all(task.layered_dist < 1e-3)

