"""The goal sheet without goal tasks (``FoldGeometry.goal_sheet``'s stage-1 half fold, the VBD goal)
against the one ``half_fold`` goal tasks give the rigid pool: CPU."""

from __future__ import annotations

import math

import torch

from isaacsimenvs.tasks.cloth.mdp.fold import FoldGeometry
from isaacsimenvs.tasks.cloth.mdp.geometry import fold_shapes as fs
from isaacsimenvs.tasks.cloth.mdp.geometry.grid import corner_indices, grid_mesh
from isaacsimenvs.tasks.cloth.mdp.state import SheetLayout
from isaacsimenvs.tasks.cloth.tests.test_goal_tasks import RES, SHAPES, _geometry, _tables, _yaw_rot

TABLE_Z = 0.4


def _stage1_geometry(num_envs: int, xy_scale: torch.Tensor | None, fold_lift: float = 0.0) -> FoldGeometry:
    """FoldGeometry without goal tasks; ``xy_scale`` ``None``: one sheet size (as VBD lays it out)."""
    size = fs.shape(SHAPES[0]).size
    verts, _ = grid_mesh(size, RES)
    layout = SheetLayout(
        rest_local=verts, keypoint_idx=corner_indices(RES, "x"), resolution=RES, fold_axis="x", size=size,
        fold_lift=fold_lift, object_scale=torch.ones(num_envs, 3), xy_scale=xy_scale,
        env_size=None if xy_scale is None else xy_scale * size,
    )
    return FoldGeometry(layout, num_envs=num_envs, device="cpu", observed_goal_quat=lambda r: r)


def _centred(points: torch.Tensor) -> torch.Tensor:
    return points - points.mean(dim=1, keepdim=True)


def test_stage1_goal_is_the_half_fold_goal():
    goal_tasks = _tables("half_fold")
    num_tasks = goal_tasks.num_tasks
    xy_scale = torch.rand(num_tasks) * 0.8 + 0.7
    pool_geometry = _geometry(goal_tasks, xy_scale)
    goal_tasks.task = torch.arange(num_tasks)
    stage1_geometry = _stage1_geometry(num_tasks, xy_scale)

    scale3 = pool_geometry._scale3()
    yaw_rot = _yaw_rot(torch.rand(num_tasks) * 2.0 * math.pi)
    flat_sheet_w = torch.einsum("nij,npj->npi", yaw_rot, goal_tasks.p_start * scale3) + torch.tensor([0.3, -0.2, TABLE_Z])

    goal_tasks_goal_w, _ = pool_geometry.goal_sheet(flat_sheet_w)
    stage1_goal_w, stage1_goal_own_frame = stage1_geometry.goal_sheet(flat_sheet_w)
    assert float((goal_tasks_goal_w - stage1_goal_w).norm(dim=-1).max()) < 1e-4
    # In its own frame: the same folded state, up to where the frame's origin is.
    goal_tasks_goal_own_frame = goal_tasks.p_goal[goal_tasks.task] * scale3
    assert float((_centred(goal_tasks_goal_own_frame) - _centred(stage1_goal_own_frame)).norm(dim=-1).max()) < 1e-4


def test_stage1_goal_follows_the_sheet_and_lift():
    num_envs, fold_lift = 8, 0.002
    geometry = _stage1_geometry(num_envs, None, fold_lift)
    yaw_rot = _yaw_rot(torch.rand(num_envs) * 2.0 * math.pi)
    flat_sheet_w = torch.einsum("nij,pj->npi", yaw_rot, geometry.rest_local) + torch.tensor([0.1, 0.2, TABLE_Z])

    goal_w, goal_own_frame = geometry.goal_sheet(flat_sheet_w)
    assert goal_w.shape == (num_envs, RES * RES, 3) and goal_own_frame.shape == (num_envs, RES * RES, 3)
    # The stationary half and the crease stay where they lie; the moving half lands on the stationary
    # one, ``fold_lift`` above it.
    not_moving = torch.ones(RES * RES, dtype=torch.bool)
    not_moving[geometry.moving_idx] = False
    assert float((goal_w[:, not_moving] - flat_sheet_w[:, not_moving]).norm(dim=-1).max()) < 1e-5
    moving_height = goal_w[:, geometry.moving_idx, 2] - TABLE_Z
    assert float((moving_height - fold_lift).abs().max()) < 1e-5
    # A sheet folded exactly onto its goal meets the fold criterion's targets: zero fold error.
    folded_view = geometry.derive(goal_w, torch.zeros_like(goal_w))
    assert float(folded_view.fold_error.max()) < 1e-5
