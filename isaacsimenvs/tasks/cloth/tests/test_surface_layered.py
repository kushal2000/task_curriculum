"""The layered surface distance (mdp/surface_layered.py) against the unlabelled one (surface_match): CPU."""

from __future__ import annotations

import math

import torch

from isaacsimenvs.tasks.cloth.mdp.geometry.grid import grid_mesh
from isaacsimenvs.tasks.cloth.mdp.surface_layered import layered_aligned_distance

RES = 7
SIZE = 0.10
PLY_GAP = 0.003
TABLE_Z = 0.53


def _flat_sheet() -> torch.Tensor:
    """``(R, R, 3)`` the flat sheet, row-major, centred, on the table."""
    verts, _ = grid_mesh(SIZE, RES)
    grid = torch.tensor(verts, dtype=torch.float32).view(RES, RES, 3)
    grid[..., 2] = TABLE_Z
    return grid


def _half_fold(axis: str, side: str) -> torch.Tensor:
    """``(R*R, 3)`` the sheet half-folded: the ``side`` half across the middle ``axis`` line laid on the rest,
    ``PLY_GAP`` above it. Grid rows run along x (``grid_mesh``: vertex ``i * R + j`` at x_i, y_j)."""
    grid = _flat_sheet()
    mid = RES // 2
    coord = 0 if axis == "row" else 1           # the row index i is x, the column index j is y
    index = torch.arange(RES)
    folded_rows = index > mid if side == "upper" else index < mid
    sel = folded_rows.view(RES, 1).expand(RES, RES) if axis == "row" else folded_rows.view(1, RES).expand(RES, RES)
    crease = grid[mid, mid, coord]
    grid[..., coord] = torch.where(sel, 2 * crease - grid[..., coord], grid[..., coord])
    grid[..., 2] = torch.where(sel, grid[..., 2] + PLY_GAP, grid[..., 2])
    return grid.reshape(-1, 3)


def _place(points: torch.Tensor, yaw: float, shift) -> torch.Tensor:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return points @ rot.T + torch.tensor(shift)


def _top_ply_mask() -> torch.Tensor:
    """The stage-1 goal's top ply: the rows above the middle one (not the crease row)."""
    return (torch.arange(RES) > RES // 2).view(RES, 1).expand(RES, RES).reshape(-1)


def _distances(sheet: torch.Tensor, goal: torch.Tensor):
    out = layered_aligned_distance(sheet.unsqueeze(0), goal.unsqueeze(0), RES)
    return float(out["max"][0]), float(out["aligned_max"][0])


def test_every_half_fold_matches_wherever_it_lies():
    goal = _half_fold("row", "upper")                       # the stage-1 goal
    for axis in ("row", "col"):
        for side in ("upper", "lower"):
            sheet = _place(_half_fold(axis, side), yaw=1.1, shift=(0.07, -0.04, 0.0))
            layered, aligned = _distances(sheet, goal)
            assert layered < 1e-3 and aligned < 1e-3, (axis, side, layered, aligned)


def test_a_top_ply_slid_out_of_the_outline_fails_both():
    goal = _half_fold("row", "upper")
    sheet = goal.clone()
    sheet[_top_ply_mask(), 1] += 0.02                          # slid 2 cm along the crease
    layered, aligned = _distances(sheet, goal)
    assert layered > 0.01 and aligned > 0.005


def test_a_top_ply_bunched_inside_the_outline_fails_only_the_layered_distance():
    goal = _half_fold("row", "upper")
    sheet = goal.clone()
    top = _top_ply_mask()
    centre = sheet[top, :2].mean(dim=0)
    sheet[top, :2] = centre + 0.5 * (sheet[top, :2] - centre)  # the top ply bunched to half its size, in place
    layered, aligned = _distances(sheet, goal)
    assert aligned < 0.01                                        # unlabelled: still a fold within 1 cm
    assert layered > 0.015                                       # layered: the top ply's outline is 1.25 cm short
