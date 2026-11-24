"""The surface fold distance with the two plies kept apart (a stricter companion of ``surface_match``).

``surface_match.aligned_distance`` compares the sheet's surface with the goal's as two unlabelled
clouds. A half-folded goal is two plies a few millimetres apart, much closer than the tolerance, so a
goal point of the top ply that the sheet's top ply does not cover is matched to the sheet's BOTTOM ply
right under it: a top ply slid by centimetres over the bottom one still passes, as long as it stays
inside the bottom ply's outline.

Here each ply is compared only with its own ply: the sheet is split at a crease into the half that lies
on top and the half underneath, and

    layered distance = max(two-sided max(sheet top half, goal top ply),
                           two-sided max(sheet bottom half, goal bottom ply))

after the same in-plane placement ``aligned_distance`` finds for the whole sheet (turned about z and
shifted in x, y). Which half of the sheet is on top is not known -- a square sheet can be half-folded
four ways (either crease through the middle, either half on top), all of them the same two-ply
rectangle -- so all four splits are scored and the smallest is kept. Within a ply no particle
correspondence is used, as in ``surface_match``.

Half folds of a square particle grid only (the stage-1 fold and the pool's ``half_fold`` goal on a
square sheet), odd ``resolution`` (the crease row is a grid row, in both halves). Torch only;
``(N, resolution**2, 3)`` row-major grids in one frame.
"""

from __future__ import annotations

import torch

from . import surface_match

__all__ = ["layered_aligned_distance", "SHEET_SPLITS"]

#: The four half folds of a square grid: (crease axis of the grid, the half on top).
#: Axis "row": the crease is the middle row, a half is the rows above or below it;
#: axis "col": the middle column.
SHEET_SPLITS = (("row", "upper"), ("row", "lower"), ("col", "upper"), ("col", "lower"))


def _interpolation_matrix(count: int, out: int, like: torch.Tensor) -> torch.Tensor:
    """``(out, count)``: row ``k`` holds the linear interpolation weights of evenly spaced coordinate
    ``k`` over ``0 .. count - 1`` (both ends included)."""
    coord = torch.linspace(0.0, count - 1.0, out, device=like.device, dtype=like.dtype)
    lower = coord.floor().clamp(max=count - 2).long()
    frac = coord - lower
    weights = torch.zeros(out, count, device=like.device, dtype=like.dtype)
    rows = torch.arange(out, device=like.device)
    weights[rows, lower] = 1.0 - frac
    weights[rows, lower + 1] = frac
    return weights


def _dense_rect(grid: torch.Tensor, out_rows: int, out_cols: int) -> torch.Tensor:
    """``(N, out_rows * out_cols, 3)``: the ``(N, rows, cols, 3)`` grid interpolated bilinearly at evenly
    spaced grid coordinates, corners included (``surface_match.dense_grid`` for a rectangle)."""
    num_envs, rows, cols, _ = grid.shape
    row_weights = _interpolation_matrix(rows, out_rows, grid)
    col_weights = _interpolation_matrix(cols, out_cols, grid)
    samples = torch.einsum("ar,nrcd,bc->nabd", row_weights, grid, col_weights)   # (N, out_rows, out_cols, 3)
    return samples.reshape(num_envs, out_rows * out_cols, 3)


def _half(grid: torch.Tensor, axis: str, side: str) -> torch.Tensor:
    """``(N, rows, cols, 3)`` one half of a square ``(N, R, R, 3)`` grid, the crease row / column included,
    laid out with the crease along the rows (so every half has the same shape)."""
    mid = grid.shape[1] // 2
    if axis == "col":
        grid = grid.transpose(1, 2)
    return grid[:, mid:] if side == "upper" else grid[:, : mid + 1]


def layered_aligned_distance(
    sheet: torch.Tensor, goal: torch.Tensor, resolution: int, goal_top: tuple[str, str] = ("row", "upper"), *,
    fine_side: int = 13, n_uniform_starts: int = 4, icp_iters: int = 3,
) -> dict[str, torch.Tensor]:
    """The layered distance (module doc) of each env's sheet from its goal.

    ``goal_top``: which half of the GOAL grid is its top ply, as a ``SHEET_SPLITS`` entry; the stage-1
    goal (``FoldGeometry.goal_sheet``, ``fold_axis`` x) folds the rows above the middle one onto the
    rest: ``("row", "upper")``.

    Returns ``max`` ``(N,)`` the layered distance, ``split`` ``(N,)`` the index into ``SHEET_SPLITS`` of
    the half of the sheet found on top, and the placement ``aligned_distance`` found: ``aligned_max``
    ``(N,)`` the unlabelled distance there (for comparison) and ``yaw`` ``(N,)`` its turn [rad]."""
    if resolution % 2 != 1:
        raise ValueError(f"layered distance needs an odd resolution (a crease row), got {resolution}")
    num_envs = sheet.shape[0]
    placement = surface_match.aligned_distance(
        sheet, goal, resolution, fine_side=fine_side, n_uniform_starts=n_uniform_starts, icp_iters=icp_iters,
    )
    # The placement as aligned_distance applies it: both clouds centred in x, y about their own centroid,
    # z about the goal's mean height; the sheet then turned and shifted onto the goal.
    z_ref = goal[..., 2].mean(dim=1).view(-1, 1, 1)
    sheet_centred = torch.cat((sheet[..., :2] - sheet[..., :2].mean(dim=1, keepdim=True), sheet[..., 2:] - z_ref), dim=-1)
    goal_centred = torch.cat((goal[..., :2] - goal[..., :2].mean(dim=1, keepdim=True), goal[..., 2:] - z_ref), dim=-1)
    sheet_placed = surface_match._place(sheet_centred, placement["yaw"], placement["shift_xy"])

    sheet_grid = sheet_placed.view(num_envs, resolution, resolution, 3)
    goal_grid = goal_centred.view(num_envs, resolution, resolution, 3)
    half_rows, half_cols = (fine_side + 1) // 2, fine_side       # the fine grid's spacing, on half a sheet

    def dense_half(grid: torch.Tensor, axis: str, side: str) -> torch.Tensor:
        return _dense_rect(_half(grid, axis, side), half_rows, half_cols)

    goal_axis, goal_side = goal_top
    goal_bottom_side = "lower" if goal_side == "upper" else "upper"
    goal_top_ply = dense_half(goal_grid, goal_axis, goal_side)
    goal_bottom_ply = dense_half(goal_grid, goal_axis, goal_bottom_side)

    # Every split at once: (N, splits, points, 3) flattened to N * splits clouds, each against the goal ply.
    n_splits = len(SHEET_SPLITS)
    sheet_tops = torch.stack([dense_half(sheet_grid, axis, side) for axis, side in SHEET_SPLITS], dim=1)
    sheet_bottoms = torch.stack(
        [dense_half(sheet_grid, axis, "lower" if side == "upper" else "upper") for axis, side in SHEET_SPLITS], dim=1,
    )

    def per_split_distance(sheet_halves: torch.Tensor, goal_ply: torch.Tensor) -> torch.Tensor:
        goal_plies = goal_ply.unsqueeze(1).expand(-1, n_splits, -1, -1)
        dist, _ = surface_match.two_sided_distance(sheet_halves.flatten(0, 1), goal_plies.flatten(0, 1))
        return dist.view(num_envs, n_splits)

    per_split = torch.maximum(per_split_distance(sheet_tops, goal_top_ply), per_split_distance(sheet_bottoms, goal_bottom_ply))
    best, split = per_split.min(dim=1)
    return {"max": best, "split": split, "aligned_max": placement["max"], "yaw": placement["yaw"]}
