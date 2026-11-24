"""Two-sided surface distance between the sheet and its goal, with and without in-plane alignment.

The fold criterion of ``termination.success: surface`` (``mdp/task.py``): the sheet counts as folded
when its surface and the goal sheet's surface lie within a tolerance of each other, measured both
ways (every point of one has a point of the other nearby: the two-sided maximum, a Hausdorff
distance). No particle correspondence is used, so every fold that LOOKS like the goal counts: on a
square sheet the four half folds (left over right, right over left, top over bottom, bottom over
top) leave the same two-ply rectangle, only placed and turned differently on the table.

* ``in_place_distance``: the two clouds where they are.
* ``aligned_distance``: the sheet first moved in the table plane -- turned about z and shifted in
  x, y, never tilted, lifted or scaled -- to where it best fits the goal. A sheet held in the air
  stays as high as it is, so it does not match a goal lying on the table.

Alignment: both centroids (x, y) to the origin; start yaws = the difference of the two clouds' major
axes (and that + 180 deg), plus ``n_uniform_starts`` evenly spaced ones (for a goal with no clear
major axis); ``icp_iters`` rounds of ICP from each start (nearest points both ways, then the 2-D
Kabsch rotation and shift); the start whose coarse two-sided max ends smallest is kept, and the
distance is measured on the fine samples at that placement. The reported distance is the one at a
real placement of the sheet, so it can only overestimate the best placement's: an alignment that
misses can fail a fold, never pass a wrong one. (A synthetic probe on 7680 envs: the two major-axis
starts found every half fold at any yaw; 4 evenly spaced starts alone found 40%, 12 found all.)

Samples: the coarse ones are the particle grid (``resolution x resolution``), the fine ones that
grid interpolated bilinearly to ``fine_side x fine_side`` -- the same parametrisation on both sides,
so a sheet exactly on its goal is at distance 0.

Nearest points: found on the matmul form of the squared distance (``|a|^2 + |b|^2 - 2 a.b``, fast),
then each distance recomputed exactly to the point found. With TF32 matmul the search can pick a
neighbour a fraction of a millimetre farther than the nearest, never a nearer one, so a distance is
never underestimated. (``torch.cdist`` without its matmul shortcut is exact but 30x slower: 360 ms
for 7680 envs.)

Torch only, any backend: inputs are ``(N, resolution**2, 3)`` grids in one frame.
"""

from __future__ import annotations

import math

import torch

__all__ = ["aligned_distance", "dense_grid", "in_place_distance", "two_sided_distance"]


def dense_grid(grid_points: torch.Tensor, resolution: int, side: int) -> torch.Tensor:
    """``(N, side**2, 3)``: the ``(N, resolution**2, 3)`` particle grid (row-major) interpolated
    bilinearly at ``side x side`` evenly spaced grid coordinates, corners included."""
    if side == resolution:
        return grid_points
    num_envs = grid_points.shape[0]
    coord = torch.linspace(0.0, resolution - 1.0, side, device=grid_points.device)
    lower = coord.floor().clamp(max=resolution - 2).long()
    frac = (coord - lower).to(grid_points.dtype)
    grid = grid_points.view(num_envs, resolution, resolution, 3)
    w_row = frac.view(1, side, 1, 1)
    rows = grid[:, lower] * (1 - w_row) + grid[:, lower + 1] * w_row                     # (N, side, res, 3)
    w_col = frac.view(1, 1, side, 1)
    samples = rows[:, :, lower] * (1 - w_col) + rows[:, :, lower + 1] * w_col             # (N, side, side, 3)
    return samples.reshape(num_envs, side * side, 3)


def _nearest(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(index in b of each a point's nearest (N, A), index in a of each b point's nearest (N, B))``,
    searched on the matmul form of the squared distance (module doc)."""
    sq = (a * a).sum(dim=-1).unsqueeze(-1) + (b * b).sum(dim=-1).unsqueeze(-2) - 2.0 * a @ b.transpose(-1, -2)
    return sq.argmin(dim=2), sq.argmin(dim=1)


def _gather(points: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(points, 1, index.unsqueeze(-1).expand(-1, -1, 3))


def _one_way(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact distance of every a point to the b point found nearest, and of every b point likewise:
    ``((N, A), (N, B))``."""
    a_idx, b_idx = _nearest(a, b)
    return (a - _gather(b, a_idx)).norm(dim=-1), (b - _gather(a, b_idx)).norm(dim=-1)


def two_sided_distance(cloud_a: torch.Tensor, cloud_b: torch.Tensor, env_chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """``(max, mean)`` ``(N,)`` between ``cloud_a`` ``(N, A, 3)`` and ``cloud_b`` ``(N, B, 3)``, each point
    to its nearest point of the other cloud. max: the larger of the two one-way maxima (Hausdorff);
    mean: the mean of the two one-way means."""
    max_parts, mean_parts = [], []
    for start in range(0, cloud_a.shape[0], env_chunk):
        a_to_b, b_to_a = _one_way(cloud_a[start:start + env_chunk], cloud_b[start:start + env_chunk])
        max_parts.append(torch.maximum(a_to_b.amax(dim=1), b_to_a.amax(dim=1)))
        mean_parts.append(0.5 * (a_to_b.mean(dim=1) + b_to_a.mean(dim=1)))
    return torch.cat(max_parts), torch.cat(mean_parts)


def _rotate_about_z(points: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """``points`` ``(..., M, 3)`` turned about z by ``yaw`` ``(...)``."""
    cos, sin = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    x, y = points[..., 0], points[..., 1]
    return torch.stack((cos * x - sin * y, sin * x + cos * y, points[..., 2]), dim=-1)


def _rotate_xy(vec: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """``vec`` ``(K, 2)`` turned by ``yaw`` ``(K,)``."""
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((cos * vec[:, 0] - sin * vec[:, 1], sin * vec[:, 0] + cos * vec[:, 1]), dim=-1)


def _place(points: torch.Tensor, yaw: torch.Tensor, shift_xy: torch.Tensor) -> torch.Tensor:
    """``points`` turned about z by ``yaw``, then shifted by ``shift_xy`` in x, y."""
    turned = _rotate_about_z(points, yaw)
    return torch.cat((turned[..., :2] + shift_xy.unsqueeze(-2), turned[..., 2:]), dim=-1)


def _major_axis_yaw(points: torch.Tensor) -> torch.Tensor:
    """``(N,)`` direction [rad] of the in-plane major axis of the centred cloud ``points`` ``(N, M, 3)``."""
    x, y = points[..., 0], points[..., 1]
    return 0.5 * torch.atan2(2.0 * (x * y).mean(dim=1), (x * x).mean(dim=1) - (y * y).mean(dim=1))


def _icp_in_plane(sheet: torch.Tensor, goal: torch.Tensor, start_yaw: torch.Tensor, icp_iters: int):
    """ICP of ``sheet`` onto ``goal`` (both ``(N, M, 3)``, centred) from each start yaw ``(N, S)``.
    Returns ``(yaw (N, S), shift_xy (N, S, 2), coarse two-sided max (N, S))`` of each start."""
    num_envs, num_starts = start_yaw.shape
    sheet = sheet.unsqueeze(1).expand(num_envs, num_starts, -1, -1).reshape(num_envs * num_starts, -1, 3)
    goal = goal.unsqueeze(1).expand(num_envs, num_starts, -1, -1).reshape(num_envs * num_starts, -1, 3)
    yaw = start_yaw.reshape(-1)
    shift_xy = torch.zeros(yaw.shape[0], 2, device=sheet.device, dtype=sheet.dtype)
    for it in range(icp_iters + 1):
        placed = _place(sheet, yaw, shift_xy)
        if it == icp_iters:
            break
        # Pairs both ways: each sheet point with its nearest goal point, each goal point with its
        # nearest sheet point. Then the rotation about z and the shift that best map the pairs.
        nearest_goal, nearest_sheet = _nearest(placed, goal)
        src = torch.cat((placed, _gather(placed, nearest_sheet)), dim=1)[..., :2]
        dst = torch.cat((_gather(goal, nearest_goal), goal), dim=1)[..., :2]
        src_mean, dst_mean = src.mean(dim=1), dst.mean(dim=1)
        src_c, dst_c = src - src_mean.unsqueeze(1), dst - dst_mean.unsqueeze(1)
        cross = (src_c[..., 0] * dst_c[..., 1] - src_c[..., 1] * dst_c[..., 0]).sum(dim=1)
        dot = (src_c * dst_c).sum(dim=(1, 2))
        yaw_step = torch.atan2(cross, dot)
        # The new placement: turn the placed sheet by yaw_step about its pair mean, onto the goal's.
        shift_xy = _rotate_xy(shift_xy, yaw_step) - _rotate_xy(src_mean, yaw_step) + dst_mean
        yaw = yaw + yaw_step
    sheet_to_goal, goal_to_sheet = _one_way(placed, goal)
    coarse_max = torch.maximum(sheet_to_goal.amax(dim=1), goal_to_sheet.amax(dim=1))
    return yaw.view(num_envs, num_starts), shift_xy.view(num_envs, num_starts, 2), coarse_max.view(num_envs, num_starts)


def aligned_distance(sheet: torch.Tensor, goal: torch.Tensor, resolution: int, *, fine_side: int = 13,
                     n_uniform_starts: int = 4, icp_iters: int = 3, env_chunk: int = 2048) -> dict[str, torch.Tensor]:
    """The two-sided distance after the best in-plane placement of the sheet (module doc).

    ``sheet`` / ``goal``: ``(N, resolution**2, 3)`` particle grids, one frame. Returns ``max`` / ``mean``
    ``(N,)`` on the fine samples, and the placement: ``yaw`` ``(N,)`` [rad] and ``shift_xy`` ``(N, 2)``
    (the sheet's centred cloud turned by ``yaw`` then shifted, onto the goal's centred cloud)."""
    out = {"max": [], "mean": [], "yaw": [], "shift_xy": []}
    for start in range(0, sheet.shape[0], env_chunk):
        sheet_part, goal_part = sheet[start:start + env_chunk], goal[start:start + env_chunk]
        # x, y about each cloud's own centroid; z about the goal's mean height for both (a common
        # shift: distances are unchanged and the numbers stay small).
        z_ref = goal_part[..., 2].mean(dim=1).view(-1, 1, 1)
        sheet_c = torch.cat((sheet_part[..., :2] - sheet_part[..., :2].mean(dim=1, keepdim=True), sheet_part[..., 2:] - z_ref), dim=-1)
        goal_c = torch.cat((goal_part[..., :2] - goal_part[..., :2].mean(dim=1, keepdim=True), goal_part[..., 2:] - z_ref), dim=-1)

        axis_yaw = _major_axis_yaw(goal_c) - _major_axis_yaw(sheet_c)
        start_yaws = [axis_yaw, axis_yaw + math.pi]
        start_yaws += [torch.full_like(axis_yaw, 2.0 * math.pi * k / n_uniform_starts) for k in range(n_uniform_starts)]
        yaw, shift_xy, coarse_max = _icp_in_plane(sheet_c, goal_c, torch.stack(start_yaws, dim=1), int(icp_iters))

        best = coarse_max.argmin(dim=1)
        env = torch.arange(best.shape[0], device=best.device)
        yaw, shift_xy = yaw[env, best], shift_xy[env, best]
        sheet_fine = _place(dense_grid(sheet_c, resolution, fine_side), yaw, shift_xy)
        dist_max, dist_mean = two_sided_distance(sheet_fine, dense_grid(goal_c, resolution, fine_side), env_chunk=env_chunk)
        out["max"].append(dist_max)
        out["mean"].append(dist_mean)
        out["yaw"].append(torch.atan2(torch.sin(yaw), torch.cos(yaw)))
        out["shift_xy"].append(shift_xy)
    return {key: torch.cat(parts) for key, parts in out.items()}


def in_place_distance(sheet: torch.Tensor, goal: torch.Tensor, resolution: int, *, fine_side: int = 13,
                      env_chunk: int = 2048) -> torch.Tensor:
    """``(N,)`` the two-sided max distance with both grids where they are (no alignment)."""
    centre = goal.mean(dim=1, keepdim=True)          # a common shift: keeps the numbers small
    dist_max, _ = two_sided_distance(dense_grid(sheet - centre, resolution, fine_side),
                                     dense_grid(goal - centre, resolution, fine_side), env_chunk=env_chunk)
    return dist_max
