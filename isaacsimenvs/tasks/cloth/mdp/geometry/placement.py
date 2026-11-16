"""Whole-sheet placement inside the actual table, including yaw feasibility."""
from __future__ import annotations

import math

import torch


def placement_bounds(x_range, y_range, table_size=None, margin=0.005):
    """Intersect requested table-relative ranges with the usable tabletop."""
    lower, upper = [float(x_range[0]), float(y_range[0])], [float(x_range[1]), float(y_range[1])]
    if table_size is not None:
        for axis in (0, 1):
            half = float(table_size[axis]) / 2 - margin
            if not math.isfinite(half) or half <= 0:
                raise ValueError("table dimensions must exceed twice the placement margin")
            lower[axis], upper[axis] = max(lower[axis], -half), min(upper[axis], half)
    if any(not math.isfinite(lo) or not math.isfinite(hi) or lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError("placement ranges must have a finite nonempty intersection with the tabletop")
    return tuple(lower), tuple(upper)


def _extent(dimensions, yaw):
    c, s = yaw.cos().abs(), yaw.sin().abs()
    return torch.stack((dimensions[:, 0] * c + dimensions[:, 1] * s,
                        dimensions[:, 0] * s + dimensions[:, 1] * c), dim=-1)


def constrain_yaw(dimensions, yaw, lower, upper, yaw_range_deg, randomize=True):
    """Keep valid headings; redraw invalid ones uniformly conditional on fitting.

    Exact interval boundaries solve L|cos(yaw)|+W|sin(yaw)|=available width
    and its height counterpart. All environments are processed as tensor batches;
    no rejection limit or per-environment Python loop biases narrow valid ranges.
    """
    room = upper - lower
    valid = (_extent(dimensions, yaw) <= room + 1e-7).all(dim=-1)
    if bool(valid.all()):
        return yaw
    if not randomize:
        raise ValueError("cloth does not fit the placement region at the fixed reset yaw")
    low, high = map(math.radians, yaw_range_deg)
    if not math.isfinite(low) or not math.isfinite(high) or not 0 <= high - low <= 2 * math.pi + 1e-10:
        raise ValueError("yaw range must be finite, ordered and span at most 360 degrees")
    dims = dimensions[~valid]
    if high == low:
        fixed = yaw.new_full((len(dims),), low)
        if not bool((_extent(dims, fixed) <= room + 1e-7).all()):
            raise ValueError("cloth does not fit the placement region anywhere in the selected yaw range")
        result = yaw.clone()
        result[~valid] = fixed
        return result
    quadrant = torch.arange(5, device=yaw.device) + math.floor(low / (math.pi / 2))
    start = quadrant.to(yaw.dtype) * (math.pi / 2)
    even = (quadrant % 2 == 0)[None]
    length, width = dims[:, 0:1], dims[:, 1:2]
    a, b = torch.where(even, length, width), torch.where(even, width, length)
    radius = torch.sqrt(a.square() + b.square())
    phase_x, phase_y = torch.atan2(b, a), torch.atan2(a, b)
    delta_x = torch.acos((room[0] / radius).clamp(-1, 1))
    delta_y = torch.acos((room[1] / radius).clamp(-1, 1))
    roots = torch.stack((phase_x - delta_x, phase_x + delta_x,
                         phase_y - delta_y, phase_y + delta_y), dim=-1).clamp(0, math.pi / 2)
    roots = (roots + start[None, :, None]).reshape(len(dims), -1)
    edges = torch.cat((start, start[-1:] + math.pi / 2)).expand(len(dims), -1)
    boundaries = torch.cat((roots, edges), dim=1).clamp(low, high).sort(dim=1).values
    left, right = boundaries[:, :-1], boundaries[:, 1:]
    mid = (left + right) / 2
    c, s = mid.cos().abs(), mid.sin().abs()
    fits = (length * c + width * s <= room[0] + 1e-7) & (length * s + width * c <= room[1] + 1e-7)
    spans = torch.where(fits, right - left, 0.)
    total = spans.sum(dim=1)
    if bool((total <= 1e-10).any()):
        raise ValueError("cloth does not fit the placement region anywhere in the selected yaw range")
    draw = torch.rand_like(total) * total
    cumulative = spans.cumsum(dim=1)
    index = (cumulative > draw[:, None]).to(torch.int64).argmax(dim=1)
    before = cumulative - spans
    sampled = left.gather(1, index[:, None])[:, 0] + draw - before.gather(1, index[:, None])[:, 0]
    result = yaw.clone()
    result[~valid] = sampled
    return result
