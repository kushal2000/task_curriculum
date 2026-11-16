"""Quaternion helpers, (w, x, y, z) throughout.

Own helpers rather than ``isaaclab.utils.math``: Isaac Lab 2.3 (PhysX) is wxyz and 3.0 (Newton) is
xyzw, so the same import means different things on the two backends. The cloth utils are wxyz, each
env converts once when it reads the simulator (``_read_state``), and the observation is xyzw (Play's
policy convention, :func:`to_xyzw`).
"""

from __future__ import annotations

import torch


def to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return q[..., [1, 2, 3, 0]]


def to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q[..., [3, 0, 1, 2]]


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a * b`` (apply ``b`` first)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v (..., 3)`` by unit quaternions ``q (..., 4)`` (broadcast)."""
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * torch.cross(u.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(u.expand_as(v), t, dim=-1)


def from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """``angle (...,)`` [rad] about unit ``axis (..., 3)``."""
    half = 0.5 * angle.unsqueeze(-1)
    return torch.cat((torch.cos(half), torch.sin(half) * axis), dim=-1)


def yaw(angle: torch.Tensor) -> torch.Tensor:
    """Rotation by ``angle (n,)`` [rad] about world +z."""
    z = torch.zeros_like(angle)
    half = 0.5 * angle
    return torch.stack((torch.cos(half), z, z, torch.sin(half)), dim=-1)


def to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def from_matrix(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(N, 3, 3)`` to unit quaternions with ``w >= 0``.

    The largest-diagonal branch, not the plain trace formula, which loses precision (and divides by
    ~0) near 180 deg -- the rotation a completed fold produces.
    """
    n = r.shape[0]
    q = torch.zeros((n, 4), device=r.device, dtype=r.dtype)
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    big = trace > 0
    if bool(big.any()):
        s = torch.sqrt(trace[big] + 1.0) * 2.0
        q[big, 0] = 0.25 * s
        q[big, 1] = (r[big, 2, 1] - r[big, 1, 2]) / s
        q[big, 2] = (r[big, 0, 2] - r[big, 2, 0]) / s
        q[big, 3] = (r[big, 1, 0] - r[big, 0, 1]) / s
    rest = ~big
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = rest & (r[:, i, i] >= r[:, j, j]) & (r[:, i, i] >= r[:, k, k])
        rest = rest & ~sel
        if not bool(sel.any()):
            continue
        s = torch.sqrt(1.0 + r[sel, i, i] - r[sel, j, j] - r[sel, k, k]) * 2.0
        q[sel, 0] = (r[sel, k, j] - r[sel, j, k]) / s
        q[sel, 1 + i] = 0.25 * s
        q[sel, 1 + j] = (r[sel, j, i] + r[sel, i, j]) / s
        q[sel, 1 + k] = (r[sel, k, i] + r[sel, i, k]) / s
    q = torch.where(q[:, :1] < 0.0, -q, q)
    return torch.nn.functional.normalize(q, dim=-1, eps=1e-12)


def random_perturb(q: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Play's object-state rotation noise: a uniform random axis, angle U(-max, max) deg, applied
    in the world frame."""
    n = q.shape[0]
    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=q.device), dim=-1)
    angle = torch.empty(n, device=q.device).uniform_(-max_deg, max_deg) * (torch.pi / 180.0)
    return mul(from_angle_axis(angle, axis), q)
