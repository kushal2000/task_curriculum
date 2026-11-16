"""Per-reset draws: robot start state, table height, the episode's task, the cloth's start pose.

Pure tensor functions; the env writes what they return to its simulator (``ClothEnv._reset_idx``).
"""

from __future__ import annotations

import math

import torch


def robot_start(env, ids: torch.Tensor, default_pos: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
    """Play's reset: each joint ``lerp(default, U(lower, upper), interval)``, velocity U(+-v);
    the targets start at the reset pose. Returns ``(joint_pos, joint_vel)`` (Lab order)."""
    r = env.cfg.reset
    scale = torch.zeros_like(default_pos)
    scale[:, env._arm_ids] = r.reset_dof_pos_random_interval_arm
    scale[:, env._hand_ids] = r.reset_dof_pos_random_interval_fingers
    sampled = lower + (upper - lower) * torch.rand_like(default_pos)
    pos = torch.lerp(default_pos, sampled, scale).clamp(lower, upper)
    vel = torch.empty_like(default_pos).uniform_(-r.reset_dof_vel_random_interval, r.reset_dof_vel_random_interval)
    env._prev_targets[ids] = pos
    env._cur_targets[ids] = pos
    return pos, vel


def table_height(env, ids: torch.Tensor) -> torch.Tensor:
    """This episode's table z (env frame), ``table_reset_z`` + U(+-range); kept in ``env._table_z``."""
    r = env.cfg.reset
    z = r.table_reset_z + torch.empty(ids.numel(), device=env.device).uniform_(-r.table_reset_z_range, r.table_reset_z_range)
    env._table_z[ids] = z
    return z


def start_task(env, ids: torch.Tensor) -> torch.Tensor:
    """Draw each episode's task (``cloth.task=mixed``) and return the + half's start angle [rad]:
    0 (flat) for a fold, U(``unfold_start_deg``) for an unfold."""
    f = env.cfg.cloth
    n = ids.numel()
    if f.task == "mixed":
        env._unfold_env[ids] = torch.rand(n, device=env.device) < f.unfold_prob
    lo, hi = (math.radians(float(v)) for v in f.unfold_start_deg)
    angle = torch.empty(n, device=env.device).uniform_(lo, hi)
    return angle * env._unfold_env[ids].float()


def start_yaw(env, n: int) -> torch.Tensor:
    if not env.cfg.reset.random_yaw:
        return torch.zeros(n, device=env.device)
    return torch.rand(n, device=env.device) * (2.0 * math.pi)


def place_in_region(dims: torch.Tensor, yaw: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, pinned: bool) -> torch.Tensor:
    """``(n, 2)`` cloth centre: uniform over every position that keeps the yawed ``length x width``
    bounding box inside ``[lo, hi]``; centred where it does not fit, and everywhere when ``pinned``."""
    c, s = torch.cos(yaw).abs(), torch.sin(yaw).abs()
    half = torch.stack((0.5 * (dims[:, 0] * c + dims[:, 1] * s), 0.5 * (dims[:, 0] * s + dims[:, 1] * c)), dim=-1)
    room = hi - lo - 2.0 * half
    u = torch.full_like(half, 0.5) if pinned else torch.rand_like(half)
    return torch.where(room >= 0.0, lo + half + u * room.clamp(min=0.0), 0.5 * (lo + hi))


def place_with_noise(env, n: int) -> torch.Tensor:
    """``(n, 2)``: Play's U(+-reset_position_noise) around the table centre (the VBD cloth)."""
    r = env.cfg.reset
    noise = torch.tensor([r.reset_position_noise_x, r.reset_position_noise_y], device=env.device)
    return (2.0 * torch.rand(n, 2, device=env.device) - 1.0) * noise
