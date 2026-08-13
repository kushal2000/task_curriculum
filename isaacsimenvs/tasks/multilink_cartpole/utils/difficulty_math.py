"""Pure-tensor difficulty maths for the multi-link cartpole.

Torch only — no Isaac, no `env` — so `tests/test_cartpole_difficulty.py` can assert the
invariants that the whole curriculum rests on without booting Kit:

  * total chain length is the same at every difficulty (only the DOF count changes)
  * joint 0 is always free (locking it would delete the task)
  * inactive observation slots are zero
  * segment indices come back in chain order, base → tip

`reset_utils` and `obs_utils` are thin wrappers over these.
"""

from __future__ import annotations

import torch

__all__ = [
    "active_joint_count",
    "sample_free_mask",
    "segment_geometry",
    "active_mask",
    "segment_kinematics",
]


def active_joint_count(difficulty: torch.Tensor, n_max: int) -> torch.Tensor:
    """Map difficulty in [0, 1] to a free-joint count in [1, n_max].

    Difficulty 0 gives a single free joint — one rigid pendulum of the full chain length
    — and difficulty 1 frees every joint.
    """
    return (1.0 + difficulty.clamp(0.0, 1.0) * (n_max - 1)).round().long().clamp(1, n_max)


def sample_free_mask(n_active: torch.Tensor, n_max: int) -> torch.Tensor:
    """Choose which pole joints stay free: joint 0 always, then a random subset.

    Randomising *which* joints are free (rather than always taking the first `n_active`)
    is what makes the effective segment lengths vary: with `n_max=4` and `n_active=2`,
    freeing {0, 1} gives a (1L, 3L) double pendulum while freeing {0, 3} gives (3L, 1L).
    """
    m = n_active.shape[0]
    device = n_active.device
    keys = torch.rand(m, n_max, device=device)
    keys[:, 0] = -1.0  # rank joint 0 first unconditionally

    order = keys.argsort(dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(n_max, device=device).expand(m, n_max))
    return rank < n_active.unsqueeze(1)


def segment_geometry(
    free_mask: torch.Tensor, link_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Effective segment lengths, and the joint index heading each segment.

    A locked joint fuses its two links, so a "segment" is a maximal run of links whose
    interior joints are all locked. Both outputs are ordered base → tip and padded past
    the active count — lengths with zeros, indices with the locked joints in ascending
    order (which the caller masks off).
    """
    m, n_max = free_mask.shape
    device = free_mask.device

    # Link j belongs to the segment opened by the most recent free joint at or before j.
    # Joint 0 is always free, so this is >= 0 everywhere.
    seg_id = free_mask.long().cumsum(dim=1) - 1
    seg_lengths = torch.zeros(m, n_max, device=device)
    seg_lengths.scatter_add_(1, seg_id, link_lengths.expand(m, n_max))

    idx = torch.arange(n_max, device=device).expand(m, n_max)
    seg_joint_idx = torch.where(free_mask, idx, idx + n_max).argsort(dim=1)
    return seg_lengths, seg_joint_idx


def active_mask(n_active: torch.Tensor, n_max: int) -> torch.Tensor:
    """Boolean mask over the padded segment slots that are actually in use."""
    idx = torch.arange(n_max, device=n_active.device).expand(n_active.shape[0], n_max)
    return idx < n_active.unsqueeze(1)


def segment_kinematics(
    pole_joint_pos: torch.Tensor,
    pole_joint_vel: torch.Tensor,
    seg_joint_idx: torch.Tensor,
    seg_lengths: torch.Tensor,
    active: torch.Tensor,
    total_length: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-segment absolute angle, absolute angular velocity, and normalised tip height.

    A URDF revolute joint reports its angle relative to its parent, so the absolute angle
    of link `j` from vertical is the running sum along the chain — and locked joints
    contribute ~0 to that sum, which is precisely why locking behaves like a rigid weld.

    Height is the sum of each segment's vertical extent over the chain's total length:
    +1 straight up, -1 straight down. Unlike a per-joint angle threshold it means the
    same thing at every link count, so one reward and one termination rule serve the
    whole curriculum.
    """
    abs_angle = pole_joint_pos.cumsum(dim=1)
    abs_ang_vel = pole_joint_vel.cumsum(dim=1)

    zeros = torch.zeros_like(abs_angle)
    seg_angles = torch.where(active, abs_angle.gather(1, seg_joint_idx), zeros)
    seg_ang_vels = torch.where(active, abs_ang_vel.gather(1, seg_joint_idx), zeros)

    # `seg_lengths` is already zero past the active count, so no extra masking needed.
    height_norm = (seg_lengths * seg_angles.cos()).sum(dim=1) / total_length
    return seg_angles, seg_ang_vels, height_norm
