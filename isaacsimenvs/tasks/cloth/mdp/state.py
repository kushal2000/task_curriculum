"""Common world-frame state exchanged by physics, task, and policy code.

Cloth points are surface samples: rigid plates carry them; VBD meshes supply them.
Quaternions are wxyz and joints use Isaac Lab order at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SimState:
    joint_pos: torch.Tensor  # (N, J) Lab order
    joint_vel: torch.Tensor  # (N, J)
    palm_pos_w: torch.Tensor  # (N, 3) palm BODY origin (the centre offset is the obs's)
    palm_quat_w: torch.Tensor  # (N, 4) wxyz
    palm_vel_w: torch.Tensor  # (N, 6) linear, angular
    fingertip_pos_w: torch.Tensor  # (N, 5, 3) fingertip BODY origins, policy order
    fingertip_quat_w: torch.Tensor  # (N, 5, 4) wxyz
    hand_pos_w: torch.Tensor  # (N, H, 3) origins of the release-test bodies
    points_w: torch.Tensor  # (N, P, 3) cloth points (VBD particles / rigid plate grid)
    points_vel_w: torch.Tensor  # (N, P, 3)
    table_top_z: torch.Tensor  # (N,) world z of the table's top surface
    keypoints_w: torch.Tensor | None = None  # (N, 4, 3) optional exact moving-half corners


@dataclass
class TaskView:
    """What the task derives from one ``SimState``: computed once per hook, read by obs, reward
    and termination alike (so they can never disagree about where the cloth is)."""

    frame_r: torch.Tensor  # (N, 3, 3) stationary half, rest -> current
    frame_c: torch.Tensor  # (N, 3) its centroid
    pos_w: torch.Tensor  # (N, 3) cloth centre (all points)
    quat_w: torch.Tensor  # (N, 4) wxyz, the stationary half's orientation
    lin_vel_w: torch.Tensor  # (N, 3)
    ang_vel_w: torch.Tensor  # (N, 3)
    corners_w: torch.Tensor  # (N, 4, 3) explicit moving-half landmarks, or legacy grid corners
    targets_w: torch.Tensor  # (N, 4, 3) where they belong
    fold_error: torch.Tensor  # (N,)
    footprint: torch.Tensor  # (N,)
    pads_w: torch.Tensor  # (N, 5, 3) fingertip pads (or body origins)
    fingertip_dist: torch.Tensor  # (N, 5) pad -> cloth centre


@dataclass
class JointCommands:
    position: torch.Tensor
    effort: torch.Tensor


@dataclass
class ResetState:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    table_z: torch.Tensor
    yaw: torch.Tensor
    fold_angle: torch.Tensor


@dataclass
class TaskTransition:
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
