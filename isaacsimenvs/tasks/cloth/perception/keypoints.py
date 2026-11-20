"""``KeypointObservation``: the sheet as a keypoint tracker reports it -- the moving half seen as a
rigid object and four keypoints on it -- with the keypoint DR (``domain_randomization.keypoints``).

The pose comes from the simulator's state, not from a sensor model: the moving half's corner
centroid and its Kabsch-fitted rotation (``FoldView.object_*``); the keypoints are
``centre + R * offsets`` (``task.keypoints_world``), for the sheet and for the fold goal. The same
keypoints are the task's truth for the reward and the success test (``mdp/task.py``); this module
is only what the policy reads of them.

DR on the pose the keypoints are built from: a delay and per-step noise, then the episode's bias.
The per-episode draws (the lag, the bias) are :class:`~isaacsimenvs.tasks.cloth.mdp.randomization.DomainRandomization`'s.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaacsimenvs.tasks.cloth.mdp.delay import DelayLine
from isaacsimenvs.tasks.cloth.mdp.geometry import quat
from isaacsimenvs.tasks.cloth.mdp.randomization import perturb_quat
from isaacsimenvs.tasks.cloth.mdp.task import keypoints_world

__all__ = ["AUX_FIELDS", "FIELD_SIZES", "NUM_KEYPOINTS", "OBJECT_FIELDS", "KeypointObservation", "KeypointReading"]

NUM_KEYPOINTS = 4

#: The policy fields that are the sheet's: built from the keypoints, or replaced by the point cloud.
#: Every other policy field goes through the proprioception's delay (``perception/proprio.py``).
OBJECT_FIELDS = ("object_rot", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales")
#: What a point-cloud student's auxiliary head regresses: the keypoint teacher's view of the sheet.
AUX_FIELDS = ("keypoints_rel_palm", "keypoints_rel_goal")

#: The fields this sensor gives and their widths.
FIELD_SIZES: dict[str, int] = {
    "object_rot": 4,
    "object_vel": 6,
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,
    "object_scales": 3,
}


def rel_palm(keypoints: torch.Tensor, palm_pos: torch.Tensor) -> torch.Tensor:
    """``keypoints_rel_palm``: (N, K, 3) keypoints minus the (N, 3) palm, both in the env frame."""
    return keypoints - palm_pos.unsqueeze(1)


@dataclass
class KeypointReading:
    """One step of the sheet. ``clean``: the true fields (critic, teacher); ``observed``: what the
    policy reads; ``observed_keypoints``: the observed keypoints (N, K, 3), env frame, for
    ``keypoints_rel_palm`` against a delayed palm."""

    clean: dict[str, torch.Tensor]
    observed: dict[str, torch.Tensor]
    observed_keypoints: torch.Tensor


class KeypointObservation:
    def __init__(self, cfg, dr, *, num_envs: int, device) -> None:
        """``dr``: the env's ``DomainRandomization`` (the episode's lag and bias)."""
        self.cfg, self.dr = cfg, dr
        self.delay = DelayLine(num_envs, cfg.domain_randomization.keypoints.delay_max, 13, device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.delay.clear(env_ids)

    def _observed_pose(self, start, obj_pos, obj_rot, obj_linvel, obj_angvel):
        """The sheet pose and velocity the keypoints are built from: delayed and noised every step
        (``keypoints.delay_max``, ``pos_noise_std``, ``rot_noise_deg``), then offset by the episode's
        bias (``pos_bias``, ``rot_bias_deg``). All neutral when off. Returns ``(pos, quat wxyz, vel (N, 6))``."""
        dr = self.cfg.domain_randomization.keypoints
        pos, rot = obj_pos, obj_rot
        vel = torch.cat([obj_linvel, obj_angvel], dim=-1)
        if dr.delay_max > 0 or dr.pos_noise_std > 0.0 or dr.rot_noise_deg > 0.0:
            state = torch.cat([obj_pos, obj_rot, vel], dim=-1)
            delayed = self.delay.push_and_read(state, start, self.dr.keypoints_lag)
            pos = delayed[:, 0:3] + torch.randn_like(delayed[:, 0:3]) * dr.pos_noise_std
            rot = perturb_quat(delayed[:, 3:7], dr.rot_noise_deg)
            vel = delayed[:, 7:13]
        pos = pos + self.dr.keypoints_bias_pos
        rot = quat.quat_mul(self.dr.keypoints_bias_quat, rot)
        return pos, rot, vel

    def observe(self, view, task, env_origins: torch.Tensor, palm_pos: torch.Tensor, start: torch.Tensor) -> KeypointReading:
        """The sheet's fields. ``palm_pos``: the true palm centre, env frame. ``start``: the envs on
        their episode's first step. Quaternions are observed in (x, y, z, w)."""
        obj_pos = view.object_pos_w - env_origins
        obj_rot = view.object_quat_w
        obj_linvel = view.object_lin_vel_w
        obj_angvel = view.object_ang_vel_w
        goal_pos = task.goal_pos_w - env_origins
        kp_offsets = task.keypoint_offsets
        obj_kp = keypoints_world(obj_pos, obj_rot, kp_offsets)
        goal_kp = keypoints_world(goal_pos, task.goal_quat, kp_offsets)
        seen_pos, seen_rot, seen_vel = self._observed_pose(start, obj_pos, obj_rot, obj_linvel, obj_angvel)
        seen_kp = keypoints_world(seen_pos, seen_rot, kp_offsets)
        clean = {
            "object_rot": quat.convert_quat(obj_rot, to="xyzw"),
            "object_vel": torch.cat([obj_linvel, obj_angvel], dim=-1),
            "keypoints_rel_palm": rel_palm(obj_kp, palm_pos),
            "keypoints_rel_goal": obj_kp - goal_kp,
            "object_scales": task.object_scale,
        }
        observed = {
            **clean,
            "object_rot": quat.convert_quat(seen_rot, to="xyzw"),
            "object_vel": seen_vel,
            "keypoints_rel_palm": rel_palm(seen_kp, palm_pos),
            "keypoints_rel_goal": seen_kp - goal_kp,
        }
        return KeypointReading(clean, observed, seen_kp)
