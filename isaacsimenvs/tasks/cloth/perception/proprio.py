"""``ProprioObservation``: what the robot reports of itself -- the joint encoders, the last joint
targets, and through forward kinematics the palm and the fingertips -- with the proprioception DR
(``domain_randomization.proprio``).

The encoding is the one the Play prior was trained on: joints in the policy's (canonical) order,
``joint_pos`` normalised to [-1, 1] over the joint limits, quaternions in (x, y, z, w), and
``palm_pos`` as if the robot base stood at the Play task's position (``PLAY_ROBOT_ROOT_POS``).

DR: joint noise every step, and one delay line for every policy field that is not the sheet's
(``perception/keypoints.OBJECT_FIELDS``) -- the task fields a policy might read ride along. The
per-episode lag is :class:`~isaacsimenvs.tasks.cloth.mdp.randomization.DomainRandomization`'s.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaacsimenvs.tasks.cloth.mdp.delay import DelayLine
from isaacsimenvs.tasks.cloth.mdp.geometry import quat
from isaacsimenvs.tasks.cloth.mdp.robot import NUM_FINGERTIPS
from isaacsimenvs.tasks.cloth.mdp.task import apply_local_offset

__all__ = ["FIELD_SIZES", "PER_JOINT_FIELDS", "PLAY_ROBOT_ROOT_POS", "ProprioObservation", "ProprioReading"]

#: Where the Play task puts the robot base; ``palm_pos`` is observed relative to it.
PLAY_ROBOT_ROOT_POS: tuple[float, float, float] = (0.0, 0.8, 0.0)

#: Fields with one entry per robot joint; their size follows the robot.
PER_JOINT_FIELDS = ("joint_pos", "joint_vel", "prev_action_targets")

#: The other fields this sensor gives and their widths.
FIELD_SIZES: dict[str, int] = {
    "palm_pos": 3,
    "palm_rot": 4,
    "palm_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,
}


@dataclass
class ProprioReading:
    """One step of the robot. ``fields``: the clean fields; ``palm_pos``: the true palm centre, env
    frame; ``palm_frame_offset``: ``palm_pos`` minus the observed ``palm_pos`` (the Play frame shift)."""

    fields: dict[str, torch.Tensor]
    palm_pos: torch.Tensor
    palm_frame_offset: torch.Tensor


class ProprioObservation:
    def __init__(self, cfg, robot, controller, dr, *, num_envs: int, device, delayed_fields, width: int) -> None:
        """``robot``: the ``RobotLayout``; ``controller``: the ``JointController`` (its last targets);
        ``dr``: the env's ``DomainRandomization``. ``delayed_fields`` (``width`` wide in all) go
        through the delay line."""
        self.cfg, self.robot, self.controller, self.dr = cfg, robot, controller, dr
        self.num_envs = num_envs
        self.delayed_fields = list(delayed_fields)
        self.delay = DelayLine(num_envs, cfg.domain_randomization.proprio.delay_max, width, device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.delay.clear(env_ids)

    def observe(self, state, env_origins: torch.Tensor) -> ProprioReading:
        """The clean fields of ``state`` (``SimState``)."""
        robot, n = self.robot, self.num_envs
        perm = robot.perm_lab_to_canon
        joint_pos_raw = state.joint_pos[:, perm]
        joint_pos = 2.0 * (joint_pos_raw - robot.joint_lower_canon) / (robot.joint_upper_canon - robot.joint_lower_canon) - 1.0

        palm_state = state.body_state_w[:, robot.palm_body_id, :]  # (N, 13)
        palm_rot = palm_state[:, 3:7]  # wxyz
        palm_pos = apply_local_offset(palm_state[:, 0:3], palm_rot, robot.spec.palm_center_offset, (n,)) - env_origins
        # Observed as if the robot base stood at Play's position: zero unless cfg.robot_root_pos moves it.
        palm_pos_obs = palm_pos - (
            torch.as_tensor(self.cfg.robot_root_pos, device=palm_pos.device, dtype=palm_pos.dtype)
            - torch.as_tensor(PLAY_ROBOT_ROOT_POS, device=palm_pos.device, dtype=palm_pos.dtype)
        )

        ft_state = state.body_state_w[:, robot.fingertip_body_ids, :]  # (N, 5, 13)
        ft_pos_w = apply_local_offset(
            ft_state[:, :, 0:3], ft_state[:, :, 3:7], robot.spec.fingertip_offset, (n, NUM_FINGERTIPS)
        )
        fields = {
            "joint_pos": joint_pos,
            "joint_vel": state.joint_vel[:, perm],
            "prev_action_targets": self.controller.prev_targets[:, perm],
            "palm_pos": palm_pos_obs,
            "palm_rot": quat.convert_quat(palm_rot, to="xyzw"),
            "palm_vel": palm_state[:, 7:13],
            "fingertip_pos_rel_palm": (ft_pos_w - env_origins.unsqueeze(1)) - palm_pos.unsqueeze(1),
        }
        return ProprioReading(fields, palm_pos, palm_pos - palm_pos_obs)

    def noised(self, fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The joint fields with this step's encoder noise (``joint_pos_noise_std`` [rad],
        ``joint_vel_noise_std`` [rad/s]); only the noised fields are returned."""
        dr, out = self.cfg.domain_randomization.proprio, {}
        if dr.joint_vel_noise_std > 0:
            out["joint_vel"] = fields["joint_vel"] + torch.randn_like(fields["joint_vel"]) * dr.joint_vel_noise_std
        if dr.joint_pos_noise_std > 0:
            # Noise in radians, carried through the [-1, 1] normalisation.
            span = self.robot.joint_upper_canon - self.robot.joint_lower_canon
            pos = fields["joint_pos"]
            out["joint_pos"] = pos + torch.randn_like(pos) * dr.joint_pos_noise_std * 2.0 / span
        return out

    def delayed(self, fields: dict[str, torch.Tensor], start: torch.Tensor) -> dict[str, torch.Tensor]:
        """``delayed_fields`` read ``proprio_lag`` steps back (a pass-through without a delay).
        ``start``: the envs on their episode's first step."""
        names = self.delayed_fields
        flat = torch.cat([fields[f].reshape(self.num_envs, -1) for f in names], dim=-1)
        old = self.delay.push_and_read(flat, start, self.dr.proprio_lag)
        out, begin = {}, 0
        for f in names:
            width = fields[f].reshape(self.num_envs, -1).shape[1]
            out[f] = old[:, begin:begin + width].reshape(fields[f].shape)
            begin += width
        return out
