"""Policy action -> joint position targets: ``JointController``.

The policy acts in canonical joint order, every entry in [-1, 1]:

* **arm**: a joint velocity command. ``target += dof_speed_scale * dt * a``, clamped to the joint
  limits, then low-passed (``arm_moving_average``) against the previous target;
* **hand**: an absolute position, ``a`` mapped linearly onto the joint range, then low-passed
  (``hand_moving_average``).

Domain randomization on the way (``domain_randomization.action``): a delay line (``delay_max``, one
lag per episode) and Gaussian noise (``noise_std``) on the action that is executed.

``targets`` is what the env writes every physics step; ``prev_targets`` is the smoothing state and
the ``prev_action_targets`` observation.
"""

from __future__ import annotations

import torch

from .randomization import DelayLine, DomainRandomization
from .robot import RobotLayout

__all__ = ["JointController"]


class JointController:
    def __init__(self, cfg, layout: RobotLayout, dr: DomainRandomization, *, step_dt: float, num_envs: int, device) -> None:
        self.cfg, self.layout, self.dr = cfg, layout, dr
        self.step_dt = float(step_dt)
        self.num_envs, self.device = num_envs, device
        n_joints = layout.num_joints
        self.targets = torch.zeros(num_envs, n_joints, device=device)
        self.prev_targets = torch.zeros(num_envs, n_joints, device=device)
        self.action_delay = DelayLine(num_envs, cfg.domain_randomization.action.delay_max, n_joints, device)

    def process(self, actions: torch.Tensor, start: torch.Tensor) -> None:
        """This policy step's ``actions`` (N, J, canonical) -> ``targets`` (Lab order). ``start``:
        the envs on their episode's first step (the delay line flushes there)."""
        dr = self.cfg.domain_randomization.action
        act_cfg = self.cfg.action
        layout = self.layout
        n_arm = len(layout.arm_joint_ids)

        actions = actions.clone().to(self.device)[:, layout.perm_canon_to_lab]

        if dr.delay_max > 0:
            actions = self.action_delay.push_and_read(actions, start, self.dr.action_lag)

        # On the executed (delayed) action, kept in the policy's [-1, 1].
        if dr.noise_std > 0:
            actions = (actions + torch.randn_like(actions) * dr.noise_std).clamp(-1.0, 1.0)

        prev = self.prev_targets
        arm_raw = prev[:, :n_arm] + act_cfg.dof_speed_scale * self.step_dt * actions[:, :n_arm]
        arm_raw = torch.clamp(arm_raw, layout.arm_lower, layout.arm_upper)
        arm = act_cfg.arm_moving_average * arm_raw + (1.0 - act_cfg.arm_moving_average) * prev[:, :n_arm]
        arm = torch.clamp(arm, layout.arm_lower, layout.arm_upper)

        hand_raw = layout.hand_lower + 0.5 * (actions[:, n_arm:] + 1.0) * (layout.hand_upper - layout.hand_lower)
        hand = act_cfg.hand_moving_average * hand_raw + (1.0 - act_cfg.hand_moving_average) * prev[:, n_arm:]
        hand = torch.clamp(hand, layout.hand_lower, layout.hand_upper)

        self.targets[:, layout.arm_joint_ids] = arm
        self.targets[:, layout.hand_joint_ids] = hand
        self.prev_targets = self.targets.clone()

    def spring_effort(self, joint_pos: torch.Tensor) -> torch.Tensor | None:
        return self.layout.spring_effort(joint_pos)

    def sample_reset(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint state at reset: the default pose blended toward a uniform draw within the limits,
        and a uniform joint velocity. ``(joint_pos, joint_vel)``, Lab order."""
        cfg = self.cfg.reset
        layout = self.layout
        default_pos = layout.default_joint_pos[env_ids]
        lower = layout.joint_pos_limits[env_ids, :, 0]
        upper = layout.joint_pos_limits[env_ids, :, 1]

        reset_scale = torch.zeros_like(default_pos)
        reset_scale[:, layout.arm_joint_ids] = cfg.reset_dof_pos_random_interval_arm
        reset_scale[:, layout.hand_joint_ids] = cfg.reset_dof_pos_random_interval_fingers

        sampled_pos = lower + (upper - lower) * torch.rand_like(default_pos)
        joint_pos = torch.lerp(default_pos, sampled_pos, reset_scale).clamp(lower, upper)
        joint_vel = torch.empty_like(default_pos).uniform_(
            -cfg.reset_dof_vel_random_interval, cfg.reset_dof_vel_random_interval
        )
        return joint_pos, joint_vel

    def reset(self, env_ids: torch.Tensor, joint_pos: torch.Tensor) -> None:
        """The targets start at the reset pose; the action delay line is cleared."""
        self.prev_targets[env_ids] = joint_pos
        self.targets[env_ids] = joint_pos
        self.action_delay.clear(env_ids)
