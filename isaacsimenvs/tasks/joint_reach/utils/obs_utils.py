"""Joint-reach observations.

Policy observation (4 x 27 = 108), each block canonical order:
    joint_pos            normalised to [-1, 1] by the joint limits (obs_utils._canonical_joint_obs)
    joint_vel            rad/s
    prev_action_targets  rad, the action pipeline's last joint targets
    goal                 normalised like joint_pos

The critic adds STATE_EXTRA privileged inputs to the noise- and delay-free policy blocks.
"""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.play.utils.obs_utils import _apply_obs_delay, _canonical_joint_obs

__all__ = ["STATE_EXTRA", "build_observations"]

#: Extra critic inputs: goal error (27) + log gain multipliers (4) + obs / action delay (2) + hold phase (1).
STATE_EXTRA = 27 + 4 + 2 + 1


def build_observations(env) -> dict[str, torch.Tensor]:
    dr = env.cfg.domain_randomization
    joint_pos, joint_vel, prev_targets = _canonical_joint_obs(env)
    span = env._hi - env._lo
    goal = 2.0 * (env._goal - env._lo) / span - 1.0

    pos_obs, vel_obs = joint_pos, joint_vel
    if dr.joint_pos_obs_noise_std > 0:
        pos_obs = joint_pos + torch.randn_like(joint_pos) * dr.joint_pos_obs_noise_std * 2.0 / span
    if dr.joint_velocity_obs_noise_std > 0:
        vel_obs = joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
    policy = torch.cat([pos_obs, vel_obs, prev_targets, goal], dim=-1)
    if dr.use_obs_delay:
        policy = _apply_obs_delay(env, policy)

    phase = (1.0 - env._hold_left.float() / env._hold_len.float()).unsqueeze(-1)
    state = torch.cat(
        [
            joint_pos, joint_vel, prev_targets, goal, env._err,
            torch.log(env._gain_scale),
            env._obs_delay_idx.float().unsqueeze(-1),
            env._action_delay_idx.float().unsqueeze(-1),
            phase,
        ],
        dim=-1,
    )
    clip = env.cfg.obs.clamp_abs_observations
    return {"policy": policy.clamp(-clip, clip), "critic": state.clamp(-clip, clip)}
