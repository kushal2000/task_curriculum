"""Padded, difficulty-conditioned observations for the multi-link cartpole.

The observation is sized for `n_max` segments no matter how many are actually free, and
the inactive slots are zero-filled. That padding is what makes a stage-2 run able to
restore a stage-1 checkpoint with `--checkpoint_load_mode weights`: the network's input
width never changes as the curriculum advances.

Segment lengths ride along in the observation so the policy can tell which morphology it
is currently driving — without them, a padded observation is ambiguous between "a short
2-link chain" and "a long 2-link chain" and the policy can only learn the average.
"""

from __future__ import annotations

import torch

from .difficulty_math import active_mask, segment_kinematics

__all__ = [
    "compute_obs_dim",
    "compute_state_dim",
    "compute_intermediate_values",
    "build_observations",
]


def compute_obs_dim(cfg) -> int:
    """Policy observation width implied by the config. Called before `DirectRLEnv.__init__`."""
    per_segment = 2  # angle + angular velocity
    if cfg.obs.include_segment_lengths:
        per_segment += 1
    if cfg.obs.include_free_mask:
        per_segment += 1
    return 2 + per_segment * cfg.geometry.n_max


def compute_state_dim(cfg) -> int:
    """Critic observation width: the policy view plus privileged morphology.

    The critic additionally sees the raw per-joint angles and velocities (unmasked, all
    `n_max` of them) and the free/locked mask — i.e. exactly which joints are welded this
    episode. The actor only ever sees the folded per-*segment* view, so this is genuine
    asymmetry rather than a padded copy: the critic can tell a locked joint from a free
    one held near zero, which is the main thing that makes value estimation hard here.

    A non-zero `state_space` is also a hard requirement for SAPG. The vendored fork's
    `a2c_common.env_reset` does an unconditional
    `obs['states'] = cat([obs['states'], intr_reward_coef_embd])` whenever
    `expl_type` is a `mixed_expl*` variant, so a task with no critic observation raises
    `KeyError: 'states'` the moment SAPG training starts.
    """
    n_max = cfg.geometry.n_max
    extra = 2 * n_max  # raw joint angles + velocities
    if not cfg.obs.include_free_mask:
        extra += n_max  # the mask, only if the actor is not already getting it
    return compute_obs_dim(cfg) + extra


def compute_intermediate_values(env) -> None:
    """Refresh the per-segment kinematics every step, before rewards and dones read them."""
    joint_pos = env.cartpole.data.joint_pos
    joint_vel = env.cartpole.data.joint_vel

    env._seg_angles, env._seg_ang_vels, env._height_norm = segment_kinematics(
        pole_joint_pos=joint_pos[:, env._pole_dof_ids],
        pole_joint_vel=joint_vel[:, env._pole_dof_ids],
        seg_joint_idx=env._seg_joint_idx,
        seg_lengths=env._seg_lengths,
        active=active_mask(env._n_active, env.cfg.geometry.n_max),
        total_length=env._total_length,
    )


def build_observations(env) -> dict[str, torch.Tensor]:
    obs_cfg = env.cfg.obs
    joint_pos = env.cartpole.data.joint_pos
    joint_vel = env.cartpole.data.joint_vel

    cart_pos = joint_pos[:, env._cart_dof_idx[0]]
    if obs_cfg.normalize_cart_pos:
        cart_pos = cart_pos / (0.5 * env.cfg.geometry.rail_length)
    cart_vel = joint_vel[:, env._cart_dof_idx[0]]

    parts = [cart_pos.unsqueeze(1), cart_vel.unsqueeze(1), env._seg_angles, env._seg_ang_vels]
    if obs_cfg.include_segment_lengths:
        parts.append(env._seg_lengths)
    if obs_cfg.include_free_mask:
        parts.append(env._free_mask.float())

    obs = torch.cat(parts, dim=-1)

    # Privileged critic view: the policy observation plus the *raw* joint state. The
    # free mask is appended only when the actor does not already have it, so the two
    # never carry a duplicated slice. See `compute_state_dim`.
    extra = [joint_pos[:, env._pole_dof_ids], joint_vel[:, env._pole_dof_ids]]
    if not obs_cfg.include_free_mask:
        extra.insert(0, env._free_mask.float())
    state = torch.cat([obs, *extra], dim=-1)

    if obs_cfg.clamp_abs_observations > 0.0:
        clamp = obs_cfg.clamp_abs_observations
        obs = obs.clamp(-clamp, clamp)
        state = state.clamp(-clamp, clamp)
    return {"policy": obs, "critic": state}
