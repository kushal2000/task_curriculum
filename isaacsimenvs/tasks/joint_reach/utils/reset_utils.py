"""Joint-reach buffers, robot placement, per-episode reset and joint gain randomisation.

Replaces Play's reset_env_state: the robot starts either at the default pose (Play's reset noise) at
the start of a default-seeded chain, or exactly at a random goal of a random chain; the object is parked.
"""

from __future__ import annotations

import math

import torch

from isaacsimenvs.tasks.play.utils.reset_utils import _draw_episode_dr, _reset_table_pose

from .goal_utils import advance_goals
from .scene_utils import park_object

__all__ = ["allocate_reach_buffers", "draw_gains", "place_robot", "reset_reach_state"]


def _log_uniform(lo_hi, n: int, device) -> torch.Tensor:
    lo, hi = (float(v) for v in lo_hi)
    return torch.exp(torch.empty(n, device=device).uniform_(math.log(lo), math.log(hi)))


def allocate_reach_buffers(env) -> None:
    n_env, n = env.num_envs, env.robot_spec.num_joints
    dev = env.device
    env._goal = torch.zeros(n_env, n, device=dev)
    env._hold_left = torch.zeros(n_env, dtype=torch.long, device=dev)
    env._hold_len = torch.ones(n_env, dtype=torch.long, device=dev)
    env._actions = torch.zeros(n_env, n, device=dev)
    env._prev_actions = torch.zeros(n_env, n, device=dev)
    env._err = torch.zeros(n_env, n, device=dev)
    env._in_tol = torch.zeros(n_env, dtype=torch.bool, device=dev)
    # When True the env never draws goals itself: the caller writes _goal (eval / replay).
    env.external_goals = False
    env._chain_id = torch.zeros(n_env, dtype=torch.long, device=dev)
    env._chain_idx = torch.zeros(n_env, dtype=torch.long, device=dev)
    env._chain_dir = torch.ones(n_env, dtype=torch.long, device=dev)

    stiff, damp = env.robot.data.joint_stiffness, env.robot.data.joint_damping
    env._base_k_lab = (stiff.torch if hasattr(stiff, "torch") else stiff).clone()
    env._base_d_lab = (damp.torch if hasattr(damp, "torch") else damp).clone()
    env._arm_mask_lab = torch.zeros(n, dtype=torch.bool, device=dev)
    env._arm_mask_lab[env._arm_joint_ids] = True
    # Columns: arm stiffness, hand stiffness, arm damping, hand damping.
    env._gain_scale = torch.ones(n_env, 4, device=dev)

    env._lo = env._joint_lower_canon
    env._hi = env._joint_upper_canon
    env._arm_tol = math.radians(env.cfg.reach_reward.arm_tol_deg)
    env._hand_tol = math.radians(env.cfg.reach_reward.hand_tol_deg)

    # Per-episode tallies over completed goal segments (logged at episode end).
    env._seg_done = torch.zeros(n_env, device=dev)
    env._seg_reached = torch.zeros(n_env, device=dev)
    env._seg_arm_err = torch.zeros(n_env, device=dev)
    env._seg_hand_err = torch.zeros(n_env, device=dev)
    env._seg_first_in_tol = torch.full((n_env,), -1, dtype=torch.long, device=dev)
    env._seg_settle_sum = torch.zeros(n_env, device=dev)
    env._steps_in_tol = torch.zeros(n_env, device=dev)
    env._near_table_steps = torch.zeros(n_env, device=dev)
    env._reward_terms: dict[str, torch.Tensor] = {}


def place_robot(env, env_ids: torch.Tensor, q_canon: torch.Tensor) -> None:
    """Write a joint configuration (canonical order) as the robot state and the pipeline's targets."""
    q = q_canon[:, env._perm_canon_to_lab]
    env.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    env._prev_targets[env_ids] = q
    env._cur_targets[env_ids] = q


def draw_gains(env, env_ids: torch.Tensor) -> None:
    gd = env.cfg.gain_dr
    if not gd.enabled:
        return
    n = env_ids.numel()
    scale = torch.stack(
        [
            _log_uniform(gd.arm_stiffness, n, env.device),
            _log_uniform(gd.hand_stiffness, n, env.device),
            _log_uniform(gd.arm_damping, n, env.device),
            _log_uniform(gd.hand_damping, n, env.device),
        ],
        dim=-1,
    )
    fixed = torch.tensor([float(v) for v in gd.fixed_scales], device=env.device)
    scale = torch.where(fixed > 0.0, fixed.expand(n, -1), scale)
    env._gain_scale[env_ids] = scale
    arm = env._arm_mask_lab.unsqueeze(0)
    k = env._base_k_lab[env_ids] * torch.where(arm, scale[:, 0:1], scale[:, 1:2])
    d = env._base_d_lab[env_ids] * torch.where(arm, scale[:, 2:3], scale[:, 3:4])
    env.robot.write_joint_stiffness_to_sim(k.contiguous(), env_ids=env_ids)
    env.robot.write_joint_damping_to_sim(d.contiguous(), env_ids=env_ids)


def reset_reach_state(env, env_ids: torch.Tensor) -> None:
    n = env_ids.numel()
    g, rc = env.cfg.goal, env.cfg.reset
    dev = env.device
    n_chain, length = env._chains.shape[:2]

    # Either the default pose (Play's reset noise) at the start of a default-seeded chain, or a
    # random goal of a random chain, exactly.
    from_default = torch.rand(n, device=dev) < g.default_start_prob
    chain_id = torch.where(
        from_default,
        torch.randint(0, max(env._n_default_chains, 1), (n,), device=dev),
        torch.randint(0, n_chain, (n,), device=dev),
    )
    idx = torch.where(from_default, torch.zeros_like(chain_id), torch.randint(0, length, (n,), device=dev))
    default = env.robot.data.default_joint_pos[env_ids]
    lower = env.robot.data.joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.joint_pos_limits[env_ids, :, 1]
    scale = torch.zeros_like(default)
    scale[:, env._arm_joint_ids] = rc.reset_dof_pos_random_interval_arm
    scale[:, env._hand_joint_ids] = rc.reset_dof_pos_random_interval_fingers
    q_lab = torch.lerp(default, lower + (upper - lower) * torch.rand_like(default), scale).clamp(lower, upper)
    q = q_lab[:, env._perm_lab_to_canon]
    q = torch.where(from_default.unsqueeze(1), q, env._chains[chain_id, idx])
    place_robot(env, env_ids, q)

    _reset_table_pose(env, env_ids)
    park_object(env, env_ids)

    env._chain_id[env_ids] = chain_id
    env._chain_idx[env_ids] = idx
    env._chain_dir[env_ids] = torch.where(torch.rand(n, device=dev) < 0.5, 1, -1)
    env._chain_dir[env_ids] = torch.where(from_default, torch.ones_like(chain_id), env._chain_dir[env_ids])
    env._goal[env_ids] = q
    if not env.external_goals:
        advance_goals(env, env_ids)
    env._actions[env_ids] = 0.0
    env._prev_actions[env_ids] = 0.0
    env._action_queue[env_ids] = 0.0
    env._obs_queue[env_ids] = 0.0
    _draw_episode_dr(env, env_ids)  # delays; the gain ranges there are (1, 1)
    draw_gains(env, env_ids)

    for buf in (
        env._seg_done, env._seg_reached, env._seg_arm_err, env._seg_hand_err,
        env._seg_settle_sum, env._steps_in_tol, env._near_table_steps,
    ):
        buf[env_ids] = 0.0
    env._seg_first_in_tol[env_ids] = -1
