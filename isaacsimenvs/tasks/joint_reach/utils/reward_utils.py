"""Joint-reach reward: joint error to the goal, smoothness penalties and a table-proximity penalty."""

from __future__ import annotations

import torch

from .scene_utils import robot_points, table_top_and_halfsize

__all__ = ["compute_rewards"]


def compute_rewards(env) -> torch.Tensor:
    """Sum of the reward terms. Also stores env._reward_terms and env._in_tol, and counts the steps in
    tolerance and near the table."""
    rc = env.cfg.reach_reward
    err = env._err.abs()
    arm_mean, hand_mean = err[:, :7].mean(dim=-1), err[:, 7:].mean(dim=-1)
    in_tol = (err[:, :7].max(dim=-1).values < env._arm_tol) & (err[:, 7:].max(dim=-1).values < env._hand_tol)

    _, hx, hy = table_top_and_halfsize(env)
    _, points = robot_points(env)
    over = (points[..., 0].abs() <= hx) & (points[..., 1].abs() <= hy)
    top_now = (env._table_z_per_env + 0.5 * env.cfg.goal.table_box_size[2]).unsqueeze(1)
    near_table = (over & (points[..., 2] < top_now + rc.table_proximity)).any(dim=-1)

    qd = env.robot.data.joint_vel
    terms = {
        "arm_error": -rc.arm_error_scale * arm_mean,
        "hand_error": -rc.hand_error_scale * hand_mean,
        "arm_exp": rc.arm_exp_weight * torch.exp(-arm_mean / rc.arm_exp_width),
        "hand_exp": rc.hand_exp_weight * torch.exp(-hand_mean / rc.hand_exp_width),
        "within_tol": rc.within_tol_bonus * in_tol.float(),
        "action_rate": -rc.action_rate_scale * (env._actions - env._prev_actions).pow(2).mean(dim=-1),
        "arm_action": -rc.arm_action_scale * env._actions[:, :7].pow(2).mean(dim=-1),
        "joint_vel": -rc.joint_vel_scale * qd.pow(2).mean(dim=-1),
        "near_table": -rc.table_proximity_penalty * near_table.float(),
    }
    reward = torch.stack(list(terms.values())).sum(dim=0)
    env._reward_terms = terms

    env._in_tol = in_tol
    env._steps_in_tol += in_tol.float()
    env._near_table_steps += near_table.float()
    return reward
