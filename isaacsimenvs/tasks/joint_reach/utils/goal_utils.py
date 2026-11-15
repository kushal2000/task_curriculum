"""Per-episode goal stepping: each env walks its chain, holding every goal for a timed segment.

A segment's hold time is the largest arm joint move at the action pipeline's speed limit plus a random
settle time; when it runs out the segment is tallied and the next goal along the chain is set, turning
back at either end of the chain. With env.external_goals the caller writes env._goal and none of this runs.
"""

from __future__ import annotations

import torch

__all__ = ["advance_goals", "hold_steps", "update_goals"]


def hold_steps(env, goal_from: torch.Tensor, goal_to: torch.Tensor, settle_s: torch.Tensor) -> torch.Tensor:
    """Policy steps to hold `goal_to` after `goal_from`: the largest arm joint move at the action
    pipeline's arm speed limit, plus `settle_s`."""
    a = env.cfg.action
    v = a.dof_speed_scale * a.arm_moving_average  # rad/s the joint targets can move
    travel = (goal_to[:, :7] - goal_from[:, :7]).abs().max(dim=-1).values / v
    return ((travel + settle_s) / env.step_dt).round().long().clamp(min=1)


def set_goal(env, env_ids: torch.Tensor, goal_from: torch.Tensor) -> None:
    """Write the chain goal at _chain_idx as the goal of `env_ids` and start its hold timer."""
    lo_s, hi_s = env.cfg.goal.settle_time_range_s
    goal = env._chains[env._chain_id[env_ids], env._chain_idx[env_ids]]
    env._goal[env_ids] = goal
    settle = torch.empty(env_ids.numel(), device=env.device).uniform_(lo_s, hi_s)
    env._hold_len[env_ids] = hold_steps(env, goal_from, goal, settle)
    env._hold_left[env_ids] = env._hold_len[env_ids]
    env._seg_first_in_tol[env_ids] = -1


def advance_goals(env, env_ids: torch.Tensor) -> None:
    """Next goal along each env's chain, turning back at either end."""
    length = env._chains.shape[1]
    idx, direction = env._chain_idx[env_ids], env._chain_dir[env_ids]
    nxt = idx + direction
    turn = (nxt < 0) | (nxt >= length)
    direction = torch.where(turn, -direction, direction)
    env._chain_dir[env_ids] = direction
    env._chain_idx[env_ids] = torch.where(turn, idx + direction, nxt).clamp(0, length - 1)
    set_goal(env, env_ids, env._goal[env_ids].clone())


def close_segments(env, env_ids: torch.Tensor) -> None:
    """Tally the goal segments that end now (their hold time ran out)."""
    err = env._err[env_ids].abs()
    arm_max = err[:, :7].max(dim=-1).values
    hand_max = err[:, 7:].max(dim=-1).values
    reached = (arm_max < env._arm_tol) & (hand_max < env._hand_tol)
    env._seg_done[env_ids] += 1
    env._seg_reached[env_ids] += reached.float()
    env._seg_arm_err[env_ids] += torch.rad2deg(arm_max)
    env._seg_hand_err[env_ids] += torch.rad2deg(hand_max)
    first = env._seg_first_in_tol[env_ids]
    env._seg_settle_sum[env_ids] += torch.where(
        first >= 0, first.float(), env._hold_len[env_ids].float()
    ) * env.step_dt


def update_goals(env) -> None:
    """Segment bookkeeping for this step (env._in_tol from compute_rewards), then the next goals for
    segments whose hold time ran out."""
    if env.external_goals:
        return
    elapsed = env._hold_len - env._hold_left
    first = env._in_tol & (env._seg_first_in_tol < 0)
    env._seg_first_in_tol = torch.where(first, elapsed, env._seg_first_in_tol)
    env._hold_left -= 1
    ended = (env._hold_left <= 0).nonzero(as_tuple=False).squeeze(-1)
    if ended.numel() > 0:
        close_segments(env, ended)
        advance_goals(env, ended)
