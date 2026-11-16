"""The reward: fingertip approach + fold progress + the fold bonus + joint-velocity penalties.

* ``fingertip_delta_rew``: each pad's best distance to the sheet centre so far this episode; pays
  ``distance_delta_rew_scale`` x the summed improvement (clamped >= 0): find the sheet.
* ``keypoint_rew``: the same ratchet on the fold error (the success test's own quantity):
  ``keypoint_rew_scale`` x improvement on the best so far. Monotone and clamped, so oscillating
  cannot farm it and regressing costs nothing.
* ``bonus_rew``: ``reach_goal_bonus`` on the step the fold is entered (``_bonus_mask``, set by
  ``termination_utils.compute`` earlier in the same step); with ``fold.release`` the entry bonus,
  release and persist terms instead.
* ``kuka_actions_penalty`` / ``hand_actions_penalty``: L1 joint velocity (Play's names).

A non-finite reward, or one beyond ``REWARD_SANITY_LIMIT`` (a diverging articulation's action
penalties reach 1e5-1e11 while staying finite), is zeroed and its env flagged for reset.
"""

from __future__ import annotations

import torch

from .state import ClothState, TaskView

#: The largest legitimate one-step reward is the bonus (1000) plus shaping of order 100.
REWARD_SANITY_LIMIT = 1.0e4


def ratchet(best: torch.Tensor, current: torch.Tensor, cap: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(improvement clamped to [0, cap], new best)``; ``best < 0`` = not yet observed."""
    best = torch.where(best < 0.0, current, best)
    return torch.clamp(best - current, 0.0, cap), torch.minimum(best, current)


def guard(env, reward: torch.Tensor) -> torch.Tensor:
    """Zero and flag non-finite or implausibly large rewards."""
    bad = ~torch.isfinite(reward) | (reward.abs() > REWARD_SANITY_LIMIT)
    env.extras["insane_reward"] = float(bad.float().mean())
    if not bool(bad.any()):
        return reward
    worst = reward[bad].abs().nan_to_num(nan=float("inf")).max()
    print(f"[cloth] bad reward in {int(bad.sum())}/{env.num_envs} envs (worst |r| {float(worst):.3e}) "
          f"-- zeroed and flagged for reset", flush=True)
    env._nonfinite_envs |= bad
    env._insane_count += int(bad.sum())
    return torch.where(bad, torch.zeros_like(reward), reward)


def compute(env, s: ClothState, v: TaskView) -> torch.Tensor:
    rew, rel = env.cfg.reward, env.cfg.fold.release

    ft_gain, env._closest_fingertip_dist = ratchet(env._closest_fingertip_dist, v.fingertip_dist, 10.0)
    fingertip = rew.distance_delta_rew_scale * ft_gain.sum(dim=-1)

    fold_err = v.fold_error
    # A non-finite env is being reset this step; keep the ratchet finite (no progress paid).
    fold_err = torch.where(torch.isfinite(fold_err), fold_err, env._closest_fold_err.clamp(min=0.0))
    fold_gain, env._closest_fold_err = ratchet(env._closest_fold_err, fold_err, 100.0)
    keypoint = rew.keypoint_rew_scale * fold_gain

    arm = -rew.kuka_actions_penalty_scale * s.joint_vel[:, env._arm_ids].abs().sum(dim=-1)
    hand = -rew.hand_actions_penalty_scale * s.joint_vel[:, env._hand_ids].abs().sum(dim=-1)

    terms = {"fingertip_delta_rew": fingertip, "keypoint_rew": keypoint,
             "kuka_actions_penalty": arm, "hand_actions_penalty": hand}
    if rel.enabled:
        terms["bonus_rew"] = env._bonus_mask.float() * rel.fold_entry_bonus
        terms["release_rew"] = env._release_rew_step + env._release_latch_mask.float() * rel.release_bonus
        terms["persist_rew"] = (env._persist_step_mask.float() * rel.persist_rew_per_step
                                + env._persist_done.float() * rel.persist_bonus
                                + env._persist_fail.float() * rel.persist_fail_penalty)
    else:
        terms["bonus_rew"] = env._bonus_mask.float() * rew.reach_goal_bonus

    reward = guard(env, sum(terms.values()))
    terms["total_reward"] = reward
    env._reward_terms = terms
    env.extras["fold_err_mean"] = float(fold_err.mean())
    env.extras["fold_rate"] = float(env._bonus_mask.float().mean())
    return reward
