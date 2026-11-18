"""The reward, and the per-step hand and goal measurements it shares with the observation.

    reward = fingertip_approach + fold_progress + fold_bonus - arm_joint_speed - hand_joint_speed

* ``fingertip_approach``: pay for every fingertip getting closer to the sheet than it has been
  this episode (``distance_delta_rew_scale``), until the sheet is lifted;
* ``fold_progress``: pay for every new best fold error this episode (``keypoint_rew_scale``);
* ``fold_bonus``: ``reach_goal_bonus`` on the step the fold criterion is met (``terminations``);
* the joint-speed penalties: L1 joint velocity, arm and hand (``*_actions_penalty_scale``).

The reward is then guarded: a non-finite or implausibly large value (a diverging env) is zeroed and
the env flagged for reset.

:func:`update_hand_and_goal_metrics` also keeps the Play task's goal measurement -- the keypoints of
the fitted sheet pose against the goal pose's -- which the critic observes
(``closest_keypoint_max_dist``) and ``terminations`` reads.
"""

from __future__ import annotations

import torch

from .observations import apply_local_offset, keypoints_world
from .robot import NUM_FINGERTIPS

__all__ = ["REWARD_SANITY_LIMIT", "compute_reward", "update_hand_and_goal_metrics"]

#: Any |reward| above this is a diverging env, not a task outcome: the most the task can pay in one
#: step is ``reach_goal_bonus`` (1000) plus shaping of order 100.
REWARD_SANITY_LIMIT = 1.0e4


def update_hand_and_goal_metrics(env) -> None:
    """Fingertip-to-sheet distances, the goal keypoint distance, and the near-goal counter."""
    reward_cfg, term_cfg = env.cfg.reward, env.cfg.termination
    origins = env.scene.env_origins
    sheet_pos = env.object.data.root_pos_w - origins
    sheet_rot = env.object.data.root_quat_w
    goal_pos = env.goal_viz.data.root_pos_w - origins
    goal_rot = env.goal_viz.data.root_quat_w

    fingertips = env.robot.data.body_state_w[:, env._fingertip_body_ids, :]
    fingertip_pos_w = fingertips[:, :, 0:3]
    if reward_cfg.fingertip_distance_from_pad:
        fingertip_pos_w = apply_local_offset(
            fingertip_pos_w, fingertips[:, :, 3:7], env.robot_spec.fingertip_offset, (env.num_envs, NUM_FINGERTIPS)
        )
    fingertip_pos = fingertip_pos_w - origins.unsqueeze(1)
    env._curr_fingertip_distances = torch.norm(fingertip_pos - sheet_pos.unsqueeze(1), dim=-1)

    sheet_kp = keypoints_world(sheet_pos, sheet_rot, env._keypoint_offsets)
    goal_kp = keypoints_world(goal_pos, goal_rot, env._keypoint_offsets)
    env._keypoints_max_dist = torch.norm(sheet_kp - goal_kp, dim=-1).max(dim=-1).values

    # -1 means "not observed yet this goal": the first value becomes the best so far.
    unset = env._closest_keypoint_max_dist < 0.0
    env._closest_keypoint_max_dist = torch.where(unset, env._keypoints_max_dist, env._closest_keypoint_max_dist)
    unset = env._closest_fingertip_dist < 0.0
    env._closest_fingertip_dist = torch.where(unset, env._curr_fingertip_distances, env._closest_fingertip_dist)

    near_goal = env._keypoints_max_dist <= env._current_success_tolerance * reward_cfg.keypoint_scale
    if term_cfg.force_consecutive_near_goal_steps:
        env._near_goal_steps = (env._near_goal_steps + near_goal.long()) * near_goal.long()
    else:
        env._near_goal_steps = env._near_goal_steps + near_goal.long()
    env._near_goal = near_goal
    env._is_success = env._near_goal_steps >= term_cfg.success_steps


def compute_reward(env) -> torch.Tensor:
    """This step's reward (N,); its terms go to ``env._reward_terms``."""
    reward_cfg = env.cfg.reward

    # The lift latch (Play's): set once the sheet's centre rises past `lifting_bonus_threshold`. It
    # gates the fingertip approach term and is observed by the critic (`lifted_object`).
    sheet_z = (env.object.data.root_pos_w - env.scene.env_origins)[:, 2]
    lift = 0.05 + sheet_z - env._object_init_z
    env._lifted_object = (lift > reward_cfg.lifting_bonus_threshold) | env._lifted_object
    not_lifted = (~env._lifted_object).float()

    approach = torch.clamp(env._closest_fingertip_dist - env._curr_fingertip_distances, 0.0, 10.0)
    fingertip_approach = approach.sum(dim=-1) * not_lifted * reward_cfg.distance_delta_rew_scale
    env._closest_fingertip_dist = torch.minimum(env._closest_fingertip_dist, env._curr_fingertip_distances)
    env._closest_keypoint_max_dist = torch.minimum(env._closest_keypoint_max_dist, env._keypoints_max_dist)

    joint_speed = env.robot.data.joint_vel.abs()
    arm_penalty = -reward_cfg.kuka_actions_penalty_scale * joint_speed[:, env._arm_joint_ids].sum(dim=-1)
    hand_penalty = -reward_cfg.hand_actions_penalty_scale * joint_speed[:, env._hand_joint_ids].sum(dim=-1)

    # Fold progress, ratcheted on the best fold error this episode: regression is free, and
    # oscillating cannot farm it. A non-finite error (a diverged sheet, reset this step anyway)
    # counts as a flat sheet, so it can neither pay nor poison the ratchet.
    fold_error = env.fold_error()
    fold_error = torch.where(torch.isfinite(fold_error), fold_error, torch.full_like(fold_error, env._flat_fold_err))
    unset = env._closest_fold_err < 0.0
    env._closest_fold_err = torch.where(unset, fold_error, env._closest_fold_err)
    fold_progress = torch.clamp(env._closest_fold_err - fold_error, 0.0, 100.0) * float(reward_cfg.keypoint_rew_scale)
    env._closest_fold_err = torch.minimum(env._closest_fold_err, fold_error)

    fold_bonus = env._fold_bonus_mask.float() * float(reward_cfg.reach_goal_bonus)

    reward = fingertip_approach + fold_progress + fold_bonus + arm_penalty + hand_penalty
    reward = _guard_reward(env, reward)
    env._reward_terms = {
        "fingertip_delta_rew": fingertip_approach,
        "keypoint_rew": fold_progress,
        "bonus_rew": fold_bonus,
        "kuka_actions_penalty": arm_penalty,
        "hand_actions_penalty": hand_penalty,
        "total_reward": reward,
    }
    env.extras["fold_err_mean"] = float(fold_error.mean())
    env.extras["fold_rate"] = float(env._fold_bonus_mask.float().mean())
    return reward


def _guard_reward(env, reward: torch.Tensor) -> torch.Tensor:
    """Zero a non-finite or implausibly large reward and flag the env for reset.

    Finiteness alone is not enough: a diverging articulation drives the joint-speed penalties to
    -1e5 .. -1e11 while every value is still finite (seen in jobs 197715 and 199350). Either way the
    number would reach the advantage and the gradient, so it contributes nothing and the env resets
    through the normal path (``terminations`` consumes ``_nonfinite_envs``).
    """
    nonfinite = ~torch.isfinite(reward)
    if bool(nonfinite.any()):
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        env._nonfinite_envs |= nonfinite
        print(f"[cloth] non-finite REWARD in {int(nonfinite.sum())}/{env.num_envs} envs -- zeroed and flagged for reset", flush=True)
    env.extras["nonfinite_reward"] = float(nonfinite.float().mean())

    insane = reward.abs() > REWARD_SANITY_LIMIT
    if bool(insane.any()):
        worst = float(reward[insane].abs().max())
        reward = torch.where(insane, torch.zeros_like(reward), reward)
        env._nonfinite_envs |= insane
        print(
            f"[cloth] INSANE REWARD in {int(insane.sum())}/{env.num_envs} envs (worst |r|={worst:.3e}, "
            f"limit {REWARD_SANITY_LIMIT:.0e}) -- zeroed and flagged for reset",
            flush=True,
        )
    env.extras["insane_reward"] = float(insane.float().mean())
    env._insane_count += int(insane.sum())
    env.extras["insane_total"] = float(env._insane_count)
    return reward
