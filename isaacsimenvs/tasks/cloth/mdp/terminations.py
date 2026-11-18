"""When an episode ends, and why.

Terminated on:

* ``fold``: the fold criterion (``ClothTaskMixin.is_folded``) has held for
  ``termination.success_steps`` steps; this is also the step the fold bonus pays on;
* ``fall``: the sheet's centre dropped below 0.1 m (off the table);
* ``hand_far``: a fingertip is more than 1.5 m from the sheet;
* ``nonfinite``: the state, observation or reward of the env diverged (flagged during the step);

truncated on ``timeout`` (``termination.episode_length``). ``fold_held`` (the criterion held for
``HELD_FOLD_STEPS``) is reported, not acted on.

The Play task's goal hit is kept as it was trained with: when its goal measurement
(``rewards.update_hand_and_goal_metrics``) comes within ``success_tolerance * keypoint_scale`` for
``success_steps`` steps, the episode clock and the approach trackers restart. It never ends an
episode and never counts as a success.
"""

from __future__ import annotations

import torch

from .randomization import rng_parity_goal
from .rewards import update_hand_and_goal_metrics

__all__ = ["HELD_FOLD_STEPS", "compute_dones", "update_tolerance_curriculum"]

#: Steps the fold criterion must hold to count as a held fold. Independent of
#: ``termination.success_steps``, which a finetune recipe may set to 1 so the bonus pays on first entry.
HELD_FOLD_STEPS = 10


def update_tolerance_curriculum(env) -> None:
    """Shrink the Play goal tolerance when finished episodes average enough successes; evaluation
    pins it (``eval_success_tolerance``)."""
    term = env.cfg.termination
    env._frame_counter += 1
    if env._frame_counter - env._last_curriculum_update >= term.tolerance_curriculum_interval:
        successes = env._prev_episode_successes.float()
        if successes.numel() > 0 and successes.mean().item() >= term.tolerance_curriculum_success_threshold:
            tolerance = env._current_success_tolerance * term.tolerance_curriculum_increment
            env._current_success_tolerance = max(min(tolerance, term.success_tolerance), term.target_success_tolerance)
            env._last_curriculum_update = env._frame_counter
    if term.eval_success_tolerance is not None:
        env._current_success_tolerance = float(term.eval_success_tolerance)


def compute_dones(env) -> tuple[torch.Tensor, torch.Tensor]:
    """``(terminated, truncated)``; sets ``_termination_reasons``, ``_fold_bonus_mask`` and
    ``_successes``."""
    update_tolerance_curriculum(env)
    update_hand_and_goal_metrics(env)

    goal_hit = env._is_success.nonzero(as_tuple=False).squeeze(-1)
    if goal_hit.numel() > 0:
        env._closest_keypoint_max_dist[goal_hit] = -1.0
        env._closest_fingertip_dist[goal_hit] = -1.0
        env._near_goal_steps[goal_hit] = 0
        env.episode_length_buf[goal_hit] = 0
        rng_parity_goal(env, goal_hit.numel(), env.cfg.reset.goal_sampling_type)

    sheet_z = env.object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    fall = sheet_z < 0.1
    hand_far = env._curr_fingertip_distances.max(dim=-1).values > 1.5

    folded = env.is_folded()
    env._fold_hold = torch.where(folded, env._fold_hold + 1, torch.zeros_like(env._fold_hold))
    fold = env._fold_hold >= int(env.cfg.termination.success_steps)
    fold_held = env._fold_hold >= HELD_FOLD_STEPS
    env._fold_bonus_mask.copy_(fold)
    env._successes += fold.long()

    # Consumed here: the reset restores a valid state.
    nonfinite = env._nonfinite_envs.clone()
    env._nonfinite_envs.zero_()

    truncated = env.episode_length_buf >= env.max_episode_length
    env._termination_reasons = {
        "fall": fall,
        "hand_far": hand_far,
        "timeout": truncated,
        "fold": fold,
        "fold_held": fold_held,
        "nonfinite": nonfinite,
    }
    return fall | hand_far | fold | nonfinite, truncated
