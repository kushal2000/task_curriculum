"""When an episode ends, and why.

Terminated on:

* ``fold``: the fold criterion (``ClothTaskMixin.is_folded``) has held for
  ``termination.success_steps`` steps; this is also the step the fold bonus pays on;
* ``fall``: the sheet's centre dropped below 0.1 m (off the table);
* ``hand_far``: a fingertip is more than 1.5 m from the sheet;
* ``nonfinite``: the state, observation or reward of the env diverged (flagged during the step);

truncated on ``timeout`` (``termination.episode_length``). ``fold_held`` (the criterion held for
``HELD_FOLD_STEPS``) is reported, not acted on.
"""

from __future__ import annotations

import torch

from .rewards import update_hand_and_goal_metrics

__all__ = ["HELD_FOLD_STEPS", "compute_dones"]

#: Steps the fold criterion must hold to count as a held fold. Independent of
#: ``termination.success_steps``, which a finetune recipe may set to 1 so the bonus pays on first entry.
HELD_FOLD_STEPS = 10


def compute_dones(env) -> tuple[torch.Tensor, torch.Tensor]:
    """``(terminated, truncated)``; sets ``_termination_reasons``, ``_fold_bonus_mask`` and
    ``_successes``."""
    update_hand_and_goal_metrics(env)

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
