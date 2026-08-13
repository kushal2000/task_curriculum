"""Termination rules for bottle flipping.

An episode ends as soon as its outcome is decided rather than running to the time limit:
once the bottle has come to rest the result is known, and holding thousands of envs idle
on a settled bottle is pure wasted throughput. Note that this terminates *both* a
successful upright landing and a failed one — the two are distinguished by
`_bf_is_success`, not by how the episode ended.
"""

from __future__ import annotations

import torch

__all__ = ["compute_terminations"]


def compute_terminations(env) -> tuple[torch.Tensor, torch.Tensor]:
    term = env.cfg.termination
    origins = env.scene.env_origins
    obj_z = env.object.data.root_pos_w[:, 2] - origins[:, 2]

    fell_off_table = obj_z < (env._table_z_per_env - term.fall_below_table)
    outcome_decided = env._bf_settled_steps >= term.terminate_settle_steps

    env._bf_termination_reasons["fell_off_table"] = fell_off_table
    env._bf_termination_reasons["landed_settled"] = outcome_decided

    time_out = env.episode_length_buf >= env.max_episode_length - 1
    return fell_off_table | outcome_decided, time_out
