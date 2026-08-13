"""Termination rules and the per-step balance counter.

Both conditions are expressed in link-count-independent terms — cart position, and
normalised tip height — so the same rule scores a 1-link and a 4-link episode the same
way. That matters for the experiment: if termination got stricter as `n` rose, an
apparent curriculum effect could just be a moving goalpost.
"""

from __future__ import annotations

import torch

__all__ = ["compute_terminations"]


def compute_terminations(env) -> tuple[torch.Tensor, torch.Tensor]:
    term = env.cfg.termination
    cart_pos = env.cartpole.data.joint_pos[:, env._cart_dof_idx[0]]

    # Count this step toward the episode's balance fraction before deciding anything —
    # `reset_utils.episode_success` divides by `episode_length_buf`, which includes it.
    env._upright_steps += (env._height_norm >= term.success_height_frac).float()

    out_of_bounds = cart_pos.abs() > term.max_cart_pos
    fell = env._height_norm < term.fall_height_frac

    env._termination_reasons["cart_out_of_bounds"] = out_of_bounds
    env._termination_reasons["pole_fell"] = fell

    time_out = env.episode_length_buf >= env.max_episode_length - 1
    return out_of_bounds | fell, time_out
