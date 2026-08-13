"""Reward terms for the multi-link cartpole.

Generalised from Isaac Lab's `direct/cartpole/cartpole_env.py`. The one substantive
change is that the stock per-joint `pole_pos` penalty is replaced by normalised tip
height: with links able to fold, "every joint angle near zero" and "the pole is up" stop
being the same statement, and only the latter is the task.

Every term is routed through `curriculum.reward_scale`, so any of them can be annealed
over the curriculum from `cfg.curriculum.reward_schedule` without touching this file.
"""

from __future__ import annotations

import torch

from isaacsimenvs.curriculum import reward_scale

__all__ = ["compute_rewards"]


def compute_rewards(env) -> torch.Tensor:
    cfg = env.cfg.reward
    joint_pos = env.cartpole.data.joint_pos
    joint_vel = env.cartpole.data.joint_vel

    cart_pos = joint_pos[:, env._cart_dof_idx[0]]
    cart_vel = joint_vel[:, env._cart_dof_idx[0]]
    terminated = env.reset_terminated.float()

    terms = {
        "alive": cfg.alive * reward_scale(env, "alive") * (1.0 - terminated),
        "terminated": cfg.terminated * reward_scale(env, "terminated") * terminated,
        "upright": cfg.upright * reward_scale(env, "upright") * env._height_norm,
        "cart_pos": cfg.cart_pos
        * reward_scale(env, "cart_pos")
        * (cart_pos / (0.5 * env.cfg.geometry.rail_length)).square(),
        "cart_vel": cfg.cart_vel * reward_scale(env, "cart_vel") * cart_vel.abs(),
        "pole_vel": cfg.pole_vel
        * reward_scale(env, "pole_vel")
        * env._seg_ang_vels.abs().sum(dim=1),
        "action_rate": cfg.action_rate
        * reward_scale(env, "action_rate")
        * (env._actions - env._prev_actions).square().sum(dim=1),
    }

    env._reward_terms = terms
    return torch.stack(list(terms.values()), dim=0).sum(dim=0)
