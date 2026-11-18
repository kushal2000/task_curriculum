"""Policy action -> joint position targets.

The policy acts in canonical joint order, every entry in [-1, 1]:

* **arm**: a joint velocity command. ``target += dof_speed_scale * dt * a``, clamped to the joint
  limits, then low-passed (``arm_moving_average``) against the previous target;
* **hand**: an absolute position, ``a`` mapped linearly onto the joint range, then low-passed
  (``hand_moving_average``).

Domain randomization on the way: a delay line (``use_action_delay``; per step or per episode, see
``delay_mode``) and Gaussian noise (``action_noise_std``) on the action that is executed.

``_cur_targets`` is what ``_apply_action`` writes every physics step; ``_prev_targets`` is the
smoothing state and the ``prev_action_targets`` observation.
"""

from __future__ import annotations

import torch

from .randomization import episode_start

__all__ = ["process_actions"]


def process_actions(env, actions: torch.Tensor) -> None:
    """Turn this policy step's ``actions`` (N, J) into ``env._cur_targets`` (Lab order)."""
    dr = env.cfg.domain_randomization
    act_cfg = env.cfg.action
    dt = env.step_dt
    n_arm = len(env._arm_joint_ids)

    actions = actions.clone().to(env.device)[:, env._perm_canon_to_lab]

    if dr.use_action_delay and dr.action_delay_max > 0:
        start = episode_start(env)
        if start.any():
            env._action_queue[start] = actions[start].unsqueeze(1)
        env._action_queue = torch.roll(env._action_queue, shifts=1, dims=1)
        env._action_queue[:, 0, :] = actions
        if dr.delay_mode == "per_episode":
            delay_idx = env._action_delay_idx
        else:
            delay_idx = torch.randint(0, env._action_queue.shape[1], (env.num_envs,), device=env.device)
        actions = env._action_queue[torch.arange(env.num_envs, device=env.device), delay_idx]

    # On the executed (delayed) action, kept in the policy's [-1, 1].
    if dr.action_noise_std > 0:
        actions = (actions + torch.randn_like(actions) * dr.action_noise_std).clamp(-1.0, 1.0)

    prev = env._prev_targets
    arm_raw = prev[:, :n_arm] + act_cfg.dof_speed_scale * dt * actions[:, :n_arm]
    arm_raw = torch.clamp(arm_raw, env._arm_lower, env._arm_upper)
    arm = act_cfg.arm_moving_average * arm_raw + (1.0 - act_cfg.arm_moving_average) * prev[:, :n_arm]
    arm = torch.clamp(arm, env._arm_lower, env._arm_upper)

    hand_raw = env._hand_lower + 0.5 * (actions[:, n_arm:] + 1.0) * (env._hand_upper - env._hand_lower)
    hand = act_cfg.hand_moving_average * hand_raw + (1.0 - act_cfg.hand_moving_average) * prev[:, n_arm:]
    hand = torch.clamp(hand, env._hand_lower, env._hand_upper)

    env._cur_targets[:, env._arm_joint_ids] = arm
    env._cur_targets[:, env._hand_joint_ids] = hand
    env._prev_targets = env._cur_targets.clone()
