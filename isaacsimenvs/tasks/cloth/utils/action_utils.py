"""Play's action map, unchanged in meaning (the Play prior's actions must mean the same here).

The policy acts in the canonical joint order (arm first, ``robot.joint_names_canonical``), in
[-1, 1]: the arm as a joint-target VELOCITY (``prev + dof_speed_scale * dt * a``), the hand as
ABSOLUTE targets over its limits, both through an EMA and clamped. Optional per-step action delay.
"""

from __future__ import annotations

import torch


def joint_maps(lab_names: list[str], spec) -> dict[str, torch.Tensor | list[int]]:
    """Permutations between Isaac Lab's joint order and the canonical one, and the arm / hand ids.

    Raises if the asset's joints are not exactly the spec's, or the arm is not first in Lab order
    (the action map indexes ``[:7]`` / ``[7:]`` after permuting)."""
    canon = list(spec.joint_names_canonical)
    if sorted(lab_names) != sorted(canon):
        raise ValueError(f"robot joints {sorted(lab_names)} are not the spec's {sorted(canon)}")
    canon_to_lab = [canon.index(n) for n in lab_names]  # lab_vec = canon_vec[:, canon_to_lab]
    lab_to_canon = [lab_names.index(n) for n in canon]  # canon_vec = lab_vec[:, lab_to_canon]
    arm = sorted(lab_names.index(n) for n in spec.arm_joint_names)
    hand = sorted(lab_names.index(n) for n in spec.hand_joint_names)
    if arm != list(range(len(arm))) or hand != list(range(len(arm), len(lab_names))):
        raise ValueError(f"expected the arm joints first in Lab order, got arm {arm}, hand {hand}")
    return {"canon_to_lab": canon_to_lab, "lab_to_canon": lab_to_canon, "arm": arm, "hand": hand}


def delayed(queue: torch.Tensor, values: torch.Tensor, flush: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Push ``values`` into a rolling ``(N, L, D)`` queue and read each env at a random lag in
    [0, L). ``flush`` envs (episode start) fill the whole queue with ``values`` first."""
    n = values.shape[0]
    if bool(flush.any()):
        queue[flush] = values[flush].unsqueeze(1).expand(-1, queue.shape[1], -1)
    queue = torch.roll(queue, shifts=1, dims=1)
    queue[:, 0] = values
    lag = torch.randint(0, queue.shape[1], (n,), device=values.device)
    return queue, queue[torch.arange(n, device=values.device), lag]


def apply_actions(env, actions: torch.Tensor) -> None:
    """Policy actions (canonical order) -> ``env._cur_targets`` (Lab order); also ``_prev_targets``."""
    dr, act = env.cfg.domain_randomization, env.cfg.action
    a = actions.to(env.device).clone()[:, env._perm_canon_to_lab]  # canonical -> Lab order

    if dr.use_action_delay and dr.action_delay_max > 0:
        env._action_queue, a = delayed(env._action_queue, a, env.episode_length_buf == 0)
    if dr.action_noise_std > 0:
        a = (a + torch.randn_like(a) * dr.action_noise_std).clamp(-1.0, 1.0)

    arm, hand, prev = env._arm_ids, env._hand_ids, env._prev_targets
    arm_raw = (prev[:, arm] + act.dof_speed_scale * env.step_dt * a[:, arm]).clamp(env._arm_lower, env._arm_upper)
    arm_t = act.arm_moving_average * arm_raw + (1.0 - act.arm_moving_average) * prev[:, arm]
    hand_raw = env._hand_lower + 0.5 * (a[:, hand] + 1.0) * (env._hand_upper - env._hand_lower)
    hand_t = act.hand_moving_average * hand_raw + (1.0 - act.hand_moving_average) * prev[:, hand]

    env._cur_targets[:, arm] = arm_t.clamp(env._arm_lower, env._arm_upper)
    env._cur_targets[:, hand] = hand_t.clamp(env._hand_lower, env._hand_upper)
    env._prev_targets = env._cur_targets.clone()


def spring_constants(lab_names: list[str], spec, device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(ks, ref)`` per Lab joint for the hand's passive spring (0 / 0 on the arm)."""
    ks = torch.tensor([spec.hand_spring_stiffness.get(n, 0.0) for n in lab_names], device=device)
    ref = torch.tensor([spec.hand_spring_ref.get(n, 0.0) for n in lab_names], device=device)
    return ks, ref


def spring_effort(env, joint_pos: torch.Tensor) -> torch.Tensor:
    """``ks (ref - q)``: applied as a joint effort every physics step, next to the PD drive and not
    folded into it (the drive's effort limit clips only the PD, as the firmware's current limit
    does on the real hand, where the spring is mechanical)."""
    return env._spring_ks * (env._spring_ref - joint_pos)
