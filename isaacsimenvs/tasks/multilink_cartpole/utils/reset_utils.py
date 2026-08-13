"""Buffers, difficulty → joint-lock translation, and episode reset.

`apply_difficulty` is where this task's difficulty scalar becomes physics. Given
`d ∈ [0, 1]` for an env it picks how many pole joints stay free:

    n_active = 1 + round(d * (n_max - 1))

joint 0 is always free (locking it would weld the whole pole to the cart and delete the
task), and the remaining `n_active - 1` free joints are drawn uniformly from the rest.
Because a locked joint fuses its two links into one rigid body, *which* joints are free
also determines the effective segment lengths — so an env at `n_active = 2` might be a
(0.25 m, 0.75 m) double pendulum on one episode and a (0.75 m, 0.25 m) one on the next.
Both `n` and the effective link lengths therefore vary per-env and per-episode, from a
single spawned USD.
"""

from __future__ import annotations

import torch

from isaacsimenvs.curriculum import record_episode_success, sample_difficulty

from .difficulty_math import active_joint_count, sample_free_mask, segment_geometry
from .generate_cartpole import CART_JOINT_NAME, pole_joint_name

__all__ = [
    "allocate_state_buffers",
    "apply_difficulty",
    "reset_env_state",
    "episode_success",
]


def allocate_state_buffers(env) -> None:
    """Populate every per-env buffer. Called once from `__init__` after `_setup_scene`."""
    geom = env.cfg.geometry
    n_max = geom.n_max
    device = env.device

    env._cart_dof_idx = env.cartpole.find_joints(CART_JOINT_NAME)[0]
    # Resolve pole joints one at a time so the ids come back in chain order (base → tip)
    # rather than whatever order the URDF parser happened to produce.
    env._pole_dof_ids = [env.cartpole.find_joints(pole_joint_name(i))[0][0] for i in range(n_max)]
    if len(env._pole_dof_ids) != n_max:
        raise RuntimeError(f"Expected {n_max} pole joints, resolved {len(env._pole_dof_ids)}.")

    env._link_lengths = torch.tensor(geom.link_lengths, device=device, dtype=torch.float32)
    env._total_length = float(env._link_lengths.sum().item())

    # Baseline (unlocked) limits, captured before anything is narrowed, so a joint can
    # be restored exactly when it becomes free again.
    env._base_joint_limits = env.cartpole.data.joint_pos_limits[0, env._pole_dof_ids].clone()

    # --- Difficulty realisation ---
    env._free_mask = torch.ones(env.num_envs, n_max, dtype=torch.bool, device=device)
    env._n_active = torch.full((env.num_envs,), n_max, dtype=torch.long, device=device)
    env._seg_lengths = torch.zeros(env.num_envs, n_max, device=device)
    env._seg_joint_idx = torch.zeros(env.num_envs, n_max, dtype=torch.long, device=device)

    # --- Intermediates, filled every step by obs_utils.compute_intermediate_values ---
    env._seg_angles = torch.zeros(env.num_envs, n_max, device=device)
    env._seg_ang_vels = torch.zeros(env.num_envs, n_max, device=device)
    env._height_norm = torch.zeros(env.num_envs, device=device)

    # --- Actions ---
    env._actions = torch.zeros(env.num_envs, env.cfg.action_space, device=device)
    env._prev_actions = torch.zeros_like(env._actions)

    # --- Episode statistics ---
    env._upright_steps = torch.zeros(env.num_envs, device=device)
    env._prev_episode_success = torch.zeros(env.num_envs, device=device)
    env._reward_terms: dict[str, torch.Tensor] = {}
    env._termination_reasons: dict[str, torch.Tensor] = {
        "cart_out_of_bounds": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "pole_fell": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
    }


def apply_difficulty(env, env_ids: torch.Tensor) -> None:
    """Translate `env._difficulty` into joint locks and write them to PhysX."""
    geom = env.cfg.geometry
    n_max = geom.n_max
    if env_ids.numel() == 0:
        return

    difficulty = env._difficulty[env_ids, 0].clamp(0.0, 1.0)
    n_active = active_joint_count(difficulty, n_max)

    free_mask = sample_free_mask(n_active, n_max)
    seg_lengths, seg_joint_idx = segment_geometry(free_mask, env._link_lengths)

    env._n_active[env_ids] = n_active
    env._free_mask[env_ids] = free_mask
    env._seg_lengths[env_ids] = seg_lengths
    env._seg_joint_idx[env_ids] = seg_joint_idx

    # A locked joint is held by a stiff PD *and* pinned by a narrow position limit. The
    # limit is a solver constraint and does the real work; the PD keeps it centred so
    # the constraint is not permanently saturated.
    stiffness = torch.where(
        free_mask,
        torch.zeros_like(seg_lengths),
        torch.full_like(seg_lengths, geom.lock_stiffness),
    )
    damping = torch.where(
        free_mask,
        torch.full_like(seg_lengths, geom.pole_joint_damping),
        torch.full_like(seg_lengths, geom.lock_damping),
    )
    env.cartpole.write_joint_stiffness_to_sim(stiffness, joint_ids=env._pole_dof_ids, env_ids=env_ids)
    env.cartpole.write_joint_damping_to_sim(damping, joint_ids=env._pole_dof_ids, env_ids=env_ids)

    if geom.use_joint_limit_lock:
        base = env._base_joint_limits.unsqueeze(0).expand(env_ids.numel(), n_max, 2)
        locked = torch.tensor(
            [-geom.lock_limit_rad, geom.lock_limit_rad], device=env.device
        ).expand(env_ids.numel(), n_max, 2)
        limits = torch.where(free_mask.unsqueeze(-1), base, locked).contiguous()
        env.cartpole.write_joint_position_limit_to_sim(
            limits,
            joint_ids=env._pole_dof_ids,
            env_ids=env_ids,
            # Defaults are all zero and every limit brackets zero, so this can only fire
            # spuriously.
            warn_limit_violation=False,
        )


def episode_success(env, env_ids: torch.Tensor, episode_lengths: torch.Tensor) -> torch.Tensor:
    """Did the just-finished episode count as a success, per env?

    Success requires *both* surviving to the time limit (never fell, never ran off the
    rail) and having actually been upright for most of it — a policy that hangs near the
    fall threshold for the whole episode is not balancing.

    `episode_lengths` is passed in rather than read from `env.episode_length_buf`
    because `DirectRLEnv._reset_idx` has already zeroed that buffer by this point.
    """
    term = env.cfg.termination
    steps = episode_lengths.clamp(min=1).float()
    upright_frac = env._upright_steps[env_ids] / steps
    timed_out = env.reset_time_outs[env_ids].float()
    return timed_out * (upright_frac >= term.success_min_upright_frac).float()


def reset_env_state(env, env_ids: torch.Tensor, episode_lengths: torch.Tensor) -> None:
    """Score the finished episode, redraw its difficulty, and write the new state."""
    if env_ids.numel() == 0:
        return
    geom = env.cfg.geometry
    reset_cfg = env.cfg.reset
    n_envs = env_ids.numel()

    # 1. Score the episode that just ended, and feed it to the curriculum.
    success = episode_success(env, env_ids, episode_lengths)
    record_episode_success(env, env_ids, success)
    env._prev_episode_success[env_ids] = success
    env._upright_steps[env_ids] = 0.0

    # 2. Redraw difficulty and realise it as joint locks.
    sample_difficulty(env, env_ids)
    apply_difficulty(env, env_ids)

    # 3. Initial joint state. Harder envs also start further from upright, so difficulty
    #    moves the basin of attraction as well as the morphology.
    joint_pos = torch.zeros(n_envs, env.cartpole.num_joints, device=env.device)
    joint_vel = torch.zeros_like(joint_pos)

    cart_lo, cart_hi = reset_cfg.cart_pos_range
    joint_pos[:, env._cart_dof_idx[0]] = (
        torch.rand(n_envs, device=env.device) * (cart_hi - cart_lo) + cart_lo
    )
    vel_lo, vel_hi = reset_cfg.cart_vel_range
    joint_vel[:, env._cart_dof_idx[0]] = (
        torch.rand(n_envs, device=env.device) * (vel_hi - vel_lo) + vel_lo
    )

    difficulty = env._difficulty[env_ids, 0].clamp(0.0, 1.0).unsqueeze(1)
    spread = reset_cfg.pole_angle_range_easy + difficulty * (
        reset_cfg.pole_angle_range_hard - reset_cfg.pole_angle_range_easy
    )
    free_mask = env._free_mask[env_ids]
    pole_pos = (torch.rand(n_envs, geom.n_max, device=env.device) * 2.0 - 1.0) * spread
    pole_vel = (
        (torch.rand(n_envs, geom.n_max, device=env.device) * 2.0 - 1.0)
        * reset_cfg.pole_vel_range
    )
    # A locked joint must start exactly at 0 — it is meant to be a rigid weld, and a
    # non-zero start would bake a permanent kink into the "rigid" segment.
    joint_pos[:, env._pole_dof_ids] = torch.where(free_mask, pole_pos, torch.zeros_like(pole_pos))
    joint_vel[:, env._pole_dof_ids] = torch.where(free_mask, pole_vel, torch.zeros_like(pole_vel))

    root_state = env.cartpole.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    env.cartpole.write_root_pose_to_sim(root_state[:, :7], env_ids)
    env.cartpole.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    env.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    env._actions[env_ids] = 0.0
    env._prev_actions[env_ids] = 0.0
