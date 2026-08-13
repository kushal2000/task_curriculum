"""Flip-specific buffers, difficulty realisation, and per-episode reset.

Runs *after* play's `reset_env_state`, so the robot pose, table, bottle start pose and
observation queues are all already handled by the inherited code; everything here is
additive.

The difficulty scalar drives four things:

    fill_fraction     more liquid  → higher centre of mass → harder to land upright
    required_turns    0 turns (toss and land) → 1+ full flips
    upright_tolerance how far off vertical still counts as landed upright
    settle_speed      how still the bottle must be to count as settled

Fill is interpolated from `fill_easy` (near the centre-of-mass minimum, ~24% for the
default bottle) upward, which is the range over which CoM height — and therefore
difficulty — increases monotonically. See `generate_bottle.bottle_inertial_properties`.
"""

from __future__ import annotations

import math

import torch

from isaacsimenvs.curriculum import record_episode_success, sample_difficulty

from .generate_bottle import bottle_inertial_properties

__all__ = [
    "allocate_flip_buffers",
    "apply_difficulty",
    "apply_fill_level",
    "write_landing_goal",
    "reset_flip_state",
    "episode_success",
]


def allocate_flip_buffers(env) -> None:
    """Per-env flip state. Called from `__init__` after play's buffers exist."""
    n, device = env.num_envs, env.device

    # Difficulty-derived task parameters, refreshed at every reset.
    env._bf_fill_fraction = torch.full((n,), env.cfg.assets.nominal_fill_fraction, device=device)
    env._bf_required_turns = torch.zeros(n, device=device)
    env._bf_upright_cos_tol = torch.ones(n, device=device)
    env._bf_settle_speed = torch.full((n,), env.cfg.termination.settle_speed_easy, device=device)

    # Episode progress.
    env._bf_released = torch.zeros(n, dtype=torch.bool, device=device)
    env._bf_flip_turns = torch.zeros(n, device=device)
    env._bf_turn_progress = torch.zeros(n, device=device)
    env._bf_turn_delta = torch.zeros(n, device=device)
    env._bf_landed_steps = torch.zeros(n, dtype=torch.long, device=device)
    env._bf_settled_steps = torch.zeros(n, dtype=torch.long, device=device)
    env._bf_is_success = torch.zeros(n, dtype=torch.bool, device=device)
    env._bf_prev_episode_success = torch.zeros(n, device=device)
    env._bf_upright_cos = torch.zeros(n, device=device)
    env._bf_landed = torch.zeros(n, dtype=torch.bool, device=device)
    env._bf_airborne = torch.zeros(n, dtype=torch.bool, device=device)
    env._bf_land_bonus_paid = torch.zeros(n, dtype=torch.bool, device=device)
    env._bf_success_paid = torch.zeros(n, dtype=torch.bool, device=device)

    env._bf_termination_reasons = {
        "fell_off_table": torch.zeros(n, dtype=torch.bool, device=device),
        "landed_settled": torch.zeros(n, dtype=torch.bool, device=device),
    }


def apply_difficulty(env, env_ids: torch.Tensor) -> None:
    """Turn `env._difficulty` into concrete task parameters for `env_ids`."""
    if env_ids.numel() == 0:
        return
    assets = env.cfg.assets
    term = env.cfg.termination
    d = env._difficulty[env_ids, 0].clamp(0.0, 1.0)

    env._bf_fill_fraction[env_ids] = (
        assets.fill_fraction_easy + d * (assets.fill_fraction_hard - assets.fill_fraction_easy)
    )
    env._bf_required_turns[env_ids] = (
        term.required_turns_easy + d * (term.required_turns_hard - term.required_turns_easy)
    )
    tol_deg = term.upright_tolerance_deg_easy + d * (
        term.upright_tolerance_deg_hard - term.upright_tolerance_deg_easy
    )
    env._bf_upright_cos_tol[env_ids] = torch.cos(tol_deg * math.pi / 180.0)
    env._bf_settle_speed[env_ids] = (
        term.settle_speed_easy + d * (term.settle_speed_hard - term.settle_speed_easy)
    )

    if assets.runtime_fill:
        apply_fill_level(env, env_ids)


def apply_fill_level(env, env_ids: torch.Tensor) -> None:
    """Write the sampled fill level into PhysX as mass, centre of mass, and inertia.

    This is what makes fill a genuine per-env, per-episode curriculum knob rather than a
    property frozen into the USD at spawn: the bottle's *shape* is fixed, but how much
    liquid is in it is not.

    Caveat, and the reason `cfg.assets.runtime_fill` exists as a switch: PhysX exposes
    these three quantities only through CPU tensors (Isaac Lab's own
    `randomize_rigid_body_mass` / `_com` have the same constraint), so each call is a
    host round-trip. It is one call per reset batch, not per env, but if throughput
    suffers at high `num_envs`, turning `runtime_fill` off falls back to the nominal fill
    baked into the USD pool and costs nothing.
    """
    assets = env.cfg.assets
    view = env.object.root_physx_view
    ids_cpu = env_ids.cpu()

    mass, com_z, ixx, iyy, izz = bottle_inertial_properties(
        env._bottle_height[env_ids],
        env._bottle_radius[env_ids],
        assets.bottle_wall_thickness,
        env._bf_fill_fraction[env_ids],
        shell_density=assets.bottle_shell_density,
        liquid_density=assets.bottle_liquid_density,
    )

    masses = view.get_masses().clone()
    masses[ids_cpu, 0] = mass.cpu()
    view.set_masses(masses, ids_cpu)

    # RigidBodyView reports inertia as a flattened 3x3 per body.
    inertias = view.get_inertias().clone()
    diag = torch.zeros(env_ids.numel(), 9)
    diag[:, 0] = ixx.cpu()
    diag[:, 4] = iyy.cpu()
    diag[:, 8] = izz.cpu()
    inertias[ids_cpu] = diag.reshape(inertias[ids_cpu].shape)
    view.set_inertias(inertias, ids_cpu)

    # `get_coms` is (count, 7) — position + quaternion — for a rigid body view, but
    # (count, num_bodies, 7) for an articulation view. Handle both so this keeps working
    # if the bottle is ever promoted to an articulation.
    coms = view.get_coms().clone()
    if coms.dim() == 3:
        coms[ids_cpu, 0, 2] = com_z.cpu()
    else:
        coms[ids_cpu, 2] = com_z.cpu()
    view.set_coms(coms, ids_cpu)


def write_landing_goal(env, env_ids: torch.Tensor) -> None:
    """Place GoalViz as an upright bottle standing on the table.

    Play's inherited observation includes `keypoints_rel_goal`, and its reward machinery
    measures keypoint distance to GoalViz. Pointing that goal at "upright, on the table"
    means the flip task gets play's whole shaping stack for free, and — more importantly
    — the observation layout stays byte-identical to play2perfect's, which is what lets a
    play2perfect checkpoint load into this task.
    """
    cfg = env.cfg.bottle_flip
    n = env_ids.numel()
    origins = env.scene.env_origins[env_ids]

    noise = torch.empty(n, 2, device=env.device).uniform_(-1.0, 1.0)
    pos_local = torch.stack(
        (
            noise[:, 0] * cfg.landing_xy_range[0],
            noise[:, 1] * cfg.landing_xy_range[1],
            env._table_z_per_env[env_ids] + 0.5 * env._bottle_height[env_ids],
        ),
        dim=-1,
    )
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0  # identity (w, x, y, z) — bottle standing upright

    env.goal_viz.write_root_pose_to_sim(torch.cat([pos_local + origins, quat], dim=-1), env_ids)
    env.goal_viz.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids)


def episode_success(env, env_ids: torch.Tensor) -> torch.Tensor:
    """Whether the finished episode ended in a sustained upright landing."""
    return env._bf_is_success[env_ids].float()


def reset_flip_state(env, env_ids: torch.Tensor) -> None:
    """Score the finished episode, redraw difficulty, and clear per-episode trackers."""
    if env_ids.numel() == 0:
        return

    success = episode_success(env, env_ids)
    record_episode_success(env, env_ids, success)
    env._bf_prev_episode_success[env_ids] = success

    sample_difficulty(env, env_ids)
    apply_difficulty(env, env_ids)
    write_landing_goal(env, env_ids)

    env._bf_released[env_ids] = False
    env._bf_flip_turns[env_ids] = 0.0
    env._bf_turn_progress[env_ids] = 0.0
    env._bf_turn_delta[env_ids] = 0.0
    env._bf_landed_steps[env_ids] = 0
    env._bf_settled_steps[env_ids] = 0
    env._bf_is_success[env_ids] = False
    env._bf_landed[env_ids] = False
    env._bf_airborne[env_ids] = False
    env._bf_land_bonus_paid[env_ids] = False
    env._bf_success_paid[env_ids] = False
