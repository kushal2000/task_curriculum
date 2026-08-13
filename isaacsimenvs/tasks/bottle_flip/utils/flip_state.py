"""The flip state machine: grasped → released → airborne → landed → settled.

Called once per policy step from `_get_dones`, after play's `compute_intermediate_values`
has refreshed fingertip distances and the lifted flag, so those are reused rather than
recomputed.

Rotation is tracked by integrating the magnitude of the bottle's *horizontal* angular
velocity, not by unwrapping its orientation. A flip is a rotation about a horizontal
axis, and integrating the rate handles multi-turn flips without any of the branch-cut
bookkeeping that angle unwrapping needs.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply

__all__ = ["compute_flip_state"]

_TWO_PI = 2.0 * math.pi


def compute_flip_state(env) -> None:
    cfg = env.cfg.bottle_flip
    term = env.cfg.termination

    origins = env.scene.env_origins
    obj_pos = env.object.data.root_pos_w - origins
    obj_quat = env.object.data.root_quat_w
    lin_vel = env.object.data.root_lin_vel_w
    ang_vel = env.object.data.root_ang_vel_w

    # --- Uprightness: the bottle's own +z against world +z. ---
    local_z = torch.zeros_like(obj_pos)
    local_z[:, 2] = 1.0
    env._bf_upright_cos = quat_apply(obj_quat, local_z)[:, 2]

    # --- Release: lifted, and no fingertip still near the bottle. ---
    min_ft_dist = env._curr_fingertip_distances.min(dim=1).values
    env._bf_released = env._bf_released | (env._lifted_object & (min_ft_dist > cfg.release_distance))

    # --- Rotation accumulated since release. ---
    horiz_rate = ang_vel[:, :2].norm(dim=-1)
    env._bf_flip_turns = env._bf_flip_turns + torch.where(
        env._bf_released,
        horiz_rate * env.step_dt / _TWO_PI,
        torch.zeros_like(horiz_rate),
    )

    # --- Airborne vs. down. ---
    land_z = env._table_z_per_env + 0.5 * env._bottle_height + cfg.landing_z_margin
    is_down = obj_pos[:, 2] <= land_z
    env._bf_airborne = env._bf_released & ~is_down

    settled = (lin_vel.norm(dim=-1) < env._bf_settle_speed) & (
        ang_vel.norm(dim=-1) < env._bf_settle_speed * term.settle_ang_speed_ratio
    )
    env._bf_landed = env._bf_released & is_down & settled

    # --- Success: upright, enough turns, and holding still for a few steps. ---
    upright_ok = env._bf_upright_cos >= env._bf_upright_cos_tol
    turns_ok = env._bf_flip_turns >= env._bf_required_turns
    good = env._bf_landed & upright_ok & turns_ok

    # Consecutive counters: a bottle that wobbles through vertical on its way over has
    # not landed upright, so the count resets the moment the condition lapses.
    env._bf_landed_steps = (env._bf_landed_steps + good.long()) * good.long()
    env._bf_settled_steps = (env._bf_settled_steps + env._bf_landed.long()) * env._bf_landed.long()
    env._bf_is_success = env._bf_is_success | (env._bf_landed_steps >= term.success_steps)

    # Progress toward the required turn count, saturating at the requirement so the
    # reward cannot be farmed by spinning indefinitely. The reward consumes the
    # per-step *delta*, which is why it is published separately here rather than
    # differenced later.
    progress = torch.minimum(env._bf_flip_turns, env._bf_required_turns)
    env._bf_turn_delta = (progress - env._bf_turn_progress).clamp(min=0.0)
    env._bf_turn_progress = progress
