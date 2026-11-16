"""The per-step snapshot the utils read, and the task's per-env buffers.

``ClothState`` is the whole simulator-facing contract: each env's ``_read_state`` fills it (world
frame, quaternions wxyz, joints in Isaac Lab's order) and nothing in ``utils/`` reads the
simulator otherwise. ``allocate`` puts the task buffers on the env; ``clear`` resets them per env.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import geometry, quat


@dataclass
class ClothState:
    joint_pos: torch.Tensor          # (N, J) Lab order
    joint_vel: torch.Tensor          # (N, J)
    palm_pos_w: torch.Tensor         # (N, 3) palm BODY origin (the centre offset is the obs's)
    palm_quat_w: torch.Tensor        # (N, 4) wxyz
    palm_vel_w: torch.Tensor         # (N, 6) linear, angular
    fingertip_pos_w: torch.Tensor    # (N, 5, 3) fingertip BODY origins, policy order
    fingertip_quat_w: torch.Tensor   # (N, 5, 4) wxyz
    hand_pos_w: torch.Tensor         # (N, H, 3) origins of the release-test bodies
    points_w: torch.Tensor           # (N, P, 3) sheet points (VBD particles / rigid plate grid)
    points_vel_w: torch.Tensor       # (N, P, 3)
    table_top_z: torch.Tensor        # (N,) world z of the table's top surface


@dataclass
class TaskView:
    """What the task derives from one ``ClothState``: computed once per hook, read by obs, reward
    and termination alike (so they can never disagree about where the sheet is)."""

    frame_r: torch.Tensor        # (N, 3, 3) stationary half, rest -> current
    frame_c: torch.Tensor        # (N, 3) its centroid
    pos_w: torch.Tensor          # (N, 3) sheet centre (all points)
    quat_w: torch.Tensor         # (N, 4) wxyz, the stationary half's orientation
    lin_vel_w: torch.Tensor      # (N, 3)
    ang_vel_w: torch.Tensor      # (N, 3)
    keypoints_w: torch.Tensor    # (N, 4, 3) the tracked moving-half corners
    targets_w: torch.Tensor      # (N, 4, 3) where they belong
    fold_error: torch.Tensor     # (N,)
    footprint: torch.Tensor      # (N,)
    pads_w: torch.Tensor         # (N, 5, 3) fingertip pads (or body origins)
    fingertip_dist: torch.Tensor  # (N, 5) pad -> sheet centre


def derive(env, s: ClothState) -> TaskView:
    sheet = env._sheet_geom
    r, c = geometry.stationary_frame(sheet, s.points_w)
    pos, q, lin, ang = geometry.sheet_frame(s.points_w, s.points_vel_w, r)
    kp = s.points_w[:, sheet.kp_idx]
    targets = geometry.keypoint_targets(sheet, r, c, env._unfold_env)
    pads = s.fingertip_pos_w
    if env.cfg.reward.fingertip_distance_from_pad:
        pads = pads + quat.apply(s.fingertip_quat_w, env._fingertip_offsets.expand_as(pads))
    return TaskView(
        frame_r=r, frame_c=c, pos_w=pos, quat_w=q, lin_vel_w=lin, ang_vel_w=ang,
        keypoints_w=kp, targets_w=targets, fold_error=geometry.fold_error(kp, targets),
        footprint=geometry.footprint_ratio(sheet, s.points_w, r),
        pads_w=pads, fingertip_dist=(pads - pos.unsqueeze(1)).norm(dim=-1),
    )


def allocate(env) -> None:
    """Task buffers on ``env`` (``num_envs``, ``device``, ``cfg`` and ``_spec`` set)."""
    n, dev, cfg = env.num_envs, env.device, env.cfg
    j = env._spec.num_joints
    dr = cfg.domain_randomization
    f32 = dict(device=dev, dtype=torch.float32)

    env._cur_targets = torch.zeros(n, j, **f32)
    env._prev_targets = torch.zeros(n, j, **f32)
    env._action_queue = torch.zeros(n, max(1, int(dr.action_delay_max)), j, **f32)
    env._obs_queue = torch.zeros(n, max(1, int(dr.obs_delay_max)), int(cfg.observation_space), **f32)
    # Sheet frame (pos 3, quat 4, lin 3, ang 3) + the 4 corners in that frame (12).
    env._object_state_queue = torch.zeros(n, max(1, int(dr.object_state_delay_max)), 25, **f32)

    env._closest_fold_err = torch.full((n,), -1.0, **f32)       # -1 = not yet observed
    env._closest_fingertip_dist = torch.full((n, 5), -1.0, **f32)
    env._fold_hold = torch.zeros(n, dtype=torch.long, device=dev)
    env._successes = torch.zeros(n, dtype=torch.long, device=dev)
    env._prev_episode_successes = torch.zeros(n, dtype=torch.long, device=dev)
    env._bonus_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    env._nonfinite_envs = torch.zeros(n, dtype=torch.bool, device=dev)
    env._nonfinite_count = 0
    env._insane_count = 0
    env._unfold_env = torch.full((n,), cfg.fold.task == "unfold", dtype=torch.bool, device=dev)
    env._table_z = torch.full((n,), float(cfg.reset.table_reset_z), **f32)

    # fold.release (per episode)
    env._entry_paid = torch.zeros(n, dtype=torch.bool, device=dev)
    env._released = torch.zeros(n, dtype=torch.bool, device=dev)
    env._rel_streak = torch.zeros(n, dtype=torch.long, device=dev)
    env._persist_steps = torch.zeros(n, dtype=torch.long, device=dev)
    env._best_clear = torch.full((n,), -1.0, **f32)
    env._release_rew_step = torch.zeros(n, **f32)
    env._release_latch_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    env._persist_step_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    env._persist_done = torch.zeros(n, dtype=torch.bool, device=dev)
    env._persist_fail = torch.zeros(n, dtype=torch.bool, device=dev)

    env._reward_terms = {}
    env._termination_reasons = {}


def clear(env, ids: torch.Tensor) -> None:
    """Per-episode buffers back to their start values for ``ids``."""
    env._prev_episode_successes[ids] = env._successes[ids]
    env._successes[ids] = 0
    env._closest_fold_err[ids] = -1.0
    env._closest_fingertip_dist[ids] = -1.0
    env._fold_hold[ids] = 0
    env._bonus_mask[ids] = False
    for q in (env._action_queue, env._obs_queue, env._object_state_queue):
        q[ids] = 0.0
    for b in (env._entry_paid, env._released, env._release_latch_mask, env._persist_step_mask,
              env._persist_done, env._persist_fail):
        b[ids] = False
    env._rel_streak[ids] = 0
    env._persist_steps[ids] = 0
    env._release_rew_step[ids] = 0.0
    env._best_clear[ids] = -1.0
