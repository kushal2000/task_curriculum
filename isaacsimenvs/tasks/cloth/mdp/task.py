"""``ClothTask``: the fold task -- where an episode starts, when it ends, what it pays -- on top of
the fold geometry (``fold.py``). Simulator-free: it reads ``SimState`` / ``FoldView`` and hands
back tensors; the backend writes what it draws.

Reward::

    reward = fingertip_approach + fold_progress + fold_bonus - arm_joint_speed - hand_joint_speed - slide

* ``fingertip_approach``: pay for every fingertip getting closer to the sheet than it has been
  this episode (``distance_delta_rew_scale``), until the sheet is lifted;
* ``fold_progress``: pay for every new best fold error this episode (``keypoint_rew_scale``);
* ``fold_bonus``: ``reach_goal_bonus`` on the step the fold criterion is met;
* the joint-speed penalties: L1 joint velocity, arm and hand (``*_actions_penalty_scale``);
* ``slide``: the stationary half's in-plane travel this step, past a deadband and capped per step
  (``slide_penalty_scale`` per metre, ``slide_deadband`` m/s, ``slide_step_cap`` m; 0 by default).

**The fold criterion** (``termination.success``). ``fold_error``: the moved particles within
``cloth.keypoint_tolerance`` of the designated goal and the footprint halved (look-alikes with
``accept_lookalike``). ``surface``: the goal sheet is frozen where it is at ``surface.ref_step``;
the sheet is folded when its surface is within ``surface.tolerance_frac`` x its side of the frozen goal's,
both ways, after the best placement in the table plane (``mdp/surface_match.py``): any fold that
looks like the goal counts, wherever it was made. With ``surface.layered`` (the default) each ply is
compared only with the goal's same ply under that placement (``mdp/surface_layered.py``), so a top ply
misaligned over the bottom one fails. ``termination.max_in_place_dist`` (``.inf`` by default)
also requires the two-sided max distance to the frozen goal WITHOUT alignment below it: the in-place
gate. With ``surface`` the old criterion is still reported (``fold_designated``), not acted on; the
fold progress reward stays the designated fold error.

**Drift** (logged). Once the sheet has landed (``termination.drift_settle_steps`` into the episode)
the stationary half's grid points are latched as its anchor; ``drift`` is their mean in-plane (xy)
distance from it, ``yaw`` their in-plane rotation about their centroid (least-squares 2-D fit,
|rad|), both 0 before the latch. ``slide`` pays the same points' per-step travel, which the policy
can sense, not the accumulated drift, which it cannot.

**Gate curriculum.** ``termination.gate_curriculum.max_in_place_dist`` (a list; empty = off) replaces
``max_in_place_dist`` and ``slide_penalty_scale`` by stages, stage ``k`` using entry ``k`` of each list.
The stage advances when the gated fold rate over a window of ``window_episodes`` finished episodes
reaches ``advance_rate``; ``start_stage`` restarts a resumed run where it stood.

A non-finite or implausibly large reward (a diverging env) is zeroed and the env flagged for reset.

Terminated on ``fold`` (the criterion held ``termination.success_steps`` steps; the step the bonus
pays on), ``fall`` (the sheet's centre below 0.1 m), ``hand_far`` (a fingertip more than 1.5 m from
the sheet), ``nonfinite`` (the env diverged this step); truncated on ``timeout``. ``fold_held``
(the criterion held ``HELD_FOLD_STEPS``) is reported, not acted on.

The goal is a pose -- the fold targets' centre and the folded half's orientation -- kept as STATE:
refreshed at reset and with every observation, and read one step old by the critic's goal
distance (``closest_keypoint_max_dist``), as the Play prior's critic learnt it.
"""

from __future__ import annotations

import torch

from . import surface_layered, surface_match
from .fold import FoldGeometry
from .geometry import quat
from .robot import NUM_FINGERTIPS, RobotLayout
from .state import FoldView, SheetLayout, SimState

__all__ = ["HELD_FOLD_STEPS", "REWARD_SANITY_LIMIT", "ClothTask", "apply_local_offset", "keypoints_world"]

#: Any |reward| above this is a diverging env, not a task outcome: the most the task can pay in one
#: step is ``reach_goal_bonus`` (1000) plus shaping of order 100.
REWARD_SANITY_LIMIT = 1.0e4

#: Steps the fold criterion must hold to count as a held fold. Independent of
#: ``termination.success_steps``, which a finetune recipe may set to 1 so the bonus pays on first entry.
HELD_FOLD_STEPS = 10


def apply_local_offset(pos_w, rot_wxyz, offset, batch_shape: tuple[int, ...]) -> torch.Tensor:
    """Shift batched world positions by one offset given in each pose's own frame."""
    offset_t = torch.as_tensor(offset, device=pos_w.device, dtype=pos_w.dtype).expand(*batch_shape, 3)
    shifted = quat.quat_apply(rot_wxyz.reshape(-1, 4), offset_t.reshape(-1, 3)).reshape(*batch_shape, 3)
    return pos_w + shifted


def keypoints_world(center_pos: torch.Tensor, center_rot: torch.Tensor, kp_offsets: torch.Tensor) -> torch.Tensor:
    """``centre + R * offsets``: (N, 3), (N, 4) wxyz, (N, K, 3) -> (N, K, 3)."""
    n, k, _ = kp_offsets.shape
    rot = center_rot.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    return center_pos.unsqueeze(1) + quat.quat_apply(rot, kp_offsets.reshape(-1, 3)).reshape(n, k, 3)


def ratchet_unset(best: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """``best`` with every not-yet-observed entry (``< 0``) set to ``current``."""
    return torch.where(best < 0.0, current, best)


class ClothTask:
    def __init__(
        self, cfg, layout: SheetLayout, robot: RobotLayout, *, num_envs: int, device, env_origins, extras: dict,
        observed_goal_quat,
    ) -> None:
        self.cfg, self.robot = cfg, robot
        self.num_envs, self.device = num_envs, device
        self.env_origins = env_origins
        self.extras = extras
        self.geometry = FoldGeometry(layout, num_envs=num_envs, device=device, observed_goal_quat=observed_goal_quat)
        self.object_scale = layout.object_scale
        self._allocate()

    def _allocate(self) -> None:
        n, dev = self.num_envs, self.device
        reset_cfg = self.cfg.reset
        # Folds this episode, and in the finished one (the logger's `successes`).
        self.successes = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_episode_successes = torch.zeros(n, dtype=torch.long, device=dev)
        # The fold: steps the criterion has held, the bonus mask, best fold error (-1 = unset).
        self.fold_hold = torch.zeros(n, dtype=torch.long, device=dev)
        self.fold_bonus_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self.closest_fold_err = torch.full((n,), -1.0, device=dev)
        # The hand and goal measurements, with their best-so-far values (-1 = unset).
        self.curr_fingertip_distances = torch.zeros(n, NUM_FINGERTIPS, device=dev)
        self.closest_fingertip_dist = torch.full((n, NUM_FINGERTIPS), -1.0, device=dev)
        self.keypoints_max_dist = torch.zeros(n, device=dev)
        self.closest_keypoint_max_dist = torch.full((n,), -1.0, device=dev)
        self.lifted_object = torch.zeros(n, dtype=torch.bool, device=dev)
        self.object_init_z = torch.full((n,), reset_cfg.table_reset_z + reset_cfg.table_object_z_offset, device=dev)
        self.table_z = torch.full((n,), float(reset_cfg.table_reset_z), device=dev)
        # Divergence guards: envs flagged this step (consumed by the dones), and running counts.
        self.nonfinite_envs = torch.zeros(n, dtype=torch.bool, device=dev)
        self.nonfinite_count = 0
        self.insane_count = 0
        # The goal pose (see the module doc) and the per-step outputs.
        self.goal_pos_w = torch.zeros(n, 3, device=dev)
        self.goal_quat = torch.zeros(n, 4, device=dev)
        self.reward_terms: dict[str, torch.Tensor] = {}
        self.termination_reasons: dict[str, torch.Tensor] = {}
        #: Fold error of a flat sheet: what a non-finite one counts as (set by the env at startup).
        self.flat_fold_err = 0.0
        # Drift (module doc): the stationary half's latched xy anchor, its previous xy (the per-step
        # travel), the latch, this step's travel, and the drift and yaw with their episode maxima.
        n_stationary = int(self.geometry.stationary_idx.numel())
        self.drift_anchor_xy = torch.zeros(n, n_stationary, 2, device=dev)
        self.drift_prev_xy = torch.zeros(n, n_stationary, 2, device=dev)
        self.drift_anchored = torch.zeros(n, dtype=torch.bool, device=dev)
        self.slide_step = torch.zeros(n, device=dev)
        self.drift = torch.zeros(n, device=dev)
        self.drift_max = torch.zeros(n, device=dev)
        self.yaw = torch.zeros(n, device=dev)
        self.yaw_max = torch.zeros(n, device=dev)
        # The surface criterion (module doc): the frozen goal sheet (world frame, particle grid) and
        # this step's distances to it [m] -- after the best in-plane placement (the unlabelled one and,
        # with ``surface.layered``, the layered one), and as it is.
        self.surface = self.cfg.termination.success == "surface"
        self.surface_layered = self.surface and bool(self.cfg.termination.surface.layered)
        n_particles = int(self.geometry.layout.resolution) ** 2
        self.frozen_goal_w = torch.zeros(n, n_particles, 3, device=dev)
        self.goal_is_frozen = torch.zeros(n, dtype=torch.bool, device=dev)
        self.aligned_dist = torch.full((n,), float("inf"), device=dev)
        self.layered_dist = torch.full((n,), float("inf"), device=dev)
        self.in_place_dist = torch.full((n,), float("inf"), device=dev)
        self.aligned_yaw = torch.zeros(n, device=dev)
        #: The old criterion (``fold_error``), reported for comparison when ``surface`` decides.
        self.designated_folded = torch.zeros(n, dtype=torch.bool, device=dev)
        # The gate curriculum (module doc): stage, and the current window's finished / folded episodes.
        curriculum = self.cfg.termination.gate_curriculum
        self.gate_stages = len(curriculum.max_in_place_dist)
        self.gate_stage = min(int(curriculum.start_stage), max(self.gate_stages - 1, 0))
        self.gate_window_episodes = 0
        self.gate_window_folds = 0
        self.gate_last_rate = 0.0

    # ------------------------------------------------------------------ the in-place gate

    def gate(self) -> tuple[float, float]:
        """``(max_in_place_dist m, slide_penalty_scale)`` in force now."""
        if self.gate_stages:
            curriculum, stage = self.cfg.termination.gate_curriculum, self.gate_stage
            return float(curriculum.max_in_place_dist[stage]), float(curriculum.slide_penalty_scale[stage])
        return float(self.cfg.termination.max_in_place_dist), float(self.cfg.reward.slide_penalty_scale)

    def _advance_gate(self, done: torch.Tensor, fold: torch.Tensor) -> None:
        """Count this step's finished episodes; at a full window, advance the stage if folds reached the rate."""
        if not self.gate_stages:
            return
        curriculum = self.cfg.termination.gate_curriculum
        self.gate_window_episodes += int(done.sum())
        self.gate_window_folds += int((done & fold).sum())
        if self.gate_window_episodes >= int(curriculum.window_episodes):
            self.gate_last_rate = self.gate_window_folds / self.gate_window_episodes
            if self.gate_last_rate >= float(curriculum.advance_rate) and self.gate_stage < self.gate_stages - 1:
                self.gate_stage += 1
                max_in_place, slide_scale = self.gate()
                print(f"[cloth] gate curriculum -> stage {self.gate_stage}: fold rate {self.gate_last_rate:.3f}; "
                      f"in place < {max_in_place * 100:.1f} cm, slide {slide_scale:.0f}/m", flush=True)
            self.gate_window_episodes = self.gate_window_folds = 0
        max_in_place, slide_scale = self.gate()
        self.extras["gate_stage"] = float(self.gate_stage)
        self.extras["gate_max_in_place_dist"] = max_in_place
        self.extras["gate_slide_penalty"] = slide_scale
        self.extras["gate_window_rate"] = self.gate_last_rate

    # ------------------------------------------------------------------ geometry queries

    def derive(self, state: SimState) -> FoldView:
        return self.geometry.derive(state.points_w, state.points_vel_w)

    def sheet_size(self) -> torch.Tensor:
        """``(N,)`` each env's sheet side [m]."""
        layout = self.geometry.layout
        if layout.env_size is not None:
            return layout.env_size.to(self.device).float()
        return torch.full((self.num_envs,), float(layout.size), device=self.device)

    @property
    def surface_dist(self) -> torch.Tensor:
        """``(N,)`` the distance the surface criterion holds below its tolerance: the layered one with
        ``surface.layered``, else the unlabelled aligned one."""
        return self.layered_dist if self.surface_layered else self.aligned_dist

    def surface_tolerance(self) -> torch.Tensor:
        """``(N,)`` the surface criterion's tolerance [m]: ``surface.tolerance_frac`` x each sheet's side."""
        return float(self.cfg.termination.surface.tolerance_frac) * self.sheet_size()

    def is_folded(self, view: FoldView) -> torch.Tensor:
        """The fold criterion (``termination.success``, module doc); the surface one reads the
        distances ``_update_surface`` measured this step."""
        if self.surface:
            folded = self.goal_is_frozen & (self.surface_dist < self.surface_tolerance())
            max_in_place, _ = self.gate()
            if max_in_place < float("inf"):
                folded = folded & (self.in_place_dist < max_in_place)
            return folded
        return self.is_folded_designated(view)

    def is_folded_designated(self, view: FoldView) -> torch.Tensor:
        """Both conditions: keypoints on target AND the footprint halved.

        With goal tasks (``fold_pool.goal_task``): ``any_fold`` needs only the fold error (the
        smaller of the designated and the look-alike one) within tolerance; ``half_fold`` keeps
        both conditions for the designated fold, and a look-alike within tolerance also counts."""
        c = self.cfg.cloth
        tol = c.keypoint_tolerance
        if view.designated_error is None:
            return (view.fold_error < tol) & (view.footprint < c.max_folded_footprint)
        if self.geometry.goal_tasks.any_fold:
            return view.fold_error < tol
        return ((view.designated_error < tol) & (view.footprint < c.max_folded_footprint)) | (view.lookalike_error < tol)

    @property
    def keypoint_offsets(self) -> torch.Tensor:
        return self.geometry.keypoint_offsets

    def update_goal(self, view: FoldView) -> torch.Tensor:
        """Refresh the goal pose from ``view``; returns the goal marker's true pose (N, 7)."""
        self.goal_pos_w, self.goal_quat, marker = self.geometry.goal_pose(view)
        return marker

    # ------------------------------------------------------------------ the episode's start

    def sample_table(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Draw the table pose of ``env_ids``, (n, 7) env frame, wxyz: at the env origin, its height
        jittered by ``table_reset_z_range``. Drawn before the sheet, whose height it sets."""
        cfg = self.cfg.reset
        dev = self.device
        n = env_ids.numel()
        dz = torch.empty(n, device=dev).uniform_(-cfg.table_reset_z_range, cfg.table_reset_z_range)
        table_z = cfg.table_reset_z + dz
        self.table_z[env_ids] = table_z

        pos = torch.zeros(n, 3, device=dev)
        pos[:, 2] = table_z
        rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).unsqueeze(0).expand(n, -1)
        return torch.cat([pos, rot], dim=-1)

    def sample_sheet(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Draw the sheet's start pose of ``env_ids``, (n, 7) env frame, wxyz: at
        ``reset_position_noise_{x,y,z}`` around the env centre, at a uniformly random orientation of
        which the sheet keeps only the yaw (a uniform SO(3) draw has a uniform yaw).
        ``fixed_start_pose`` pins it (the evaluation protocol does). Its height is the lift latch's
        reference."""
        cfg = self.cfg.reset
        dev = self.device
        n = env_ids.numel()
        if cfg.fixed_start_pose is not None:
            fixed = torch.as_tensor(cfg.fixed_start_pose, device=dev, dtype=torch.float32)
            pos = fixed[:3].unsqueeze(0).expand(n, -1)
            rot = fixed[3:].unsqueeze(0).expand(n, -1)
        else:
            noise = torch.empty(n, 3, device=dev).uniform_(-1.0, 1.0)
            pos = torch.stack(
                (
                    noise[:, 0] * cfg.reset_position_noise_x,
                    noise[:, 1] * cfg.reset_position_noise_y,
                    self.table_z[env_ids] + cfg.table_object_z_offset + noise[:, 2] * cfg.reset_position_noise_z,
                ),
                dim=-1,
            )
            rot = quat.random_orientation(n, device=dev)
        self.object_init_z[env_ids] = pos[:, 2]
        return torch.cat([pos, rot], dim=-1)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Per-episode state back to its start: successes (kept as the finished episode's count),
        the best-so-far trackers (-1 = not observed yet), the lift latch and the fold counters."""
        self.prev_episode_successes[env_ids] = self.successes[env_ids]
        self.successes[env_ids] = 0
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.closest_fingertip_dist[env_ids] = -1.0
        self.lifted_object[env_ids] = False
        self.fold_hold[env_ids] = 0
        self.closest_fold_err[env_ids] = -1.0
        self.fold_bonus_mask[env_ids] = False
        self.drift_anchored[env_ids] = False
        self.slide_step[env_ids] = 0.0
        self.drift[env_ids] = 0.0
        self.drift_max[env_ids] = 0.0
        self.yaw[env_ids] = 0.0
        self.yaw_max[env_ids] = 0.0
        self.goal_is_frozen[env_ids] = False
        self.aligned_dist[env_ids] = float("inf")
        self.in_place_dist[env_ids] = float("inf")

    def episode_start(self, episode_length_buf: torch.Tensor) -> torch.Tensor:
        """``(N,)`` bool: the episode's first step (delay lines flush on it)."""
        return (episode_length_buf == 0) & (self.successes == 0)

    # ------------------------------------------------------------------ dones

    def fingertip_pads_w(self, state: SimState) -> torch.Tensor:
        """``(N, 5, 3)`` fingertip positions, at the pad point the policy observes when
        ``reward.fingertip_distance_from_pad`` (``RobotSpec.fingertip_offset``)."""
        fingertips = state.body_state_w[:, self.robot.fingertip_body_ids, :]
        pos = fingertips[:, :, 0:3]
        if self.cfg.reward.fingertip_distance_from_pad:
            pos = apply_local_offset(pos, fingertips[:, :, 3:7], self.robot.spec.fingertip_offset, (self.num_envs, NUM_FINGERTIPS))
        return pos

    def _update_hand_and_goal_metrics(self, state: SimState, view: FoldView) -> None:
        """Fingertip-to-sheet distances and the goal keypoint distance, with their best-so-far values."""
        origins = self.env_origins
        sheet_pos = view.object_pos_w - origins
        sheet_rot = view.object_quat_w
        goal_pos = self.goal_pos_w - origins
        fingertip_pos = self.fingertip_pads_w(state) - origins.unsqueeze(1)
        self.curr_fingertip_distances = torch.norm(fingertip_pos - sheet_pos.unsqueeze(1), dim=-1)

        sheet_kp = keypoints_world(sheet_pos, sheet_rot, self.keypoint_offsets)
        goal_kp = keypoints_world(goal_pos, self.goal_quat, self.keypoint_offsets)
        self.keypoints_max_dist = torch.norm(sheet_kp - goal_kp, dim=-1).max(dim=-1).values
        self.closest_keypoint_max_dist = ratchet_unset(self.closest_keypoint_max_dist, self.keypoints_max_dist)
        self.closest_fingertip_dist = ratchet_unset(self.closest_fingertip_dist, self.curr_fingertip_distances)

    def _update_drift(self, view: FoldView, episode_length_buf: torch.Tensor) -> None:
        """Latch the anchor once landed; this step's travel, the drift, the yaw and their maxima (module doc)."""
        stationary_xy = view.points_w[:, self.geometry.stationary_idx, :2]
        latch = ~self.drift_anchored & (episode_length_buf >= int(self.cfg.termination.drift_settle_steps))
        self.drift_anchor_xy = torch.where(latch.view(-1, 1, 1), stationary_xy, self.drift_anchor_xy)
        self.drift_prev_xy = torch.where(latch.view(-1, 1, 1), stationary_xy, self.drift_prev_xy)
        self.drift_anchored |= latch
        anchored = self.drift_anchored.float()
        travel = (stationary_xy - self.drift_prev_xy).norm(dim=-1).mean(dim=-1) * anchored
        drift = (stationary_xy - self.drift_anchor_xy).norm(dim=-1).mean(dim=-1) * anchored
        # In-plane rotation: the angle that best maps the centred anchor onto the centred points.
        anchor_c = self.drift_anchor_xy - self.drift_anchor_xy.mean(dim=1, keepdim=True)
        now_c = stationary_xy - stationary_xy.mean(dim=1, keepdim=True)
        cross = (anchor_c[..., 0] * now_c[..., 1] - anchor_c[..., 1] * now_c[..., 0]).sum(dim=-1)
        dot = (anchor_c * now_c).sum(dim=(-1, -2))
        yaw = torch.atan2(cross, dot).abs() * anchored
        # A diverged sheet (reset this step anyway) neither pays nor poisons the maximum.
        self.slide_step = torch.nan_to_num(travel, nan=0.0, posinf=0.0, neginf=0.0)
        self.drift = torch.nan_to_num(drift, nan=0.0, posinf=0.0, neginf=0.0)
        self.drift_max = torch.maximum(self.drift_max, self.drift)
        self.yaw = torch.nan_to_num(yaw, nan=0.0, posinf=0.0, neginf=0.0)
        self.yaw_max = torch.maximum(self.yaw_max, self.yaw)
        self.drift_prev_xy = stationary_xy.clone()

    def goal_now_w(self, view: FoldView) -> torch.Tensor:
        """``(N, P, 3)`` world: the designated goal sheet carried on the stationary half's live pose
        (what the fold progress reward measures against)."""
        goal_now, _goal_rest = self.geometry.goal_sheet(view.points_w)
        return goal_now

    def goal_reference_w(self, view: FoldView) -> torch.Tensor:
        """``(N, P, 3)`` world: the frozen goal sheet (``surface.ref_step``), or before it is frozen
        the live one."""
        return torch.where(self.goal_is_frozen.view(-1, 1, 1), self.frozen_goal_w, self.goal_now_w(view))

    def _update_surface(self, view: FoldView, episode_length_buf: torch.Tensor) -> None:
        """Freeze the goal sheet at ``surface.ref_step``; this step's aligned (and layered) and in-place
        distances."""
        surface_cfg = self.cfg.termination.surface
        freeze_now = ~self.goal_is_frozen & (episode_length_buf >= int(surface_cfg.ref_step))
        self.frozen_goal_w = torch.where(freeze_now.view(-1, 1, 1), self.goal_now_w(view), self.frozen_goal_w)
        self.goal_is_frozen |= freeze_now

        resolution = int(self.geometry.layout.resolution)
        sheet = torch.nan_to_num(view.points_w, nan=0.0, posinf=0.0, neginf=0.0)
        match_args = dict(
            fine_side=int(surface_cfg.fine_side), n_uniform_starts=int(surface_cfg.uniform_start_yaws),
            icp_iters=int(surface_cfg.icp_iters),
        )
        if self.surface_layered:
            # The goal's top ply is the half the stage-1 / half_fold goal folds over, the sheet's moving
            # half (``FoldGeometry.goal_sheet``): the rows above the middle one for ``fold_axis`` x (row
            # i is x_i), the columns right of it for y.
            goal_top = ("row" if self.geometry.layout.fold_axis == "x" else "col", "upper")
            layered = surface_layered.layered_aligned_distance(
                sheet, self.frozen_goal_w, resolution, goal_top=goal_top, **match_args,
            )
            aligned = {"max": layered["aligned_max"], "yaw": layered["yaw"]}
        else:
            aligned = surface_match.aligned_distance(sheet, self.frozen_goal_w, resolution, **match_args)
        in_place = surface_match.in_place_distance(sheet, self.frozen_goal_w, resolution, fine_side=int(surface_cfg.fine_side))
        # No distance before the goal is frozen, nor for a diverged sheet (reset this step anyway).
        no_distance = ~torch.isfinite(view.points_w).all(dim=-1).all(dim=-1) | ~self.goal_is_frozen
        inf = torch.full_like(in_place, float("inf"))
        self.aligned_dist = torch.where(no_distance, inf, aligned["max"])
        if self.surface_layered:
            self.layered_dist = torch.where(no_distance, inf, layered["max"])
        self.in_place_dist = torch.where(no_distance, inf, in_place)
        self.aligned_yaw = aligned["yaw"]

    def compute_dones(self, state: SimState, view: FoldView, episode_length_buf, max_episode_length):
        """``(terminated, truncated)``; sets ``termination_reasons``, ``fold_bonus_mask`` and ``successes``."""
        self._update_hand_and_goal_metrics(state, view)
        self._update_drift(view, episode_length_buf)
        if self.surface:
            self._update_surface(view, episode_length_buf)

        sheet_z = view.object_pos_w[:, 2] - self.env_origins[:, 2]
        fall = sheet_z < 0.1
        hand_far = self.curr_fingertip_distances.max(dim=-1).values > 1.5

        folded = self.is_folded(view)
        self.fold_hold = torch.where(folded, self.fold_hold + 1, torch.zeros_like(self.fold_hold))
        fold = self.fold_hold >= int(self.cfg.termination.success_steps)
        fold_held = self.fold_hold >= HELD_FOLD_STEPS
        self.fold_bonus_mask.copy_(fold)
        self.successes += fold.long()

        # Consumed here: the reset restores a valid state.
        nonfinite = self.nonfinite_envs.clone()
        self.nonfinite_envs.zero_()

        truncated = episode_length_buf >= max_episode_length
        self._advance_gate(fall | hand_far | fold | nonfinite | truncated, fold)
        self.termination_reasons = {
            "fall": fall,
            "hand_far": hand_far,
            "timeout": truncated,
            "fold": fold,
            "fold_held": fold_held,
            "nonfinite": nonfinite,
        }
        if self.surface:
            # The old criterion, reported for comparison (not acted on).
            self.designated_folded = self.is_folded_designated(view)
            self.termination_reasons["fold_designated"] = self.designated_folded
        return fall | hand_far | fold | nonfinite, truncated

    # ------------------------------------------------------------------ reward

    def compute_reward(self, state: SimState, view: FoldView) -> torch.Tensor:
        """This step's reward (N,); its terms go to ``reward_terms``."""
        reward_cfg = self.cfg.reward

        # The lift latch (Play's): set once the sheet's centre rises past `lifting_bonus_threshold`. It
        # gates the fingertip approach term and is observed by the critic (`lifted_object`).
        sheet_z = (view.object_pos_w - self.env_origins)[:, 2]
        lift = 0.05 + sheet_z - self.object_init_z
        self.lifted_object = (lift > reward_cfg.lifting_bonus_threshold) | self.lifted_object
        not_lifted = (~self.lifted_object).float()

        approach = torch.clamp(self.closest_fingertip_dist - self.curr_fingertip_distances, 0.0, 10.0)
        fingertip_approach = approach.sum(dim=-1) * not_lifted * reward_cfg.distance_delta_rew_scale
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, self.curr_fingertip_distances)
        self.closest_keypoint_max_dist = torch.minimum(self.closest_keypoint_max_dist, self.keypoints_max_dist)

        joint_speed = state.joint_vel.abs()
        arm_penalty = -reward_cfg.kuka_actions_penalty_scale * joint_speed[:, self.robot.arm_joint_ids].sum(dim=-1)
        hand_penalty = -reward_cfg.hand_actions_penalty_scale * joint_speed[:, self.robot.hand_joint_ids].sum(dim=-1)

        # Fold progress, ratcheted on the best fold error this episode: regression is free, and
        # oscillating cannot farm it. A non-finite error (a diverged sheet, reset this step anyway)
        # counts as a flat sheet, so it can neither pay nor poison the ratchet.
        fold_error = view.fold_error
        fold_error = torch.where(torch.isfinite(fold_error), fold_error, torch.full_like(fold_error, self.flat_fold_err))
        self.closest_fold_err = ratchet_unset(self.closest_fold_err, fold_error)
        fold_progress = torch.clamp(self.closest_fold_err - fold_error, 0.0, 100.0) * float(reward_cfg.keypoint_rew_scale)
        self.closest_fold_err = torch.minimum(self.closest_fold_err, fold_error)

        fold_bonus = self.fold_bonus_mask.float() * float(reward_cfg.reach_goal_bonus)

        policy_dt = float(self.cfg.sim.dt) * int(self.cfg.decimation)
        # Capped per step: a sheet thrown off the table must not outweigh the task.
        slide = torch.clamp(self.slide_step - float(reward_cfg.slide_deadband) * policy_dt, 0.0, float(reward_cfg.slide_step_cap))
        _, slide_scale = self.gate()
        slide_penalty = -slide_scale * slide

        reward = fingertip_approach + fold_progress + fold_bonus + arm_penalty + hand_penalty
        if slide_scale != 0.0:
            reward = reward + slide_penalty
        reward = self._guard_reward(reward)
        self.reward_terms = {
            "fingertip_delta_rew": fingertip_approach,
            "keypoint_rew": fold_progress,
            "bonus_rew": fold_bonus,
            "kuka_actions_penalty": arm_penalty,
            "hand_actions_penalty": hand_penalty,
            "slide_penalty": slide_penalty,
            "total_reward": reward,
        }
        self.extras["fold_err_mean"] = float(fold_error.mean())
        self.extras["fold_rate"] = float(self.fold_bonus_mask.float().mean())
        if self.surface:
            # Means over the envs whose goal is frozen (the others have no distance yet).
            frozen = self.goal_is_frozen
            n_frozen = frozen.float().sum().clamp(min=1.0)
            self.extras["aligned_dist_mean"] = float(torch.where(frozen, self.aligned_dist, 0.0).sum() / n_frozen)
            if self.surface_layered:
                self.extras["layered_dist_mean"] = float(torch.where(frozen, self.layered_dist, 0.0).sum() / n_frozen)
            self.extras["in_place_dist_mean"] = float(torch.where(frozen, self.in_place_dist, 0.0).sum() / n_frozen)
            self.extras["fold_designated_rate"] = float(self.designated_folded.float().mean())
        return reward

    def _guard_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Zero a non-finite or implausibly large reward and flag the env for reset.

        Finiteness alone is not enough: a diverging articulation drives the joint-speed penalties to
        -1e5 .. -1e11 while every value is still finite (seen in jobs 197715 and 199350). Either way
        the number would reach the advantage and the gradient, so it contributes nothing and the env
        resets through the normal path (the dones consume ``nonfinite_envs``).
        """
        nonfinite = ~torch.isfinite(reward)
        if bool(nonfinite.any()):
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
            self.nonfinite_envs |= nonfinite
            print(f"[cloth] non-finite REWARD in {int(nonfinite.sum())}/{self.num_envs} envs -- zeroed and flagged for reset", flush=True)
        self.extras["nonfinite_reward"] = float(nonfinite.float().mean())

        insane = reward.abs() > REWARD_SANITY_LIMIT
        if bool(insane.any()):
            worst = float(reward[insane].abs().max())
            reward = torch.where(insane, torch.zeros_like(reward), reward)
            self.nonfinite_envs |= insane
            print(
                f"[cloth] INSANE REWARD in {int(insane.sum())}/{self.num_envs} envs (worst |r|={worst:.3e}, "
                f"limit {REWARD_SANITY_LIMIT:.0e}) -- zeroed and flagged for reset",
                flush=True,
            )
        self.extras["insane_reward"] = float(insane.float().mean())
        self.insane_count += int(insane.sum())
        self.extras["insane_total"] = float(self.insane_count)
        return reward
