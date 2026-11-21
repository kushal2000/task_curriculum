"""``ClothTask``: the fold task -- where an episode starts, when it ends, what it pays -- on top of
the fold geometry (``fold.py``). Simulator-free: it reads ``SimState`` / ``FoldView`` and hands
back tensors; the backend writes what it draws.

Reward::

    reward = fingertip_approach + fold_progress + fold_bonus - arm_joint_speed - hand_joint_speed

* ``fingertip_approach``: pay for every fingertip getting closer to the sheet than it has been
  this episode (``distance_delta_rew_scale``), until the sheet is lifted;
* ``fold_progress``: pay for every new best fold error this episode (``keypoint_rew_scale``);
* ``fold_bonus``: ``reach_goal_bonus`` on the step the fold criterion is met;
* the joint-speed penalties: L1 joint velocity, arm and hand (``*_actions_penalty_scale``).

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

    # ------------------------------------------------------------------ geometry queries

    def derive(self, state: SimState) -> FoldView:
        return self.geometry.derive(state.points_w, state.points_vel_w)

    def is_folded(self, view: FoldView) -> torch.Tensor:
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
        eval_poses = getattr(self, "evaluation_start_poses", None)
        if eval_poses is not None:
            pos, rot = eval_poses[env_ids, :3], eval_poses[env_ids, 3:]
        elif cfg.fixed_start_pose is not None:
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

    def compute_dones(self, state: SimState, view: FoldView, episode_length_buf, max_episode_length):
        """``(terminated, truncated)``; sets ``termination_reasons``, ``fold_bonus_mask`` and ``successes``."""
        self._update_hand_and_goal_metrics(state, view)

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
        self.termination_reasons = {
            "fall": fall,
            "hand_far": hand_far,
            "timeout": truncated,
            "fold": fold,
            "fold_held": fold_held,
            "nonfinite": nonfinite,
        }
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

        reward = fingertip_approach + fold_progress + fold_bonus + arm_penalty + hand_penalty
        reward = self._guard_reward(reward)
        self.reward_terms = {
            "fingertip_delta_rew": fingertip_approach,
            "keypoint_rew": fold_progress,
            "bonus_rew": fold_bonus,
            "kuka_actions_penalty": arm_penalty,
            "hand_actions_penalty": hand_penalty,
            "total_reward": reward,
        }
        self.extras["fold_err_mean"] = float(fold_error.mean())
        self.extras["fold_rate"] = float(self.fold_bonus_mask.float().mean())
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
