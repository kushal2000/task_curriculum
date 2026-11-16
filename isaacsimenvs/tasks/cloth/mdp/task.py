"""Fold/unfold goals and episode state, independent of the simulator.

`evaluate` advances episode history once per policy step. `derive` and `is_folded`
are read-only queries used by observations, evaluation, and visualization.
"""

from __future__ import annotations

import math

import torch

from ..backends.robot import ROBOT
from .geometry import fold, rotations
from .state import ResetState, SimState, TaskTransition, TaskView

REWARD_SANITY_LIMIT = 1.0e4


def ratchet(best: torch.Tensor, current: torch.Tensor, cap: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(improvement clamped to [0, cap], new best)``; ``best < 0`` = not yet observed."""
    best = torch.where(best < 0.0, current, best)
    return torch.clamp(best - current, 0.0, cap), torch.minimum(best, current)


class ClothTask:
    def __init__(self, cfg, cloth_grid, origins, arm_ids, hand_ids):
        self.cfg, self._cloth_grid, self.origins = cfg, cloth_grid, origins
        self.device, self.num_envs = origins.device, origins.shape[0]
        self._arm_ids, self._hand_ids = arm_ids, hand_ids
        self._fingertip_offsets = torch.tensor(ROBOT.fingertip_offsets, device=self.device)
        self._object_scales = cloth_grid.dims / float(cfg.reward.object_base_size)
        self.extras = {}
        self.reward_buf = torch.zeros(self.num_envs, device=self.device)
        self._allocate()

    def evaluate(self, state, episode_length, max_episode_length, touching=None) -> TaskTransition:
        self.episode_length_buf = episode_length
        self.max_episode_length = max_episode_length
        self.hand_touching = touching
        view = self.derive(state)
        terminated, truncated = self.compute_terminations(state, view)
        self.reward_buf = self.compute_reward(state, view)
        self.log_step()
        return TaskTransition(self.reward_buf, terminated, truncated)

    def sample_reset(self, ids, joint_pos, joint_vel) -> ResetState:
        reset = self.cfg.reset
        table_z = reset.table_reset_z + torch.empty(ids.numel(), device=self.device).uniform_(
            -reset.table_reset_z_range, reset.table_reset_z_range
        )
        angle = self.start_task(ids)
        yaw = self.start_yaw(ids.numel())
        return ResetState(joint_pos, joint_vel, table_z, yaw, angle)

    def _allocate(self) -> None:
        """Allocate per-environment task history."""
        n, dev, cfg = self.num_envs, self.device, self.cfg
        f32 = dict(device=dev, dtype=torch.float32)

        self._closest_fold_err = torch.full((n,), -1.0, **f32)  # -1 = not yet observed
        self._closest_fingertip_dist = torch.full((n, 5), -1.0, **f32)
        self._fold_hold = torch.zeros(n, dtype=torch.long, device=dev)
        self._successes = torch.zeros(n, dtype=torch.long, device=dev)
        self._prev_episode_successes = torch.zeros(n, dtype=torch.long, device=dev)
        self._bonus_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self._nonfinite_envs = torch.zeros(n, dtype=torch.bool, device=dev)
        self._nonfinite_count = 0
        self._insane_count = 0
        self._unfold_env = torch.full((n,), cfg.task.mode == "unfold", dtype=torch.bool, device=dev)

        # task.release (per episode)
        self._entry_paid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._released = torch.zeros(n, dtype=torch.bool, device=dev)
        self._rel_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._persist_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._best_clear = torch.full((n,), -1.0, **f32)
        self._release_rew_step = torch.zeros(n, **f32)
        self._release_latch_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self._persist_step_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self._persist_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self._persist_fail = torch.zeros(n, dtype=torch.bool, device=dev)

        self._reward_terms = {}
        self._termination_reasons = {}

    def reset(self, ids: torch.Tensor) -> None:
        """Per-episode buffers back to their start values for ``ids``."""
        self._prev_episode_successes[ids] = self._successes[ids]
        self._successes[ids] = 0
        self._closest_fold_err[ids] = -1.0
        self._closest_fingertip_dist[ids] = -1.0
        self._fold_hold[ids] = 0
        self._bonus_mask[ids] = False
        for b in (
            self._entry_paid,
            self._released,
            self._release_latch_mask,
            self._persist_step_mask,
            self._persist_done,
            self._persist_fail,
        ):
            b[ids] = False
        self._rel_streak[ids] = 0
        self._persist_steps[ids] = 0
        self._release_rew_step[ids] = 0.0
        self._best_clear[ids] = -1.0

    def derive(self, state: SimState) -> TaskView:
        cg = self._cloth_grid
        r, c = fold.stationary_frame(cg, state.points_w)
        pos, q, lin, ang = fold.cloth_frame(state.points_w, state.points_vel_w, r)
        kp = state.points_w[:, cg.corner_idx]
        targets = fold.corner_targets(cg, r, c, self._unfold_env)
        pads = state.fingertip_pos_w
        if self.cfg.reward.fingertip_distance_from_pad:
            pads = pads + rotations.apply(state.fingertip_quat_w, self._fingertip_offsets.expand_as(pads))
        return TaskView(
            frame_r=r,
            frame_c=c,
            pos_w=pos,
            quat_w=q,
            lin_vel_w=lin,
            ang_vel_w=ang,
            corners_w=kp,
            targets_w=targets,
            fold_error=fold.fold_error(kp, targets),
            footprint=fold.footprint_ratio(cg, state.points_w, r),
            pads_w=pads,
            fingertip_dist=(pads - pos.unsqueeze(1)).norm(dim=-1),
        )

    def start_task(self, ids: torch.Tensor) -> torch.Tensor:
        """Draw each episode's task (``task.mode=mixed``) and return the + half's start angle [rad]:
        0 (flat) for a fold, U(``unfold_start_deg``) for an unfold."""
        f = self.cfg.task
        n = ids.numel()
        if f.mode == "mixed":
            self._unfold_env[ids] = torch.rand(n, device=self.device) < f.unfold_prob
        lo, hi = (math.radians(float(view)) for view in f.unfold_start_deg)
        angle = torch.empty(n, device=self.device).uniform_(lo, hi)
        return angle * self._unfold_env[ids].float()

    def start_yaw(self, n: int) -> torch.Tensor:
        if not self.cfg.reset.random_yaw:
            return torch.zeros(n, device=self.device)
        return torch.rand(n, device=self.device) * (2.0 * math.pi)

    def is_folded(self, state: SimState, view: TaskView) -> torch.Tensor:
        f = self.cfg.task
        unfold = self._unfold_env
        footprint_ok = torch.where(
            unfold, view.footprint > f.min_unfolded_footprint, view.footprint < f.max_folded_footprint
        )
        ok = (view.fold_error < f.keypoint_tolerance) & footprint_ok
        if f.unfold_on_table:
            ok = ok & (self.on_table(state) | ~unfold)
        return ok

    def on_table(self, state: SimState) -> torch.Tensor:
        f = self.cfg.task
        spread, lift = fold.flat_on_table(state.points_w, state.table_top_z)
        return (spread < f.on_table_max_spread) & (lift < f.on_table_max_lift)

    def hand_sheet_distance(self, state: SimState) -> torch.Tensor:
        """Nearest hand body origin to any cloth point [m]; a non-finite self reads as touching."""
        d = torch.cdist(state.hand_pos_w.float(), state.points_w.float()).amin(dim=(1, 2))
        return torch.nan_to_num(d, nan=0.0)

    def update_release(
        self, state: SimState, view: TaskView, folded: torch.Tensor, entered: torch.Tensor
    ) -> torch.Tensor:
        """``task.release``: advance the release / persist state; returns the success mask (the fold
        held ``persist_window_steps`` with the hand clear). The fold must be ENTERED first (its bonus
        is paid once), so a hand that never touched the cloth releases nothing."""
        c = self.cfg.task.release
        self._bonus_mask.copy_(entered & ~self._entry_paid)
        self._entry_paid |= entered

        d = self.hand_sheet_distance(state)
        if c.touch == "contact":
            clear = ~self.hand_touching
            self.extras["hand_cloth_touch"] = float((~clear).float().mean())
        else:
            clear = d > c.clearance

        # Release shaping: the running max of the capped clearance while folded and not yet released.
        track = folded & self._entry_paid & ~self._released
        d_cap = d.clamp(max=c.clearance)
        self._best_clear = torch.where(track & (self._best_clear < 0.0), d_cap, self._best_clear)
        gain = torch.where(track, (d_cap - self._best_clear).clamp(min=0.0), torch.zeros_like(d))
        self._best_clear = torch.where(track, torch.maximum(self._best_clear, d_cap), self._best_clear)
        self._release_rew_step.copy_(gain * c.release_dist_rew_scale)

        latch_ok = folded & clear & self._entry_paid
        if c.fold_tolerance > 0.0:
            latch_ok = latch_ok & (view.fold_error < c.fold_tolerance)
        self._rel_streak = torch.where(latch_ok, self._rel_streak + 1, torch.zeros_like(self._rel_streak))
        newly = ~self._released & (self._rel_streak >= c.latch_steps)
        self._release_latch_mask.copy_(newly)
        self._released |= newly

        ok = self._released & folded & clear
        self._persist_step_mask.copy_(ok)
        if c.persist_consecutive:
            self._persist_steps = torch.where(ok, self._persist_steps + 1, torch.zeros_like(self._persist_steps))
        else:
            self._persist_steps += ok.long()
        self._persist_done.copy_(self._released & (self._persist_steps >= c.persist_window_steps))
        self._persist_fail.copy_(self._released & ~folded)
        self.extras["hand_cloth_dist"] = float(d.mean())
        return self._persist_done.clone()

    def compute_terminations(self, state: SimState, view: TaskView) -> tuple[torch.Tensor, torch.Tensor]:
        """``(terminated, truncated)``; sets ``_bonus_mask`` for this step's reward."""
        cfg = self.cfg
        folded = self.is_folded(state, view)
        self._fold_hold = torch.where(folded, self._fold_hold + 1, torch.zeros_like(self._fold_hold))
        entered = self._fold_hold >= cfg.termination.success_steps
        held = self._fold_hold >= cfg.termination.held_steps

        release = cfg.task.release.enabled
        if release:
            success = self.update_release(state, view, folded, entered)
        else:
            self._bonus_mask.copy_(entered)
            success = entered
        self._successes += success.long()

        fall = (view.pos_w[:, 2] - self.origins[:, 2]) < cfg.termination.fall_height
        hand_far = view.fingertip_dist.amax(dim=-1) > cfg.termination.hand_far_distance
        nonfinite = self._nonfinite_envs.clone()
        self._nonfinite_envs.zero_()
        truncated = self.episode_length_buf >= self.max_episode_length

        reasons = {
            "fold": entered,
            "fold_held": held,
            "fall": fall,
            "hand_far": hand_far,
            "nonfinite": nonfinite,
            "timeout": truncated,
        }
        if cfg.task.mode != "fold":
            u = self._unfold_env
            reasons["on_table"] = entered & self.on_table(state)
            reasons["task_unfold"] = u.clone()
            reasons["unfold_ok"] = entered & u
            reasons["fold_ok"] = entered & ~u
        end = entered
        if release:
            reasons["released"] = self._released.clone()
            reasons["persist"] = self._persist_done.clone()
            reasons["persist_fail"] = self._persist_fail.clone()
            end = self._persist_done | (
                self._persist_fail if cfg.task.release.persist_fail_terminates else torch.zeros_like(fall)
            )
        self._termination_reasons = reasons
        return fall | hand_far | nonfinite | end, truncated

    def guard_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Zero and flag non-finite or implausibly large rewards."""
        bad = ~torch.isfinite(reward) | (reward.abs() > REWARD_SANITY_LIMIT)
        self.extras["insane_reward"] = float(bad.float().mean())
        if not bool(bad.any()):
            return reward
        worst = reward[bad].abs().nan_to_num(nan=float("inf")).max()
        print(
            f"[cloth] bad reward in {int(bad.sum())}/{self.num_envs} envs (worst |r| {float(worst):.3e}) "
            f"-- zeroed and flagged for reset",
            flush=True,
        )
        self._nonfinite_envs |= bad
        self._insane_count += int(bad.sum())
        return torch.where(bad, torch.zeros_like(reward), reward)

    def compute_reward(self, state: SimState, view: TaskView) -> torch.Tensor:
        rew, rel = self.cfg.reward, self.cfg.task.release

        ft_gain, self._closest_fingertip_dist = ratchet(self._closest_fingertip_dist, view.fingertip_dist, 10.0)
        fingertip = rew.distance_delta_rew_scale * ft_gain.sum(dim=-1)

        fold_err = view.fold_error
        # A non-finite self is being reset this step; keep the ratchet finite (no progress paid).
        fold_err = torch.where(torch.isfinite(fold_err), fold_err, self._closest_fold_err.clamp(min=0.0))
        fold_gain, self._closest_fold_err = ratchet(self._closest_fold_err, fold_err, 100.0)
        keypoint = rew.keypoint_rew_scale * fold_gain

        arm = -rew.kuka_actions_penalty_scale * state.joint_vel[:, self._arm_ids].abs().sum(dim=-1)
        hand = -rew.hand_actions_penalty_scale * state.joint_vel[:, self._hand_ids].abs().sum(dim=-1)

        terms = {
            "fingertip_delta_rew": fingertip,
            "keypoint_rew": keypoint,
            "kuka_actions_penalty": arm,
            "hand_actions_penalty": hand,
        }
        if rel.enabled:
            terms["bonus_rew"] = self._bonus_mask.float() * rel.fold_entry_bonus
            terms["release_rew"] = self._release_rew_step + self._release_latch_mask.float() * rel.release_bonus
            terms["persist_rew"] = (
                self._persist_step_mask.float() * rel.persist_rew_per_step
                + self._persist_done.float() * rel.persist_bonus
                + self._persist_fail.float() * rel.persist_fail_penalty
            )
        else:
            terms["bonus_rew"] = self._bonus_mask.float() * rew.reach_goal_bonus

        reward = self.guard_reward(sum(terms.values()))
        terms["total_reward"] = reward
        self._reward_terms = terms
        self.extras["fold_err_mean"] = float(fold_err.mean())
        self.extras["fold_rate"] = float(self._bonus_mask.float().mean())
        return reward

    def log_step(self) -> None:
        final = {"successes": self._successes.float()}
        final.update({f"done_{k}": view.float() for k, view in self._termination_reasons.items()})
        self.extras["episode_cumulative"] = self._reward_terms
        self.extras["episode_final"] = final
        self.extras["successes"] = self._prev_episode_successes.float()
        self.extras["nonfinite_total"] = float(self._nonfinite_count)
        self.extras["insane_total"] = float(self._insane_count)
