"""Success and episode ends.

**Success** (``is_folded``) is two independent conditions: every tracked corner within
``fold.keypoint_tolerance`` of its target, AND the extent along the sheet's own fold axis halved
(fold: below ``max_folded_footprint`` of the length; unfold: back above ``min_unfolded_footprint``).
Keypoints alone pass a dragged, crumpled sheet; the footprint alone passes a crumple.

**Ends**: the fold entered (``termination.success_steps`` in a row) -- or, with ``fold.release``,
the fold persisted with the hand clear -- the sheet off the table, a fingertip too far, a
non-finite state, or the time limit. Each env's reason is in ``env._termination_reasons`` (logged
as ``done_<reason>``).
"""

from __future__ import annotations

import torch

from . import geometry
from .state import ClothState, TaskView


def is_folded(env, s: ClothState, v: TaskView) -> torch.Tensor:
    f = env.cfg.fold
    unfold = env._unfold_env
    footprint_ok = torch.where(unfold, v.footprint > f.min_unfolded_footprint, v.footprint < f.max_folded_footprint)
    ok = (v.fold_error < f.keypoint_tolerance) & footprint_ok
    if f.unfold_on_table:
        ok = ok & (on_table(env, s) | ~unfold)
    return ok


def on_table(env, s: ClothState) -> torch.Tensor:
    f = env.cfg.fold
    spread, lift = geometry.flat_on_table(s.points_w, s.table_top_z)
    return (spread < f.on_table_max_spread) & (lift < f.on_table_max_lift)


def hand_sheet_distance(s: ClothState) -> torch.Tensor:
    """Nearest hand body origin to any sheet point [m]; a non-finite env reads as touching."""
    d = torch.cdist(s.hand_pos_w.float(), s.points_w.float()).amin(dim=(1, 2))
    return torch.nan_to_num(d, nan=0.0)


def update_release(env, s: ClothState, v: TaskView, folded: torch.Tensor, entered: torch.Tensor) -> torch.Tensor:
    """``fold.release``: advance the release / persist state; returns the success mask (the fold
    held ``persist_window_steps`` with the hand clear). The fold must be ENTERED first (its bonus
    is paid once), so a hand that never touched the sheet releases nothing."""
    c = env.cfg.fold.release
    env._bonus_mask.copy_(entered & ~env._entry_paid)
    env._entry_paid |= entered

    d = hand_sheet_distance(s)
    if c.touch == "contact":
        clear = ~env.hand_cloth_touching()
        env.extras["hand_cloth_touch"] = float((~clear).float().mean())
    else:
        clear = d > c.clearance

    # Release shaping: the running max of the capped clearance while folded and not yet released.
    track = folded & env._entry_paid & ~env._released
    d_cap = d.clamp(max=c.clearance)
    env._best_clear = torch.where(track & (env._best_clear < 0.0), d_cap, env._best_clear)
    gain = torch.where(track, (d_cap - env._best_clear).clamp(min=0.0), torch.zeros_like(d))
    env._best_clear = torch.where(track, torch.maximum(env._best_clear, d_cap), env._best_clear)
    env._release_rew_step.copy_(gain * c.release_dist_rew_scale)

    latch_ok = folded & clear & env._entry_paid
    if c.fold_tolerance > 0.0:
        latch_ok = latch_ok & (v.fold_error < c.fold_tolerance)
    env._rel_streak = torch.where(latch_ok, env._rel_streak + 1, torch.zeros_like(env._rel_streak))
    newly = ~env._released & (env._rel_streak >= c.latch_steps)
    env._release_latch_mask.copy_(newly)
    env._released |= newly

    ok = env._released & folded & clear
    env._persist_step_mask.copy_(ok)
    if c.persist_consecutive:
        env._persist_steps = torch.where(ok, env._persist_steps + 1, torch.zeros_like(env._persist_steps))
    else:
        env._persist_steps += ok.long()
    env._persist_done.copy_(env._released & (env._persist_steps >= c.persist_window_steps))
    env._persist_fail.copy_(env._released & ~folded)
    env.extras["hand_cloth_dist"] = float(d.mean())
    return env._persist_done.clone()


def compute(env, s: ClothState, v: TaskView) -> tuple[torch.Tensor, torch.Tensor]:
    """``(terminated, truncated)``; sets ``_bonus_mask`` for this step's reward."""
    cfg = env.cfg
    folded = is_folded(env, s, v)
    env._fold_hold = torch.where(folded, env._fold_hold + 1, torch.zeros_like(env._fold_hold))
    entered = env._fold_hold >= cfg.termination.success_steps
    held = env._fold_hold >= cfg.termination.held_steps

    release = cfg.fold.release.enabled
    if release:
        success = update_release(env, s, v, folded, entered)
    else:
        env._bonus_mask.copy_(entered)
        success = entered
    env._successes += success.long()

    fall = (v.pos_w[:, 2] - env.scene.env_origins[:, 2]) < cfg.termination.fall_height
    hand_far = v.fingertip_dist.amax(dim=-1) > cfg.termination.hand_far_distance
    nonfinite = env._nonfinite_envs.clone()
    env._nonfinite_envs.zero_()
    truncated = env.episode_length_buf >= env.max_episode_length

    reasons = {"fold": entered, "fold_held": held, "fall": fall, "hand_far": hand_far,
               "nonfinite": nonfinite, "timeout": truncated}
    if cfg.fold.task != "fold":
        u = env._unfold_env
        reasons["on_table"] = entered & on_table(env, s)
        reasons["task_unfold"] = u.clone()
        reasons["unfold_ok"] = entered & u
        reasons["fold_ok"] = entered & ~u
    end = entered
    if release:
        reasons["released"] = env._released.clone()
        reasons["persist"] = env._persist_done.clone()
        reasons["persist_fail"] = env._persist_fail.clone()
        end = env._persist_done | (env._persist_fail if cfg.fold.release.persist_fail_terminates else torch.zeros_like(fall))
    env._termination_reasons = reasons
    return fall | hand_far | nonfinite | end, truncated
