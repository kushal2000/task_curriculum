"""Per-env difficulty buffers, the adaptation step, and telemetry.

Wiring, per task (both envs in this repo do exactly this):

    __init__      →  allocate_curriculum_buffers(self)
    _reset_idx    →  record_episode_success(self, env_ids, success)   # before resetting
                     sample_difficulty(self, env_ids)
                     <task>.apply_difficulty(self, env_ids)
    _get_dones    →  update_curriculum(self)
    _get_rewards  →  reward_scale(self, "<term>")   ... then log_curriculum(self)

`update_curriculum` sits at the same hook play2perfect used for its tolerance
curriculum (`tasks/play/utils/termination_utils.py:update_tolerance_curriculum`), and
the optional `_curriculum_*` subclass hooks from that file are honoured here too so
task-specific eligibility rules keep working.
"""

from __future__ import annotations

import torch

from .schedulers import STEP_FNS

__all__ = [
    "allocate_curriculum_buffers",
    "sample_difficulty",
    "record_episode_success",
    "update_curriculum",
    "curriculum_progress",
    "reward_scale",
    "log_curriculum",
]


def allocate_curriculum_buffers(env) -> None:
    """Create every mutable buffer the curriculum owns. Call once from ``__init__``."""
    cfg = env.cfg.curriculum

    env._difficulty = torch.zeros(
        env.num_envs, cfg.difficulty_dim, device=env.device, dtype=torch.float32
    )
    env._curr_lo = float(cfg.init_range[0])
    env._curr_hi = float(cfg.init_range[1])
    # Policy steps since training began; drives `linear` and gates `adaptive`.
    env._curr_frame = 0
    env._curr_last_update = 0
    # Success accumulator over the window since the last adaptation.
    env._curr_succ_sum = 0.0
    env._curr_succ_count = 0
    # Last computed window success rate, kept only so it can be logged.
    env._curr_succ_rate = 0.0

    # Seed every env's difficulty so the first episode is already on-distribution.
    sample_difficulty(env, torch.arange(env.num_envs, device=env.device))


def sample_difficulty(env, env_ids: torch.Tensor) -> None:
    """Draw fresh difficulty vectors for ``env_ids`` from the current range."""
    cfg = env.cfg.curriculum
    if env_ids.numel() == 0:
        return

    if cfg.eval_difficulty >= 0.0:
        env._difficulty[env_ids] = float(cfg.eval_difficulty)
        return

    if not cfg.enabled or not cfg.resample_on_reset:
        # `enabled=False` pins everything to the hardest instance — i.e. the
        # from-scratch control arm, which must see the target task from step 0.
        if not cfg.enabled:
            env._difficulty[env_ids] = float(cfg.final_range[1])
        return

    levels = int(cfg.difficulty_levels)
    if levels > 1:
        # Discrete difficulty: draw uniformly from the grid points currently unlocked,
        # so every level is equally likely. See `CurriculumCfg.difficulty_levels` for
        # why continuous sampling under-weights the two endpoints.
        grid = torch.linspace(0.0, 1.0, levels, device=env.device)
        eps = 0.5 / (levels - 1)  # half a grid spacing of tolerance
        unlocked = grid[(grid >= env._curr_lo - eps) & (grid <= env._curr_hi + eps)]
        if unlocked.numel() == 0:  # degenerate range — fall back to the nearest point
            unlocked = grid[(grid - env._curr_hi).abs().argmin()].reshape(1)
        idx = torch.randint(
            unlocked.numel(), (env_ids.numel(), cfg.difficulty_dim), device=env.device
        )
        env._difficulty[env_ids] = unlocked[idx]
        return

    env._difficulty[env_ids] = (
        torch.rand(env_ids.numel(), cfg.difficulty_dim, device=env.device)
        * (env._curr_hi - env._curr_lo)
        + env._curr_lo
    )


def record_episode_success(env, env_ids: torch.Tensor, success: torch.Tensor) -> None:
    """Accumulate completed-episode outcomes for the adaptive scheduler.

    ``success`` is a per-env value in [0, 1] for the episode that just ended. Call from
    ``_reset_idx`` *before* the env state is overwritten.
    """
    if env_ids.numel() == 0 or env._curr_frame == 0:
        # frame == 0 is the startup reset of every env — no episode actually ran, and
        # counting those zeros would trigger a spurious regress on the first check.
        return

    success = success.float().reshape(-1)

    # SAPG: score only the leader block, matching what rl_games itself reports. See
    # `CurriculumCfg.score_last_n_envs`.
    keep_n = int(env.cfg.curriculum.score_last_n_envs)
    if keep_n > 0:
        first_scored = env.num_envs - int(keep_n)
        leader = env_ids >= first_scored
        env_ids = env_ids[leader]
        success = success[leader]
        if env_ids.numel() == 0:
            return

    # Honour play2perfect's opt-in narrowing hooks: a task may declare that only some
    # envs should steer the curriculum (e.g. only the insertion-goal envs).
    if hasattr(env, "_curriculum_eligible_mask"):
        mask = env._curriculum_eligible_mask()
        if mask is not None:
            keep = mask[env_ids]
            success = success[keep]

    env._curr_succ_sum += float(success.sum().item())
    env._curr_succ_count += int(success.numel())


def update_curriculum(env) -> None:
    """Advance the difficulty range. Call once per policy step from ``_get_dones``."""
    cfg = env.cfg.curriculum
    env._curr_frame += 1

    if not cfg.enabled or cfg.eval_difficulty >= 0.0 or cfg.mode == "fixed":
        return

    if env._curr_frame - env._curr_last_update < cfg.adapt_interval:
        return

    if env._curr_succ_count > 0:
        env._curr_succ_rate = env._curr_succ_sum / env._curr_succ_count

    threshold = cfg.adapt_success_threshold
    if hasattr(env, "_curriculum_success_threshold"):
        custom = env._curriculum_success_threshold()
        if custom is not None:
            threshold = float(custom)

    step_fn = STEP_FNS[cfg.mode]
    env._curr_lo, env._curr_hi = step_fn(
        env._curr_lo,
        env._curr_hi,
        init_range=tuple(cfg.init_range),
        final_range=tuple(cfg.final_range),
        frame=env._curr_frame,
        anneal_steps=cfg.anneal_steps,
        success_rate=env._curr_succ_rate,
        num_episodes=env._curr_succ_count,
        threshold=threshold,
        min_episodes=cfg.adapt_min_episodes,
        step=cfg.adapt_step,
        allow_regress=cfg.adapt_allow_regress,
        regress_ratio=cfg.adapt_regress_ratio,
        advance_lo_with_hi=cfg.advance_lo_with_hi,
    )

    env._curr_last_update = env._curr_frame
    env._curr_succ_sum = 0.0
    env._curr_succ_count = 0


def curriculum_progress(env) -> float:
    """How far the range's ceiling has travelled from `init_range` to `final_range`, in [0, 1].

    This is the single scalar that reward schedules interpolate on, so shaping and
    difficulty always move together no matter which scheduler is driving.
    """
    cfg = env.cfg.curriculum
    lo, hi = float(cfg.init_range[1]), float(cfg.final_range[1])
    if hi <= lo:
        return 1.0
    return max(0.0, min(1.0, (env._curr_hi - lo) / (hi - lo)))


def reward_scale(env, term: str, default: float = 1.0) -> float:
    """Current multiplier for a scheduled reward term.

    Terms absent from ``curriculum.reward_schedule`` return ``default``, so a task can
    call this on every term unconditionally and only schedule the ones it cares about.
    """
    schedule = env.cfg.curriculum.reward_schedule
    if not schedule or term not in schedule:
        return default
    start, end = schedule[term]
    return float(start) + curriculum_progress(env) * (float(end) - float(start))


def log_curriculum(env) -> None:
    """Publish curriculum scalars into ``extras``.

    `EnvStatsAlgoObserver` (utils/rlgames_utils.py) flattens `extras` and forwards every
    scalar to the summary writer, so these need no observer changes to reach
    tensorboard/wandb.
    """
    env.extras["curriculum/range_lo"] = env._curr_lo
    env.extras["curriculum/range_hi"] = env._curr_hi
    env.extras["curriculum/mean_difficulty"] = float(env._difficulty.mean().item())
    env.extras["curriculum/max_difficulty"] = float(env._difficulty.max().item())
    env.extras["curriculum/window_success_rate"] = env._curr_succ_rate
    env.extras["curriculum/progress"] = curriculum_progress(env)
