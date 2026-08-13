"""Metric publishing for bottle flipping.

Same `extras` contract play uses, so `EnvStatsAlgoObserver` picks everything up without
modification. `success` is the key the curriculum's adaptive scheduler is scored on, so
it is reported both per-episode and as the running vector.
"""

from __future__ import annotations

from isaacsimenvs.curriculum import log_curriculum

__all__ = ["log_step_metrics"]


def log_step_metrics(env) -> None:
    env.extras["episode_cumulative"] = env._reward_terms
    env.extras["episode_final"] = {
        "success": env._bf_prev_episode_success,
        "flip_turns": env._bf_flip_turns,
        "upright_cos": env._bf_upright_cos,
        "fill_fraction": env._bf_fill_fraction,
        "required_turns": env._bf_required_turns,
        "released": env._bf_released.float(),
        **{f"done_{name}": v.float() for name, v in env._bf_termination_reasons.items()},
    }
    env.extras["successes"] = env._bf_prev_episode_success
    env.extras["mean_fill_fraction"] = float(env._bf_fill_fraction.mean().item())
    env.extras["mean_required_turns"] = float(env._bf_required_turns.mean().item())
    log_curriculum(env)
