"""Metric publishing for the multi-link cartpole.

Keys follow the contract `EnvStatsAlgoObserver` (`isaacsimenvs/utils/rlgames_utils.py`)
already understands: `episode_cumulative` sums a term over an episode, `episode_final`
snapshots a value at termination, and bare scalars go straight to the summary writer.
`log_curriculum` adds the `curriculum/*` scalars on top.
"""

from __future__ import annotations

from isaacsimenvs.curriculum import log_curriculum

__all__ = ["log_step_metrics"]


def log_step_metrics(env) -> None:
    env.extras["episode_cumulative"] = env._reward_terms
    env.extras["episode_final"] = {
        "success": env._prev_episode_success,
        "height_norm": env._height_norm,
        "n_active": env._n_active.float(),
        **{f"done_{name}": value.float() for name, value in env._termination_reasons.items()},
    }
    env.extras["successes"] = env._prev_episode_success
    env.extras["n_active_mean"] = float(env._n_active.float().mean().item())
    log_curriculum(env)
