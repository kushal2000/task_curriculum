"""Metric publishing for joint reach, on Play's `extras` contract so EnvStatsAlgoObserver picks it up."""

from __future__ import annotations

__all__ = ["log_step_metrics"]


def log_step_metrics(env) -> None:
    segs = env._seg_done.clamp(min=1.0)
    steps = env.episode_length_buf.float().clamp(min=1.0)
    env.extras["episode_cumulative"] = env._reward_terms
    env.extras["episode_final"] = {
        "segments": env._seg_done.clone(),
        "segment_reached": env._seg_reached / segs,
        "segment_arm_err_deg": env._seg_arm_err / segs,
        "segment_hand_err_deg": env._seg_hand_err / segs,
        "segment_settle_s": env._seg_settle_sum / segs,
        "frac_steps_in_tol": env._steps_in_tol / steps,
        "frac_steps_near_table": env._near_table_steps / steps,
    }
