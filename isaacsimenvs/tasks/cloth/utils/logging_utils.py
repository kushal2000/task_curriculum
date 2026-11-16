"""Step extras for the rl_games observer (``isaacsimenvs/utils/rlgames_utils.py``) and the eval.

* ``episode_cumulative``: the reward terms, summed per episode by the observer;
* ``episode_final``: ``successes`` and ``done_<reason>`` per env, read at episode end;
* ``successes``: the previous episode's count per env.
"""

from __future__ import annotations


def log_step(env) -> None:
    final = {"successes": env._successes.float()}
    final.update({f"done_{k}": v.float() for k, v in env._termination_reasons.items()})
    env.extras["episode_cumulative"] = env._reward_terms
    env.extras["episode_final"] = final
    env.extras["successes"] = env._prev_episode_successes.float()
    env.extras["nonfinite_total"] = float(env._nonfinite_count)
    env.extras["insane_total"] = float(env._insane_count)
