"""Curriculum configuration — shared by every task in this repo.

One `@configclass` section, mounted as `curriculum:` on each task's env cfg, so it
overlays from `cfg/task/<Task>.yaml` and from the Hydra CLI like every other section
(`env.curriculum.mode=linear`, `env.curriculum.init_range=[0.0,0.4]`, ...).

The difficulty abstraction is deliberately thin: every task carries a per-env vector
`env._difficulty` in `[0, 1]^difficulty_dim`, where **0 is the easiest task instance
and 1 the hardest**. The curriculum only ever moves the *sampling range* that vector is
drawn from; translating a difficulty value into concrete physics (link counts, bottle
fill, tolerances) is each task's own job, in its `reset_utils.apply_difficulty`.

That split is what lets one controller drive two unrelated environments.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    """Per-env difficulty sampling + reward-shaping schedule."""

    enabled: bool = True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    mode: str = "adaptive"
    """How the difficulty range advances.

    - ``"fixed"``    — range stays at ``init_range``. Use for the from-scratch control
                       arm (set ``init_range = final_range = [1.0, 1.0]``) and for eval.
    - ``"linear"``   — range interpolates ``init_range → final_range`` over
                       ``anneal_steps`` policy steps, regardless of performance.
    - ``"adaptive"`` — the range's upper bound advances only when the recent success
                       rate clears ``adapt_success_threshold``. Self-paced.
    """

    difficulty_dim: int = 1
    """Length of the per-env difficulty vector. Tasks that only need one knob leave
    this at 1 and read ``env._difficulty[:, 0]``."""

    init_range: tuple[float, float] = (0.0, 0.2)
    """Difficulty range at the start of training, as (lo, hi) in [0, 1]."""

    final_range: tuple[float, float] = (0.0, 1.0)
    """Difficulty range the curriculum is allowed to grow to."""

    resample_on_reset: bool = True
    """Redraw a terminating env's difficulty at every reset. When False, each env keeps
    the difficulty it was first assigned — useful for ablations that want a static mix."""

    # ------------------------------------------------------------------
    # mode="linear"
    # ------------------------------------------------------------------
    anneal_steps: int = 200_000
    """Policy steps over which `linear` mode walks init_range → final_range."""

    # ------------------------------------------------------------------
    # mode="adaptive"
    # ------------------------------------------------------------------
    adapt_interval: int = 3000
    """Policy steps between adaptation checks. Mirrors play2perfect's
    `tolerance_curriculum_interval`."""

    adapt_success_threshold: float = 0.7
    """Mean episode success over the window required to make the task harder."""

    adapt_min_episodes: int = 64
    """Don't adapt on a handful of episodes — the success estimate would be noise."""

    adapt_step: float = 0.05
    """How far `range_hi` moves per successful adaptation, in difficulty units."""

    adapt_allow_regress: bool = True
    """Also *lower* `range_hi` when success collapses below
    ``adapt_success_threshold * adapt_regress_ratio``. Guards against a curriculum that
    ratchets past what the policy can hold."""

    adapt_regress_ratio: float = 0.4

    # ------------------------------------------------------------------
    # Reward shaping schedule
    # ------------------------------------------------------------------
    reward_schedule: dict[str, tuple[float, float]] = {}
    """Maps a reward-term name to ``(scale_at_easiest, scale_at_hardest)``. Read via
    :func:`isaacsimenvs.curriculum.core.reward_scale`; interpolated on the same
    normalised progress that drives the difficulty range. The usual use is annealing a
    dense shaping term to zero as the task gets hard, e.g.
    ``{"reach": [1.0, 0.2], "success_bonus": [1.0, 3.0]}``."""

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    eval_difficulty: float | None = None
    """When set, every env is pinned to this difficulty and no adaptation happens —
    the counterpart of play2perfect's `eval_success_tolerance`. Set it for eval runs so
    both experiment arms are scored on identical task instances."""


__all__ = ["CurriculumCfg"]
