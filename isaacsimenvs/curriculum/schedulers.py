"""The three ways a difficulty range is allowed to move.

Each scheduler is a pure function of the curriculum's own scalar state — no torch, no
env — so `tests/test_curriculum.py` exercises them without booting Kit.

All of them return the new ``(lo, hi)``. Only ``hi`` is normally advanced: keeping ``lo``
at its initial value means the easy instances never disappear from the batch, which is
what stops the policy forgetting them. A task that wants a moving *window* instead of a
growing one sets ``init_range``/``final_range`` with a non-zero ``lo`` and gets the
interpolated behaviour from ``linear``.
"""

from __future__ import annotations

__all__ = ["step_fixed", "step_linear", "step_adaptive", "STEP_FNS"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def step_fixed(
    lo: float,
    hi: float,
    *,
    init_range: tuple[float, float],
    **_unused,
) -> tuple[float, float]:
    """Never move. The control arm and eval runs use this."""
    return float(init_range[0]), float(init_range[1])


def step_linear(
    lo: float,
    hi: float,
    *,
    init_range: tuple[float, float],
    final_range: tuple[float, float],
    frame: int,
    anneal_steps: int,
    **_unused,
) -> tuple[float, float]:
    """Interpolate init_range → final_range on wall-clock training progress."""
    t = _clamp(frame / max(1, anneal_steps), 0.0, 1.0)
    new_lo = init_range[0] + t * (final_range[0] - init_range[0])
    new_hi = init_range[1] + t * (final_range[1] - init_range[1])
    return float(new_lo), float(new_hi)


def step_adaptive(
    lo: float,
    hi: float,
    *,
    init_range: tuple[float, float],
    final_range: tuple[float, float],
    success_rate: float,
    num_episodes: int,
    threshold: float,
    min_episodes: int,
    step: float,
    allow_regress: bool,
    regress_ratio: float,
    **_unused,
) -> tuple[float, float]:
    """Advance only on demonstrated competence; optionally back off on collapse.

    Returns the range unchanged when too few episodes have completed to trust the
    success estimate — advancing on 3 episodes is how a curriculum runs away from a
    policy that never actually learned the easy case.
    """
    if num_episodes < min_episodes:
        return lo, hi

    if success_rate >= threshold:
        hi = hi + step
    elif allow_regress and success_rate < threshold * regress_ratio:
        hi = hi - step

    # Never shrink below where we started, never grow past the configured ceiling.
    hi = _clamp(hi, init_range[1], final_range[1])
    return lo, float(hi)


STEP_FNS = {
    "fixed": step_fixed,
    "linear": step_linear,
    "adaptive": step_adaptive,
}
