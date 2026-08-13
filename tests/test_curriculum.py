"""Scheduler behaviour — the part of the curriculum that decides when a task gets harder.

Kept to the pure schedulers: `core.py` needs a live `env`, so its wiring is checked in
the smoke run described in the README rather than here.
"""

from __future__ import annotations

import pytest

INIT = (0.0, 0.2)
FINAL = (0.0, 1.0)

ADAPT = dict(
    init_range=INIT, final_range=FINAL, threshold=0.7, min_episodes=64,
    step=0.05, allow_regress=True, regress_ratio=0.4,
)


def test_fixed_never_moves(schedulers):
    assert schedulers.step_fixed(0.0, 0.9, init_range=INIT) == INIT


def test_linear_walks_init_to_final(schedulers):
    assert schedulers.step_linear(
        0.0, 0.0, init_range=INIT, final_range=FINAL, frame=0, anneal_steps=1000
    ) == pytest.approx(INIT)
    assert schedulers.step_linear(
        0.0, 0.0, init_range=INIT, final_range=FINAL, frame=1000, anneal_steps=1000
    ) == pytest.approx(FINAL)
    mid = schedulers.step_linear(
        0.0, 0.0, init_range=INIT, final_range=FINAL, frame=500, anneal_steps=1000
    )
    assert mid[1] == pytest.approx(0.6)


def test_linear_clamps_past_the_anneal_horizon(schedulers):
    assert schedulers.step_linear(
        0.0, 0.0, init_range=INIT, final_range=FINAL, frame=10**9, anneal_steps=1000
    ) == pytest.approx(FINAL)


def test_adaptive_advances_on_success(schedulers):
    _, hi = schedulers.step_adaptive(0.0, 0.2, success_rate=0.9, num_episodes=100, **ADAPT)
    assert hi == pytest.approx(0.25)


def test_adaptive_regresses_on_collapse(schedulers):
    _, hi = schedulers.step_adaptive(0.0, 0.5, success_rate=0.1, num_episodes=100, **ADAPT)
    assert hi == pytest.approx(0.45)


def test_adaptive_holds_in_the_middle_band(schedulers):
    """Between the regress ratio and the threshold the policy is learning — leave it be."""
    _, hi = schedulers.step_adaptive(0.0, 0.5, success_rate=0.5, num_episodes=100, **ADAPT)
    assert hi == pytest.approx(0.5)


def test_adaptive_ignores_a_thin_sample(schedulers):
    """Advancing on a handful of episodes is how a curriculum runs away from a policy
    that never actually solved the easy case."""
    _, hi = schedulers.step_adaptive(0.0, 0.2, success_rate=1.0, num_episodes=3, **ADAPT)
    assert hi == pytest.approx(0.2)


def test_adaptive_never_leaves_the_configured_bounds(schedulers):
    _, hi = 0.0, 0.99
    for _ in range(50):
        _, hi = schedulers.step_adaptive(0.0, hi, success_rate=1.0, num_episodes=100, **ADAPT)
    assert hi == pytest.approx(FINAL[1])

    hi = 0.21
    for _ in range(50):
        _, hi = schedulers.step_adaptive(0.0, hi, success_rate=0.0, num_episodes=100, **ADAPT)
    assert hi == pytest.approx(INIT[1]), "must not regress below where the curriculum began"


def test_adaptive_can_be_told_not_to_regress(schedulers):
    opts = {**ADAPT, "allow_regress": False}
    _, hi = schedulers.step_adaptive(0.0, 0.5, success_rate=0.0, num_episodes=100, **opts)
    assert hi == pytest.approx(0.5)


def test_every_mode_is_reachable_from_the_registry(schedulers):
    assert set(schedulers.STEP_FNS) == {"fixed", "linear", "adaptive"}


def test_adaptive_staircase_advances_the_whole_band(schedulers):
    """`advance_lo_with_hi` steps lo and hi together — the literal "advance n" mode."""
    opts = {**ADAPT, "init_range": (0.0, 0.0), "advance_lo_with_hi": True}
    lo, hi = 0.0, 0.0
    for _ in range(3):
        lo, hi = schedulers.step_adaptive(lo, hi, success_rate=1.0, num_episodes=100, **opts)
    assert lo == pytest.approx(hi), "staircase must keep the band collapsed"
    assert hi == pytest.approx(0.15)


def test_adaptive_default_widens_instead_of_stepping(schedulers):
    """Default keeps lo pinned, so easy instances stay in the batch."""
    lo, hi = 0.0, 0.2
    for _ in range(3):
        lo, hi = schedulers.step_adaptive(lo, hi, success_rate=1.0, num_episodes=100, **ADAPT)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.35)


def test_one_link_per_advance_with_matched_step(schedulers):
    """adapt_step = 1/(n_max-1) makes each advance worth exactly one link."""
    n_max = 8
    opts = {**ADAPT, "init_range": (0.0, 0.0), "final_range": (0.0, 1.0),
            "step": 1.0 / (n_max - 1), "advance_lo_with_hi": True}
    lo, hi = 0.0, 0.0
    seen = [1 + round(hi * (n_max - 1))]
    for _ in range(9):  # more advances than rungs, to check the ceiling holds
        lo, hi = schedulers.step_adaptive(lo, hi, success_rate=1.0, num_episodes=100, **opts)
        seen.append(1 + round(hi * (n_max - 1)))
    assert seen[:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert seen[-1] == 8, "must saturate at n_max, not overshoot"
