"""Scheduler behaviour — the part of the curriculum that decides when a task gets harder.

Kept to the pure schedulers: `core.py` needs a live `env`, so its wiring is checked in
the smoke run described in the README rather than here.
"""

from __future__ import annotations

import pytest

from conftest import REPO_ROOT  # noqa: E402  pytest puts tests/ on sys.path

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


# --- discrete difficulty levels -------------------------------------------------

class _FakeCfg:
    def __init__(self, **kw):
        self.enabled = True; self.eval_difficulty = -1.0; self.resample_on_reset = True
        self.difficulty_dim = 1; self.difficulty_levels = 0
        self.final_range = (0.0, 1.0); self.score_last_n_envs = 0
        self.__dict__.update(kw)


class _FakeEnv:
    """Minimal stand-in so sample_difficulty can be exercised without Kit."""
    def __init__(self, n, lo, hi, **cfgkw):
        torch = pytest.importorskip("torch")
        self.num_envs = n; self.device = "cpu"
        self._difficulty = torch.zeros(n, 1)
        self._curr_lo, self._curr_hi = lo, hi
        self.cfg = type("C", (), {"curriculum": _FakeCfg(**cfgkw)})()


def _levels_seen(env, core, n_max=8):
    torch = pytest.importorskip("torch")
    core.sample_difficulty(env, torch.arange(env.num_envs))
    d = env._difficulty[:, 0]
    return (1 + torch.round(d * (n_max - 1)).long())


@pytest.fixture(scope="session")
def core():
    import sys, types
    # core imports .schedulers relatively; give it a package to live in.
    from conftest import load_module
    pkg = types.ModuleType("curric"); pkg.__path__ = [str(REPO_ROOT / "isaacsimenvs/curriculum")]
    sys.modules["curric"] = pkg
    load_module("isaacsimenvs/curriculum/schedulers.py", "curric.schedulers")
    return load_module("isaacsimenvs/curriculum/core.py", "curric.core")


def test_discrete_levels_hit_every_integer_n(core):
    torch = pytest.importorskip("torch")
    env = _FakeEnv(20_000, 0.0, 1.0, difficulty_levels=8)
    n = _levels_seen(env, core)
    assert sorted(n.unique().tolist()) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_discrete_levels_are_uniform_including_endpoints(core):
    """Continuous sampling + rounding halves the weight of n=1 and n=X; the grid must not."""
    torch = pytest.importorskip("torch")
    env = _FakeEnv(80_000, 0.0, 1.0, difficulty_levels=8)
    n = _levels_seen(env, core)
    frac = [(n == k).float().mean().item() for k in range(1, 9)]
    for f in frac:
        assert abs(f - 1 / 8) < 0.01, f"non-uniform: {[round(x,4) for x in frac]}"


def test_discrete_levels_respect_the_frontier(core):
    """Only levels up to the current range_hi are drawn — this is the "1..X" mixture."""
    torch = pytest.importorskip("torch")
    for x in (1, 2, 4, 8):
        hi = (x - 1) / 7
        env = _FakeEnv(20_000, 0.0, hi, difficulty_levels=8)
        n = _levels_seen(env, core)
        assert sorted(n.unique().tolist()) == list(range(1, x + 1)), f"X={x}"


def test_continuous_sampling_underweights_the_endpoints(core):
    """Documents why difficulty_levels exists at all."""
    torch = pytest.importorskip("torch")
    env = _FakeEnv(80_000, 0.0, 1.0, difficulty_levels=0)
    n = _levels_seen(env, core)
    frac = [(n == k).float().mean().item() for k in range(1, 9)]
    assert frac[0] < 0.6 * frac[3], "endpoint should be about half-weight without the grid"
    assert frac[-1] < 0.6 * frac[3]
