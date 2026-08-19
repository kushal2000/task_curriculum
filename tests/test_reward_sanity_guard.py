"""The reward guard must catch MAGNITUDE, not only non-finiteness.

Both training runs (197715, 199350) were compromised by rewards of -1e5 to -1e11 that were
perfectly finite: a diverging articulation grows huge joint velocities and actions before anything
becomes NaN, the action penalties are sums of squares, and `torch.nan_to_num` leaves the result
untouched. `nonfinite_reward` read 0.0000 throughout.
"""

from __future__ import annotations

import torch
import yaml

from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv

LIMIT = ClothEnv.REWARD_SANITY_LIMIT


def _apply_guard(reward: torch.Tensor):
    """The guard exactly as `_get_rewards` applies it."""
    insane = reward.abs() > LIMIT
    guarded = torch.where(insane, torch.zeros_like(reward), reward)
    return guarded, insane


def test_nan_to_num_does_not_catch_huge_finite_rewards():
    """The negative control: the OLD guard, on the values actually observed.

    Without this, the test below could pass for the wrong reason -- one has to show the previous
    guard genuinely let these through.
    """
    observed = torch.tensor([-784_354.0, -74_109.88, -36_772_568.0, -110_317_699_072.0])
    survived = torch.nan_to_num(observed, nan=0.0, posinf=0.0, neginf=0.0)
    assert torch.allclose(survived, observed), "nan_to_num should leave finite values untouched"
    assert bool(torch.isfinite(observed).all()), "the observed blow-ups were all FINITE"


def test_guard_zeroes_the_observed_blow_ups():
    observed = torch.tensor([-784_354.0, -74_109.88, -36_772_568.0, -110_317_699_072.0])
    guarded, insane = _apply_guard(observed)
    assert bool(insane.all()), "every observed blow-up must be flagged"
    assert torch.count_nonzero(guarded) == 0


def test_guard_does_not_touch_legitimate_rewards():
    """A fold pays reach_goal_bonus in a single step. That must survive untouched."""
    cfg = yaml.safe_load(open("isaacsimenvs/cfg/task/Cloth.yaml"))
    bonus = float(cfg["reward"]["reach_goal_bonus"])
    legit = torch.tensor([0.0, -50.0, 12.5, bonus, bonus + 250.0, -bonus])
    guarded, insane = _apply_guard(legit)
    assert not bool(insane.any()), f"legitimate rewards flagged: {legit[insane].tolist()}"
    assert torch.allclose(guarded, legit)


def test_limit_leaves_headroom_over_anything_the_task_can_pay():
    cfg = yaml.safe_load(open("isaacsimenvs/cfg/task/Cloth.yaml"))
    bonus = float(cfg["reward"]["reach_goal_bonus"])
    # Bonus plus generous shaping; the limit should sit well above it but far below the blow-ups.
    plausible_max = bonus + 1000.0
    assert LIMIT > plausible_max, "limit must not clip a legitimate fold"
    assert LIMIT >= 5 * plausible_max, "want real headroom, not a hair's breadth"
    assert LIMIT < 74_109.88, "limit must be BELOW the smallest observed blow-up"


def test_guard_is_elementwise_and_spares_healthy_envs():
    """One diverging env must not zero the rest of the batch."""
    reward = torch.tensor([5.0, -1e9, 12.0, 1000.0, -3e7])
    guarded, insane = _apply_guard(reward)
    assert insane.tolist() == [False, True, False, False, True]
    assert torch.allclose(guarded, torch.tensor([5.0, 0.0, 12.0, 1000.0, 0.0]))


def test_nonfinite_still_handled_too():
    """The magnitude guard must not have displaced the finiteness guard."""
    reward = torch.tensor([float("nan"), float("inf"), float("-inf"), 7.0])
    nonfinite = ~torch.isfinite(reward)
    assert nonfinite.tolist() == [True, True, True, False]
    cleaned = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    guarded, insane = _apply_guard(cleaned)
    assert bool(torch.isfinite(guarded).all())
    assert not bool(insane.any()), "nan_to_num output is already in range"
