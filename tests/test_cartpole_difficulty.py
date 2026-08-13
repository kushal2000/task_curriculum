"""Invariants the cartpole curriculum depends on.

The central claim of this env is that difficulty varies *only* the number of unactuated
degrees of freedom — total length, mass and inertia are held constant. If that stops
being true, an apparent curriculum effect could just be the task getting shorter, so
these are the assertions worth having.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


N_MAX = 4
LINKS = [0.25, 0.25, 0.25, 0.25]


@pytest.fixture
def lengths():
    return torch.tensor(LINKS)


def test_difficulty_maps_monotonically_to_free_joint_count(difficulty_math):
    d = torch.linspace(0.0, 1.0, 21)
    n = difficulty_math.active_joint_count(d, N_MAX)
    assert n.min().item() == 1 and n.max().item() == N_MAX
    assert torch.all(n[1:] >= n[:-1]), "harder must never mean fewer free joints"


def test_joint_zero_is_always_free(difficulty_math):
    """Locking the base joint would weld the pole to the cart and delete the task."""
    n_active = difficulty_math.active_joint_count(torch.rand(512), N_MAX)
    free = difficulty_math.sample_free_mask(n_active, N_MAX)
    assert free[:, 0].all()
    assert torch.equal(free.sum(dim=1), n_active)


def test_total_chain_length_is_invariant_across_difficulty(difficulty_math, lengths):
    """The load-bearing invariant: only the DOF count changes with difficulty."""
    n_active = difficulty_math.active_joint_count(torch.rand(512), N_MAX)
    free = difficulty_math.sample_free_mask(n_active, N_MAX)
    seg_lengths, _ = difficulty_math.segment_geometry(free, lengths)
    assert torch.allclose(seg_lengths.sum(dim=1), lengths.sum().expand(512), atol=1e-6)


def test_inactive_slots_are_zero_padded(difficulty_math, lengths):
    n_active = difficulty_math.active_joint_count(torch.rand(256), N_MAX)
    free = difficulty_math.sample_free_mask(n_active, N_MAX)
    seg_lengths, _ = difficulty_math.segment_geometry(free, lengths)
    active = difficulty_math.active_mask(n_active, N_MAX)
    assert (seg_lengths[~active] == 0).all()
    assert (seg_lengths[active] > 0).all()


def test_segment_indices_are_the_free_joints_in_chain_order(difficulty_math, lengths):
    n_active = difficulty_math.active_joint_count(torch.rand(64), N_MAX)
    free = difficulty_math.sample_free_mask(n_active, N_MAX)
    _, seg_idx = difficulty_math.segment_geometry(free, lengths)
    for i in range(free.shape[0]):
        expected = free[i].nonzero().flatten().tolist()
        assert seg_idx[i, : n_active[i]].tolist() == expected


def test_segment_lengths_vary_at_a_fixed_free_joint_count(difficulty_math, lengths):
    """Randomising *which* joints are free is what makes effective link lengths vary —
    the second half of "n and L vary per-env"."""
    n_active = torch.full((512,), 2, dtype=torch.long)
    free = difficulty_math.sample_free_mask(n_active, N_MAX)
    seg_lengths, _ = difficulty_math.segment_geometry(free, lengths)
    distinct = {tuple(round(v, 4) for v in row[:2].tolist()) for row in seg_lengths}
    assert len(distinct) > 1, f"expected several length partitions, saw {distinct}"


def test_upright_chain_has_normalised_height_one(difficulty_math, lengths):
    n_active = difficulty_math.active_joint_count(torch.rand(128), N_MAX)
    free = difficulty_math.sample_free_mask(n_active, N_MAX)
    seg_lengths, seg_idx = difficulty_math.segment_geometry(free, lengths)
    active = difficulty_math.active_mask(n_active, N_MAX)

    zeros = torch.zeros(128, N_MAX)
    _, _, height = difficulty_math.segment_kinematics(
        zeros, zeros, seg_idx, seg_lengths, active, float(lengths.sum())
    )
    assert torch.allclose(height, torch.ones(128), atol=1e-6)


def test_folding_the_base_joint_flat_drops_height_to_zero(difficulty_math, lengths):
    """Rotating the base joint 90 degrees swings the whole chain horizontal, whatever
    the difficulty — the height measure has to agree at every link count."""
    n_active = difficulty_math.active_joint_count(torch.rand(128), N_MAX)
    free = difficulty_math.sample_free_mask(n_active, N_MAX)
    seg_lengths, seg_idx = difficulty_math.segment_geometry(free, lengths)
    active = difficulty_math.active_mask(n_active, N_MAX)

    pos = torch.zeros(128, N_MAX)
    pos[:, 0] = math.pi / 2
    _, _, height = difficulty_math.segment_kinematics(
        pos, torch.zeros_like(pos), seg_idx, seg_lengths, active, float(lengths.sum())
    )
    assert torch.allclose(height, torch.zeros(128), atol=1e-6)


def test_inactive_segment_angles_are_masked_out(difficulty_math, lengths):
    n_active = torch.full((64,), 1, dtype=torch.long)
    free = difficulty_math.sample_free_mask(n_active, N_MAX)
    seg_lengths, seg_idx = difficulty_math.segment_geometry(free, lengths)
    active = difficulty_math.active_mask(n_active, N_MAX)

    pos = torch.randn(64, N_MAX)
    angles, ang_vels, _ = difficulty_math.segment_kinematics(
        pos, pos, seg_idx, seg_lengths, active, float(lengths.sum())
    )
    assert (angles[~active] == 0).all()
    assert (ang_vels[~active] == 0).all()
