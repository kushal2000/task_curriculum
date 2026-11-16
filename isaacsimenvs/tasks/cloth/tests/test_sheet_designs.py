"""The rigid-sheet design grid and the single-hinge sheet: what the env builds and the fold targets assume."""

from __future__ import annotations

import math

import numpy as np
import pytest

from isaacsimenvs.tasks.cloth.utils import sheet_designs as sd


def test_the_grid_is_4096_distinct_designs_spanning_the_ranges():
    g = sd.design_grid()
    assert g.shape == (4096, 3)
    assert len({tuple(r) for r in np.round(g, 9)}) == 4096
    for col, (lo, hi) in enumerate((sd.LENGTH_RANGE, sd.WIDTH_RANGE, sd.THICKNESS_RANGE)):
        levels = np.unique(np.round(g[:, col], 9))
        assert len(levels) == 16
        assert levels[0] == pytest.approx(lo) and levels[-1] == pytest.approx(hi)


@pytest.mark.parametrize("start", [0, 4096, 8192, 20480])
def test_every_sapg_block_of_4096_envs_holds_each_design_once(start):
    d = sd.design_of(np.arange(start, start + 4096), 4096)
    assert sorted(d.tolist()) == list(range(4096))


def test_a_small_run_spans_every_dimension():
    """256 envs (a smoke test) must not all be one length: the env -> design map is permuted."""
    g = sd.design_grid()[sd.design_of(np.arange(256), 4096)]
    for col in range(3):
        assert len(np.unique(np.round(g[:, col], 9))) == 16


def test_a_half_block_holds_half_the_designs_once():
    """2 GPUs x 12288 envs: SAPG blocks of 2048 see 2048 distinct designs, not a repeat."""
    d = sd.design_of(np.arange(2048, 4096), 4096)
    assert len(set(d.tolist())) == 2048


def test_single_hinge_plates_tile_the_sheet():
    s = sd.single_hinge(0.24, 0.12, 0.01)
    lo = [np.subtract(p.center, p.half_extents) for p in s.plates]
    hi = [np.add(p.center, p.half_extents) for p in s.plates]
    assert np.min(lo, axis=0) == pytest.approx([-0.12, -0.06, -0.005])
    assert np.max(hi, axis=0) == pytest.approx([0.12, 0.06, 0.005])
    assert hi[0][0] == pytest.approx(lo[1][0]) == pytest.approx(0.0)   # they meet at the crease
    assert s.ply_gap == pytest.approx(0.01)


@pytest.mark.parametrize("dims", [(0.10, 0.30, 0.002), (0.30, 0.10, 0.020), (0.17, 0.23, 0.0092)])
def test_a_180_deg_fold_lands_every_moving_point_on_its_target(dims):
    """The fold-target rule (reflect across the crease, lift by the ply gap) is exactly where the
    hinge puts the plate: a perfect fold scores 0, and the plate sits on top, not inside."""
    s = sd.single_hinge(*dims)
    length, width, t = dims
    x = np.linspace(1e-6, length / 2, 5)
    y = np.linspace(-width / 2, width / 2, 4)
    pts = np.array([(a, b, 0.0) for a in x for b in y])
    folded = sd.plate_points(s, pts, math.pi)
    target = pts * [-1.0, 1.0, 1.0] + [0.0, 0.0, s.ply_gap]
    assert np.abs(folded - target).max() < 1e-9
    # Folding flips the plate: its TOP face comes down onto the stationary plate's top face
    # (z = t/2), touching it without overlap, and its bottom face ends up on top at 3t/2.
    top = sd.plate_points(s, pts + [0, 0, t / 2], math.pi)
    bottom = sd.plate_points(s, pts + [0, 0, -t / 2], math.pi)
    assert top[:, 2] == pytest.approx(np.full(len(pts), t / 2))
    assert bottom[:, 2] == pytest.approx(np.full(len(pts), 1.5 * t))


def test_positive_angles_lift_the_moving_plate_and_leave_the_other():
    s = sd.single_hinge(0.2, 0.2, 0.01)
    pts = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    out = sd.plate_points(s, pts, math.radians(90))
    assert out[0, 2] > 0.09
    assert np.allclose(out[1], pts[1])


def test_a_mid_plane_hinge_would_fold_the_plate_into_the_other():
    """Negative control for the hinge height: about the mid-plane the folded points land at z = 0,
    inside the stationary plate, and miss the lifted targets by a whole thickness."""
    s = sd.single_hinge(0.2, 0.2, 0.01)
    mid = sd.SheetSpec(s.length, s.width, s.thickness, s.plates,
                       (sd.Hinge("minus", "plus", (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),), s.ply_gap)
    pts = np.array([[0.05, 0.02, 0.0]])
    folded = sd.plate_points(mid, pts, math.pi)
    target = pts * [-1.0, 1.0, 1.0] + [0.0, 0.0, s.ply_gap]
    assert np.abs(folded - target).max() == pytest.approx(0.01)


def test_mass_is_per_area_and_inertia_is_the_box_formula():
    s = sd.single_hinge(0.2, 0.1, 0.01)
    m, com, i = sd.box_mass_properties(s.plates[1], density=2.0)
    assert m == pytest.approx(2.0 * 0.1 * 0.1)
    assert com == pytest.approx([0.05, 0.0, 0.0])
    assert i == pytest.approx(m / 12 * np.array([0.1**2 + 0.01**2, 0.1**2 + 0.01**2, 0.1**2 + 0.1**2]))
