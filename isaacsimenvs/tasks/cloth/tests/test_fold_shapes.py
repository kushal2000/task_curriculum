"""The stage-2 fold-shape pool: every named fold is kinematically legal and lands where it says.

``fold_shapes`` is the definition the viewer and any future env read, so the failure that matters
is a fold that LOOKS fine in a simulator only because the solver papered over it -- a seam held
closed by a soft constraint the fold was fighting, or a flap that swung through the table on the
way. Both are asserted on the analytic kinematics, along the whole path, not just at the end.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import load_module  # noqa: E402  pytest puts tests/ on sys.path

L = 0.05

#: Loaded at import time too, because the fold list parametrises a test.
_POOL = load_module("isaacsimenvs/tasks/cloth/utils/fold_shapes.py", "fold_shapes").POOL


@pytest.fixture(scope="module")
def pool(fold_shapes):
    return fold_shapes.POOL


def _path(shape, stages, n=41):
    """Every configuration a stage-by-stage linear ramp passes through."""
    ang = {h.child: 0.0 for h in shape.hinges}
    for st in stages:
        start = dict(ang)
        for s in np.linspace(0.0, 1.0, n):
            cur = dict(start)
            cur.update({k: start[k] + s * (v - start[k]) for k, v in st.items()})
            yield cur
        ang.update(st)


def test_pool_names(pool):
    assert set(pool) == {
        "box2-mid", "box2diag4-mid", "box2x8-mid", "cross4-mid", "corners8-mid", "box4-mid", "star8-mid"
    }


def _footprint(sh, ang):
    tf = sh.transforms(ang)
    return np.array(
        [tf[p.name][0] @ np.array([*v, 0.0]) + tf[p.name][1] for p in sh.plates for v in p.polygon]
    )


def test_every_shape_half_folds(pool):
    """The membership rule: +x half over onto -x, landing on the left half of the sheet."""
    for sh in pool.values():
        half = next(f for f in sh.folds if f in ("half", "half: right over left"))
        xy = _footprint(sh, sh.fold_angles(half))
        assert xy[:, 0].max() < 1e-12 and xy[:, 0].min() > -L - 1e-12, sh.name
        assert abs(xy[:, 1]).max() < L + 1e-12, sh.name


def test_positive_angle_folds_up(pool):
    """+angle lifts every child off the table -- the convention every fold dict relies on."""
    for sh in pool.values():
        for h in sh.hinges:
            c = sh.plate(h.child).centroid
            assert sh.point(h.child, c, {h.child: 0.5})[2] > 1e-3, (sh.name, h.child)


@pytest.mark.parametrize("name,fold", [(n, f) for n, sh in _POOL.items() for f in sh.folds])
def test_fold_is_legal(pool, name, fold):
    """Seams stay closed and nothing goes below the table, along the whole path; ends flat."""
    sh = pool[name]
    for cur in _path(sh, sh.folds[fold]):
        assert sh.seam_error(cur) < 1e-9
        tf = sh.transforms(cur)
        for p in sh.plates:
            for v in p.polygon:
                r, t = tf[p.name]
                assert (r @ np.array([*v, 0.0]) + t)[2] > -1e-9
    tf = sh.transforms(sh.fold_angles(fold))
    for p in sh.plates:
        for v in p.polygon:
            r, t = tf[p.name]
            assert abs((r @ np.array([*v, 0.0]) + t)[2]) < 1e-9


def test_corners_meet_in_the_centre(pool):
    sh = pool["corners8-mid"]
    ang = sh.fold_angles("all four corners")
    for k, corner in {"ne": (L, L), "nw": (-L, L), "sw": (-L, -L), "se": (L, -L)}.items():
        np.testing.assert_allclose(sh.point(k, corner, ang), 0.0, atol=1e-12)


@pytest.mark.parametrize("name", ["cross4-mid", "corners8-mid", "star8-mid"])
def test_quarter_fold_lands_on_the_root_quarter(pool, name):
    """Bottom up, then right over: everything ends on the -x/+y quarter."""
    sh = pool[name]
    xy = _footprint(sh, sh.fold_angles("quarter (bottom up, then right over)"))
    assert xy[:, 0].max() < 1e-12 and xy[:, 0].min() > -L - 1e-12
    assert xy[:, 1].min() > -1e-12 and xy[:, 1].max() < L + 1e-12


def test_diagonal_fold_moves_both_layers_together(pool):
    """After the half fold, r_out lies on l_out and must stay on it through the diagonal fold."""
    sh = pool["box2diag4-mid"]
    (half, diag) = sh.folds["half, then diagonal (both layers)"]
    for f in np.linspace(0.0, 1.0, 21):
        ang = {**half, **{k: f * v for k, v in diag.items()}}
        for x, y in [(L, -L), (L, L), (0.5 * L, -0.5 * L)]:
            np.testing.assert_allclose(
                sh.point("r_out", (x, y), ang), sh.point("l_out", (-x, y), ang), atol=1e-12
            )


@pytest.mark.parametrize(
    "fold,pairs",
    [
        (
            "half, then diagonal A (both layers)",
            [("r_b", "l_b", (0.6 * L, -0.9 * L)), ("r_r", "l_l", (0.9 * L, 0.1 * L))],
        ),
        (
            "half, then diagonal B (both layers)",
            [("r_t", "l_t", (0.5 * L, 0.8 * L)), ("r_r", "l_l", (0.9 * L, 0.1 * L))],
        ),
    ],
)
def test_x_halves_fold_both_layers_together(pool, fold, pairs):
    sh = pool["box2x8-mid"]
    (half, diag) = sh.folds[fold]
    for f in np.linspace(0.0, 1.0, 21):
        ang = {**half, **{k: f * v for k, v in diag.items()}}
        for right, left, (x, y) in pairs:
            np.testing.assert_allclose(
                sh.point(right, (x, y), ang), sh.point(left, (-x, y), ang), atol=1e-12
            )


def test_box4_outer_strips_fold_both_layers_together(pool):
    """After the half fold, s4 lies on s1 and must stay on it through the outer-strip fold."""
    sh = pool["box4-mid"]
    (half, outer) = sh.folds["half, then outer strip (both layers)"]
    for f in np.linspace(0.0, 1.0, 21):
        ang = {**half, **{k: f * v for k, v in outer.items()}}
        for x, y in [(L, -L), (0.75 * L, 0.3 * L), (0.5 * L, L)]:
            np.testing.assert_allclose(
                sh.point("s4", (x, y), ang), sh.point("s1", (-x, y), ang), atol=1e-12
            )


@pytest.mark.parametrize(
    "fold,mirror",
    [
        ("diagonal: lower right over upper left", lambda x, y: (y, x)),
        ("diagonal: lower left over upper right", lambda x, y: (-y, -x)),
    ],
)
def test_star_diagonal_folds_land_on_the_other_half(pool, fold, mirror):
    """A diagonal fold is a mirror across that diagonal: every moved point lands on its image."""
    sh = pool["star8-mid"]
    ang = sh.fold_angles(fold)
    tf = sh.transforms(ang)
    flipped = set()
    for p in sh.plates:
        r, t = tf[p.name]
        if r[2, 2] < 0:
            flipped.add(p.name)
        for v in [*p.polygon, tuple(p.centroid)]:
            got = r @ np.array([*v, 0.0]) + t
            want = (*mirror(*v), 0.0) if r[2, 2] < 0 else (*v, 0.0)
            np.testing.assert_allclose(got, want, atol=1e-12)
    assert len(flipped) == 4


def test_half_folds_cannot_mix(pool):
    """The seam is what makes the cross a sheet: half of each fold at once tears it open."""
    sh = pool["corners8-mid"]
    half = math.pi / 2.0
    assert sh.seam_error({"c_ne": half}) < 1e-12
    assert sh.seam_error({"c_ne": half, "c_sw": half, "c_se": half}) > 0.01


def test_tree_without_seam_rejected(fold_shapes):
    with pytest.raises(ValueError, match="two parent hinges"):
        fold_shapes.FoldShape(
            name="bad",
            label="bad",
            plates=(
                fold_shapes.Plate("a", ((-L, -L), (0, -L), (0, L), (-L, L))),
                fold_shapes.Plate("b", ((0, -L), (L, -L), (L, L), (0, L))),
            ),
            hinges=(
                fold_shapes.Hinge("b", "a", (0, -L), (0, L)),
                fold_shapes.Hinge("b", "a", (0, -L), (0, 0)),
            ),
        )


def test_start_poses_are_never_the_target_and_stay_legal_at_any_angles(pool):
    """Every folded start (``init_fold_starts: other``), each part at its own random angle up to a
    full fold: seams closed, nothing below the table, and the target hinge never moves."""
    rng = np.random.default_rng(0)
    for sh in pool.values():
        target = set(sh.folds["half: right over left"][0])
        for name, parts in sh.start_poses().items():
            moved = {h for part in parts for h in part}
            assert not moved & target, (sh.name, name)
            for _ in range(20):
                ang = {}
                for part in parts:  # one angle per part
                    a = rng.uniform(0.0, math.pi)
                    ang.update({h: a for h in part})
                assert sh.seam_error(ang) < 1e-9, (sh.name, name)
                assert _footprint(sh, ang)[:, 2].min() > -1e-9, (sh.name, name)


def test_start_pose_counts(pool):
    """box2's only fold is the target; a corner folded on a quarter cannot ride a bottom-up fold
    (it would turn into the table), and fold-shape seams forbid mixing crossing lines."""
    counts = {n: len(sh.start_poses()) for n, sh in pool.items()}
    assert counts == {
        "box2-mid": 0, "box2diag4-mid": 3, "box2x8-mid": 8, "cross4-mid": 1,
        "corners8-mid": 19, "box4-mid": 3, "star8-mid": 3,
    }
    assert "half: bottom up + corner sw" not in pool["corners8-mid"].start_poses()
