"""Reachable folded states, one-fold moves and look-alike classes of the pool shapes
(mdp/geometry/fold_states.py): numpy, CPU."""

from __future__ import annotations

import math

import pytest

from isaacsimenvs.tasks.cloth.mdp.geometry import fold_shapes as fs
from isaacsimenvs.tasks.cloth.mdp.geometry import fold_states as ft

SHAPES = ["box2-mid", "box2diag4-mid", "box2x8-mid", "cross4-mid", "corners8-mid", "box4-mid", "star8-mid"]


@pytest.fixture(scope="module")
def tasks():
    return {n: ft.enumerate_tasks(fs.shape(n)) for n in SHAPES}


def test_task_counts(tasks):
    # Flat included; every task one fold from a reachable state (res 7: 209 over the pool).
    assert {n: len(st.states) for n, st in tasks.items()} == {
        "box2-mid": 2, "box2diag4-mid": 6, "box2x8-mid": 12, "cross4-mid": 4,
        "corners8-mid": 64, "box4-mid": 8, "star8-mid": 11,
    }
    assert sum(len(ft.task_arrays(st, 7)) for st in tasks.values()) == 209


def test_flat_and_half_fold_everywhere(tasks):
    for name, st in tasks.items():
        assert not st.states[0].folded
        half = ft.task_arrays(st, 7, only_half=True)
        assert len(half) == 1, name
        assert abs(abs(half[0]["direction"]) - math.pi) < 1e-9      # +x half travels to -x
        assert half[0]["members"][0][0] == half[0]["goal"]


def test_a_move_is_one_rigid_fold(tasks):
    st = tasks["box2diag4-mid"]
    names = [h.child for h in st.shape.hinges]
    from_flat = [set(c for c, _ in m.hinges) for m in st.moves if m.start == 0]
    assert {"l_out", "r_out"} not in from_flat                      # two creases: two folds
    assert {"r_in"} in from_flat and {"l_out"} in from_flat
    # after the half fold the two diagonals lie on each other: both layers fold as one piece
    half = next(m.goal for m in st.moves if m.start == 0 and [c for c, _ in m.hinges] == ["r_in"])
    assert any(m.start == half and set(c for c, _ in m.hinges) == {"l_out", "r_out"} for m in st.moves)
    assert len(names) == 3


def test_a_covered_piece_cannot_fold(tasks):
    st = tasks["box4-mid"]
    half = next(m.goal for m in st.moves if m.start == 0 and [c for c, _ in m.hinges] == ["s3"])
    after = [set(c for c, _ in m.hinges) for m in st.moves if m.start == half]
    assert {"s1"} not in after                                      # s4 lies on s1 after the half fold
    assert {"s4"} in after and {"s1", "s4"} in after


def test_look_alike_classes(tasks):
    st = tasks["cross4-mid"]
    half = ft.task_arrays(st, 7, only_half=True)[0]
    assert len(half["members"]) == 2                                # right-over-left and bottom-up
    st = tasks["box4-mid"]
    half = ft.task_arrays(st, 7, only_half=True)[0]
    assert len(half["members"]) >= 2                                # the gate fold is the same rectangle
    assert len(ft.task_arrays(tasks["box2-mid"], 7)[0]["members"]) == 1
