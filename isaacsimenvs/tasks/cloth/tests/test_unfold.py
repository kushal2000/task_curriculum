"""``cloth.task: unfold`` -- the VBD sheet's folded start and the config switch.

The folded start (``cloth_adapter.fold_rest``) turns the ``+`` half rigidly about an axis
``lift / 2`` above the crease: at 0 deg it is the rest sheet, at 180 deg it lies reflected onto the
``-`` half exactly ``lift`` above it, and the half itself is never stretched.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

ft = pytest.importorskip("isaacsimenvs.tasks.cloth.fold_task")
geo = adapter = ft

SIZE, RES, LIFT = 0.10, 7, 0.002


def _rest() -> torch.Tensor:
    verts, _ = geo.grid_mesh(SIZE, RES)
    return torch.tensor(verts, dtype=torch.float32)


def _fold(deg: float) -> torch.Tensor:
    return adapter.fold_rest(_rest(), torch.tensor([math.radians(deg)]), 0, LIFT)[0]


def test_zero_angle_is_the_rest_sheet():
    assert torch.allclose(_fold(0.0), _rest(), atol=1e-7)


def test_full_fold_is_the_reflection_one_lift_up():
    rest, out = _rest(), _fold(180.0)
    moving = rest[:, 0] > 1e-9
    want = rest[moving].clone()
    want[:, 0] = -want[:, 0]
    want[:, 2] += LIFT
    assert torch.allclose(out[moving], want, atol=1e-6)
    assert torch.equal(out[~moving], rest[~moving])  # stationary half and crease untouched


def test_the_moving_half_moves_rigidly():
    rest = _rest()
    moving = rest[:, 0] > 1e-9
    d0 = torch.cdist(rest[moving], rest[moving])
    for deg in (30.0, 90.0, 150.0, 180.0):
        out = _fold(deg)
        assert torch.allclose(torch.cdist(out[moving], out[moving]), d0, atol=1e-6), deg


def test_far_corner_distance_from_its_flat_target():
    """``size * sin(angle / 2)`` up to the lift -- the rule `_check_unfold_starts` uses."""
    rest = _rest()
    far = int(rest[:, 0].argmax())
    for deg in (30.0, 60.0, 120.0, 180.0):
        d = float((_fold(deg)[far] - rest[far]).norm())
        assert abs(d - SIZE * math.sin(math.radians(deg) / 2)) < LIFT, deg


def test_cloth_cfg_unfold_switch_is_validated():
    c = ft.ClothCfg()
    assert c.task == "fold"
    c.task = "unfold"
    c.check_task()
    c.task = "refold"
    with pytest.raises(ValueError):
        c.check_task()
    c.task, c.unfold_start_deg = "unfold", [0.0, 180.0]
    with pytest.raises(ValueError):
        c.check_task()
    c.unfold_start_deg = [180.0, 180.0]
    c.task = "mixed"
    c.check_task()
    for bad in (0.0, 1.0):
        c.unfold_prob = bad
        with pytest.raises(ValueError):
            c.check_task()
