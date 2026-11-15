"""``cloth.task: unfold`` -- the VBD sheet's folded start and the config switch.

The folded start (``cloth_adapter.fold_rest``) turns the ``+`` half rigidly about an axis
``lift / 2`` above the crease: at 0 deg it is the rest sheet, at 180 deg it lies reflected onto the
``-`` half exactly ``lift`` above it, and the half itself is never stretched.
"""

from __future__ import annotations

import math

import pytest
from conftest import load_module  # noqa: E402  pytest puts tests/ on sys.path

torch = pytest.importorskip("torch")

geo = load_module("isaacsimenvs/tasks/cloth/utils/cloth_geometry.py", "cloth_geometry")
adapter = load_module("isaacsimenvs/tasks/cloth/utils/cloth_adapter.py", "cloth_adapter")
rc = load_module("isaacsimenvs/tasks/cloth/utils/rigid_cloth.py", "rigid_cloth")
fs = load_module("isaacsimenvs/tasks/cloth/utils/fold_shapes.py", "fold_shapes")

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
    cfg_mod = pytest.importorskip("isaacsimenvs.tasks.cloth.cloth_cfg")
    c = cfg_mod.ClothCfg()
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


def test_fold_pool_target_starts_reach_180():
    cfg_mod = pytest.importorskip("isaacsimenvs.tasks.cloth.fold_pool_cfg")
    cfg_mod.FoldPoolCfg(init_fold_starts="target", init_fold_prob=1.0, init_fold_deg=[60.0, 180.0])
    with pytest.raises(ValueError):
        cfg_mod.FoldPoolCfg(init_fold_starts="single", init_fold_deg=[60.0, 180.0])


def test_box3_mid_full_fold_stacks_the_plates_without_overlap():
    """The single chain's folded start (``chain_fold_angles``, scaled to 180 deg): both bar hinges at
    90 deg put the moving plate one bar width (2 mm) above the root plate, face on face, its whole
    width over the root plate. ``box2-mid``'s 180 deg fold has no such gap: the plates coincide."""
    spec = rc.variant("box3-mid")
    root, wall, top = rc.chain_frames(spec, rc.chain_fold_angles(spec))
    assert abs(math.degrees(abs(wall[2])) - 90.0) < 1e-9 and abs(math.degrees(abs(top[2])) - 180.0) < 1e-9
    lift = top[1] - root[1]
    assert abs(lift - spec.widths[1]) < 1e-12
    assert lift - spec.thickness >= -1e-12  # the plates' facing surfaces touch, they do not cross
    # The moving plate's frame sits on its inboard edge, over the crease; flipped, it extends to -x.
    x0, x1 = sorted((top[0], top[0] - spec.widths[2]))
    assert -spec.span / 2 - 1e-12 <= x0 and x1 <= 0.0
    assert fs.POOL["box2-mid"].ply_gap == 0.0
