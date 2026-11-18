"""Cloth tracker observation DR (isaacsimenvs/tasks/cloth/mdp/tracker.py): Kit-free, CPU."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from isaacsimenvs.conftest import load_module

ct = load_module("isaacsimenvs/tasks/cloth/mdp/tracker.py", "cloth_tracker")

R = 7
SIZE = 0.15


def _grid(size=SIZE, r=R) -> torch.Tensor:
    step = size / (r - 1)
    return torch.tensor(
        [(i * step - size / 2, j * step - size / 2, 0.0) for i in range(r) for j in range(r)], dtype=torch.float32
    )


def _cfg(**kw) -> SimpleNamespace:
    """The config with every random range collapsed: an exact pipeline unless a test opens one."""
    base = dict(
        inset=0.1, n_min=4, min_spread=0.02,
        cam_a_base=(0.645, -0.963, 0.856), cam_b_base=(-0.525, -0.936, 0.876), cam_pos_noise=(0.0, 0.0, 0.0),
        hand_occlusion=False, margin_range=(0.0, 0.0), finger_radius=0.009, palm_radius=0.02,
        wrist_radius=0.04, arm_radius=0.05, end_pullback=0.005,
        reacquire=False, reacquire_delay_s=(0.0, 0.0), lost_prob_range=(0.0, 0.0),
        self_occlusion=False, self_occlusion_gap=0.001,
        noise_std_range=(0.0, 0.0), reacquire_bias_range=(0.0, 0.0),
        calib_xyz=(0.0, 0.0, 0.0), calib_rot_deg=0.0,
        frame_period_choices=(1,), latency_choices=(0,), size_scale_range=(1.0, 1.0), lost_duration_s=(1.0, 3.0),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _rot_z(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


CAMS = torch.tensor([[0.645, -0.453, 1.056], [-0.525, -0.426, 1.076]])
CENTRE = torch.tensor([0.0, 0.0, 0.53])


def _sheet(n: int, rot: torch.Tensor | None = None, shift=(0.0, 0.0, 0.0)) -> torch.Tensor:
    g = _grid()
    if rot is not None:
        g = g @ rot.T
    return (g + CENTRE + torch.tensor(shift)).unsqueeze(0).repeat(n, 1, 1)


def test_inset_weights_shrink_the_grid():
    w = ct.inset_grid_weights(R, 0.1)
    assert torch.allclose(w.sum(dim=1), torch.ones(R * R))
    g = _grid()
    assert torch.allclose(w @ g, 0.8 * g, atol=1e-6)


def test_halves_exclude_the_crease_column():
    mv, st = ct.inset_halves(R)
    assert len(mv) == len(st) == 21
    xs = (ct.inset_grid_weights(R, 0.1) @ _grid())[:, 0]
    assert (xs[mv] > 0).all() and (xs[st] < 0).all()


def test_weighted_kabsch_recovers_transform_from_a_subset():
    gen = torch.Generator().manual_seed(0)
    rest = torch.randn(5, 21, 3, generator=gen)
    q = torch.nn.functional.normalize(torch.randn(5, 4, generator=gen), dim=-1)
    w_, x, y, z = q.unbind(-1)
    rot = torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w_ * z), 2 * (x * z + w_ * y),
        2 * (x * y + w_ * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w_ * x),
        2 * (x * z - w_ * y), 2 * (y * z + w_ * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).view(5, 3, 3)
    t = torch.randn(5, 3, generator=gen)
    cur = torch.einsum("nij,nkj->nki", rot, rest) + t.unsqueeze(1)
    w = (torch.rand(5, 21, generator=gen) < 0.4).float()
    w[:, :4] = 1.0
    cur = cur + (1 - w).unsqueeze(-1) * 10.0  # garbage at unweighted points must not matter
    r_fit, t_fit = ct.weighted_kabsch(rest, cur, w)
    assert torch.allclose(r_fit, rot, atol=1e-4) and torch.allclose(t_fit, t, atol=1e-4)


def test_segment_distance_matches_sampling():
    gen = torch.Generator().manual_seed(1)
    p1, q1, p2, q2 = (torch.randn(200, 3, generator=gen) for _ in range(4))
    got = ct.segment_distance_sq(p1, q1, p2, q2).sqrt()
    s = torch.linspace(0, 1, 401)
    a = p1.unsqueeze(1) + (q1 - p1).unsqueeze(1) * s.view(1, -1, 1)
    b = p2.unsqueeze(1) + (q2 - p2).unsqueeze(1) * s.view(1, -1, 1)
    brute = torch.cdist(a, b).amin(dim=(1, 2))
    assert (got <= brute + 1e-5).all() and (brute - got < 5e-3).all()


def test_ray_triangle_hit_and_gap():
    tri = [torch.tensor([[-1.0, -1.0, 0.0]]), torch.tensor([[1.0, -1.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]])]
    o = torch.tensor([[0.0, 0.0, 1.0]])
    assert ct.ray_hits_triangles(o, torch.tensor([[0.0, 0.0, -0.01]]), *tri, gap=0.001).item()
    assert not ct.ray_hits_triangles(o, torch.tensor([[0.0, 0.0, 0.0005]]), *tri, gap=0.001).item()
    assert not ct.ray_hits_triangles(o, torch.tensor([[3.0, 0.0, -0.01]]), *tri, gap=0.001).item()


def test_minor_spread_rejects_a_line():
    xy = torch.tensor([[[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [0.03, 0.0]]])
    assert ct.minor_spread(xy, torch.ones(1, 4)).item() < 1e-6
    xy2 = torch.tensor([[[0.0, 0.0], [0.03, 0.0], [0.0, 0.03], [0.03, 0.03]]])
    assert abs(ct.minor_spread(xy2, torch.ones(1, 4)).item() - 0.03) < 1e-5


def _run(cfg, n=2):
    return ct.ClothTracker(cfg, n, R, CAMS, "cpu")


def test_exact_pipeline_recovers_the_sheet_pose():
    tr = _run(_cfg())
    n = 2
    rest = _grid().unsqueeze(0).repeat(n, 1, 1)
    rot = _rot_z(0.4)
    parts = _sheet(n, rot, (0.02, -0.01, 0.0))
    start = torch.tensor([True, True])
    r_mv, t_mv, r_st, t_st, s = tr.step(torch.zeros(n, dtype=torch.long), start, parts, rest, None)
    want_t = CENTRE + torch.tensor([0.02, -0.01, 0.0])
    for r_, t_ in ((r_mv, t_mv), (r_st, t_st)):
        assert torch.allclose(r_, rot.expand(n, 3, 3), atol=1e-4)
        assert torch.allclose(t_, want_t.expand(n, 3), atol=1e-4)


def test_frame_period_and_latency():
    tr = _run(_cfg(frame_period_choices=(3,), latency_choices=(2,)), n=1)
    rest = _grid().unsqueeze(0)
    t = torch.zeros(1, dtype=torch.long)
    tr.step(t, torch.tensor([True]), _sheet(1), rest, None)
    tr.phase[:] = 0
    seen = []
    for step in range(1, 12):
        parts = _sheet(1, shift=(0.001 * step, 0.0, 0.0))
        _, t_mv, _, _, _ = tr.step(torch.tensor([step]), torch.tensor([False]), parts, rest, None)
        seen.append(round(float(t_mv[0, 0]) * 1000))
    # Frames at steps 3, 6, 9, each read 2 steps later; the start frame (shift 0) before that.
    assert seen == [0, 0, 0, 0, 3, 3, 3, 6, 6, 6, 9]


def test_hand_over_a_half_freezes_the_whole_sheet():
    tr = _run(_cfg(hand_occlusion=True), n=1)
    rest = _grid().unsqueeze(0)
    tr.step(torch.zeros(1, dtype=torch.long), torch.tensor([True]), _sheet(1), rest, None)
    # A fat capsule lying on the stationary (-x) half, then the sheet moves.
    a = torch.tensor([[[-0.04, -0.08, 0.56]]])
    b = torch.tensor([[[-0.04, 0.08, 0.56]]])
    rad = torch.tensor([0.06])
    _, t_mv, _, _, _ = tr.step(torch.tensor([1]), torch.tensor([False]), _sheet(1, shift=(0.01, 0, 0)), rest, None, a, b, rad)
    assert abs(float(t_mv[0, 0])) < 1e-6  # held: the moving half's estimate froze too
    assert not tr.fresh[0, 21:].any()
    # Hand gone: the estimate catches up.
    far_a, far_b = a + 1.0, b + 1.0
    _, t_mv, _, _, _ = tr.step(torch.tensor([2]), torch.tensor([False]), _sheet(1, shift=(0.01, 0, 0)), rest, None, far_a, far_b, rad)
    assert abs(float(t_mv[0, 0]) - 0.01) < 1e-5


def test_reacquire_lost_points_stay_lost():
    tr = _run(_cfg(hand_occlusion=True, reacquire=True, lost_prob_range=(1.0, 1.0), lost_duration_s=(1.0, 1.0)), n=1)
    rest = _grid().unsqueeze(0)
    tr.step(torch.zeros(1, dtype=torch.long), torch.tensor([True]), _sheet(1), rest, None)
    a, b, rad = torch.tensor([[[-0.04, -0.08, 0.56]]]), torch.tensor([[[-0.04, 0.08, 0.56]]]), torch.tensor([0.06])
    tr.step(torch.tensor([1]), torch.tensor([False]), _sheet(1), rest, None, a, b, rad)
    tr.step(torch.tensor([2]), torch.tensor([False]), _sheet(1), rest, None, a + 1, b + 1, rad)
    assert not tr.fresh[0, 21:].any()  # uncovered, but not re-acquired yet
    assert tr.frozen_frames[0].item() == 2.0
    tr.step(torch.tensor([62]), torch.tensor([False]), _sheet(1), rest, None, a + 1, b + 1, rad)
    assert tr.fresh[0, 21:].all()  # back after 1 s


def test_folded_flap_covers_the_stationary_half():
    tr = _run(_cfg(self_occlusion=True), n=1)
    rest = _grid().unsqueeze(0)
    tr.step(torch.zeros(1, dtype=torch.long), torch.tensor([True]), _sheet(1), rest, None)
    folded = _grid().clone()
    mv = folded[:, 0] > 1e-9
    folded[mv, 0] = -folded[mv, 0]
    folded[mv, 2] = 0.005
    parts = (folded + CENTRE).unsqueeze(0)
    tr.step(torch.tensor([1]), torch.tensor([False]), parts, rest, None)
    assert tr.fresh[0, :21].all()  # the flap itself is on top
    assert not tr.fresh[0, 21:].any()
    assert tr.frozen_frames[0].item() == 1.0
