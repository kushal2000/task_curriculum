"""Cloth point-cloud observation (isaacsimenvs/tasks/cloth/perception/pointcloud.py): Kit-free, CPU."""

from __future__ import annotations

import torch

from isaacsimenvs.conftest import load_module

pc = load_module("isaacsimenvs/tasks/cloth/perception/pointcloud.py", "cloth_pointcloud")

R = 7
SIZE = 0.15
CAMS = torch.tensor([[0.645, -0.453, 1.056], [-0.525, -0.426, 1.076]])
CENTRE = torch.tensor([0.0, 0.0, 0.53])


def _grid(fold_deg: float = 0.0) -> torch.Tensor:
    """The R x R sheet; the +x half rotated ``fold_deg`` about the crease (y axis through x = 0)."""
    step = SIZE / (R - 1)
    g = torch.tensor([(i * step - SIZE / 2, j * step - SIZE / 2, 0.0) for i in range(R) for j in range(R)])
    a = torch.deg2rad(torch.tensor(fold_deg))
    mv = g[:, 0] > 1e-9
    x = g[:, 0].clone()
    g[mv, 0] = x[mv] * torch.cos(a)
    g[mv, 2] = x[mv] * torch.sin(a)
    return g + CENTRE


def _make(n=3, **kw) -> "pc.ClothPointCloud":
    cfg = pc.PointCloudCfg(**kw)
    cloud = pc.ClothPointCloud(cfg, n, R, CAMS, torch.tensor([0.01, 0.01]), "cpu")
    cloud.reset(torch.arange(n), CENTRE.expand(n, 3))
    return cloud


def _caps(n, at=(0.5, 0.5, 0.9)):
    a = torch.tensor(at).view(1, 1, 3).repeat(n, 2, 1)
    return a, a + torch.tensor([0.0, 0.02, 0.0])


def test_samples_lie_on_the_sheet_and_are_labelled_by_half():
    cloud = _make()
    parts = _grid().unsqueeze(0).repeat(3, 1, 1)
    pts, moving = cloud.sample(parts)
    assert pts.shape == (3, 192, 3)
    assert torch.allclose(pts[..., 2], torch.full_like(pts[..., 2], 0.53), atol=1e-6)
    assert bool((pts[..., :2].abs() <= SIZE / 2 + 1e-6).all())
    assert bool(((pts[..., 0] > 0) == moving).all())


def test_flat_sheet_is_fully_visible():
    cloud = _make()
    parts = _grid().unsqueeze(0).repeat(3, 1, 1)
    a, b = _caps(3)
    _, mask, _ = cloud.cloth_cloud(parts, a, b)
    assert bool(mask.all())


def test_folded_flap_hides_the_stationary_half_only():
    # 180 deg: the moving half lies IN the stationary half's plane (PhysX mid hinge, ply gap 0).
    # 175 deg: the flap's free edge is 6.5 mm up, and the oblique cameras see a little under it.
    for deg, limit in ((180.0, 0.02), (175.0, 0.2)):
        cloud = _make()
        parts = _grid(deg).unsqueeze(0).repeat(3, 1, 1)
        a, b = _caps(3)
        _, mask, moving = cloud.cloth_cloud(parts, a, b)
        assert bool(mask[moving].all()), deg
        assert float(mask[~moving].float().mean()) < limit, deg


def test_hand_capsule_hides_points_behind_it():
    cloud = _make(n=1)
    parts = _grid().unsqueeze(0)
    # A capsule on the line of sight of BOTH cameras cannot exist for all points; put one just
    # above the sheet centre: points under it are hidden from both cameras, far ones are not.
    a = (CENTRE + torch.tensor([0.0, -0.01, 0.01])).view(1, 1, 3).repeat(1, 2, 1)
    b = a + torch.tensor([0.0, 0.02, 0.0])
    cloud.cap_r = torch.tensor([0.02, 0.02])
    pts, mask, _ = cloud.cloth_cloud(parts, a, b)
    near = (pts[0, :, :2] - CENTRE[:2]).norm(dim=-1) < 0.008
    far = (pts[0, :, :2] - CENTRE[:2]).norm(dim=-1) > 0.06
    assert bool(near.any()) and not bool(mask[0][near].any())
    assert bool(mask[0][far].all())


def test_visibility_none_keeps_everything():
    cloud = _make(visibility="none")
    parts = _grid(180.0).unsqueeze(0).repeat(3, 1, 1)
    _, mask, _ = cloud.cloth_cloud(parts)
    assert bool(mask.all())


def test_observe_layout_and_labels():
    n = 2
    parts = _grid().unsqueeze(0).repeat(n, 1, 1)
    a, b = _caps(n)
    palm = CENTRE.expand(n, 3) + torch.tensor([0.0, 0.0, 0.1])
    goal = torch.rand(n, 4, 3)
    st = CENTRE.expand(n, 3)
    step = torch.zeros(n, dtype=torch.long)
    for mode, extra in (("g0", 4), ("g3", 4), ("g1", 192), ("gl", 0)):
        cloud = _make(n=n, goal_mode=mode)
        points, mask = cloud.observe(step, parts, palm, goal, st, a, b)
        assert points.shape == (n, 192 + 6 + extra, 3 + len(pc.LABELS)), mode
        assert mask.shape == points.shape[:2]
        lab = points[..., 3:].argmax(dim=-1)
        assert bool((lab[:, 192:198] == 3).all())
        if mode == "gl":
            assert set(lab[:, :192].unique().tolist()) == {1, 2}
        else:
            assert bool((lab[:, :192] == 0).all())
        if mode in ("g0", "g3"):
            assert bool((lab[:, 198:] == 6).all())
            assert torch.allclose(points[:, 198:, :3], (goal - palm.unsqueeze(1)) * 10.0, atol=1e-5)
        if mode == "g1":
            assert set(lab[:, 198:].unique().tolist()) == {4, 5}


def test_g3_noise_moves_the_goal_in_the_table_plane():
    n = 4
    parts = _grid().unsqueeze(0).repeat(n, 1, 1)
    palm = torch.zeros(n, 3)
    goal = CENTRE + torch.tensor([[-0.05, -0.05, 0.0], [-0.05, 0.05, 0.0], [-0.01, -0.05, 0.0], [-0.01, 0.05, 0.0]])
    goal = goal.unsqueeze(0).repeat(n, 1, 1)
    cfg = pc.g3_noise_preset(pc.PointCloudCfg(goal_mode="g3"))
    cloud = pc.ClothPointCloud(cfg, n, R, CAMS, None, "cpu")
    cloud.reset(torch.arange(n), CENTRE.expand(n, 3))
    points, _ = cloud.observe(torch.zeros(n, dtype=torch.long), parts, palm, goal, CENTRE.expand(n, 3))
    got = points[:, 192:, :3] / 10.0
    err = (got - goal).norm(dim=-1)
    assert float(err.max()) < 0.03 and float(err.max()) > 1e-4
    assert torch.allclose(got[..., 2], goal[..., 2], atol=1e-6)


def test_g1_reference_is_frozen_until_reset():
    n = 2
    cloud = _make(n=n, goal_mode="g1", ref_step=2)
    a, b = _caps(n)
    palm = torch.zeros(n, 3)
    flat = _grid().unsqueeze(0).repeat(n, 1, 1)
    step = torch.zeros(n, dtype=torch.long)
    _, mask = cloud.observe(step, flat, palm, caps_a=a, caps_b=b)
    assert not bool(mask[:, 198:].any())              # before ref_step: no reference yet
    p1, m1 = cloud.observe(step + 2, flat, palm, caps_a=a, caps_b=b)
    assert bool(m1[:, 198:].all())
    p2, _ = cloud.observe(step + 3, _grid(120.0).unsqueeze(0).repeat(n, 1, 1), palm, caps_a=a, caps_b=b)
    assert torch.equal(p1[:, 198:], p2[:, 198:])
    cloud.reset(torch.tensor([0]), CENTRE.view(1, 3))
    _, m3 = cloud.observe(step, flat, palm, caps_a=a, caps_b=b)
    assert not bool(m3[0, 198:].any()) and bool(m3[1, 198:].all())


def test_noise_preset_perturbs_and_drops():
    n = 8
    cfg = pc.noise_preset(pc.PointCloudCfg())
    cloud = pc.ClothPointCloud(cfg, n, R, CAMS, torch.tensor([0.01, 0.01]), "cpu")
    cloud.reset(torch.arange(n), CENTRE.expand(n, 3))
    parts = _grid().unsqueeze(0).repeat(n, 1, 1)
    a, b = _caps(n)
    pts, mask, _ = cloud.cloth_cloud(parts, a, b, tips=CENTRE.view(1, 1, 3).repeat(n, 5, 1))
    assert 0.4 < float(mask.float().mean()) < 1.0
    dz = (pts[..., 2] - 0.53).abs()
    assert 1e-4 < float(dz.mean()) < 0.03


def test_hand_cam_points_carry_the_calibration_offset():
    n = 6
    cfg = pc.noise_preset(pc.PointCloudCfg(goal_mode="gl", hand_cam=True))
    cfg.depth_noise_std_range, cfg.drop_prob_range, cfg.calib_rot_deg = (0.0, 0.0), (0.0, 0.0), 0.0
    cloud = pc.ClothPointCloud(cfg, n, R, CAMS, torch.tensor([0.01, 0.01]), "cpu")
    cloud.reset(torch.arange(n), CENTRE.expand(n, 3))
    parts = _grid().unsqueeze(0).repeat(n, 1, 1)
    a, b = _caps(n)
    points, mask = cloud.observe(torch.zeros(n, dtype=torch.long), parts, torch.zeros(n, 3), caps_a=a, caps_b=b)
    assert points.shape == (n, 192 + 6 + 6, 3 + len(pc.LABELS)) and bool(mask[:, 192:].all())
    assert bool((points[:, 198:, 3:].argmax(dim=-1) == 6).all())
    off = (points[:, 198:, :3] - points[:, 192:198, :3]) / 10.0          # one of the two cameras' shifts
    d = (off.unsqueeze(2) - cloud.calib_t.unsqueeze(1)).norm(dim=-1).amin(dim=-1)
    assert float(d.max()) < 1e-5 and float(off.norm(dim=-1).max()) > 1e-3


def test_hand_capsules_cover_the_arm_and_five_fingers():
    caps = pc.hand_capsules(pc.PointCloudCfg(), "r_")
    assert len(caps) == 24
    bodies = {b for a, _, c, _, _ in caps for b in (a, c)}
    assert {"link5", "link6", "link7", "r_thumb_distal", "r_pinky_proximal"} <= bodies
    assert all(r > 0 for *_, r in caps)
