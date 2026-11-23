"""The point cloud (tasks/cloth/perception/pointcloud.py, robot_surface.py; mdp/delay.py): Kit-free, CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from isaacsimenvs.conftest import REPO_ROOT, load_module

sp = load_module("isaacsimenvs/tasks/cloth/perception/pointcloud.py", "pointcloud")
dl = load_module("isaacsimenvs/tasks/cloth/mdp/delay.py", "delay")
rs = load_module("isaacsimenvs/tasks/cloth/perception/robot_surface.py", "robot_surface")

R = 7
SIZE = 0.15
TABLE_Z = 0.53
CAMS = torch.tensor([[0.645, -0.453, 1.056], [-0.525, -0.426, 1.076]])
CENTRE = torch.tensor([0.0, 0.0, TABLE_Z + 0.008])


def _grid(n: int, at=CENTRE) -> torch.Tensor:
    step = SIZE / (R - 1)
    g = torch.tensor([(i * step - SIZE / 2, j * step - SIZE / 2, 0.0) for i in range(R) for j in range(R)])
    return (g + at).unsqueeze(0).repeat(n, 1, 1)


def _plate(centre, half: float, count: int = 4000) -> torch.Tensor:
    """A square horizontal plate of 'robot' samples, as offsets in one body's frame."""
    torch.manual_seed(0)
    xy = (torch.rand(count, 2) * 2 - 1) * half
    return torch.cat((xy, torch.zeros(count, 1)), dim=1) + torch.as_tensor(centre)


def _make(n=2, plate=None, **kw):
    cfg = sp.PointCloudSetup(**kw)
    off = _plate((0.0, 0.0, 5.0), 0.01, 64) if plate is None else plate          # default: far away, unseen
    area = 1e-4 if plate is None else 0.09
    fk = torch.arange(cfg.fk_points) if cfg.fk_points else None
    cloud = sp.PointCloud(cfg, n, R, CAMS, TABLE_Z, torch.zeros(len(off), dtype=torch.long), off,
                          torch.full((len(off),), area / len(off)), "cpu", fk_idx=fk)
    cloud.reset(torch.arange(n))
    return cloud


def _bodies(n):
    return torch.zeros(n, 1, 3), torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1, 1)


def _xyz(cloud, points):
    return points[..., :3] / cloud.cfg.scale + cloud.centre


def test_plain_cloud_is_xyz_inside_the_box():
    cloud = _make()
    pos, quat = _bodies(2)
    points, mask = cloud.observe(torch.zeros(2, dtype=torch.long), _grid(2), pos, quat)
    assert points.shape == (2, 1024, 3) and mask.shape == (2, 1024)
    xyz = _xyz(cloud, points)[mask]
    assert mask.sum(dim=1).min() > 750                       # the whole sheet is in view of a camera
    assert (xyz[:, 0].abs() <= SIZE / 2 + 1e-5).all() and (xyz[:, 2] - CENTRE[2]).abs().max() < 1e-5
    assert (points[~mask] == 0).all()


def test_box_crops_and_camera_view_ends():
    cloud = _make()
    pos, quat = _bodies(2)
    step = torch.zeros(2, dtype=torch.long)
    off_table = _grid(2, at=torch.tensor([0.0, 0.45, TABLE_Z + 0.008]))        # past the robot-side edge
    assert cloud.observe(step, off_table, pos, quat)[1].sum() == 0
    high = _grid(2, at=torch.tensor([0.0, 0.0, TABLE_Z + 1.5]))               # above both images
    assert cloud.observe(step, high, pos, quat)[1].sum() == 0


def test_plate_hides_the_sheet_below():
    plate = _plate((0.0, 0.0, TABLE_Z + 0.10), 0.15)
    cloud = _make(plate=plate)
    pos, quat = _bodies(2)
    points, mask = cloud.observe(torch.zeros(2, dtype=torch.long), _grid(2), pos, quat)
    z = _xyz(cloud, points)[..., 2]
    on_plate = mask & (z > TABLE_Z + 0.05)
    assert on_plate.sum(dim=1).min() > 700
    # Only the sheet's rim can be seen under a 30 cm plate 10 cm above it.
    assert cloud.last_visible_frac.max() < 0.35
    assert (mask & ~on_plate).sum(dim=1).max() < 150


def test_points_are_drawn_in_proportion_to_area():
    plate = _plate((0.0, -0.05, TABLE_Z + 0.008), 0.15)                        # 0.09 m^2 in the sheet's plane
    cloud = _make(n=8, plate=plate, n_points=256)
    pos, quat = _bodies(8)
    sheet = _grid(8, at=torch.tensor([0.0, 0.15, TABLE_Z + 0.008]))            # beside the plate, 0.0225 m^2
    cloud.observe(torch.zeros(8, dtype=torch.long), sheet, pos, quat)
    share = float(cloud.last_cloth_share.mean())
    assert abs(share - 0.0225 / (0.0225 + 0.09)) < 0.05, share


def test_first_frame_reference_and_fk_labels():
    cloud = _make(goal_mode="label", fk_points=16)
    assert cloud.labels == ("live", "ref_moving", "ref_stationary", "fk") and cloud.point_dim == 7
    pos, quat = _bodies(2)
    p0, m0 = cloud.observe(torch.zeros(2, dtype=torch.long), _grid(2), pos, quat)
    assert p0.shape == (2, 1024 + 192 + 16, 7)
    ref = slice(1024, 1024 + 192)
    assert m0[:, ref].all() and (p0[:, ref, 3] == 0).all()
    moving = p0[:, ref, 4] == 1
    assert ((_xyz(cloud, p0)[:, ref, 0] > 0) == moving).all()                  # the +x half moves
    assert (p0[:, :1024, 3][m0[:, :1024]] == 1).all() and (p0[:, -16:, 6] == 1).all()
    shifted = _grid(2, at=CENTRE + torch.tensor([0.05, 0.0, 0.0]))
    p1, _ = cloud.observe(torch.ones(2, dtype=torch.long), shifted, pos, quat)
    assert torch.equal(p1[:, ref], p0[:, ref])                                 # frozen
    cloud.reset(torch.tensor([1]))
    p2, _ = cloud.observe(torch.tensor([2, 0]), shifted, pos, quat)
    assert torch.equal(p2[0, ref], p0[0, ref]) and not torch.equal(p2[1, ref], p0[1, ref])


def test_goal_clouds():
    """goal_fixed: the goal sheet where it is, taken at ref_step and frozen; goal_any: its shape about
    the box centre at a random yaw, there from the first step. One ``goal`` label."""
    pos, quat = _bodies(2)
    goal_now = _grid(2, at=CENTRE + torch.tensor([0.03, -0.02, 0.0]))
    goal_rest = _grid(2, at=torch.zeros(3))
    goal_rest[:, :, 0] *= 0.5                                                  # a folded sheet: half as wide
    for mode in ("goal_fixed", "goal_any"):
        cloud = _make(goal_mode=mode, ref_step=3)
        assert cloud.labels == ("live", "goal") and cloud.point_dim == 5
        ref = slice(1024, 1024 + 192)
        p0, m0 = cloud.observe(torch.zeros(2, dtype=torch.long), _grid(2), pos, quat, goal_now=goal_now, goal_rest=goal_rest)
        assert p0.shape == (2, 1024 + 192, 5)
        if mode == "goal_fixed":
            assert not m0[:, ref].any()                                        # not before ref_step
            p0, m0 = cloud.observe(torch.full((2,), 3), _grid(2), pos, quat, goal_now=goal_now, goal_rest=goal_rest)
        assert m0[:, ref].all() and (p0[:, ref, 4] == 1).all()
        xyz = _xyz(cloud, p0)[:, ref]
        if mode == "goal_fixed":                                               # in place
            assert (xyz[..., :2].mean(dim=1) - goal_now[:, :, :2].mean(dim=1)).abs().max() < 5e-3
        else:                                                                  # about the box centre, its extent
            assert (xyz[..., :2].mean(dim=1) - cloud.centre[:2]).abs().max() < 5e-3
            r = (xyz[..., :2] - cloud.centre[:2]).norm(dim=-1).amax(dim=1)
            assert (r < (0.25 * SIZE**2 + 0.0625 * SIZE**2) ** 0.5 + 1e-4).all()
        moved = _grid(2, at=CENTRE + torch.tensor([0.05, 0.0, 0.0]))
        p1, _ = cloud.observe(torch.full((2,), 5), moved, pos, quat, goal_now=moved, goal_rest=goal_rest)
        assert torch.equal(p1[:, ref], p0[:, ref])                             # frozen for the episode


def test_delay_returns_the_cloud_of_lag_steps_ago():
    cloud = _make(n=3, delay_max=4)
    cloud.lag[:] = torch.tensor([0, 2, 4])
    pos, quat = _bodies(3)
    xs = []
    for t in range(7):
        sheet = _grid(3, at=CENTRE + torch.tensor([0.01 * t, 0.0, 0.0]))
        points, mask = cloud.observe(torch.full((3,), t), sheet, pos, quat)
        xs.append(torch.stack([_xyz(cloud, points)[e, :, 0][mask[e]].mean() for e in range(3)]))
    x = torch.stack(xs)                                                       # (T, env) mean x of the cloud
    for e, lag in enumerate((0, 2, 4)):
        for t in range(7):
            assert abs(float(x[t, e]) - 0.01 * max(t - lag, 0)) < 5e-3, (e, t)   # 768 uniform samples: sd 1.6 mm


def test_noise_moves_points_and_keeps_them_in_the_box():
    torch.manual_seed(0)
    cfg_kw = dict(calib_xyz=(0.015, 0.015, 0.01), calib_rot_deg=2.0, depth_noise_std_range=(0.004, 0.004),
                  drop_prob_range=(0.3, 0.3), cam_pos_noise=(0.05, 0.05, 0.05))
    cloud = _make(n=4, **cfg_kw)
    pos, quat = _bodies(4)
    points, mask = cloud.observe(torch.zeros(4, dtype=torch.long), _grid(4), pos, quat)
    z = _xyz(cloud, points)[..., 2][mask]
    assert 0.002 < float(z.std()) < 0.03
    assert 0.55 < float(mask.sum()) / (4 * cloud.n_cloth) < 0.85              # ~30% dropped
    assert (z > TABLE_Z - cloud.cfg.box_below - 1e-6).all()


def test_look_at_and_delay_line():
    rot = sp.look_at(CAMS, torch.tensor([0.0, 0.0, TABLE_Z]).expand(2, 3))
    assert torch.allclose(rot.transpose(-1, -2) @ rot, torch.eye(3).expand(2, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(rot), torch.ones(2), atol=1e-5)
    assert (rot[:, 2, 1] < 0).all()                                            # image y points down
    buf = dl.DelayLine(2, 3, (1,), "cpu")
    lag = torch.tensor([1, 3])
    out = [buf.push_and_read(torch.full((2, 1), float(t)), torch.tensor([t == 0, t == 0]), lag) for t in range(6)]
    assert [float(o[0]) for o in out] == [0, 0, 1, 2, 3, 4] and [float(o[1]) for o in out] == [0, 0, 0, 0, 1, 2]


def test_robot_surface_resolves_fixed_links():
    urdf = REPO_ROOT / "assets/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf"
    data = np.load(urdf.with_suffix(".surface.npz"))
    links = [str(v) for v in data["links"]]
    assert "r_wrist" in links and "link2" not in links
    bodies = [k for k in links if k not in ("r_mount", "r_wrist") and not k.endswith("tip_sensor_frame")]
    surf = rs.load(urdf, bodies)
    wrist = surf["offset"][surf["link"] == links.index("r_wrist")]
    assert set(surf["body"][surf["link"] == links.index("r_wrist")]) == {bodies.index("link7")}
    # flange 0.124 along link7's z, then the mount and the wrist (the palm).
    assert 0.12 < wrist[:, 2].min() and wrist[:, 2].max() < 0.25
    assert abs(float(surf["weight"].sum()) - float(data["weight"].sum())) < 1e-6
    fk = rs.fk_subset(surf, 128)
    assert len(set(fk.tolist())) == 128
    assert not any(links[i].startswith(("link3", "link4")) for i in surf["link"][fk])


def test_reset_mask_matches_reset():
    """``reset_mask`` (no host sync) leaves unmasked envs untouched and redraws the masked ones
    within ``reset``'s ranges, as ``reset(mask.nonzero())`` would."""
    import math

    setup = sp.PointCloudSetup(n_points=16, cam_pos_noise=(0.05, 0.05, 0.05), calib_xyz=(0.015, 0.015, 0.01),
                            calib_rot_deg=2.0, depth_noise_std_range=(0.001, 0.004), drop_prob_range=(0.0, 0.3),
                            delay_max=6)
    n = 64
    cams = torch.tensor([[0.6, -0.9, 0.85], [-0.5, -0.9, 0.86]])
    pc = sp.PointCloud(setup, n, 2, cams, 0.28, torch.zeros(4, dtype=torch.long), torch.zeros(4, 3),
                    torch.ones(4), "cpu")
    pc.reset(torch.arange(n))
    before = {k: getattr(pc, k).clone() for k in ("cams", "rots", "calib_r", "calib_t", "depth_std", "drop_p", "lag")}
    pc.fresh[:] = False
    pc.ref_valid[:] = True
    mask = torch.zeros(n, dtype=torch.bool)
    mask[::3] = True
    pc.reset_mask(mask)
    for k, v in before.items():
        new = getattr(pc, k)
        assert torch.equal(new[~mask], v[~mask]), k
        assert not torch.equal(new[mask], v[mask]), k
    assert torch.equal(pc.fresh, mask) and torch.equal(pc.ref_valid, ~mask)
    assert ((pc.cams[mask] - pc.cams0).abs() <= 0.05 + 1e-6).all()
    assert ((pc.calib_t[mask].abs() <= torch.tensor([0.015, 0.015, 0.01]) + 1e-6)).all()
    assert ((pc.depth_std[mask] >= 0.001) & (pc.depth_std[mask] <= 0.004)).all()
    assert ((pc.lag[mask] >= 0) & (pc.lag[mask] <= 6)).all()
