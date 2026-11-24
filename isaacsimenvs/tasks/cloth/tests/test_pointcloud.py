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


def test_one_camera_sees_less_and_marks_its_points():
    torch.manual_seed(0)
    plate = _plate((0.0, 0.0, TABLE_Z + 0.10), 0.05)
    pos, quat = _bodies(2)
    for cams, want in (("a", 0), ("b", 1)):
        cloud = _make(plate=plate, cameras=cams)
        _, mask = cloud.observe(torch.zeros(2, dtype=torch.long), _grid(2), pos, quat)
        assert (cloud.last_cam[:, :1024][mask] == want).all()


def test_robot_points_near_the_table_are_dropped_cloth_stays():
    torch.manual_seed(0)
    low = _plate((0.21, 0.0, TABLE_Z + 0.01), 0.04)                           # 1 cm over the table, beside the sheet
    kw = dict(near_table_drop_episode_prob=1.0, near_table_drop_height_range=(0.03, 0.03),
              near_table_drop_prob_range=(1.0, 1.0))
    pos, quat = _bodies(2)
    counts = []
    for extra in ({}, kw):
        cloud = _make(plate=low, **extra)
        points, mask = cloud.observe(torch.zeros(2, dtype=torch.long), _grid(2), pos, quat)
        x = _xyz(cloud, points)[..., 0]
        counts.append((int((mask & (x > 0.16)).sum()), int((mask & (x < 0.15)).sum())))
    assert counts[0][0] > 100 and counts[1][0] == 0                           # the low plate is gone
    assert counts[1][1] >= counts[0][1] > 100                                 # the sheet is not


def test_outliers_are_robot_points_moved_along_the_ray():
    torch.manual_seed(0)
    plate = _plate((0.12, 0.0, TABLE_Z + 0.10), 0.04)
    cloud = _make(plate=plate, outlier_frac_range=(0.05, 0.05))
    pos, quat = _bodies(2)
    points, mask = cloud.observe(torch.zeros(2, dtype=torch.long), _grid(2), pos, quat)
    out = cloud.last_outlier
    share = float(out.sum()) / float(mask.sum())
    assert 0.03 < share < 0.07, share
    xyz = _xyz(cloud, points)[:, :1024]
    assert (xyz[out][:, 0] > 0.05).all()                                      # robot points only
    dz = (xyz[out][:, 2] - (TABLE_Z + 0.10)).abs()
    assert float(dz.min()) > 0.008 and float(dz.max()) < 0.05                 # 2-5 cm along a slanted ray


def test_cloth_weight_raises_the_cloth_share():
    torch.manual_seed(0)
    plate = _plate((0.15, 0.0, TABLE_Z + 0.10), 0.15)                         # 0.09 m^2 beside a 0.04 m^2 sheet
    pos, quat = _bodies(2)
    shares = []
    for rng in ((1.0, 1.0), (1.8, 1.8)):
        cloud = _make(plate=plate, n_points=256, cloth_weight_range=rng)
        cloud.observe(torch.zeros(2, dtype=torch.long), _grid(2), pos, quat)
        shares.append(float(cloud.last_cloth_share.mean()))
    assert shares[1] > shares[0] * 1.25, shares


def test_one_camera_goal_carries_the_calibration_error():
    torch.manual_seed(0)
    cloud = _make(goal_mode="goal_fixed", cameras="a")
    cloud.calib_t[:, 0] = torch.tensor([[0.05, -0.04, 0.0], [-0.03, 0.05, 0.0]])
    pos, quat = _bodies(2)
    rest = _grid(2, at=torch.zeros(3))
    points, mask = cloud.observe(torch.zeros(2, dtype=torch.long), _grid(2), pos, quat, goal_now=_grid(2), goal_rest=rest)
    goal = _xyz(cloud, points)[:, 1024:]
    off = goal.mean(dim=1)[:, :2] - CENTRE[:2]
    assert torch.allclose(off, cloud.calib_t[:, 0, :2], atol=0.012)


def test_manipuland_only_keeps_cloth_points_and_robot_still_hides():
    """``manipuland_only``: no robot point is kept, and the plate (robot) still hides the sheet below it."""
    plate = _plate((0.0, 0.0, TABLE_Z + 0.10), 0.15)
    pos, quat = _bodies(2)
    step = torch.zeros(2, dtype=torch.long)
    torch.manual_seed(1)                                   # the same cloth candidates in both clouds
    scene = _make(plate=plate)
    torch.manual_seed(2)
    scene.observe(step, _grid(2), pos, quat)
    torch.manual_seed(1)
    cloud = _make(plate=plate, manipuland_only=True)
    torch.manual_seed(2)
    points, mask = cloud.observe(step, _grid(2), pos, quat)
    z = _xyz(cloud, points)[..., 2]
    assert mask.any() and (z[mask] < TABLE_Z + 0.05).all()                    # sheet points only
    assert float(cloud.last_cloth_share.min()) == 1.0
    # The same occlusion as the full cloud: the sheet is still seen only around the plate's rim.
    assert torch.allclose(cloud.last_visible_frac, scene.last_visible_frac)
    assert cloud.last_visible_frac.max() < 0.35


def test_manipuland_only_count_is_capped_by_the_visible_candidates():
    cloud = _make(manipuland_only=True, n_cloth_cand=300)
    pos, quat = _bodies(2)
    _, mask = cloud.observe(torch.zeros(2, dtype=torch.long), _grid(2), pos, quat)
    assert mask.sum(dim=1).max() <= 300
