"""The Warp visibility kernels (perception/visibility_warp.py) give the torch path's answer."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp kernels run on the GPU")


def _cloud(n=64):
    from isaacsimenvs.tasks.cloth.perception.pointcloud import PointCloud, PointCloudSetup

    cfg = PointCloudSetup(n_points=64, n_cloth_cand=512, box_y_near=0.49)
    cams = torch.stack((torch.tensor(cfg.cam_a_base), torch.tensor(cfg.cam_b_base))) + torch.tensor([0.0, 0.8, 0.0])
    g = torch.Generator().manual_seed(0)
    pool_body = torch.randint(0, 6, (3000,), generator=g)
    pool_off = torch.randn(3000, 3, generator=g) * 0.04
    pc = PointCloud(cfg, n, 2, cams, 0.28, pool_body, pool_off, torch.full((3000,), 1e-5), "cuda",
                    cam_rots=(cfg.cam_a_rot, cfg.cam_b_rot))
    return pc


def _scene(n, k=512):
    g = torch.Generator(device="cuda").manual_seed(1)
    tool = torch.tensor([0.0, 0.3, 0.32], device="cuda") + torch.randn(n, k, 3, device="cuda", generator=g) * 0.04
    body_pos = torch.tensor([0.0, 0.33, 0.42], device="cuda") + torch.randn(n, 6, 3, device="cuda", generator=g) * 0.03
    body_quat = torch.nn.functional.normalize(torch.randn(n, 6, 4, device="cuda", generator=g), dim=-1)
    return tool, body_pos, body_quat


def test_visible_matches_torch():
    from isaacsimenvs.tasks.cloth.perception.visibility_warp import visible_warp

    pc = _cloud()
    tool, bp, bq = _scene(pc.n)
    pts = torch.cat((tool, pc.robot_points(bp, bq)), dim=1)
    ref = pc.visible(pts, pc.cams, pc.rots)
    out = visible_warp(pts, pc.cams, pc.rots, pc.f, pc.w, pc.h, pc.cfg.min_depth, pc.cfg.depth_tol)
    assert ref.any() and (~ref).any()                       # the scene has both visible and hidden points
    assert (out != ref).float().mean().item() < 1e-5        # depth ties may round differently


def test_bodies_in_kernel_match_built_points():
    """Robot occluders moved in the kernel == built as points and splatted (the manipuland's verdict)."""
    from isaacsimenvs.tasks.cloth.perception.visibility_warp import visible_warp

    pc = _cloud()
    tool, bp, bq = _scene(pc.n)
    k = tool.shape[1]
    built = visible_warp(torch.cat((tool, pc.robot_points(bp, bq)), dim=1), pc.cams, pc.rots, pc.f, pc.w, pc.h,
                         pc.cfg.min_depth, pc.cfg.depth_tol)[:, :, :k]
    kernel = visible_warp(tool, pc.cams, pc.rots, pc.f, pc.w, pc.h, pc.cfg.min_depth, pc.cfg.depth_tol,
                          body_pos=bp, body_quat=bq, pool_body=pc.pool_body, pool_off=pc.pool_off)
    hidden_by_robot = (visible_warp(tool, pc.cams, pc.rots, pc.f, pc.w, pc.h, pc.cfg.min_depth, pc.cfg.depth_tol) & ~built).sum()
    assert hidden_by_robot > 0                              # the robot does occlude part of the tool here
    assert (kernel != built).float().mean().item() < 1e-4
