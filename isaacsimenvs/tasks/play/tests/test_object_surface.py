"""``EnvShapes`` (the ShapeBank bound to each env's fixed tool, batched SDF) against the per-call
``ShapeBank`` methods it replaces: the same SDF, surface distance, canonical goal and samples."""

from __future__ import annotations

import torch

from isaacsimenvs.conftest import load_module

os_ = load_module("isaacsimenvs/tasks/play/utils/object_surface.py", "object_surface")

_LINK = """<?xml version="1.0"?><robot name="t"><link name="object_root">
<inertial><origin xyz="0 0 0"/><mass value="0.1"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
{cols}</link></robot>"""
_BOX = '<collision><origin xyz="{x} 0 0" rpy="0 0 0"/><geometry><box size="{a} {b} {c}"/></geometry></collision>'
_CYL = '<collision><origin xyz="0 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="{r}" length="{l}"/></geometry></collision>'


def _tools(tmp_path):
    specs = [
        _CYL.format(r=0.011, l=0.13) + _BOX.format(x=0.09, a=0.054, b=0.032, c=0.041),   # hammer-like
        _CYL.format(r=0.009, l=0.16),                                                  # one primitive (padded slot)
        _BOX.format(x=0.0, a=0.12, b=0.02, c=0.02) + _BOX.format(x=0.07, a=0.03, b=0.06, c=0.01),
    ]
    paths = []
    for i, cols in enumerate(specs):
        p = tmp_path / f"{i:03d}_tool_handle.urdf"
        p.write_text(_LINK.format(cols=cols))
        paths.append(str(p))
    return paths


def _quat(n, g):
    q = torch.randn(n, 4, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def test_env_shapes_match_bank(tmp_path):
    bank = os_.ShapeBank(_tools(tmp_path), "cpu", k_fixed=64, k_pool=256, seed=0)
    g = torch.Generator().manual_seed(1)
    n = 30
    asset = torch.arange(n) % 3
    es = os_.EnvShapes(bank, asset)
    x = torch.randn(n, 50, 3, generator=g) * 0.08
    assert torch.allclose(es.sdf(x), bank.sdf(asset, x), atol=1e-6)
    pa, pb = torch.randn(n, 3, generator=g) * 0.1, torch.randn(n, 3, generator=g) * 0.1
    qa, qb = _quat(n, g), _quat(n, g)
    for got, want in zip(es.surface_distance(pa, qa, pb, qb), bank.surface_distance(asset, pa, qa, pb, qb)):
        assert torch.allclose(got, want, atol=1e-6)
    assert torch.allclose(es.points_w(pa, qa, "fixed"), bank.points_w(asset, pa, qa, "fixed"), atol=1e-6)
    assert torch.allclose(es.fixed_w_first(pa, qa, 10), bank.points_w(asset, pa, qa, "fixed")[:, :10], atol=1e-6)
    kp = torch.randn(n, 4, 3, generator=g) * 0.05
    for got, want in zip(es.canonical_goal_pose(pa, qa, pb, qb, kp), bank.canonical_goal_pose(asset, pa, qa, pb, qb, kp)):
        assert torch.allclose(got, want, atol=1e-5)
    # pool draws: each point is one of its env's tool's pool samples (posed)
    torch.manual_seed(0)
    pts = es.points_w(torch.zeros(n, 3), torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1), "pool", 16)
    for i in range(n):
        d = torch.cdist(pts[i], bank.pool[asset[i]])
        assert (d.min(dim=1).values < 1e-4).all()          # cdist (matmul form) rounds at ~3e-5
