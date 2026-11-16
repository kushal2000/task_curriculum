"""Conservative release distance must cover complete collision shapes and sparse rows."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from isaacsimenvs.tasks.cloth.backends.rigid import clearance, designs
from isaacsimenvs.tasks.cloth.mdp.geometry import rotations


def test_rotated_world_box_encloses_every_actual_local_corner():
    torch.manual_seed(7)
    center = torch.randn(1, 5, 3)
    extent = torch.rand(1, 5, 3)
    position = torch.randn(4, 5, 3)
    quaternion = torch.nn.functional.normalize(torch.randn(4, 5, 4), dim=-1)
    world_center, world_extent = clearance.world_bounds(center, extent, position, quaternion)
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                points = (center + extent * torch.tensor([x, y, z])).expand_as(position)
                actual = position + rotations.apply(quaternion, points)
                assert torch.all((actual-world_center).abs() <= world_extent + 1e-6)


def test_long_robot_shape_can_be_near_when_body_origin_is_far():
    # The origin is 1 m away; the actual elongated collider is only 5 cm away.
    robot = torch.tensor([[[1., 0., 0.]]])
    robot_extent = torch.tensor([[[.85, .02, .02]]])
    cloth, cloth_extent = torch.zeros(1, 1, 3), torch.full((1, 1, 3), .1)
    distance = clearance.box_clearance(robot, robot_extent, cloth, cloth_extent)
    assert distance.item() == pytest.approx(.05)
    assert distance.item() < .1


def test_pairwise_minimum_includes_base_arm_and_fingers_and_is_chunk_invariant():
    robot = torch.tensor([[[4., 4., 0.], [1., 0., 0.], [.3, .4, 0.]]])
    robot_extent = torch.full_like(robot, .05)
    cloth = torch.zeros(1, 2, 3)
    cloth[:, 1, 1] = 2.
    cloth_extent = torch.full_like(cloth, .1)
    expected = np.sqrt(.15**2 + .25**2)
    for chunk in (1, 2, 16):
        assert clearance.box_clearance(robot, robot_extent, cloth, cloth_extent, chunk).item() == pytest.approx(expected)
    robot[:, 0] = 0.  # A base collider overlaps, regardless of where fingers are.
    assert clearance.box_clearance(robot, robot_extent, cloth, cloth_extent).item() == 0.
    robot[:, 1] = float('nan')
    assert clearance.box_clearance(robot, robot_extent, cloth, cloth_extent).item() == 0.


def test_sparse_mixed_design_readout_uses_geometry_and_global_environment_order(monkeypatch):
    monkeypatch.setattr(clearance, 'robot_collision_bounds', lambda robot: ([0], [[0., 0., 0.]], [[.05, .05, .05]]))
    origins = torch.tensor([[0., 0., 0.], [3., 4., 0.], [-2., 8., 0.], [7., -3., 0.]])
    gaps = torch.tensor([.02, .2, .11, .001])
    robot_position = origins[:, None].clone()
    robot_position[:, 0, 0] += .15 + gaps
    quaternion = torch.tensor([1., 0., 0., 0.])
    backend = NS(device='cpu', num_envs=4,
                 _dims=torch.tensor([[.2, .2, .004]]*4),
                 robot=NS(data=NS(body_link_pos_w=robot_position,
                                  body_link_quat_w=quaternion.expand(4, 1, 4))), groups=[])
    for name, env_ids in [('single_hinge', [3, 1]), ('flexible_corners', [2, 0])]:
        design = designs.load_design(name)
        ids = torch.tensor(env_ids)
        # Noncanonical simulator body order must be handled by the supplied map.
        bodies = len(design.plates)
        body_ids = list(reversed(range(bodies)))
        positions = origins[ids, None].expand(-1, bodies, -1).clone()
        data = NS(body_link_pos_w=positions, body_link_quat_w=quaternion.expand(2, bodies, 4))
        backend.groups.append(NS(design=design, env_ids=ids, body_ids=body_ids, articulation=NS(data=data)))
    helper = clearance.RobotClothClearance(backend)
    torch.testing.assert_close(helper.distance(), gaps, atol=1e-6, rtol=0)


@pytest.mark.parametrize('empty_cache', [False, True])
def test_usd_guide_collision_mesh_bounds_and_point_fallback(monkeypatch, empty_cache):
    import sys
    import types
    pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, '/Robot/body')
    body.AddTranslateOp().Set(Gf.Vec3d(10., 20., 30.))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, '/Robot/body/collisions/convex_mesh/node_STL_BINARY_')
    points = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    mesh.AddTranslateOp().Set(Gf.Vec3d(.4, .5, .6))
    mesh.AddScaleOp().Set(Gf.Vec3f(.1, .2, .3))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    fake = types.ModuleType('isaaclab.sim.utils')
    fake.get_current_stage = lambda: stage
    monkeypatch.setitem(sys.modules, 'isaaclab.sim.utils', fake)
    if empty_cache:
        cache = NS(ComputeRelativeBound=lambda *args: NS(ComputeAlignedBox=lambda: Gf.Range3d()))
        monkeypatch.setattr(UsdGeom, 'BBoxCache', lambda *args, **kwargs: cache)
    robot = NS(root_physx_view=NS(link_paths=[['/Robot/body']]))
    bodies, centers, extents = clearance.robot_collision_bounds(robot)
    assert bodies == [0]
    np.testing.assert_allclose(centers, [[.4, .5, .6]], atol=1e-7)
    np.testing.assert_allclose(extents, [[.1, .2, .3]], atol=1e-7)
