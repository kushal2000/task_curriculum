"""Historical contact draws stay independent of topology and reset grouping."""
from types import SimpleNamespace as NS

import pytest
import torch

from isaacsimenvs.tasks.cloth.backends.rigid import randomization
from isaacsimenvs.tasks.cloth.backends.rigid.scene import (
    bind_physics_material,
    material_combine,
)
from isaacsimenvs.tasks.cloth.tests.test_rigid_backend_readout import _backend


def _case(names=("single_hinge", "strip_8", "flexible_corners"), enabled=True):
    env = _backend(names, num_envs=6)
    cfg, rd = env.cfg.rigid_cloth, env.cfg.rigid_cloth.randomization
    cfg.hinge_stiffness_override, cfg.hinge_damping_override, cfg.hinge_armature = 0., .05, .001
    cfg.rest_offset, cfg.contact_offset = .007, .012
    rd.profile, rd.enabled = 'yibo', enabled
    rd.yibo_cloth_friction, rd.yibo_robot_friction, rd.yibo_table_friction = [.7, 1.5], [.35, .65], [.7, 1.3]
    rd.yibo_rest_offset, rd.yibo_friction_seed = [.004, .010], 0
    for group in env.groups:
        v = group.articulation.root_physx_view
        v.rest_offsets = torch.full(v.materials.shape[:2], .007)
        v.contact_offsets = torch.full_like(v.rest_offsets, .012)
        v.get_rest_offsets = lambda v=v: v.rest_offsets.clone()
        v.get_contact_offsets = lambda v=v: v.contact_offsets.clone()
        v.set_rest_offsets = lambda values, ids, v=v: v.rest_offsets.__setitem__(ids, values[ids])
        v.set_contact_offsets = lambda values, ids, v=v: v.contact_offsets.__setitem__(ids, values[ids])
    randomization.rigid_init(env)
    return env


def _shells(env):
    values = torch.empty(env.num_envs)
    for group in env.groups:
        rest = group.articulation.root_physx_view.get_rest_offsets()
        torch.testing.assert_close(rest, rest[:, :1].expand_as(rest))
        values[group.env_ids] = rest[:, 0]
    return values


def test_startup_friction_matches_historical_seeded_bucket_draw_order_and_stays_fixed():
    env = _case()
    gen = torch.Generator().manual_seed(0)
    for name, bounds in (("cloth", [.7, 1.5]), ("robot", [.35, .65]), ("table", [.7, 1.3])):
        expected = torch.linspace(*bounds, 16)[torch.randint(16, (6,), generator=gen)]
        torch.testing.assert_close(getattr(env, f'_yibo_{name}_friction'), expected)
    before = [g.articulation.root_physx_view.materials.clone() for g in env.groups]
    for ids in (torch.tensor([5, 0, 3]), torch.arange(6), torch.tensor([2])):
        randomization.rigid_randomize(env, ids)
        randomization.rigid_check(env, torch.arange(6))
    for group, expected in zip(env.groups, before, strict=True):
        torch.testing.assert_close(group.articulation.root_physx_view.materials, expected)
    other = _case(("hinge_band", "single_hinge"))
    for name in ('cloth', 'robot', 'table'):
        torch.testing.assert_close(getattr(env, f'_yibo_{name}_friction'), getattr(other, f'_yibo_{name}_friction'))


def test_rest_shell_reset_maps_unsorted_sparse_global_ids_and_leaves_physics_fixed():
    env = _case()
    before = _shells(env)
    masses = [g.articulation.root_physx_view.masses.clone() for g in env.groups]
    ids = torch.tensor([5, 0, 2])
    torch.manual_seed(371)
    expected = torch.empty(3).uniform_(.004, .010)
    torch.manual_seed(371)
    randomization.rigid_randomize(env, ids)
    actual = _shells(env)
    torch.testing.assert_close(actual[ids], expected)
    torch.testing.assert_close(actual[[1, 3, 4]], before[[1, 3, 4]])
    for group, mass in zip(env.groups, masses, strict=True):
        view = group.articulation.root_physx_view
        torch.testing.assert_close(view.masses, mass)
        torch.testing.assert_close(view.stiffness, torch.zeros_like(view.stiffness))
        torch.testing.assert_close(view.damping, torch.full_like(view.damping, .05))
        torch.testing.assert_close(view.armature, torch.full_like(view.armature, .001))
        torch.testing.assert_close(view.contact_offsets, torch.full_like(view.contact_offsets, .012))
    randomization.rigid_check(env, torch.arange(6))
    env.groups[-1].articulation.root_physx_view.rest_offsets[0, 0] += .001
    with pytest.raises(RuntimeError, match='rest offset'):
        randomization.rigid_check(env, torch.arange(6))


def test_disabled_yibo_uses_nominal_materials_and_shell_without_draws():
    env = _case(enabled=False)
    torch.testing.assert_close(_shells(env), torch.full((6,), .007))
    for name, expected in (('cloth', 1.), ('robot', .8), ('table', .4)):
        torch.testing.assert_close(getattr(env, f'_yibo_{name}_friction'), torch.full((6,), expected))
    torch.manual_seed(42)
    state = torch.random.get_rng_state().clone()
    randomization.rigid_randomize(env, torch.tensor([4, 1]))
    assert torch.equal(state, torch.random.get_rng_state())
    randomization.rigid_check(env, torch.arange(6))


def test_yibo_material_combine_is_authored_on_collision_binding():
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    cfg = NS(randomization=NS(profile='yibo'))
    stage = Usd.Stage.CreateInMemory()
    for asset, expected in (('cloth', 'min'), ('robot', 'average'), ('table', 'average')):
        UsdGeom.Xform.Define(stage, f'/{asset}')
        collider = UsdGeom.Cube.Define(stage, f'/{asset}/collision')
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        bind_physics_material(stage, f'/{asset}', (.8, .8, 0.), combine=material_combine(cfg, asset))
        material, _ = UsdShade.MaterialBindingAPI(collider.GetPrim()).ComputeBoundMaterial('physics')
        assert material.GetPrim().GetAttribute('physxMaterial:frictionCombineMode').Get() == expected
    cfg.randomization.profile = 'material'
    assert material_combine(cfg, 'cloth') == 'multiply'
    cfg.randomization.profile = 'unknown'
    with pytest.raises(ValueError, match='profile'):
        material_combine(cfg, 'cloth')
