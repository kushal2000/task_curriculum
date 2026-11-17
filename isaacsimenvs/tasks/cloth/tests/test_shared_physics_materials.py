"""Startup physics materials must not multiply when USD asset roots are cloned."""
from types import SimpleNamespace as NS

import pytest

Usd = pytest.importorskip('pxr.Usd')
from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade

from isaacsimenvs.tasks.cloth.backends.rigid.scene import (
    bind_material_roots,
    define_physics_material,
    shared_contact_material,
    strip_physics_materials,
)


def _cfg(profile='material'):
    return NS(table_static_friction=.4, hand_static_friction=.8, dynamic_friction_ratio=.8,
              randomization=NS(profile=profile))


@pytest.mark.parametrize('count', [1, 100])
@pytest.mark.parametrize('profile', ['material', 'yibo'])
def test_three_shared_materials_survive_asset_copy_and_references(count, profile):
    world = Usd.Stage.CreateInMemory()
    source = Usd.Stage.CreateInMemory()
    cfg = _cfg(profile)
    for kind in ('robot', 'table', 'cloth'):
        global_material = shared_contact_material(world, cfg, kind)
        root = UsdGeom.Xform.Define(source, f'/{kind}').GetPrim()
        # Exercise preservation of explicitly authored root physics APIs.
        UsdPhysics.RigidBodyAPI.Apply(root)
        box = UsdGeom.Cube.Define(source, f'/{kind}/collision').GetPrim()
        UsdPhysics.CollisionAPI.Apply(box)
        visual = UsdShade.Material.Define(source, f'/{kind}/VisualMaterial')
        UsdShade.MaterialBindingAPI.Apply(box).Bind(visual)
        bind_material_roots(source, [root.GetPath()], global_material.GetPath())
        assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    with Sdf.ChangeBlock():
        for i in range(count):
            for kind in ('robot', 'table', 'cloth'):
                dest = f'/World/envs/env_{i}/{kind}'
                Sdf.CreatePrimInLayer(world.GetRootLayer(), f'/World/envs/env_{i}')
                assert Sdf.CopySpec(source.GetRootLayer(), f'/{kind}', world.GetRootLayer(), dest)
    materials = [p for p in world.Traverse() if p.HasAPI(UsdPhysics.MaterialAPI)]
    assert len(materials) == 3
    assert all(str(p.GetPath()).startswith('/World/ClothMaterials/') for p in materials)
    for i in (0, count-1):
        for kind in ('robot', 'table', 'cloth'):
            prim = world.GetPrimAtPath(f'/World/envs/env_{i}/{kind}/collision')
            binding = UsdShade.MaterialBindingAPI(prim)
            physics, _ = binding.ComputeBoundMaterial('physics')
            visual, _ = binding.ComputeBoundMaterial()
            assert str(physics.GetPath()) == f'/World/ClothMaterials/{kind}'
            assert str(visual.GetPath()) == f'/World/envs/env_{i}/{kind}/VisualMaterial'
            mode = physics.GetPrim().GetAttribute('physxMaterial:frictionCombineMode').Get()
            assert mode == ('multiply' if profile == 'material' else 'min' if kind == 'cloth' else 'average')


def test_baked_asset_strips_inherited_physics_materials_but_keeps_visuals_and_body_density(tmp_path):
    source = Usd.Stage.CreateNew(str(tmp_path/'source.usda'))
    root = UsdGeom.Xform.Define(source, '/Robot').GetPrim()
    source.SetDefaultPrim(root)
    UsdPhysics.MassAPI.Apply(root).CreateDensityAttr(123.)
    box = UsdGeom.Cube.Define(source, '/Robot/box').GetPrim()
    UsdPhysics.CollisionAPI.Apply(box)
    material = define_physics_material(source, '/Robot/Material', (.5, .4, 0.))
    material.CreateSurfaceOutput().Set('visual-marker')
    UsdShade.MaterialBindingAPI.Apply(box).Bind(material)
    UsdShade.MaterialBindingAPI(box).Bind(material, materialPurpose='physics')
    source.GetRootLayer().Save()
    baked = Usd.Stage.CreateNew(str(tmp_path/'baked.usda'))
    baked_root = UsdGeom.Xform.Define(baked, '/Robot').GetPrim()
    baked_root.GetReferences().AddReference(str(tmp_path/'source.usda'))
    baked.SetDefaultPrim(baked_root)
    baked.GetRootLayer().Save()
    strip_physics_materials(str(tmp_path/'baked.usda'))
    baked.Reload()
    assert not any(p.HasAPI(UsdPhysics.MaterialAPI) for p in baked.Traverse())
    assert baked.GetPrimAtPath('/Robot/Material').GetAttribute('outputs:surface').Get() == 'visual-marker'
    assert UsdPhysics.MassAPI(baked.GetPrimAtPath('/Robot')).GetDensityAttr().Get() == 123.
    box = baked.GetPrimAtPath('/Robot/box')
    visual, _ = UsdShade.MaterialBindingAPI(box).ComputeBoundMaterial()
    assert str(visual.GetPath()) == '/Robot/Material'
