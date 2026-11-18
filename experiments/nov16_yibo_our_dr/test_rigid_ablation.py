"""Ablations change declared factors; contact-only composition preserves mass/gain DR."""
import pytest
import torch

from experiments.nov16_yibo_our_dr.evaluate_rigid import CASES, case_overlay, case_preset, install_contact_only_profile, merge_config
from isaacsimenvs.scripts.configs.tests.test_config_overlays import (
    _collect_classes, _env_problems, _experiment_loader, EXPERIMENT_TASKS,
)
from isaacsimenvs.tasks.cloth.backends.rigid import randomization
from isaacsimenvs.tasks.cloth.tests.test_rigid_backend_readout import _backend


def _recipe(name):
    return _experiment_loader()(name)['env']


@pytest.mark.parametrize('name', list(CASES))
def test_every_ablation_is_a_valid_config_overlay_and_retains_yibo_robot(name):
    cfg = merge_config(_recipe(case_preset(name)), case_overlay(name))
    root, modules, _ = EXPERIMENT_TASKS['Isaacsimenvs-RigidBodyCloth-Direct-v0']
    assert not _env_problems(cfg, _collect_classes(modules), root, '')
    assert cfg['robot_preset'] == 'yibo' and cfg['obs']['keypoint_mode'] == 'yibo'
    if name.startswith('yibo_with_'):
        assert cfg['rigid_cloth']['randomization']['profile'] == 'yibo'
        assert cfg['rigid_cloth']['density'] == 2.
    if name == 'contacts_yibo':
        assert cfg['rigid_cloth']['density'] == _recipe('yibo_our_dr')['rigid_cloth']['density']
        assert cfg['rigid_cloth']['randomization']['density'] == [.02, .5]


def test_contact_only_profile_preserves_all_modern_mass_gain_draws_and_rng(monkeypatch):
    original_init, original_randomize = randomization.rigid_init, randomization.rigid_randomize
    monkeypatch.setattr(randomization, 'rigid_init', original_init)
    monkeypatch.setattr(randomization, 'rigid_randomize', original_randomize)

    def make_env():
        env = _backend()
        cfg, rd = env.cfg.rigid_cloth, env.cfg.rigid_cloth.randomization
        cfg.rest_offset, cfg.contact_offset = .007, .012
        rd.profile = 'yibo'
        rd.yibo_cloth_friction, rd.yibo_robot_friction, rd.yibo_table_friction = [.7, 1.5], [.35, .65], [.7, 1.3]
        rd.yibo_rest_offset, rd.yibo_friction_seed = [.004, .010], 0
        for group in env.groups:
            view = group.articulation.root_physx_view
            view.rest = torch.full(view.materials.shape[:2], .007)
            view.contact = torch.full_like(view.rest, .012)
            view.get_rest_offsets = lambda view=view: view.rest.clone()
            view.get_contact_offsets = lambda view=view: view.contact.clone()
            view.set_rest_offsets = lambda values, ids, view=view: view.rest.__setitem__(ids, values[ids])
            view.set_contact_offsets = lambda values, ids, view=view: view.contact.__setitem__(ids, values[ids])
        return env

    plain = make_env()
    plain.cfg.rigid_cloth.randomization.profile = 'material'
    torch.manual_seed(71)
    original_init(plain)
    install_contact_only_profile(0)
    mixed = make_env()
    torch.manual_seed(71)
    randomization.rigid_init(mixed)
    for env, function in ((plain, original_randomize), (mixed, randomization.rigid_randomize)):
        torch.manual_seed(91)
        function(env, torch.tensor([5, 0, 2]))
    for field in ('bending_rigidity', 'bending_viscosity', 'density', 'armature_scale'):
        torch.testing.assert_close(getattr(plain, '_dr_' + field), getattr(mixed, '_dr_' + field))
    for one, two in zip(plain.groups, mixed.groups):
        a, b = one.articulation.root_physx_view, two.articulation.root_physx_view
        for field in ('masses', 'inertias', 'stiffness', 'damping', 'armature'):
            torch.testing.assert_close(getattr(a, field), getattr(b, field))
        assert bool(((b.rest >= .004) & (b.rest <= .010)).all())
    randomization.rigid_check(mixed, torch.arange(6))


def test_yibo_our_offsets_changes_only_offsets_and_preserves_friction_dispatch():
    base = _recipe('yibo')
    changed = merge_config(base, case_overlay('yibo_with_our_offsets'))
    assert case_preset('yibo_with_our_offsets') == 'yibo'
    expected = merge_config(base, {'rigid_cloth': {'rest_offset': 0., 'contact_offset': .002,
                                                  'randomization': {'yibo_rest_offset': [0.,0.]}}})
    assert changed == expected
    assert changed['rigid_cloth']['randomization']['profile'] == 'yibo'
    assert changed['obs'] == base['obs']
    assert changed['rigid_cloth']['randomization']['yibo_cloth_friction'] == [.7,1.5]
