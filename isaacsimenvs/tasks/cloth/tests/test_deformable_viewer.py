"""CPU checks for viewer configuration, native pose conversion and rollout bookkeeping."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from isaacsimenvs.scripts.visualization import viser_deformableCloth_viewer as viewer


@pytest.mark.parametrize('preset,size,start_height,xy,table_range', [
    ('yibo', .10, .30, .10, .01),
    ('materials', .15, .20, .05, 0.),
])
def test_presets_match_their_policy_evaluation_contract(preset, size, start_height, xy, table_range):
    cfg = viewer.build_config(viewer.parse_args(['--preset', preset]))
    assert cfg.scene.num_envs == 1
    assert cfg.robot_preset == 'yibo'
    assert cfg.obs.keypoint_mode == 'yibo'
    assert not cfg.task.release.enabled
    assert cfg.deformable_cloth.size == pytest.approx(size)
    assert cfg.deformable_cloth.start_height == pytest.approx(start_height)
    assert cfg.reset.reset_position_noise_x == pytest.approx(xy)
    assert cfg.reset.reset_position_noise_y == pytest.approx(xy)
    assert cfg.reset.table_reset_z == pytest.approx(.38)
    assert cfg.reset.table_reset_z_range == pytest.approx(table_range)


def test_material_switch_constructs_fresh_complete_physics_config():
    towel = viewer.build_config(viewer.parse_args(['--material', 'towel_measured']))
    base = viewer.build_config(viewer.parse_args(['--material', 'base']))
    assert towel is not base
    assert towel.deformable_cloth is not base.deformable_cloth
    assert towel.deformable_cloth.self_contact_margin == pytest.approx(.0055)
    assert towel.deformable_cloth.plastic_yield_curvature == pytest.approx(60.)
    # Base omits self_contact_margin in its YAML, so this specifically catches
    # reusing the previous material's config instead of starting from defaults.
    assert base.deformable_cloth.self_contact_margin == pytest.approx(.003)
    assert base.deformable_cloth.plastic_yield_curvature == pytest.approx(0.)
    assert base.deformable_cloth.density == pytest.approx(2.)
    base.deformable_cloth.self_contact_margin = .02
    assert towel.deformable_cloth.self_contact_margin == pytest.approx(.0055)


def test_explicit_geometry_and_drop_height_override_preset():
    cfg = viewer.build_config(viewer.parse_args([
        '--preset', 'materials', '--size', '.18', '--resolution', '9', '--drop-height', '.07',
        '--vbd-iterations', '80',
    ]))
    assert cfg.deformable_cloth.size == pytest.approx(.18)
    assert cfg.deformable_cloth.resolution == 9
    assert cfg.deformable_cloth.vbd_iterations == 80
    # start_height is measured from table center; the collision box is30cm high.
    assert cfg.deformable_cloth.start_height == pytest.approx(.15 + .07)


@pytest.mark.parametrize('proxy', [False, True])
@pytest.mark.parametrize('xyzw,wxyz', [
    ([0., 0., 0., 1.], [1., 0., 0., 0.]),
    ([0., 0., 2**-.5, 2**-.5], [2**-.5, 0., 0., 2**-.5]),
])
def test_asset_pose_uses_native_newton_xyzw_and_removes_env_origin(proxy, xyzw, wxyz):
    pos = torch.tensor([[1.2, -2.3, .4]])
    quat = torch.tensor([xyzw])
    if proxy:
        pos, quat = NS(torch=pos), NS(torch=quat)
    asset = NS(data=NS(root_pos_w=pos, root_quat_w=quat))
    result_pos, result_quat = viewer.asset_pose(asset, np.array([1., -2., 0.]))
    np.testing.assert_allclose(result_pos, [.2, -.3, .4], atol=1e-6)
    np.testing.assert_allclose(result_quat, wxyz, atol=1e-7)


@pytest.mark.parametrize('terminated,truncated,folded', [
    (False, False, False), (True, False, True), (False, True, False),
])
def test_playback_uses_deterministic_policy_and_resets_rnn_after_autoreset(terminated, truncated, folded):
    playback = viewer.Playback.__new__(viewer.Playback)
    playback.obs = {'policy': torch.zeros(1, 134)}
    before = playback.obs['policy']
    returned_obs = {'policy': torch.ones(1, 134)}
    action = torch.zeros(1, 27)
    calls = []

    def get_action(obs, *, deterministic):
        assert obs is before
        assert deterministic is True
        assert not torch.is_grad_enabled()
        calls.append('action')
        return action

    def step(actual_action):
        assert actual_action is action
        calls.append('step')
        return (returned_obs, torch.tensor([.75]), torch.tensor([terminated]), torch.tensor([truncated]),
                {'episode_final': {'done_fold': torch.tensor([folded])}})

    def reset_rnn(*args, **kwargs):
        calls.append('rnn_reset')

    playback.env = NS(step=step)
    playback.player = NS(get_action=get_action, reset_rnn=reset_rnn)
    playback.reward = 0.
    playback.episodes, playback.folds = 4, 2
    playback.step()
    assert playback.obs is returned_obs  # DirectRLEnv already returned the reset observation.
    assert float(playback.reward) == pytest.approx(.75)
    assert playback.episodes == 4 + int(terminated or truncated)
    assert playback.folds == 2 + int(folded)
    assert calls == ['action', 'step'] + (['rnn_reset'] if terminated or truncated else [])


def test_experiment_overrides_preset_but_cli_and_single_env_remain_authoritative(tmp_path):
    recipe = tmp_path / 'viewer.yaml'
    recipe.write_text('''env:
  scene:
    num_envs: 2048
  sim:
    device: cuda:7
  deformable_cloth:
    size: 0.12
    resolution: 6
    start_height: 0.26
  reset:
    reset_position_noise_x: 0.07
''')
    args = viewer.parse_args(['--preset', 'materials', '--experiment', str(recipe), '--device', 'cpu'])
    recipe_cfg = viewer.build_config(args)
    assert recipe_cfg.deformable_cloth.size == pytest.approx(.12)
    assert recipe_cfg.deformable_cloth.resolution == 6
    assert recipe_cfg.deformable_cloth.start_height == pytest.approx(.26)
    assert recipe_cfg.reset.reset_position_noise_x == pytest.approx(.07)
    assert recipe_cfg.scene.num_envs == 1
    assert recipe_cfg.sim.device == 'cpu'
    args = viewer.parse_args(['--experiment', str(recipe), '--size', '.17', '--resolution', '9',
                              '--drop-height', '.04'])
    explicit_cfg = viewer.build_config(args)
    assert explicit_cfg.deformable_cloth.size == pytest.approx(.17)
    assert explicit_cfg.deformable_cloth.resolution == 9
    assert explicit_cfg.deformable_cloth.start_height == pytest.approx(.19)
    assert explicit_cfg.reset.reset_position_noise_x == pytest.approx(.07)
