"""Viewer policy boundary tests; no simulator or checkpoint loading required."""
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaacsimenvs import REPO_ROOT
from isaacsimenvs.scripts.visualization.rigid_cloth.policy import (
    POLICY_FIELDS, PolicyPhysics, robot_profile, validate_policy_config,
)
from isaacsimenvs.scripts.visualization.viser_rigidCloth_viewer import Settings
from isaacsimenvs.tasks.cloth.backends.assets import urdf_box_size
from isaacsimenvs.tasks.cloth.backends.robot import ROBOT, RobotSpec
from isaacsimenvs.tasks.play.robots import FLEXIV_WUJI_RIGHT_HUMAN_LIKE


def test_profiles_preserve_complete_dynamics_and_scene_geometry():
    training, robot, table, root = robot_profile("training")
    for field in fields(RobotSpec):
        assert getattr(training, field.name) == getattr(FLEXIV_WUJI_RIGHT_HUMAN_LIKE, field.name)
    assert training.hand_friction == 0.0
    assert training.num_joints == 27
    assert root == (0.0, 0.8, 0.0)
    assert (REPO_ROOT / robot).is_file()
    assert urdf_box_size(str(REPO_ROOT / table)) == (0.475, 0.4, 0.04)
    real, robot, table, root = robot_profile("real")
    assert real is ROBOT
    assert real.hand_spring_stiffness
    assert root == (0.0, 0.51, 0.20)
    assert (REPO_ROOT / robot).is_file()
    assert urdf_box_size(str(REPO_ROOT / table)) == (0.6, 0.4, 0.3)


def test_yibo_profile_has_historical_dynamics_with_real_limits_and_table_height():
    from isaacsimenvs.tasks.cloth.backends.robot import YIBO_ROBOT, get_robot_spec
    spec, urdf, table, root = robot_profile('yibo')
    assert spec is YIBO_ROBOT is get_robot_spec('yibo')
    for field in fields(RobotSpec):
        current, historical = getattr(spec, field.name), getattr(FLEXIV_WUJI_RIGHT_HUMAN_LIKE, field.name)
        if field.name == 'self_collision_filter':
            current = {name: set(others) for name, others in current.items()}
            historical = {name: set(others) for name, others in historical.items()}
        assert current == historical, field.name
    assert urdf.endswith('real_limits.urdf')
    from isaacsimenvs.scripts.visualization.rigid_cloth.policy import table_position
    height = table_position('yibo', {})[2] + urdf_box_size(str(REPO_ROOT / table))[2] / 2 - root[2]
    assert height == pytest.approx(.33)


def test_config_rejects_same_width_reordered_observations_and_other_robots():
    saved = {"env": {"robot": "flexiv_wuji_right_human_like", "obs": {"obs_list": POLICY_FIELDS}}}
    validate_policy_config(saved)
    saved["env"]["obs"]["obs_list"] = tuple(reversed(POLICY_FIELDS))
    with pytest.raises(ValueError, match="observation order"):
        validate_policy_config(saved)
    saved["env"]["robot"] = "kuka_sharpa"
    with pytest.raises(ValueError, match="flexiv_wuji_right_human_like"):
        validate_policy_config(saved)
    with pytest.raises(ValueError, match="profile"):
        robot_profile("unknown")


@pytest.mark.parametrize("terminated,truncated", [(False, False), (True, False), (False, True)])
def test_one_policy_call_per_environment_step_and_done_clears_lstm(terminated, truncated):
    physics = PolicyPhysics.__new__(PolicyPhysics)
    initial = {"policy": torch.ones(1, 134)}
    reset_obs = {"policy": torch.full((1, 134), 2.0)}
    action = torch.zeros(1, 27)
    physics.player = Mock()
    def infer(*args, **kwargs):
        assert not torch.is_grad_enabled()
        # UI coefficient changes reset LSTM state outside the stepping context.
        assert not torch.is_inference_mode_enabled()
        return action

    physics.player.get_action.side_effect = infer
    physics.env = Mock()
    physics.env.step.return_value = (
        reset_obs, torch.tensor([3.0]), torch.tensor([terminated]), torch.tensor([truncated]),
        {"episode_final": {"successes": torch.tensor([1.0])}},
    )
    physics.obs, physics.dt, physics.time = initial, 1 / 60, 0.0
    physics.episodes = physics.successes = 0
    physics.deterministic = False
    physics.step()
    physics.player.get_action.assert_called_once_with(initial["policy"], deterministic=False)
    physics.env.step.assert_called_once_with(action)
    assert physics.obs is reset_obs
    assert physics.reward == 3.0
    done = terminated or truncated
    assert physics.episodes == physics.successes == int(done)
    assert physics.time == (0.0 if done else 1 / 60)
    assert physics.player.reset_rnn.call_count == int(done)


def test_manual_reset_uses_returned_observation_and_retains_player():
    physics = PolicyPhysics.__new__(PolicyPhysics)
    physics.env = Mock(cfg=SimpleNamespace(rigid_cloth=SimpleNamespace(), reset=SimpleNamespace()),
                       backend=SimpleNamespace(_dims=torch.tensor([[.2, .2, .004]])))
    physics.table_size = (.6, .4, .3)
    obs = {"policy": torch.zeros(1, 134)}
    physics.env.reset.return_value = (obs, {})
    physics.player = Mock()
    player = physics.player
    physics.update_material = Mock()
    settings = Settings(bend_deg=37.0, x=0.03, y=-0.02, drop_height=0.01)
    physics.reset(settings)
    assert physics.player is player
    player.reset_rnn.assert_called_once()
    physics.env.reset.assert_called_once()
    assert physics.obs is obs
    assert physics.env.viewer_settings == settings
    assert physics.env.viewer_settings is not settings
    assert physics.env.cfg.rigid_cloth.spawn_clearance == 0.01
    assert physics.time == physics.reward == 0.0



@pytest.mark.parametrize('sample_dr', [False, True])
def test_training_recipe_is_applied_before_viewer_and_checkpoint_overrides(monkeypatch, tmp_path, sample_dr):
    import sys
    import types
    import yaml
    from isaacsimenvs.scripts.visualization.rigid_cloth.policy import build_policy_env_cfg

    class Node(SimpleNamespace):
        def from_dict(self, values):
            for key, value in values.items():
                if isinstance(value, dict):
                    child = getattr(self, key, Node())
                    child.from_dict(value)
                    setattr(self, key, child)
                else:
                    setattr(self, key, value)

    def make_cfg(_):
        return Node(scene=Node(), sim=Node(), assets=Node(), reset=Node(), action=Node(), obs=Node(),
                    reward=Node(), task=Node(), domain_randomization=Node(), rigid_cloth=Node(randomization=Node()))

    module = types.ModuleType('isaacsimenvs.tasks.cloth.cloth_cfg')
    module.ClothCfg = SimpleNamespace(for_backend=make_cfg)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(yaml.safe_dump({"env": {
        "scene": {"num_envs": 999},
        "assets": {"robot_friction": .2, "finger_tip_friction": .6},
        "task": {"keypoint_tolerance": .012}, "obs": {"keypoint_scale": .7},
        "domain_randomization": {"use_obs_delay": True, "use_action_delay": True,
                                 "use_object_state_delay_noise": True},
    }}))
    saved = yaml.safe_load((REPO_ROOT / 'pretrained_policy/flexiv_wuji/config.yaml').read_text())
    cfg, spec = build_policy_env_cfg(Settings(keypoint_mode='planar', sample_material=sample_dr,
                                             sample_sensor=sample_dr), saved, 'training', 'cpu', recipe)
    assert cfg.scene.num_envs == 1 and cfg.sim.device == 'cpu'
    assert cfg.assets.robot_friction == .2 and cfg.assets.finger_tip_friction == .6
    assert cfg.task.keypoint_tolerance == .012
    assert cfg.obs.keypoint_scale == saved['env']['reward']['keypoint_scale']
    assert cfg.obs.keypoint_mode == 'planar'
    assert cfg.assets.robot_urdf.endswith('rizon4s_right_wuji_human_like_joint_limits.urdf')
    assert cfg.robot_root_pos == (0., .8, 0.)
    assert cfg.reset.table_reset_xy == (0., .29)
    assert cfg.rigid_cloth.randomization.enabled is sample_dr
    assert cfg.domain_randomization.use_obs_delay is sample_dr
    assert cfg.domain_randomization.use_action_delay is sample_dr
    assert cfg.domain_randomization.use_object_state_delay_noise is sample_dr


def test_yibo_our_dr_retains_our_cloth_recipe_but_fixes_table_and_uses_actual_thickness():
    from isaacsimenvs import CONFIG_DIR
    from isaacsimenvs.scripts.visualization.rigid_cloth.reset import load_training_env_config
    base = load_training_env_config()
    hybrid = load_training_env_config(CONFIG_DIR / 'runs/yibo_our_dr.yaml')
    assert hybrid['rigid_cloth'] == base['rigid_cloth']
    assert hybrid['robot_preset'] == 'yibo'
    assert hybrid['reset']['table_reset_z'] == .38
    assert hybrid['reset']['table_reset_z_range'] == 0.
    assert hybrid['obs']['yibo_contact_thickness'] is None
    assert all(hybrid['domain_randomization'][name] for name in (
        'use_obs_delay', 'use_action_delay', 'use_object_state_delay_noise'))


def test_live_materials_apply_effective_static_dynamic_friction_and_geometry_gains():
    from isaacsimenvs.tasks.cloth.backends.rigid.designs import get_design
    physics = PolicyPhysics.__new__(PolicyPhysics)
    physics.env = SimpleNamespace(cfg=SimpleNamespace(rigid_cloth=SimpleNamespace(), assets=SimpleNamespace()),
                                  table=Mock(), robot=Mock())
    physics.cloth = Mock(device='cpu')
    physics.spec = get_design('single_hinge').resolve(.2, .2, .002)
    physics.joint_ids = [0]
    for asset, shapes in ((physics.cloth, 2), (physics.env.table, 1), (physics.env.robot, 3)):
        asset.root_physx_view.get_material_properties.return_value = torch.ones(1, shapes, 3)
    settings = Settings(table_static_friction=.2, hand_static_friction=.9, dynamic_friction_ratio=.75)
    for _ in range(2):
        physics.update_material(settings)
        for asset, expected in ((physics.cloth, [1., 1., 0.]),
                                (physics.env.table, [.2, .15, 0.]),
                                (physics.env.robot, [.9, .675, 0.])):
            written = asset.root_physx_view.set_material_properties.call_args.args[0]
            torch.testing.assert_close(written, torch.tensor(expected).expand_as(written))
        torch.testing.assert_close(physics.cloth.write_joint_stiffness_to_sim.call_args.args[0],
                                  torch.tensor([[2 * settings.bending_rigidity]]))
        torch.testing.assert_close(physics.cloth.write_joint_damping_to_sim.call_args.args[0],
                                  torch.tensor([[2 * settings.bending_viscosity]]))


def test_sampled_materials_survive_viewer_reset_material_hook():
    physics = PolicyPhysics.__new__(PolicyPhysics)
    physics.env = SimpleNamespace(cfg=SimpleNamespace(rigid_cloth=SimpleNamespace(
        randomization=SimpleNamespace(enabled=True))))
    physics.cloth = Mock()
    # Called by every viewer reset. Yibo's startup friction draw must survive;
    # material-mode draws are likewise owned by backend randomization.
    physics.update_material(Settings(sample_material=True))
    assert physics.cloth.mock_calls == []
