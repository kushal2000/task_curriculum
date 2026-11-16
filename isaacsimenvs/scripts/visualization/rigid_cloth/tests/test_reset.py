"""Reset geometry and shared training-recipe inheritance without a simulator."""
from types import SimpleNamespace as NS

import pytest
import torch

from isaacsimenvs.scripts.visualization.rigid_cloth.reset import (
    load_training_env_config, sample_reset_pose, validate_reset_settings,
)


def settings(**overrides):
    values = dict(x=.03, y=-.04, yaw_deg=37., bend_deg=0., drop_height=.002,
                  sample_reset=True, x_range=(-.3, .3), y_range=(-.25, .25), yaw_range_deg=(-70., 150.))
    return NS(**(values | overrides))


def test_manual_pose_preserves_nonsquare_cloth_xy_and_yaw_without_rng_draws():
    dims = torch.tensor([[.2, .1, .004], [.12, .24, .008]], dtype=torch.float64)
    torch.manual_seed(17)
    rng = torch.random.get_rng_state().clone()
    xy, yaw = sample_reset_pose(settings(sample_reset=False), dims)
    assert torch.equal(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(xy, torch.tensor([[.03, -.04]] * 2, dtype=torch.float64))
    torch.testing.assert_close(yaw, torch.deg2rad(torch.tensor([37., 37.], dtype=torch.float64)))


def test_sampled_yawed_nonsquare_bounds_and_fresh_poses_on_every_call():
    dims = torch.tensor([[.2, .1, .004], [.12, .24, .008]]).repeat(128, 1)
    cfg = settings()
    torch.manual_seed(29)
    xy, yaw = sample_reset_pose(cfg, dims)
    c, s = yaw.cos().abs(), yaw.sin().abs()
    extent = torch.stack((dims[:, 0] * c + dims[:, 1] * s,
                          dims[:, 0] * s + dims[:, 1] * c), dim=-1) / 2
    lo, hi = torch.tensor([-.3, -.25]), torch.tensor([.3, .25])
    assert bool((xy - extent >= lo - 1e-7).all())
    assert bool((xy + extent <= hi + 1e-7).all())
    assert bool(((torch.rad2deg(yaw) >= -70) & (torch.rad2deg(yaw) <= 150)).all())
    next_xy, next_yaw = sample_reset_pose(cfg, dims)
    assert not torch.equal(xy, next_xy) and not torch.equal(yaw, next_yaw)


def test_impossible_region_and_fixed_heading_fail_instead_of_spawning_overhang():
    dims = torch.tensor([[.3, .2, .004]])
    for cfg in (settings(x_range=(.1, .1)),
                settings(y_range=(-.1, .1), yaw_range_deg=(90., 90.))):
        with pytest.raises(ValueError, match="does not fit"):
            sample_reset_pose(cfg, dims)


@pytest.mark.parametrize("table_size", [(.475, .4, .04), (.6, .4, .3)])
def test_actual_table_clips_recipe_region_and_conditions_yaw_on_fit(table_size):
    cfg = settings(x_range=(-.29, .29), y_range=(-.18, .19), yaw_range_deg=(0., 360.))
    dims = torch.tensor([[.3, .3, .004], [.2, .1, .004]]).repeat(256, 1)
    xy, yaw = sample_reset_pose(cfg, dims, table_size=table_size)
    c, s = yaw.cos().abs(), yaw.sin().abs()
    half = torch.stack((dims[:, 0] * c + dims[:, 1] * s,
                        dims[:, 0] * s + dims[:, 1] * c), dim=-1) / 2
    table_half = torch.tensor(table_size[:2]) / 2 - .005
    assert bool((xy.abs() + half <= table_half + 1e-7).all())
    assert bool((xy - half >= torch.tensor([-.29, -.18]) - 1e-7).all())
    assert bool((xy + half <= torch.tensor([.29, .19]) + 1e-7).all())


def test_narrow_feasible_angle_range_is_sampled_without_rejection_failure():
    cfg = settings(x_range=(-.20001, .20001), y_range=(-.05001, .05001), yaw_range_deg=(0., 360.))
    dims = torch.tensor([[.4, .1, .004]]).repeat(100, 1)
    xy, yaw = sample_reset_pose(cfg, dims)
    assert bool((yaw.sin().abs() < 1e-4).all())
    rng = torch.random.get_rng_state().clone()
    validate_reset_settings(cfg, dims)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("change", [dict(x=float('nan')), dict(drop_height=-.1),
    dict(x_range=(1., -1.)), dict(y_range=(0., float('inf'))), dict(yaw_range_deg=(0., 361.))])
def test_invalid_reset_settings_fail_before_sampling(change):
    with pytest.raises(ValueError):
        validate_reset_settings(settings(**change))


def test_recipe_inheritance_keeps_sections_and_does_not_mutate_future_reads(tmp_path):
    (tmp_path / 'base.yaml').write_text('env:\n  assets:\n    robot_friction: 0.2\n    finger_tip_friction: 0.6\n  reset:\n    random_yaw: true\n')
    child = tmp_path / 'child.yaml'
    child.write_text('base: base\nenv:\n  assets:\n    robot_friction: 0.3\n')
    first = load_training_env_config(child)
    assert first['assets'] == dict(robot_friction=.3, finger_tip_friction=.6)
    assert first['reset']['random_yaw'] is True
    first['assets']['robot_friction'] = 9
    assert load_training_env_config(child)['assets']['robot_friction'] == .3
    (tmp_path / 'base.yaml').write_text('base: child\n')
    with pytest.raises(ValueError, match='cycle'):
        load_training_env_config(child)


def test_default_recipe_all_sizes_fit_training_table_with_full_yaw_range():
    from isaacsimenvs.scripts.visualization.viser_rigidCloth_viewer import Settings
    from isaacsimenvs.tasks.cloth.backends.rigid.designs import design_grid
    recipe = load_training_env_config()
    cloth, reset = recipe['rigid_cloth'], recipe['reset']
    assert reset['yaw_range_deg'] == [0., 360.]
    dims = torch.tensor(design_grid(cloth['design_levels'], cloth['length_range'],
                                   cloth['width_range'], cloth['thickness_range']), dtype=torch.float32)
    selected = Settings(sample_reset=True, x_range=tuple(cloth['place_x']),
                        y_range=tuple(cloth['place_y']), yaw_range_deg=tuple(reset['yaw_range_deg']))
    xy, yaw = sample_reset_pose(selected, dims, table_size=(.475, .4, .04))
    c, s = yaw.cos().abs(), yaw.sin().abs()
    extent = torch.stack((dims[:, 0]*c + dims[:, 1]*s, dims[:, 0]*s + dims[:, 1]*c), -1) / 2
    lower = dims.new_tensor([cloth['place_x'][0], cloth['place_y'][0]])
    upper = dims.new_tensor([cloth['place_x'][1], cloth['place_y'][1]])
    assert torch.all(xy - extent >= lower - 1e-6)
    assert torch.all(xy + extent <= upper + 1e-6)
    # Feasibility conditioning must not impose a narrow common heading interval.
    assert yaw.min() < .2 and yaw.max() > 6.1


def test_static_and_policy_setups_share_nominal_table_centers():
    import yaml
    from isaacsimenvs import REPO_ROOT
    from isaacsimenvs.scripts.visualization.rigid_cloth.policy import robot_profile, table_position
    from isaacsimenvs.tasks.cloth.backends.assets import urdf_box_size
    source = yaml.safe_load((REPO_ROOT / 'pretrained_policy/flexiv_wuji/config.yaml').read_text())['env']
    for profile, size, center, root in (
        ('training', (.475, .4, .04), (0., .29, .26), (0., .8, 0.)),
        ('real', (.6, .4, .3), (0., 0., .38), (0., .51, .2)),
    ):
        _, _, table, actual_root = robot_profile(profile)
        assert actual_root == root
        assert urdf_box_size(str(REPO_ROOT / table)) == size
        assert table_position(profile, source) == center
