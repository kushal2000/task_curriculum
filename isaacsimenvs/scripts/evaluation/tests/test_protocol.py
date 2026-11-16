"""Evaluation disables the same shared noise without changing task-specific semantics."""

from types import SimpleNamespace as NS

import pytest

from isaacsimenvs.scripts.evaluation.protocol import disable_randomization


@pytest.mark.parametrize("kind", ["play", "rigid", "vbd"])
def test_pinned_evaluation_protocol(kind):
    cfg = NS(
        domain_randomization=NS(), reset=NS(table_reset_xy=(0.2, 0.3), table_reset_z=0.4, table_object_z_offset=0.1)
    )
    if kind != "play":
        cfg.task = NS()
    if kind == "rigid":
        cfg.rigid_cloth = NS(randomization=NS(enabled=True))
    disable_randomization(cfg)
    assert not cfg.domain_randomization.use_obs_delay
    assert not cfg.domain_randomization.use_action_delay
    assert not cfg.domain_randomization.use_object_state_delay_noise
    assert cfg.domain_randomization.action_noise_std == 0
    assert cfg.reset.reset_dof_vel_random_interval == 0
    assert cfg.reset.table_reset_z_range == 0
    if kind == "play":
        assert cfg.reset.fixed_start_pose == (0.2, 0.3, 0.5, 1.0, 0.0, 0.0, 0.0)
        assert cfg.domain_randomization.force_scale == 0
    else:
        assert not cfg.reset.random_yaw
        assert not hasattr(cfg.reset, "fixed_start_pose")
    if kind == "rigid":
        assert not cfg.rigid_cloth.randomization.enabled
