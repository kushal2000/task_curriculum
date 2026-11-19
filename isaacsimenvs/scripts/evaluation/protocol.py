"""Settings that define the comparison protocol, applied identically on every backend.

These are not task configuration and not tuning. Each one removes a source of variation that
would otherwise make two backends' numbers incomparable, so they live in one module that both the
PhysX and Newton runs import -- rather than being re-implemented per backend, which is how a
protocol difference creeps in between two experiments.
"""

from __future__ import annotations


def set_present(section, **values) -> None:
    """Set each of ``values`` on the config ``section`` that has that field; skip the rest.

    For the settings one task has and another does not (the Play task's tool pool and goal
    tolerance; the cloth task has neither), so one eval script serves both.
    """
    for name, value in values.items():
        if hasattr(section, name):
            setattr(section, name, value)


def disable_randomization(cfg) -> None:
    """Turn off every observation (proprio, keypoints, point cloud) / action / robot DR and reset-noise
    source, and pin the object's start pose. The cloth task's sheet-physics DR (``domain_randomization.cloth``) stays as the run
    configured it; ``experiments/eval.sh`` pins it for its pinned protocol.

    Domain randomisation is *on* by default in ``PlayEnvCfg`` because that is the
    training regime. For evaluation it has to be off: obs/action delays, object-state noise and
    random wrench impulses all inject variance that has nothing to do with the policy under
    test.
    """
    dr = cfg.domain_randomization
    if hasattr(dr, "proprio"):  # the cloth task: DR by group, each back to its (neutral) defaults
        for group in ("proprio", "keypoints", "pointcloud", "action", "robot"):
            setattr(dr, group, type(getattr(dr, group))())
    else:  # the Play task
        set_present(
            dr,
            use_obs_delay=False,
            use_action_delay=False,
            use_object_state_delay_noise=False,
            object_scale_noise_multiplier_range=(1.0, 1.0),
            joint_velocity_obs_noise_std=0.0,
            joint_pos_obs_noise_std=0.0,
            action_noise_std=0.0,
            object_pose_bias_xyz=(0.0, 0.0, 0.0),
            object_pose_bias_rot_degrees=0.0,
            robot_stiffness_scale_range=(1.0, 1.0),
            robot_damping_scale_range=(1.0, 1.0),
            point_dr_mode="off",
            hand_friction_scale_range=(1.0, 1.0),
            hand_effort_limit_scale_range=(1.0, 1.0),
            # The Play task's random wrench on its tool.
            force_scale=0.0,
            torque_scale=0.0,
            force_prob_range=(0.0001, 0.0001),
            torque_prob_range=(0.0001, 0.0001),
        )

    rs = cfg.reset
    rs.reset_position_noise_x = 0.0
    rs.reset_position_noise_y = 0.0
    rs.reset_position_noise_z = 0.0
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z_range = 0.0
    # Over the table's centre (the Play task's right-hand setup moves the table; the cloth task's
    # reset has no table_reset_xy and keeps the origin).
    table_xy = getattr(rs, "table_reset_xy", (0.0, 0.0))
    rs.fixed_start_pose = (
        table_xy[0],
        table_xy[1],
        rs.table_reset_z + rs.table_object_z_offset,
        1.0,
        0.0,
        0.0,
        0.0,
    )


def use_single_object_variant() -> None:
    """Play task: give every environment the same tool, by truncating each spawn list to its first
    entry. Removes object diversity as a variable when comparing two runs. (The cloth task has no
    tool; its single-USD spawns are unchanged by this.)

    Wraps ``robots.usd.build_rigid_object_cfg``, which the Play scene calls through the module.
    Idempotent.
    """
    import importlib

    robot_usd = importlib.import_module("isaacsimenvs.robots.usd")
    current = robot_usd.build_rigid_object_cfg
    if getattr(current, "_single_variant", False):
        return

    def one_variant(prim_path, usd_paths, _current=current):
        return _current(prim_path, list(usd_paths)[:1])

    one_variant._single_variant = True
    robot_usd.build_rigid_object_cfg = one_variant


__all__ = ["disable_randomization", "set_present", "use_single_object_variant"]
