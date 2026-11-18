"""The robot as the cloth task sees it: joint and body indices, the policy's joint order, joint
limits, the reset pose, and the hand's passive spring.

Two joint orders meet here. The simulator's ("Lab") order is whatever the URDF importer produced;
the policy's ("canonical") order is ``RobotSpec.joint_names_canonical``, the order every checkpoint
was trained in. ``_perm_canon_to_lab`` maps a policy action into the simulator's columns and
``_perm_lab_to_canon`` maps joint state back for the observation.

Functions on the env; the buffers live on it under the names the eval, viewer and probe scripts
read (``_arm_joint_ids``, ``_fingertip_body_ids``, ``_cur_targets``, ``_spring_ks``, ...).
"""

from __future__ import annotations

import torch

__all__ = ["NUM_FINGERTIPS", "init_robot", "reset_robot", "spring_effort", "to_torch"]

NUM_FINGERTIPS = 5


def to_torch(value) -> torch.Tensor:
    """Isaac Lab 3.0 asset data are warp-backed ProxyArrays; Isaac Lab 2.3 already gives torch."""
    return value.torch if hasattr(value, "torch") else value


def init_robot(env) -> None:
    """Cache indices, joint orders and limits, and allocate the joint target buffers."""
    spec = env.robot_spec
    robot = env.robot
    env._arm_joint_ids = robot.find_joints(spec.arm_joint_regex)[0]
    env._hand_joint_ids = robot.find_joints(spec.hand_joint_regex)[0]
    env._palm_body_id = robot.find_bodies(spec.palm_body_name)[0][0]
    # By name, in the spec's order: the order every policy observes its fingertips in. (A regex
    # lookup returns body-index order, which differs between PhysX and Newton.)
    body_names = list(robot.body_names)
    env._fingertip_body_ids = [body_names.index(n) for n in spec.fingertip_link_names]
    assert len(env._fingertip_body_ids) == NUM_FINGERTIPS

    lab_names = list(robot.data.joint_names)
    canonical = spec.joint_names_canonical
    assert set(lab_names) == set(canonical), (
        f"robot {spec.name!r} expects joints {sorted(canonical)}, but "
        f"{env.cfg.assets.robot_urdf} has {sorted(lab_names)}"
    )
    # The action code slices the arm as Lab columns [:n_arm] and the hand as the rest.
    n_arm = len(spec.arm_joint_names)
    assert list(env._arm_joint_ids) == list(range(n_arm)), env._arm_joint_ids
    assert list(env._hand_joint_ids) == list(range(n_arm, len(lab_names))), env._hand_joint_ids
    env._perm_canon_to_lab = torch.tensor([canonical.index(n) for n in lab_names], device=env.device, dtype=torch.long)
    env._perm_lab_to_canon = torch.tensor([lab_names.index(n) for n in canonical], device=env.device, dtype=torch.long)

    limits = robot.data.joint_pos_limits  # (N, J, 2), Lab order
    # Canonical order, to normalise the joint_pos observation; Lab order, to clamp targets.
    env._joint_lower_canon = limits[0, :, 0][env._perm_lab_to_canon]
    env._joint_upper_canon = limits[0, :, 1][env._perm_lab_to_canon]
    env._arm_lower = limits[:, env._arm_joint_ids, 0]
    env._arm_upper = limits[:, env._arm_joint_ids, 1]
    env._hand_lower = limits[:, env._hand_joint_ids, 0]
    env._hand_upper = limits[:, env._hand_joint_ids, 1]

    n_joints = int(env.cfg.action_space)
    env._cur_targets = torch.zeros(env.num_envs, n_joints, device=env.device)
    env._prev_targets = torch.zeros(env.num_envs, n_joints, device=env.device)
    _init_spring(env)


def _init_spring(env) -> None:
    """The hand's passive spring (``RobotSpec.hand_spring_stiffness``) per sim joint:
    ``env._spring_ks`` / ``env._spring_ref`` (J,), ``None`` without a spring. The drives carry the
    PD gains only; :func:`spring_effort` adds ``ks (ref - q)`` as a joint effort."""
    spec = env.robot_spec
    env._spring_ks = env._spring_ref = None
    if not spec.hand_spring_stiffness:
        return
    names = list(env.robot.data.joint_names)
    env._spring_ks = torch.tensor([float(spec.hand_spring_stiffness.get(n, 0.0)) for n in names], device=env.device)
    env._spring_ref = torch.tensor([float(spec.hand_spring_ref.get(n, 0.0)) for n in names], device=env.device)
    hand = env._hand_joint_ids
    kp = to_torch(env.robot.data.joint_stiffness)[0, hand]
    ratio = env._spring_ks[hand] / kp
    print(
        f"[spring] passive hand spring on {int((env._spring_ks > 0).sum())} joints as a joint effort: ks "
        f"{float(env._spring_ks[hand].min()):.4f}-{float(env._spring_ks[hand].max()):.4f} N m/rad, "
        f"ks / kp {float(ratio.min()):.3f}-{float(ratio.max()):.3f}",
        flush=True,
    )


def spring_effort(env) -> torch.Tensor | None:
    """The hand's passive spring as a joint effort, ``ks (ref - q)`` (N, J) in sim joint order, from
    this physics step's ``q``; ``None`` without a spring.

    Applied with ``set_joint_effort_target`` every physics step: an implicit actuator passes the
    effort target through unclipped (PhysX ``set_dof_actuation_forces``, Newton ``joint_f``), so the
    drive's maxForce limits only the PD, as the firmware's current limit does on the real hand.
    """
    if env._spring_ks is None:
        return None
    return env._spring_ks * (env._spring_ref - to_torch(env.robot.data.joint_pos))


def reset_robot(env, env_ids: torch.Tensor) -> None:
    """Joint state at reset: the default pose blended toward a uniform draw within the limits, and a
    uniform joint velocity. The targets start at the reset pose."""
    cfg = env.cfg.reset
    default_pos = env.robot.data.default_joint_pos[env_ids]
    lower = env.robot.data.joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.joint_pos_limits[env_ids, :, 1]

    reset_scale = torch.zeros_like(default_pos)
    reset_scale[:, env._arm_joint_ids] = cfg.reset_dof_pos_random_interval_arm
    reset_scale[:, env._hand_joint_ids] = cfg.reset_dof_pos_random_interval_fingers

    sampled_pos = lower + (upper - lower) * torch.rand_like(default_pos)
    joint_pos = torch.lerp(default_pos, sampled_pos, reset_scale).clamp(lower, upper)
    joint_vel = torch.empty_like(default_pos).uniform_(
        -cfg.reset_dof_vel_random_interval, cfg.reset_dof_vel_random_interval
    )
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env._prev_targets[env_ids] = joint_pos
    env._cur_targets[env_ids] = joint_pos
