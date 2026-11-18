"""Observation, action and robot-dynamics domain randomization (``cfg.domain_randomization``).

State lives on the env. :func:`init_dr` allocates it once, :func:`reset_dr` re-draws the
per-episode terms for the envs being reset, in a FIXED order (it sets the random stream):

    fixed delay lags      (``delay_mode: per_episode``)
    object pose bias      (``object_pose_bias_*``; the pose-level object DR)
    point-level object DR (``point_dr_mode``; see ``point_dr.py``)
    robot PD gains        (``robot_{stiffness,damping}_scale_range``)
    hand joint dynamics   (``hand_{friction,effort_limit}_scale_range``; PhysX only)

The per-step terms (delays, noise) are applied where they act: ``actions.py`` and
``observations.py``. The physics DR of the rigid sheet is ``backends/rigid/randomization.py``.
"""

from __future__ import annotations

import math

import torch

from .geometry import quat
from .point_dr import MODES, RealisticLags
from .robot import to_torch

__all__ = [
    "episode_start",
    "init_dr",
    "perturb_quat",
    "reset_dr",
    "rng_parity_goal",
    "rng_parity_reset",
    "rng_parity_step",
]


def episode_start(env) -> torch.Tensor:
    """``(N,)`` bool: this is the episode's first step (the delay lines flush on it)."""
    return (env.episode_length_buf == 0) & (env._successes == 0)


def perturb_quat(q_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Rotate wxyz quaternions by a random axis and a uniform angle in ``+-max_deg``."""
    n = q_wxyz.shape[0]
    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=q_wxyz.device), dim=-1)
    angle = torch.empty(n, device=q_wxyz.device).uniform_(-max_deg, max_deg) * (math.pi / 180.0)
    return quat.quat_mul(quat.quat_from_angle_axis(angle, axis), q_wxyz)


def _log_uniform(lo_hi, n: int) -> torch.Tensor:
    lo, hi = lo_hi
    return torch.exp(torch.empty(n).uniform_(math.log(lo + 1e-12), math.log(hi + 1e-12)))


def init_dr(env) -> None:
    """Allocate every DR buffer. Called once, after the robot (``robot.init_robot``)."""
    dr = env.cfg.domain_randomization
    dev, n = env.device, env.num_envs
    # Per-env factor on the observed sheet size (keypoint offsets and object_scales).
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier = torch.empty(n, 3, device=dev).uniform_(lo, hi)

    env._object_state_queue = torch.zeros(n, max(1, dr.object_state_delay_max), 13, device=dev)
    env._obs_queue = torch.zeros(n, max(1, dr.obs_delay_max), env.cfg.observation_space, device=dev)
    env._action_queue = torch.zeros(n, max(1, dr.action_delay_max), env.cfg.action_space, device=dev)
    # Per-episode draws, neutral until the first reset.
    env._obs_delay_idx = torch.zeros(n, dtype=torch.long, device=dev)
    env._action_delay_idx = torch.zeros(n, dtype=torch.long, device=dev)
    env._object_delay_idx = torch.zeros(n, dtype=torch.long, device=dev)
    env._object_bias_pos = torch.zeros(n, 3, device=dev)
    env._object_bias_quat = torch.zeros(n, 4, device=dev)
    env._object_bias_quat[:, 0] = 1.0
    # Columns: arm stiffness, hand stiffness, arm damping, hand damping.
    env._robot_gain_scale = torch.ones(n, 4, device=dev)
    if tuple(dr.robot_stiffness_scale_range) != (1.0, 1.0) or tuple(dr.robot_damping_scale_range) != (1.0, 1.0):
        # The nominal gains the actuators wrote at startup; every draw multiplies these.
        env._robot_base_stiffness = to_torch(env.robot.data.joint_stiffness).clone()
        env._robot_base_damping = to_torch(env.robot.data.joint_damping).clone()
        env._arm_joint_mask = torch.zeros(env._robot_base_stiffness.shape[1], dtype=torch.bool, device=dev)
        env._arm_joint_mask[env._arm_joint_ids] = True
    _init_hand_dynamics(env)
    _init_point_dr(env)
    _print_summary(env)
    rng_parity_init(env)


def _init_hand_dynamics(env) -> None:
    """Nominal hand joint friction and torque limits for ``hand_{friction,effort_limit}_scale_range``."""
    dr = env.cfg.domain_randomization
    env._hand_friction_dr = tuple(float(v) for v in dr.hand_friction_scale_range) != (1.0, 1.0)
    env._hand_effort_dr = tuple(float(v) for v in dr.hand_effort_limit_scale_range) != (1.0, 1.0)
    if not (env._hand_friction_dr or env._hand_effort_dr):
        return
    if not hasattr(env.robot, "root_physx_view"):
        raise NotImplementedError(
            "hand_friction_scale_range / hand_effort_limit_scale_range are PhysX only (the Newton "
            "articulation has no dynamic joint friction); set them to [1.0, 1.0] here."
        )
    data = env.robot.data
    env._hand_friction_base = data.joint_friction_coeff.clone()
    env._hand_dyn_friction_base = data.joint_dynamic_friction_coeff.clone()
    env._hand_effort_base = data.joint_effort_limits.clone()
    hand = env._hand_joint_ids
    print(
        f"[dr] hand dynamics: friction x{tuple(dr.hand_friction_scale_range)} per env and joint on "
        f"{float(env._hand_friction_base[0, hand].min()):.4f}-{float(env._hand_friction_base[0, hand].max()):.4f} N m, "
        f"torque limit x{tuple(dr.hand_effort_limit_scale_range)} per env on "
        f"{float(env._hand_effort_base[0, hand].min()):.3f}-{float(env._hand_effort_base[0, hand].max()):.3f} N m",
        flush=True,
    )


def _init_point_dr(env) -> None:
    """State for point-level object DR (``point_dr_mode``); validates the config."""
    dr = env.cfg.domain_randomization
    mode = str(dr.point_dr_mode)
    if mode not in MODES:
        raise ValueError(f"domain_randomization.point_dr_mode must be one of {MODES}, got {mode!r}")
    # The proprio-only delay line; also used by the cloth tracker (cloth.tracker), which leaves
    # point_dr_mode off. Allocated on first use, when its width is known.
    env._proprio_queue = None
    if mode == "off":
        return
    pose_dr = (
        dr.use_object_state_delay_noise
        or any(float(v) != 0.0 for v in dr.object_pose_bias_xyz)
        or float(dr.object_pose_bias_rot_degrees) > 0.0
    )
    if pose_dr:
        raise ValueError(
            "point_dr_mode replaces the pose-level object DR: set use_object_state_delay_noise false "
            "and object_pose_bias_xyz / object_pose_bias_rot_degrees to zero."
        )
    if mode == "simple":
        lo, hi = (int(v) for v in dr.point_delay_range)
        if not 0 <= lo <= hi:
            raise ValueError(f"point_delay_range must be 0 <= lo <= hi, got {(lo, hi)}")
        env._point_queue_len = hi + 1
    else:
        lo, hi = (int(v) for v in dr.point_base_delay_range)
        cap = int(dr.point_delay_max)
        if not 0 <= lo <= hi <= cap or int(dr.point_delay_jitter) < 0 or min(dr.point_hold_choices) < 1:
            raise ValueError(
                "realistic point DR needs 0 <= base lo <= hi <= point_delay_max, jitter >= 0, k >= 1; "
                f"got base {(lo, hi)}, max {cap}, jitter {dr.point_delay_jitter}, k {dr.point_hold_choices}"
            )
        env._point_queue_len = cap + 1
        env._point_lags = RealisticLags(env.num_envs, env.device)
    k = len(env._cloth_kp_idx)
    env._point_bias = torch.zeros(env.num_envs, k, 3, device=env.device)
    env._point_common_bias = torch.zeros(env.num_envs, 3, device=env.device)
    env._point_queue = None


def reset_dr(env, env_ids: torch.Tensor) -> None:
    """Clear the delay lines of ``env_ids`` and re-draw every per-episode term (see the module doc)."""
    dr = env.cfg.domain_randomization
    n = env_ids.numel()
    env._action_queue[env_ids] = 0.0
    env._obs_queue[env_ids] = 0.0
    env._object_state_queue[env_ids] = 0.0
    rng_parity_reset(env, n)
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier[env_ids] = torch.empty(n, 3, device=env.device).uniform_(lo, hi)
    _draw_delay_lags(env, env_ids, dr)
    _draw_object_pose_bias(env, env_ids, dr)
    _draw_point_dr(env, env_ids, dr)
    _draw_robot_gains(env, env_ids, dr)
    _draw_hand_dynamics(env, env_ids, dr)


def _symmetric_uniform(n: int, k: int | None, half_xyz, device) -> torch.Tensor:
    """Uniform in ``+-half_xyz`` per axis: ``(n, 3)``, or ``(n, k, 3)`` when ``k`` is given."""
    half = torch.as_tensor(tuple(float(v) for v in half_xyz), device=device, dtype=torch.float32)
    shape = (n, 3) if k is None else (n, k, 3)
    return (torch.rand(*shape, device=device) * 2.0 - 1.0) * half


def _draw_delay_lags(env, env_ids, dr) -> None:
    """``delay_mode: per_episode``: one fixed lag per env for the obs, action and object queues."""
    if dr.delay_mode == "per_step":
        return
    if dr.delay_mode != "per_episode":
        raise ValueError(f"domain_randomization.delay_mode must be per_step or per_episode, got {dr.delay_mode!r}")
    n = env_ids.numel()
    for idx, queue in (
        (env._obs_delay_idx, env._obs_queue),
        (env._action_delay_idx, env._action_queue),
        (env._object_delay_idx, env._object_state_queue),
    ):
        idx[env_ids] = torch.randint(0, queue.shape[1], (n,), device=env.device)


def _draw_object_pose_bias(env, env_ids, dr) -> None:
    """Pose-level object DR: a per-episode position and rotation bias on the observed object."""
    n = env_ids.numel()
    if any(float(v) != 0.0 for v in dr.object_pose_bias_xyz):
        env._object_bias_pos[env_ids] = _symmetric_uniform(n, None, dr.object_pose_bias_xyz, env.device)
    if float(dr.object_pose_bias_rot_degrees) > 0.0:
        ident = torch.zeros(n, 4, device=env.device)
        ident[:, 0] = 1.0
        env._object_bias_quat[env_ids] = perturb_quat(ident, float(dr.object_pose_bias_rot_degrees))


def _draw_point_dr(env, env_ids, dr) -> None:
    """Point-level object DR: per-point and common biases, the camera stream."""
    if str(dr.point_dr_mode) == "off":
        return
    n = env_ids.numel()
    env._point_bias[env_ids] = _symmetric_uniform(n, env._point_bias.shape[1], dr.point_bias_xyz, env.device)
    env._point_common_bias[env_ids] = _symmetric_uniform(n, None, dr.point_common_bias_xyz, env.device)
    if dr.point_dr_mode == "realistic":
        env._point_lags.draw_episode(env_ids, dr.point_base_delay_range, dr.point_hold_choices)


def _draw_robot_gains(env, env_ids, dr) -> None:
    """PD gain DR: one stiffness and one damping factor per env, arm and hand separately."""
    k_rng = tuple(float(v) for v in dr.robot_stiffness_scale_range)
    d_rng = tuple(float(v) for v in dr.robot_damping_scale_range)
    if k_rng == (1.0, 1.0) and d_rng == (1.0, 1.0):
        return
    scale = torch.empty(env_ids.numel(), 4, device=env.device)  # arm k, hand k, arm d, hand d
    scale[:, 0:2].uniform_(*k_rng)
    scale[:, 2:4].uniform_(*d_rng)
    env._robot_gain_scale[env_ids] = scale
    arm = env._arm_joint_mask.unsqueeze(0)
    k = env._robot_base_stiffness[env_ids] * torch.where(arm, scale[:, 0:1], scale[:, 1:2])
    d = env._robot_base_damping[env_ids] * torch.where(arm, scale[:, 2:3], scale[:, 3:4])
    # The plain (not `_index`) writers: the Isaac Lab in the PhysX venv predates the `_index` API.
    env.robot.write_joint_stiffness_to_sim(k.contiguous(), env_ids=env_ids)
    env.robot.write_joint_damping_to_sim(d.contiguous(), env_ids=env_ids)


def _draw_hand_dynamics(env, env_ids, dr) -> None:
    """Hand sysid DR (PhysX): Coulomb friction per env and joint, torque limit per env."""
    n, hand = env_ids.numel(), env._hand_joint_ids
    if env._hand_friction_dr:
        f = torch.empty(n, len(hand), device=env.device).uniform_(*(float(v) for v in dr.hand_friction_scale_range))
        static = env._hand_friction_base[env_ids].clone()
        dynamic = env._hand_dyn_friction_base[env_ids].clone()
        static[:, hand] *= f
        dynamic[:, hand] *= f  # the same factor, so dynamic <= static still holds
        env.robot.write_joint_friction_coefficient_to_sim(static, joint_dynamic_friction_coeff=dynamic, env_ids=env_ids)
    if env._hand_effort_dr:
        e = torch.empty(n, 1, device=env.device).uniform_(*(float(v) for v in dr.hand_effort_limit_scale_range))
        limit = env._hand_effort_base[env_ids].clone()
        limit[:, hand] *= e
        env.robot.write_joint_effort_limit_to_sim(limit, env_ids=env_ids)


def _print_summary(env) -> None:
    """One log line with every observation / action DR term, so a run's log states its regime."""
    dr = env.cfg.domain_randomization
    print(
        "[dr] obs delay "
        f"{'0-' + str(dr.obs_delay_max - 1) if dr.use_obs_delay else 'off'}, action delay "
        f"{'0-' + str(dr.action_delay_max - 1) if dr.use_action_delay else 'off'}, object state "
        + (
            f"delay 0-{dr.object_state_delay_max - 1} + noise {dr.object_state_xyz_noise_std} m / "
            f"{dr.object_state_rotation_noise_degrees} deg"
            if dr.use_object_state_delay_noise else "off"
        )
        + f" | delay_mode {dr.delay_mode} | joint vel noise {dr.joint_velocity_obs_noise_std}, "
        f"joint pos noise {dr.joint_pos_obs_noise_std} rad, action noise {dr.action_noise_std} | "
        f"object bias +-{tuple(dr.object_pose_bias_xyz)} m / {dr.object_pose_bias_rot_degrees} deg | "
        f"robot gains k x{tuple(dr.robot_stiffness_scale_range)} d x{tuple(dr.robot_damping_scale_range)}",
        flush=True,
    )
    if str(dr.point_dr_mode) != "off":
        lag = (
            f"lag {tuple(dr.point_delay_range)} per step"
            if dr.point_dr_mode == "simple"
            else f"base {tuple(dr.point_base_delay_range)} +-{dr.point_delay_jitter}, max "
            f"{dr.point_delay_max}, frame every k in {tuple(dr.point_hold_choices)}"
        )
        print(
            f"[dr] point DR {dr.point_dr_mode}: {lag} | noise {dr.point_noise_std} m, bias "
            f"+-{tuple(dr.point_bias_xyz)} m per point, common +-{tuple(dr.point_common_bias_xyz)} m "
            f"(points + goal) | obs delay on proprio only",
            flush=True,
        )


# ---------------------------------------------------------------------------------------------------
# RNG parity with the Play task (temporary). The Play task drew random numbers the cloth never uses:
# a random wrench on the object every step (inert on a sheet), a wrench probability per episode, and
# a goal pose per reset (overwritten by the fold target). Drawing them here keeps the random stream,
# and so every fixed-seed run, bit-identical to the Play-based env while the code moves over.
# ---------------------------------------------------------------------------------------------------


def rng_parity_init(env) -> None:
    dr = env.cfg.domain_randomization
    _log_uniform(dr.force_prob_range, env.num_envs)
    _log_uniform(dr.torque_prob_range, env.num_envs)


def rng_parity_reset(env, n: int) -> None:
    dr = env.cfg.domain_randomization
    _log_uniform(dr.force_prob_range, n)
    _log_uniform(dr.torque_prob_range, n)


def rng_parity_step(env) -> None:
    n, dev = env.num_envs, env.device
    torch.rand(n, device=dev)
    torch.rand(n, device=dev)
    torch.randn(n, 1, 3, device=dev)
    torch.randn(n, 1, 3, device=dev)


def rng_parity_goal(env, n: int, mode: str) -> None:
    """The draws of the Play goal sampler: ``absolute`` at every reset, ``delta`` (the
    configured ``goal_sampling_type``) at every Play goal hit."""
    if env.cfg.reset.fixed_goal_pose is not None:
        return
    dev = env.device
    if mode == "absolute":
        torch.rand(n, 3, device=dev)
        quat.random_orientation(n, device=dev)
    elif mode == "delta":
        torch.rand(n, 3, device=dev)
        torch.randn(n, 3, device=dev)
        torch.rand(n, device=dev)
    else:
        raise ValueError(f"unknown goal sampling mode: {mode}")
