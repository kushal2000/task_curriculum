"""Observation assembly and step-shared geometry caches for Play."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import convert_quat, quat_apply, quat_from_angle_axis, quat_mul

from ..robots import KUKA_SHARPA
from .point_dr import push_and_read, refit_points, simple_lags


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------


# The Kuka + Sharpa's values (robots.py); the env reads its own robot's from env.robot_spec.
NUM_JOINTS: int = KUKA_SHARPA.num_joints
NUM_FINGERTIPS: int = 5
NUM_KEYPOINTS: int = 4

# Policy was trained against the palm center, not the raw wrist body.
PALM_CENTER_OFFSET: tuple[float, float, float] = KUKA_SHARPA.palm_center_offset

# Shift fingertip body origins to the approximate pad centers.
FINGERTIP_OFFSET: tuple[float, float, float] = KUKA_SHARPA.fingertip_offset

# Where Play puts the robot base (scene_utils default); palm_pos is observed relative to it.
PLAY_ROBOT_ROOT_POS: tuple[float, float, float] = (0.0, 0.8, 0.0)

# Object-frame keypoint corners before scaling.
KEYPOINT_CORNERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (-1, -1, -1),
)

#: Fields with one entry per robot joint; their size follows the robot.
_PER_JOINT_FIELDS = ("joint_pos", "joint_vel", "prev_action_targets")

OBS_FIELD_SIZES: dict[str, int] = {
    "joint_pos": NUM_JOINTS,
    "joint_vel": NUM_JOINTS,
    "prev_action_targets": NUM_JOINTS,
    "palm_pos": 3,
    "palm_rot": 4,
    "palm_vel": 6,
    "object_rot": 4,
    "object_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,  # 15
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,  # 12
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,  # 12
    "object_scales": 3,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS,  # 5
    "lifted_object": 1,
    "progress": 1,
    "successes": 1,
    "reward": 1,
}


def compute_obs_dim(field_list, num_joints: int = NUM_JOINTS) -> int:
    """Return total tensor dim for an ordered list of obs field names."""
    sizes = {**OBS_FIELD_SIZES, **{f: num_joints for f in _PER_JOINT_FIELDS}}
    return sum(sizes[f] for f in field_list)


def _stack_obs_dict(obs_dict: dict[str, torch.Tensor], field_list) -> torch.Tensor:
    """Concatenate named tensors in config order."""
    return torch.cat(
        [obs_dict[f].reshape(obs_dict[f].shape[0], -1) for f in field_list],
        dim=-1,
    )


# ----------------------------------------------------------------------------
# Quaternion / keypoint helpers
# ----------------------------------------------------------------------------


def _perturb_quat(q_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Apply random-axis rotation noise to wxyz quaternions."""
    n = q_wxyz.shape[0]
    axis = torch.nn.functional.normalize(
        torch.randn(n, 3, device=q_wxyz.device), dim=-1
    )
    angle = torch.empty(n, device=q_wxyz.device).uniform_(
        -max_deg, max_deg
    ) * (math.pi / 180.0)
    dq = quat_from_angle_axis(angle, axis)
    return quat_mul(dq, q_wxyz)


def _apply_local_offset(
    pos_w: torch.Tensor,
    rot_wxyz: torch.Tensor,
    offset: tuple[float, float, float],
    batch_shape: tuple[int, ...],
) -> torch.Tensor:
    """Apply one local-frame offset to batched world poses."""
    offset_t = torch.as_tensor(offset, device=pos_w.device, dtype=pos_w.dtype)
    offset_t = offset_t.expand(*batch_shape, 3)
    shifted = quat_apply(
        rot_wxyz.reshape(-1, 4), offset_t.reshape(-1, 3)
    ).reshape(*batch_shape, 3)
    return pos_w + shifted


def _keypoints_world(
    center_pos: torch.Tensor,    # (N, 3)
    center_rot: torch.Tensor,    # (N, 4) wxyz
    kp_offsets: torch.Tensor,    # (N, K, 3)
) -> torch.Tensor:
    """Rotate + translate object-frame keypoints."""
    n_envs, k, _ = kp_offsets.shape
    rot_r = center_rot.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    offsets_r = kp_offsets.reshape(-1, 3)
    return center_pos.unsqueeze(1) + quat_apply(rot_r, offsets_r).reshape(n_envs, k, 3)


def _episode_start(env) -> torch.Tensor:
    return (env.episode_length_buf == 0) & (env._successes == 0)


def _sample_delay(
    queue: torch.Tensor,
    values: torch.Tensor,
    env,
    flush: torch.Tensor | None = None,
    fixed_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push current values into a rolling queue and sample per-env delay.

    ``fixed_idx`` (``delay_mode=per_episode``): read this per-env lag instead of a fresh draw.
    """
    if flush is not None and flush.any():
        queue[flush] = values[flush].unsqueeze(1).expand(-1, queue.shape[1], -1)

    queue = torch.roll(queue, shifts=1, dims=1)
    queue[:, 0, :] = values
    if fixed_idx is not None and env.cfg.domain_randomization.delay_mode == "per_episode":
        idx = fixed_idx
    else:
        idx = torch.randint(0, queue.shape[1], (env.num_envs,), device=env.device)
    delayed = queue[torch.arange(env.num_envs, device=env.device), idx]
    return queue, delayed


def _canonical_joint_obs(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return policy-order joint pos, vel, and previous targets."""
    perm = env._perm_lab_to_canon
    joint_pos_raw = env.robot.data.joint_pos[:, perm]
    joint_pos = (
        2.0 * (joint_pos_raw - env._joint_lower_canon)
        / (env._joint_upper_canon - env._joint_lower_canon)
        - 1.0
    )
    return joint_pos, env.robot.data.joint_vel[:, perm], env._prev_targets[:, perm]


# ----------------------------------------------------------------------------
# Step-shared intermediate values (feeds _get_dones + _get_rewards)
# ----------------------------------------------------------------------------


def compute_intermediate_values(env) -> None:
    """Update shared geometric state for rewards and terminations."""
    from .reward_utils import update_near_goal_steps  # local import to avoid cycle

    rew_cfg = env.cfg.reward
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins

    obj_pos = env.object.data.root_pos_w - env_origins
    obj_rot = env.object.data.root_quat_w
    goal_pos = env.goal_viz.data.root_pos_w - env_origins
    goal_rot = env.goal_viz.data.root_quat_w

    ft_state = env.robot.data.body_state_w[:, env._fingertip_body_ids, :]
    ft_pos_w = ft_state[:, :, 0:3]
    if rew_cfg.fingertip_distance_from_pad:
        ft_pos_w = _apply_local_offset(
            ft_pos_w, ft_state[:, :, 3:7], env.robot_spec.fingertip_offset,
            (env.num_envs, NUM_FINGERTIPS),
        )
    ft_pos = ft_pos_w - env_origins.unsqueeze(1)
    env._curr_fingertip_distances = torch.norm(
        ft_pos - obj_pos.unsqueeze(1), dim=-1
    )  # (N, 5)

    if rew_cfg.fixed_size_keypoint_reward:
        kp_offsets = env._keypoint_offsets_fixed
    else:
        kp_offsets = env._keypoint_offsets

    obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)

    env._keypoints_max_dist = torch.norm(obj_kp - goal_kp, dim=-1).max(dim=-1).values

    # Legacy -1 sentinel: first observed value becomes closest-so-far.
    sentinel = env._closest_keypoint_max_dist < 0.0
    env._closest_keypoint_max_dist = torch.where(
        sentinel, env._keypoints_max_dist, env._closest_keypoint_max_dist
    )
    sentinel_ft = env._closest_fingertip_dist < 0.0
    env._closest_fingertip_dist = torch.where(
        sentinel_ft, env._curr_fingertip_distances, env._closest_fingertip_dist
    )

    if hasattr(env, "_keypoint_success_tolerance_m"):
        tol = env._keypoint_success_tolerance_m()
    else:
        tol = env._current_success_tolerance * rew_cfg.keypoint_scale
    env._near_goal = env._keypoints_max_dist <= tol
    env._near_goal_steps = update_near_goal_steps(
        near_goal=env._near_goal,
        near_goal_steps=env._near_goal_steps,
        force_consecutive=term_cfg.force_consecutive_near_goal_steps,
    )
    env._is_success = env._near_goal_steps >= term_cfg.success_steps


# ----------------------------------------------------------------------------
# Observation builder (Phase D)
# ----------------------------------------------------------------------------


def _apply_object_state_dr(env, obj_pos, obj_rot, obj_linvel, obj_angvel):
    """Apply object-state delay and pose noise."""
    dr = env.cfg.domain_randomization
    state = torch.cat([obj_pos, obj_rot, obj_linvel, obj_angvel], dim=-1)
    env._object_state_queue, delayed = _sample_delay(
        env._object_state_queue, state, env, flush=_episode_start(env),
        fixed_idx=env._object_delay_idx,
    )
    noisy_pos = delayed[:, 0:3] + torch.randn_like(delayed[:, 0:3]) * dr.object_state_xyz_noise_std
    noisy_rot = _perturb_quat(delayed[:, 3:7], dr.object_state_rotation_noise_degrees)
    noisy_vel = delayed[:, 7:13]
    return noisy_pos, noisy_rot, noisy_vel


#: Under point DR, the policy fields use_obs_delay leaves alone: the point-derived ones (they carry
#: the point lag), prev_action_targets (the robot's own command, known instantly on hardware) and
#: the constant object_scales.
_UNDELAYED_IN_POINT_MODE = (
    "object_rot", "object_vel", "keypoints_rel_palm", "keypoints_rel_goal",
    "prev_action_targets", "object_scales",
)


def _apply_point_dr(env, points, rest, goal_pos, goal_rot):
    """Point-level object DR (``point_dr_mode``; see utils/point_dr.py).

    ``points`` (N, K, 3) and the goal pose share one lag; each point gets its own noise and
    per-episode bias; a per-episode common bias moves the points and the goal together. The points
    are then rigidly re-fitted to ``rest`` (N, K, 3), as the clean keypoints are built from a pose.
    Returns ``(keypoints (N, K, 3), object quat wxyz, goal pos, goal quat wxyz)``.
    """
    dr = env.cfg.domain_randomization
    n, k, _ = points.shape
    frame = torch.cat([points.reshape(n, 3 * k), goal_pos, goal_rot], dim=-1)
    start = _episode_start(env)
    if env._point_queue is None:
        env._point_queue = torch.zeros(n, env._point_queue_len, frame.shape[1], device=env.device)
    if dr.point_dr_mode == "simple":
        lags = simple_lags(n, dr.point_delay_range, env.device)
    else:
        lags = env._point_lags.step(start, int(dr.point_delay_jitter), int(dr.point_delay_max))
    env._point_lag_last = lags  # read by probes
    env._point_queue, delayed = push_and_read(env._point_queue, frame, lags, start)

    obs_points = delayed[:, : 3 * k].reshape(n, k, 3) + env._point_bias
    obs_points = obs_points + env._point_common_bias.unsqueeze(1)
    if float(dr.point_noise_std) > 0.0:
        obs_points = obs_points + torch.randn_like(obs_points) * float(dr.point_noise_std)
    kp, quat = refit_points(rest, obs_points)

    obs_goal_pos = delayed[:, 3 * k : 3 * k + 3] + env._point_common_bias
    obs_goal_rot = torch.nn.functional.normalize(delayed[:, 3 * k + 3 : 3 * k + 7], dim=-1)
    return kp, quat, obs_goal_pos, obs_goal_rot


def _observed_object_and_goal(env, obj_pos, obj_rot, obj_linvel, obj_angvel, obj_kp, goal_pos, goal_rot, kp_offsets):
    """The object and goal as the policy observes them, under one of three regimes.

    * ``point_dr_mode`` on: the tracked points and the goal go through the point DR
      (``_apply_point_dr``) and the object pose is re-fitted from the perturbed points;
    * pose-level DR (``use_object_state_delay_noise`` and/or ``object_pose_bias_*``): the object
      pose is delayed, noised and biased, and the keypoints rebuilt from it;
    * neither: the true object pose.

    Returns ``(keypoints (N, K, 3), object quat wxyz, object vel (N, 6), goal pos, goal quat wxyz)``.
    """
    dr = env.cfg.domain_randomization
    obj_vel = torch.cat([obj_linvel, obj_angvel], dim=-1)
    if str(dr.point_dr_mode) != "off":
        tracked = getattr(env, "_tracked_points_w", None)
        points = tracked() - env.scene.env_origins.unsqueeze(1) if tracked is not None else obj_kp
        kp, rot, g_pos, g_rot = _apply_point_dr(env, points, kp_offsets, goal_pos, goal_rot)
        # Same quaternion sign as the clean observation (the fit's sign is arbitrary).
        rot = torch.where((rot * obj_rot).sum(dim=-1, keepdim=True) < 0.0, -rot, rot)
        return kp, rot, obj_vel, g_pos, g_rot  # the point stream carries no velocity: obj_vel unused

    if dr.use_object_state_delay_noise:
        pos, rot, vel = _apply_object_state_dr(env, obj_pos, obj_rot, obj_linvel, obj_angvel)
    else:
        pos, rot, vel = obj_pos, obj_rot, obj_vel
    # Per-episode pose bias (zero / identity unless object_pose_bias_* is set).
    pos = pos + env._object_bias_pos
    rot = quat_mul(env._object_bias_quat, rot)
    return _keypoints_world(pos, rot, kp_offsets), rot, vel, goal_pos, goal_rot


def _delay_proprio_fields(env, obs_noisy: dict, obj_kp: torch.Tensor, palm_frame_offset: torch.Tensor) -> None:
    """Under point DR, ``use_obs_delay`` delays the proprioceptive fields only, in place.

    The point-derived fields already carry the point lag (``_UNDELAYED_IN_POINT_MODE``).
    ``keypoints_rel_palm`` is rebuilt from the DELAYED palm, as on hardware: camera points minus
    the proprio palm the policy reads at the same step. ``palm_frame_offset`` turns the observed
    palm_pos (Play frame) back into the env frame the points are in.
    """
    fields = [f for f in env.cfg.obs.obs_list if f not in _UNDELAYED_IN_POINT_MODE]
    if "palm_pos" not in fields:
        fields.append("palm_pos")
    prop = _stack_obs_dict(obs_noisy, fields)
    if env._proprio_queue is None:
        env._proprio_queue = torch.zeros(env.num_envs, env._obs_queue.shape[1], prop.shape[1], device=env.device)
    env._proprio_queue, prop = _sample_delay(
        env._proprio_queue, prop, env, flush=_episode_start(env), fixed_idx=env._obs_delay_idx
    )
    start = 0
    for f in fields:
        width = obs_noisy[f].reshape(env.num_envs, -1).shape[1]
        obs_noisy[f] = prop[:, start:start + width].reshape(obs_noisy[f].shape)
        start += width
    palm_delayed = obs_noisy["palm_pos"] + palm_frame_offset
    obs_noisy["keypoints_rel_palm"] = obj_kp - palm_delayed.unsqueeze(1)


def _apply_obs_delay(env, policy_tensor: torch.Tensor) -> torch.Tensor:
    """Apply per-env policy-observation delay."""
    env._obs_queue, delayed = _sample_delay(
        env._obs_queue, policy_tensor, env, flush=_episode_start(env),
        fixed_idx=env._obs_delay_idx,
    )
    return delayed


def build_observations(env) -> dict[str, torch.Tensor]:
    """Assemble actor-critic observations with obs-side DR."""
    dr = env.cfg.domain_randomization
    env_origins = env.scene.env_origins

    joint_pos, joint_vel, prev_targets_canon = _canonical_joint_obs(env)

    palm_state = env.robot.data.body_state_w[:, env._palm_body_id, :]  # (N, 13)
    palm_pos_w = palm_state[:, 0:3]
    palm_rot = palm_state[:, 3:7]  # wxyz (Isaac Lab convention)
    palm_vel = palm_state[:, 7:13]

    palm_center_pos_w = _apply_local_offset(
        palm_pos_w, palm_rot, env.robot_spec.palm_center_offset, (env.num_envs,)
    )
    palm_pos = palm_center_pos_w - env_origins
    # Observed as if the robot base were at Play's position: the robot's own frame, plus that
    # offset. Zero unless cfg.robot_root_pos moves the base.
    palm_pos_obs = palm_pos - (
        torch.as_tensor(env.cfg.robot_root_pos, device=palm_pos.device, dtype=palm_pos.dtype)
        - torch.as_tensor(PLAY_ROBOT_ROOT_POS, device=palm_pos.device, dtype=palm_pos.dtype)
    )

    ft_state = env.robot.data.body_state_w[:, env._fingertip_body_ids, :]  # (N, 5, 13)
    ft_body_pos_w = ft_state[:, :, 0:3]
    ft_body_rot_w = ft_state[:, :, 3:7]  # wxyz

    ft_pos_w = _apply_local_offset(
        ft_body_pos_w,
        ft_body_rot_w,
        env.robot_spec.fingertip_offset,
        (env.num_envs, NUM_FINGERTIPS),
    )

    obj_pos = env.object.data.root_pos_w - env_origins
    obj_rot = env.object.data.root_quat_w  # wxyz
    obj_linvel = env.object.data.root_lin_vel_w
    obj_angvel = env.object.data.root_ang_vel_w
    obj_vel = torch.cat([obj_linvel, obj_angvel], dim=-1)

    goal_pos = env.goal_viz.data.root_pos_w - env_origins
    goal_rot = env.goal_viz.data.root_quat_w  # wxyz

    point_mode = str(dr.point_dr_mode) != "off"
    kp_offsets = env._keypoint_offsets * env._object_scale_multiplier.unsqueeze(1)
    obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)
    noisy_obj_kp, noisy_obj_rot, noisy_obj_vel, obs_goal_pos, obs_goal_rot = _observed_object_and_goal(
        env, obj_pos, obj_rot, obj_linvel, obj_angvel, obj_kp, goal_pos, goal_rot, kp_offsets
    )

    # Optional per-env yaw noise on the observed goal (world +Z about goal_pos).
    goal_yaw_obs_noise = getattr(env, "goal_yaw_obs_noise", None)
    if goal_yaw_obs_noise is not None and torch.any(goal_yaw_obs_noise != 0):
        z_axis = torch.tensor(
            [0.0, 0.0, 1.0], device=goal_rot.device, dtype=goal_rot.dtype
        ).unsqueeze(0).expand(env.num_envs, -1)
        obs_goal_rot = quat_mul(quat_from_angle_axis(goal_yaw_obs_noise, z_axis), obs_goal_rot)
        noisy_goal_kp = _keypoints_world(obs_goal_pos, obs_goal_rot, kp_offsets)
    elif point_mode:
        noisy_goal_kp = _keypoints_world(obs_goal_pos, obs_goal_rot, kp_offsets)
    else:
        noisy_goal_kp = goal_kp

    keypoints_rel_palm_clean = obj_kp - palm_pos.unsqueeze(1)
    keypoints_rel_palm_noisy = noisy_obj_kp - palm_pos.unsqueeze(1)
    keypoints_rel_goal_clean = obj_kp - goal_kp
    keypoints_rel_goal_noisy = noisy_obj_kp - noisy_goal_kp

    fingertip_pos_rel_palm = (
        (ft_pos_w - env_origins.unsqueeze(1)) - palm_pos.unsqueeze(1)
    )  # (N, 5, 3)

    object_scales_obs = env._object_scale_per_env * env._object_scale_multiplier

    # Policy obs use legacy Isaac Gym xyzw; internal math stays wxyz.
    palm_rot_xyzw = convert_quat(palm_rot, to="xyzw")
    obj_rot_xyzw = convert_quat(obj_rot, to="xyzw")
    noisy_obj_rot_xyzw = convert_quat(noisy_obj_rot, to="xyzw")

    obs_clean: dict[str, torch.Tensor] = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "prev_action_targets": prev_targets_canon,
        "palm_pos": palm_pos_obs,
        "palm_rot": palm_rot_xyzw,
        "palm_vel": palm_vel,
        "object_rot": obj_rot_xyzw,
        "object_vel": obj_vel,
        "fingertip_pos_rel_palm": fingertip_pos_rel_palm,
        "keypoints_rel_palm": keypoints_rel_palm_clean,
        "keypoints_rel_goal": keypoints_rel_goal_clean,
        "object_scales": object_scales_obs,
        "closest_keypoint_max_dist": env._closest_keypoint_max_dist.unsqueeze(-1),
        "closest_fingertip_dist": env._closest_fingertip_dist,
        "lifted_object": env._lifted_object.float().unsqueeze(-1),
        "progress": torch.log(env.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
        "successes": torch.log(env._successes.float() + 1.0).unsqueeze(-1),
        "reward": (env.reward_buf * 0.01).unsqueeze(-1),
    }

    obs_noisy = dict(obs_clean)
    obs_noisy["object_rot"] = noisy_obj_rot_xyzw
    obs_noisy["object_vel"] = noisy_obj_vel
    obs_noisy["keypoints_rel_palm"] = keypoints_rel_palm_noisy
    obs_noisy["keypoints_rel_goal"] = keypoints_rel_goal_noisy
    if dr.joint_velocity_obs_noise_std > 0:
        obs_noisy["joint_vel"] = (
            joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
        )
    if dr.joint_pos_obs_noise_std > 0:
        # Noise in radians, carried through the [-1, 1] normalisation of _canonical_joint_obs.
        span = env._joint_upper_canon - env._joint_lower_canon
        obs_noisy["joint_pos"] = (
            joint_pos + torch.randn_like(joint_pos) * dr.joint_pos_obs_noise_std * 2.0 / span
        )

    if point_mode and dr.use_obs_delay:
        _delay_proprio_fields(env, obs_noisy, noisy_obj_kp, palm_pos - palm_pos_obs)

    state_tensor = _stack_obs_dict(obs_clean, env.cfg.obs.state_list)
    policy_tensor = _stack_obs_dict(obs_noisy, env.cfg.obs.obs_list)

    if dr.use_obs_delay and not point_mode:
        policy_tensor = _apply_obs_delay(env, policy_tensor)

    clip = env.cfg.obs.clamp_abs_observations
    policy_tensor = policy_tensor.clamp(-clip, clip)
    state_tensor = state_tensor.clamp(-clip, clip)

    return {"policy": policy_tensor, "critic": state_tensor}


def _student_proprio_dict(env) -> dict[str, torch.Tensor]:
    joint_pos, joint_vel, prev_targets_canon = _canonical_joint_obs(env)
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "prev_action_targets": prev_targets_canon,
    }


def _apply_student_tensor_delay(
    env,
    values: torch.Tensor,
    *,
    queue_attr: str,
    delay_max: int,
    enabled: bool,
) -> torch.Tensor:
    if not enabled or delay_max <= 0:
        return values

    flat_values = values.reshape(env.num_envs, -1)
    queue_len = max(1, int(delay_max))
    queue = getattr(env, queue_attr, None)
    expected_shape = (env.num_envs, queue_len, flat_values.shape[-1])
    if (
        queue is None
        or tuple(queue.shape) != expected_shape
        or queue.device != flat_values.device
        or queue.dtype != flat_values.dtype
    ):
        queue = flat_values.unsqueeze(1).expand(-1, queue_len, -1).clone()

    queue, delayed = _sample_delay(
        queue,
        flat_values,
        env,
        flush=_episode_start(env),
    )
    setattr(env, queue_attr, queue)
    return delayed.reshape_as(values)


def _apply_student_bundle_delay(
    env,
    student_obs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cfg = env.cfg.student_obs
    if not cfg.use_student_obs_delay or cfg.student_obs_delay_max <= 0:
        return student_obs

    keys = [key for key in ("image", "proprio") if key in student_obs]
    flat_parts = [student_obs[key].reshape(env.num_envs, -1) for key in keys]
    sizes = [part.shape[-1] for part in flat_parts]
    packed = torch.cat(flat_parts, dim=-1)
    delayed = _apply_student_tensor_delay(
        env,
        packed,
        queue_attr="_student_obs_queue",
        delay_max=int(cfg.student_obs_delay_max),
        enabled=True,
    )

    out = dict(student_obs)
    offset = 0
    for key, size in zip(keys, sizes):
        tensor = student_obs[key]
        out[key] = delayed[:, offset : offset + size].reshape_as(tensor)
        offset += size
    return out


def build_student_observations(env) -> dict[str, torch.Tensor]:
    """Assemble optional image/proprio observations for distillation students."""
    cfg = env.cfg.student_obs
    if not cfg.enabled:
        raise RuntimeError(
            "cfg.student_obs.enabled is false; student observations are disabled."
        )

    proprio_fields = tuple(cfg.proprio_list)
    proprio_dict = _student_proprio_dict(env)
    unsupported = [field for field in proprio_fields if field not in proprio_dict]
    if unsupported:
        raise ValueError(
            f"Unsupported cfg.student_obs.proprio_list fields: {unsupported}. "
            f"Supported fields: {sorted(proprio_dict)}."
        )

    if proprio_fields:
        proprio = torch.cat(
            [proprio_dict[field].reshape(env.num_envs, -1) for field in proprio_fields],
            dim=-1,
        )
    else:
        proprio = torch.empty(env.num_envs, 0, device=env.device)

    student_obs = {"proprio": proprio}
    if cfg.image_enabled:
        from .scene_utils import read_student_camera_image

        image = read_student_camera_image(env)
        student_obs["image"] = _apply_student_tensor_delay(
            env,
            image,
            queue_attr="_student_camera_queue",
            delay_max=int(cfg.camera_delay_max),
            enabled=bool(cfg.use_camera_delay),
        )
    return _apply_student_bundle_delay(env, student_obs)


__all__ = [
    "NUM_JOINTS",
    "NUM_FINGERTIPS",
    "NUM_KEYPOINTS",
    "KEYPOINT_CORNERS",
    "OBS_FIELD_SIZES",
    "compute_obs_dim",
    "compute_intermediate_values",
    "build_observations",
    "build_student_observations",
]
