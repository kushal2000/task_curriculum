"""The policy and critic observations.

Named fields, concatenated in the order of ``cfg.obs.obs_list`` (policy) and ``cfg.obs.state_list``
(critic). The encoding is the one the Play prior was trained on, so every cloth checkpoint reads the
same vector:

* joint state in the policy's (canonical) joint order, ``joint_pos`` normalised to [-1, 1] over the
  joint limits;
* ``palm_pos`` as if the robot base stood at the Play task's position (``PLAY_ROBOT_ROOT_POS``);
* quaternions in (x, y, z, w), the Isaac Gym order (internally everything is (w, x, y, z));
* the sheet as a rigid object: ``keypoints_rel_*`` are four keypoints built as
  ``centre + R * offsets``, from the moving half's fitted pose and the fold goal's pose
  (``env.object`` and ``env.goal_viz``; see ``mdp/task.py``).

Observation DR, at most one object regime at a time (:func:`_observed_object_and_goal`): the cloth
tracker (``cloth.tracker``), point-level DR (``point_dr_mode``), or pose-level DR (object-state
delay and noise, a per-episode pose bias). On top: joint noise, and a policy delay line
(``use_obs_delay``; under point DR it delays the proprioceptive fields only). The critic always
reads the clean values.
"""

from __future__ import annotations

import torch

from .geometry import quat
from .point_dr import push_and_read, refit_points, simple_lags
from .randomization import episode_start, perturb_quat
from .robot import NUM_FINGERTIPS

__all__ = [
    "FIELD_SIZES",
    "NUM_KEYPOINTS",
    "PLAY_ROBOT_ROOT_POS",
    "apply_local_offset",
    "build_observations",
    "guard_finite",
    "keypoints_world",
    "obs_dim",
]

NUM_KEYPOINTS = 4

#: Where the Play task puts the robot base; ``palm_pos`` is observed relative to it.
PLAY_ROBOT_ROOT_POS: tuple[float, float, float] = (0.0, 0.8, 0.0)

#: Fields with one entry per robot joint; their size follows the robot.
_PER_JOINT_FIELDS = ("joint_pos", "joint_vel", "prev_action_targets")

FIELD_SIZES: dict[str, int] = {
    "palm_pos": 3,
    "palm_rot": 4,
    "palm_vel": 6,
    "object_rot": 4,
    "object_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,
    "object_scales": 3,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS,
    "lifted_object": 1,
    "progress": 1,
    "successes": 1,
    "reward": 1,
}

#: Under point DR, the policy fields ``use_obs_delay`` leaves alone: the point-derived ones (they
#: carry the point lag), ``prev_action_targets`` (the robot's own command, known instantly on
#: hardware) and the constant ``object_scales``.
_UNDELAYED_IN_POINT_MODE = (
    "object_rot", "object_vel", "keypoints_rel_palm", "keypoints_rel_goal",
    "prev_action_targets", "object_scales",
)


def obs_dim(fields, num_joints: int) -> int:
    """Width of the vector the ordered ``fields`` make for a robot with ``num_joints`` joints."""
    sizes = {**FIELD_SIZES, **{f: num_joints for f in _PER_JOINT_FIELDS}}
    return sum(sizes[f] for f in fields)


def _stack(obs: dict[str, torch.Tensor], fields) -> torch.Tensor:
    return torch.cat([obs[f].reshape(obs[f].shape[0], -1) for f in fields], dim=-1)


# ---------------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------------


def apply_local_offset(pos_w, rot_wxyz, offset, batch_shape: tuple[int, ...]) -> torch.Tensor:
    """Shift batched world positions by one offset given in each pose's own frame."""
    offset_t = torch.as_tensor(offset, device=pos_w.device, dtype=pos_w.dtype).expand(*batch_shape, 3)
    shifted = quat.quat_apply(rot_wxyz.reshape(-1, 4), offset_t.reshape(-1, 3)).reshape(*batch_shape, 3)
    return pos_w + shifted


def keypoints_world(center_pos: torch.Tensor, center_rot: torch.Tensor, kp_offsets: torch.Tensor) -> torch.Tensor:
    """``centre + R * offsets``: (N, 3), (N, 4) wxyz, (N, K, 3) -> (N, K, 3)."""
    n, k, _ = kp_offsets.shape
    rot = center_rot.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    return center_pos.unsqueeze(1) + quat.quat_apply(rot, kp_offsets.reshape(-1, 3)).reshape(n, k, 3)


def _canonical_joint_obs(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Policy-order joint position (normalised), velocity, and previous targets."""
    perm = env._perm_lab_to_canon
    joint_pos_raw = env.robot.data.joint_pos[:, perm]
    joint_pos = 2.0 * (joint_pos_raw - env._joint_lower_canon) / (env._joint_upper_canon - env._joint_lower_canon) - 1.0
    return joint_pos, env.robot.data.joint_vel[:, perm], env._prev_targets[:, perm]


# ---------------------------------------------------------------------------------------------------
# Delay lines and observation DR
# ---------------------------------------------------------------------------------------------------


def _sample_delay(queue, values, env, flush=None, fixed_idx=None):
    """Push ``values`` into a rolling queue and read each env's delayed entry: a fresh lag every
    step, or ``fixed_idx`` (``delay_mode: per_episode``). ``flush`` rows are refilled with
    ``values`` first, so an episode never reads the previous one."""
    if flush is not None and flush.any():
        queue[flush] = values[flush].unsqueeze(1).expand(-1, queue.shape[1], -1)
    queue = torch.roll(queue, shifts=1, dims=1)
    queue[:, 0, :] = values
    if fixed_idx is not None and env.cfg.domain_randomization.delay_mode == "per_episode":
        idx = fixed_idx
    else:
        idx = torch.randint(0, queue.shape[1], (env.num_envs,), device=env.device)
    return queue, queue[torch.arange(env.num_envs, device=env.device), idx]


def _object_state_dr(env, obj_pos, obj_rot, obj_linvel, obj_angvel):
    """Pose-level object DR: object-state delay and pose noise."""
    dr = env.cfg.domain_randomization
    state = torch.cat([obj_pos, obj_rot, obj_linvel, obj_angvel], dim=-1)
    env._object_state_queue, delayed = _sample_delay(
        env._object_state_queue, state, env, flush=episode_start(env), fixed_idx=env._object_delay_idx
    )
    noisy_pos = delayed[:, 0:3] + torch.randn_like(delayed[:, 0:3]) * dr.object_state_xyz_noise_std
    noisy_rot = perturb_quat(delayed[:, 3:7], dr.object_state_rotation_noise_degrees)
    return noisy_pos, noisy_rot, delayed[:, 7:13]


def _point_dr(env, points, rest, goal_pos, goal_rot):
    """Point-level object DR (``point_dr_mode``; see ``point_dr.py``).

    ``points`` (N, K, 3) and the goal pose share one lag; each point gets its own noise and
    per-episode bias; a per-episode common bias moves the points and the goal together. The points
    are then rigidly re-fitted to ``rest`` (N, K, 3), as the clean keypoints are built from a pose.
    Returns ``(keypoints (N, K, 3), object quat wxyz, goal pos, goal quat wxyz)``.
    """
    dr = env.cfg.domain_randomization
    n, k, _ = points.shape
    frame = torch.cat([points.reshape(n, 3 * k), goal_pos, goal_rot], dim=-1)
    start = episode_start(env)
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
    kp, rot = refit_points(rest, obs_points)
    obs_goal_pos = delayed[:, 3 * k : 3 * k + 3] + env._point_common_bias
    obs_goal_rot = torch.nn.functional.normalize(delayed[:, 3 * k + 3 : 3 * k + 7], dim=-1)
    return kp, rot, obs_goal_pos, obs_goal_rot


def tracker_on(env) -> bool:
    """Whether the cloth tracker observation DR (``cloth.tracker.enabled``) drives the object."""
    return bool(env.cfg.cloth.tracker.enabled)


def _tracker_size_xy(env) -> torch.Tensor:
    """``(N, 1, 3)``: the tracker's per-env assumed-size factor on x and y, 1 on z. Read after the
    tracker has stepped this step (it is redrawn at episode starts)."""
    s = env._tracker.scale
    return torch.stack((s, s, torch.ones_like(s)), dim=-1).unsqueeze(1)


def _observed_object_and_goal(env, obj_pos, obj_rot, obj_linvel, obj_angvel, obj_kp, goal_pos, goal_rot, kp_offsets):
    """The object and goal as the policy observes them, under one of three regimes:

    * the cloth tracker (``cloth.tracker``): the estimate a stereo + point-tracker pipeline would
      report (``mdp/tracker.py``);
    * point-level DR (``point_dr_mode``): the tracked corners and the goal go through
      :func:`_point_dr` and the object pose is re-fitted from the perturbed points;
    * pose-level DR (``use_object_state_delay_noise`` and/or ``object_pose_bias_*``), or none: the
      object pose delayed, noised and biased (all neutral when off).

    Returns ``(keypoints (N, K, 3), object quat wxyz, object vel (N, 6), goal pos, goal quat wxyz)``.
    """
    dr = env.cfg.domain_randomization
    obj_vel = torch.cat([obj_linvel, obj_angvel], dim=-1)
    if tracker_on(env):
        pos, rot, g_pos, g_rot, _ = env._tracker_observe(episode_start(env))
        rot = torch.where((rot * obj_rot).sum(dim=-1, keepdim=True) < 0.0, -rot, rot)
        return keypoints_world(pos, rot, kp_offsets * _tracker_size_xy(env)), rot, obj_vel, g_pos, g_rot
    if str(dr.point_dr_mode) != "off":
        points = env._tracked_points_w() - env.scene.env_origins.unsqueeze(1)
        kp, rot, g_pos, g_rot = _point_dr(env, points, kp_offsets, goal_pos, goal_rot)
        # Same quaternion sign as the clean observation (the fit's sign is arbitrary).
        rot = torch.where((rot * obj_rot).sum(dim=-1, keepdim=True) < 0.0, -rot, rot)
        return kp, rot, obj_vel, g_pos, g_rot  # the point stream carries no velocity: obj_vel unused

    if dr.use_object_state_delay_noise:
        pos, rot, vel = _object_state_dr(env, obj_pos, obj_rot, obj_linvel, obj_angvel)
    else:
        pos, rot, vel = obj_pos, obj_rot, obj_vel
    pos = pos + env._object_bias_pos
    rot = quat.quat_mul(env._object_bias_quat, rot)
    return keypoints_world(pos, rot, kp_offsets), rot, vel, goal_pos, goal_rot


def _delay_proprio_fields(env, obs_noisy: dict, obj_kp: torch.Tensor, palm_frame_offset: torch.Tensor) -> None:
    """Under point DR, ``use_obs_delay`` delays the proprioceptive fields only, in place.

    ``keypoints_rel_palm`` is rebuilt from the DELAYED palm, as on hardware: camera points minus
    the proprio palm the policy reads at the same step. ``palm_frame_offset`` turns the observed
    palm_pos (Play frame) back into the env frame the points are in.
    """
    fields = [f for f in env.cfg.obs.obs_list if f not in _UNDELAYED_IN_POINT_MODE]
    if "palm_pos" not in fields:
        fields.append("palm_pos")
    prop = _stack(obs_noisy, fields)
    if env._proprio_queue is None:
        env._proprio_queue = torch.zeros(env.num_envs, env._obs_queue.shape[1], prop.shape[1], device=env.device)
    env._proprio_queue, prop = _sample_delay(
        env._proprio_queue, prop, env, flush=episode_start(env), fixed_idx=env._obs_delay_idx
    )
    start = 0
    for f in fields:
        width = obs_noisy[f].reshape(env.num_envs, -1).shape[1]
        obs_noisy[f] = prop[:, start:start + width].reshape(obs_noisy[f].shape)
        start += width
    obs_noisy["keypoints_rel_palm"] = obj_kp - (obs_noisy["palm_pos"] + palm_frame_offset).unsqueeze(1)


# ---------------------------------------------------------------------------------------------------
# The observation
# ---------------------------------------------------------------------------------------------------


def build_observations(env) -> dict[str, torch.Tensor]:
    """``{"policy": (N, obs_dim), "critic": (N, state_dim)}``, both clamped to
    ``obs.clamp_abs_observations``."""
    dr = env.cfg.domain_randomization
    env_origins = env.scene.env_origins

    joint_pos, joint_vel, prev_targets_canon = _canonical_joint_obs(env)

    palm_state = env.robot.data.body_state_w[:, env._palm_body_id, :]  # (N, 13)
    palm_rot = palm_state[:, 3:7]  # wxyz
    palm_vel = palm_state[:, 7:13]
    palm_pos = apply_local_offset(palm_state[:, 0:3], palm_rot, env.robot_spec.palm_center_offset, (env.num_envs,)) - env_origins
    # Observed as if the robot base stood at Play's position: zero unless cfg.robot_root_pos moves it.
    palm_pos_obs = palm_pos - (
        torch.as_tensor(env.cfg.robot_root_pos, device=palm_pos.device, dtype=palm_pos.dtype)
        - torch.as_tensor(PLAY_ROBOT_ROOT_POS, device=palm_pos.device, dtype=palm_pos.dtype)
    )

    ft_state = env.robot.data.body_state_w[:, env._fingertip_body_ids, :]  # (N, 5, 13)
    ft_pos_w = apply_local_offset(
        ft_state[:, :, 0:3], ft_state[:, :, 3:7], env.robot_spec.fingertip_offset, (env.num_envs, NUM_FINGERTIPS)
    )

    obj_pos = env.object.data.root_pos_w - env_origins
    obj_rot = env.object.data.root_quat_w
    obj_linvel = env.object.data.root_lin_vel_w
    obj_angvel = env.object.data.root_ang_vel_w
    obj_vel = torch.cat([obj_linvel, obj_angvel], dim=-1)
    goal_pos = env.goal_viz.data.root_pos_w - env_origins
    goal_rot = env.goal_viz.data.root_quat_w

    tracker = tracker_on(env)
    point_mode = str(dr.point_dr_mode) != "off" or tracker
    kp_offsets = env._keypoint_offsets * env._object_scale_multiplier.unsqueeze(1)
    obj_kp = keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = keypoints_world(goal_pos, goal_rot, kp_offsets)
    noisy_obj_kp, noisy_obj_rot, noisy_obj_vel, obs_goal_pos, obs_goal_rot = _observed_object_and_goal(
        env, obj_pos, obj_rot, obj_linvel, obj_angvel, obj_kp, goal_pos, goal_rot, kp_offsets
    )
    if point_mode:
        # The tracker's assumed sheet size scales what the POLICY sees: its keypoint offsets and
        # object_scales (x, y). Read after the tracker stepped (inside _observed_object_and_goal).
        kp_offsets_obs = kp_offsets * _tracker_size_xy(env) if tracker else kp_offsets
        noisy_goal_kp = keypoints_world(obs_goal_pos, obs_goal_rot, kp_offsets_obs)
    else:
        noisy_goal_kp = goal_kp

    object_scales = env._object_scale_per_env * env._object_scale_multiplier

    obs_clean: dict[str, torch.Tensor] = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "prev_action_targets": prev_targets_canon,
        "palm_pos": palm_pos_obs,
        "palm_rot": quat.convert_quat(palm_rot, to="xyzw"),
        "palm_vel": palm_vel,
        "object_rot": quat.convert_quat(obj_rot, to="xyzw"),
        "object_vel": obj_vel,
        "fingertip_pos_rel_palm": (ft_pos_w - env_origins.unsqueeze(1)) - palm_pos.unsqueeze(1),
        "keypoints_rel_palm": obj_kp - palm_pos.unsqueeze(1),
        "keypoints_rel_goal": obj_kp - goal_kp,
        "object_scales": object_scales,
        "closest_keypoint_max_dist": env._closest_keypoint_max_dist.unsqueeze(-1),
        "closest_fingertip_dist": env._closest_fingertip_dist,
        "lifted_object": env._lifted_object.float().unsqueeze(-1),
        "progress": torch.log(env.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
        "successes": torch.log(env._successes.float() + 1.0).unsqueeze(-1),
        "reward": (env.reward_buf * 0.01).unsqueeze(-1),
    }

    obs_noisy = dict(obs_clean)
    obs_noisy["object_rot"] = quat.convert_quat(noisy_obj_rot, to="xyzw")
    obs_noisy["object_vel"] = noisy_obj_vel
    obs_noisy["keypoints_rel_palm"] = noisy_obj_kp - palm_pos.unsqueeze(1)
    obs_noisy["keypoints_rel_goal"] = noisy_obj_kp - noisy_goal_kp
    if tracker:
        obs_noisy["object_scales"] = object_scales * _tracker_size_xy(env).squeeze(1)
    if dr.joint_velocity_obs_noise_std > 0:
        obs_noisy["joint_vel"] = joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
    if dr.joint_pos_obs_noise_std > 0:
        # Noise in radians, carried through the [-1, 1] normalisation.
        span = env._joint_upper_canon - env._joint_lower_canon
        obs_noisy["joint_pos"] = joint_pos + torch.randn_like(joint_pos) * dr.joint_pos_obs_noise_std * 2.0 / span

    if point_mode and dr.use_obs_delay:
        _delay_proprio_fields(env, obs_noisy, noisy_obj_kp, palm_pos - palm_pos_obs)

    state = _stack(obs_clean, env.cfg.obs.state_list)
    policy = _stack(obs_noisy, env.cfg.obs.obs_list)
    if dr.use_obs_delay and not point_mode:
        env._obs_queue, policy = _sample_delay(
            env._obs_queue, policy, env, flush=episode_start(env), fixed_idx=env._obs_delay_idx
        )

    clip = env.cfg.obs.clamp_abs_observations
    return {"policy": policy.clamp(-clip, clip), "critic": state.clamp(-clip, clip)}


def guard_finite(env, obs: dict[str, torch.Tensor]) -> None:
    """Keep non-finite values out of the policy, and flag the env for reset.

    A NaN in the observation surfaces in the action sampler as ``normal expects all elements of
    std >= 0.0``, which names neither the env nor the quantity. This names the most upstream
    quantity that went bad -- the robot is checked too: most of the vector is robot state, and in
    a coupled solve the articulation can diverge while the sheet stays well behaved -- zeroes the
    observation (the env is thrown away at the next reset anyway; what matters is that the number
    entering the network is finite), and flags the env (``_nonfinite_envs``, consumed by
    ``terminations.compute_dones``). ``cloth.nan_policy: raise`` raises instead.
    """
    robot = env.robot.data
    quantities = {
        "particles": env._particles_w(),
        "velocities": env.object._velocities(),
        "quat": env.object.data.root_quat_w,
        "fold_targets": env.fold_targets_w(),
        "robot_joint_pos": robot.joint_pos,
        "robot_joint_vel": robot.joint_vel,
        "robot_body_pos_w": robot.body_pos_w,
        "robot_body_quat_w": robot.body_quat_w,
        "obs": obs["policy"],
    }
    offenders = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name, tensor in quantities.items():
        flat = tensor.reshape(tensor.shape[0], -1)
        bad = ~torch.isfinite(flat).all(dim=-1)
        if not bool(bad.any()):
            continue
        offenders |= bad
        ids = torch.nonzero(bad).flatten().tolist()
        where = ""
        if name == "obs":  # which columns, to look up against cfg.obs.obs_list
            cols = torch.nonzero(~torch.isfinite(flat[ids[0]])).flatten().tolist()
            where = f" cols {cols[:12]}{'...' if len(cols) > 12 else ''} of {flat.shape[1]}"
        steps = env.episode_length_buf[bad].tolist()  # each offender's own progress
        msg = (
            f"cloth: non-finite {name} in envs {ids[:8]}{'...' if len(ids) > 8 else ''} "
            f"({len(ids)}/{env.num_envs}) at env-step {steps[:8]}{'...' if len(steps) > 8 else ''}{where}"
        )
        if env.cfg.cloth.nan_policy == "raise":
            raise RuntimeError(msg)
        print(f"[cloth] {msg} -- resetting those envs", flush=True)

    if not bool(offenders.any()):
        return
    for tensor in obs.values():
        torch.nan_to_num_(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    env._nonfinite_envs |= offenders
    env._nonfinite_count += int(offenders.sum())
    env.extras["nonfinite_envs"] = float(offenders.float().mean())
    env.extras["nonfinite_total"] = float(env._nonfinite_count)
