"""The policy and critic observations: ``ObservationBuilder``.

Named fields, concatenated in the order of ``cfg.obs.obs_list`` (policy) and ``cfg.obs.state_list``
(critic). The encoding is the one the Play prior was trained on, so every cloth checkpoint reads the
same vector:

* joint state in the policy's (canonical) joint order, ``joint_pos`` normalised to [-1, 1] over the
  joint limits;
* ``palm_pos`` as if the robot base stood at the Play task's position (``PLAY_ROBOT_ROOT_POS``);
* quaternions in (x, y, z, w), the Isaac Gym order (internally everything is (w, x, y, z));
* the sheet as a rigid object: ``keypoints_rel_*`` are four keypoints built as
  ``centre + R * offsets``, from the moving half's fitted pose and the fold goal's pose
  (``FoldView.object_*`` and ``ClothTask.goal_*``).

Observation DR, at most one object regime at a time (:meth:`ObservationBuilder._observed_object_and_goal`):
the cloth tracker (``cloth.tracker``), point-level DR (``point_dr_mode``), or pose-level DR
(object-state delay and noise, a per-episode pose bias). On top: joint noise, and a policy delay
line (``use_obs_delay``; under point DR it delays the proprioceptive fields only). The critic always
reads the clean values.
"""

from __future__ import annotations

import torch

from .actions import JointController
from .geometry import quat
from .point_dr import push_and_read, refit_points, simple_lags
from .randomization import DelayLine, DomainRandomization, perturb_quat
from .robot import NUM_FINGERTIPS, RobotLayout
from .state import FoldView, SimState
from .task import ClothTask, apply_local_offset, keypoints_world
from .tracker import TrackerObservation

__all__ = ["FIELD_SIZES", "NUM_KEYPOINTS", "PLAY_ROBOT_ROOT_POS", "ObservationBuilder", "obs_dim"]

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


class ObservationBuilder:
    def __init__(
        self, cfg, robot: RobotLayout, controller: JointController, task: ClothTask, dr: DomainRandomization, *,
        num_envs: int, device, env_origins, extras: dict,
    ) -> None:
        self.cfg, self.robot, self.controller, self.task, self.dr = cfg, robot, controller, task, dr
        self.num_envs, self.device = num_envs, device
        self.env_origins = env_origins
        self.extras = extras
        self.obs_delay = DelayLine(num_envs, dr.obs_delay_len, cfg.observation_space, device)
        self.object_delay = DelayLine(num_envs, dr.object_delay_len, 13, device)
        # The proprio-only delay line (point DR and the tracker) and the point stream; sized on
        # first use, when their widths are known. Not cleared at reset: the episode's first step
        # flushes them.
        self.proprio_delay: DelayLine | None = None
        self.point_queue: torch.Tensor | None = None
        self.tracker = TrackerObservation(
            cfg, robot, task.geometry, num_envs=num_envs, device=device, env_origins=env_origins
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self.obs_delay.clear(env_ids)
        self.object_delay.clear(env_ids)

    def tracker_on(self) -> bool:
        """Whether the cloth tracker observation DR (``cloth.tracker.enabled``) drives the object."""
        return bool(self.cfg.cloth.tracker.enabled)

    # ------------------------------------------------------------------ object DR regimes

    def _object_state_dr(self, start, obj_pos, obj_rot, obj_linvel, obj_angvel):
        """Pose-level object DR: object-state delay and pose noise."""
        dr = self.cfg.domain_randomization
        state = torch.cat([obj_pos, obj_rot, obj_linvel, obj_angvel], dim=-1)
        delayed = self.object_delay.push_and_read(state, start, self.dr.lag(self.dr.object_delay_idx))
        noisy_pos = delayed[:, 0:3] + torch.randn_like(delayed[:, 0:3]) * dr.object_state_xyz_noise_std
        noisy_rot = perturb_quat(delayed[:, 3:7], dr.object_state_rotation_noise_degrees)
        return noisy_pos, noisy_rot, delayed[:, 7:13]

    def _point_dr(self, start, points, rest, goal_pos, goal_rot):
        """Point-level object DR (``point_dr_mode``; see ``point_dr.py``).

        ``points`` (N, K, 3) and the goal pose share one lag; each point gets its own noise and
        per-episode bias; a per-episode common bias moves the points and the goal together. The points
        are then rigidly re-fitted to ``rest`` (N, K, 3), as the clean keypoints are built from a pose.
        Returns ``(keypoints (N, K, 3), object quat wxyz, goal pos, goal quat wxyz)``.
        """
        dr = self.cfg.domain_randomization
        n, k, _ = points.shape
        frame = torch.cat([points.reshape(n, 3 * k), goal_pos, goal_rot], dim=-1)
        if self.point_queue is None:
            self.point_queue = torch.zeros(n, self.dr.point_queue_len, frame.shape[1], device=self.device)
        if dr.point_dr_mode == "simple":
            lags = simple_lags(n, dr.point_delay_range, self.device)
        else:
            lags = self.dr.point_lags.step(start, int(dr.point_delay_jitter), int(dr.point_delay_max))
        self.point_queue, delayed = push_and_read(self.point_queue, frame, lags, start)

        obs_points = delayed[:, : 3 * k].reshape(n, k, 3) + self.dr.point_bias
        obs_points = obs_points + self.dr.point_common_bias.unsqueeze(1)
        if float(dr.point_noise_std) > 0.0:
            obs_points = obs_points + torch.randn_like(obs_points) * float(dr.point_noise_std)
        kp, rot = refit_points(rest, obs_points)
        obs_goal_pos = delayed[:, 3 * k : 3 * k + 3] + self.dr.point_common_bias
        obs_goal_rot = torch.nn.functional.normalize(delayed[:, 3 * k + 3 : 3 * k + 7], dim=-1)
        return kp, rot, obs_goal_pos, obs_goal_rot

    def _observed_object_and_goal(self, state, view, episode_length_buf, start, obj_pos, obj_rot, obj_linvel,
                                  obj_angvel, goal_pos, goal_rot, kp_offsets):
        """The object and goal as the policy observes them, under one of three regimes:

        * the cloth tracker (``cloth.tracker``): the estimate a stereo + point-tracker pipeline would
          report (``mdp/tracker.py``);
        * point-level DR (``point_dr_mode``): the tracked corners and the goal go through
          :meth:`_point_dr` and the object pose is re-fitted from the perturbed points;
        * pose-level DR (``use_object_state_delay_noise`` and/or ``object_pose_bias_*``), or none: the
          object pose delayed, noised and biased (all neutral when off).

        Returns ``(keypoints (N, K, 3), object quat wxyz, object vel (N, 6), goal pos, goal quat wxyz)``.
        """
        dr = self.cfg.domain_randomization
        obj_vel = torch.cat([obj_linvel, obj_angvel], dim=-1)
        if self.tracker_on():
            pos, rot, g_pos, g_rot, _ = self.tracker.observe(state, episode_length_buf, start)
            rot = torch.where((rot * obj_rot).sum(dim=-1, keepdim=True) < 0.0, -rot, rot)
            return keypoints_world(pos, rot, kp_offsets * self.tracker.size_xy()), rot, obj_vel, g_pos, g_rot
        if str(dr.point_dr_mode) != "off":
            # The moving-half corners themselves, in keypoint order -- what a point tracker sees --
            # rather than the rigid reconstruction.
            points = view.keypoints_w - self.env_origins.unsqueeze(1)
            kp, rot, g_pos, g_rot = self._point_dr(start, points, kp_offsets, goal_pos, goal_rot)
            # Same quaternion sign as the clean observation (the fit's sign is arbitrary).
            rot = torch.where((rot * obj_rot).sum(dim=-1, keepdim=True) < 0.0, -rot, rot)
            return kp, rot, obj_vel, g_pos, g_rot  # the point stream carries no velocity: obj_vel unused

        if dr.use_object_state_delay_noise:
            pos, rot, vel = self._object_state_dr(start, obj_pos, obj_rot, obj_linvel, obj_angvel)
        else:
            pos, rot, vel = obj_pos, obj_rot, obj_vel
        pos = pos + self.dr.object_bias_pos
        rot = quat.quat_mul(self.dr.object_bias_quat, rot)
        return keypoints_world(pos, rot, kp_offsets), rot, vel, goal_pos, goal_rot

    def _delay_proprio_fields(self, start, obs_noisy: dict, obj_kp: torch.Tensor, palm_frame_offset: torch.Tensor) -> None:
        """Under point DR, ``use_obs_delay`` delays the proprioceptive fields only, in place.

        ``keypoints_rel_palm`` is rebuilt from the DELAYED palm, as on hardware: camera points minus
        the proprio palm the policy reads at the same step. ``palm_frame_offset`` turns the observed
        palm_pos (Play frame) back into the env frame the points are in.
        """
        fields = [f for f in self.cfg.obs.obs_list if f not in _UNDELAYED_IN_POINT_MODE]
        if "palm_pos" not in fields:
            fields.append("palm_pos")
        prop = _stack(obs_noisy, fields)
        if self.proprio_delay is None:
            self.proprio_delay = DelayLine(self.num_envs, self.obs_delay.length, prop.shape[1], self.device)
        prop = self.proprio_delay.push_and_read(prop, start, self.dr.lag(self.dr.obs_delay_idx))
        begin = 0
        for f in fields:
            width = obs_noisy[f].reshape(self.num_envs, -1).shape[1]
            obs_noisy[f] = prop[:, begin:begin + width].reshape(obs_noisy[f].shape)
            begin += width
        obs_noisy["keypoints_rel_palm"] = obj_kp - (obs_noisy["palm_pos"] + palm_frame_offset).unsqueeze(1)

    # ------------------------------------------------------------------ the observation

    def build(self, state: SimState, view: FoldView, episode_length_buf: torch.Tensor, reward_buf: torch.Tensor,
              start: torch.Tensor) -> dict[str, torch.Tensor]:
        """``{"policy": (N, obs_dim), "critic": (N, state_dim)}``, both clamped to
        ``obs.clamp_abs_observations``. ``start``: the envs on their episode's first step."""
        cfg, dr, task, robot = self.cfg, self.cfg.domain_randomization, self.task, self.robot
        env_origins = self.env_origins
        n = self.num_envs

        perm = robot.perm_lab_to_canon
        joint_pos_raw = state.joint_pos[:, perm]
        joint_pos = 2.0 * (joint_pos_raw - robot.joint_lower_canon) / (robot.joint_upper_canon - robot.joint_lower_canon) - 1.0
        joint_vel = state.joint_vel[:, perm]
        prev_targets_canon = self.controller.prev_targets[:, perm]

        palm_state = state.body_state_w[:, robot.palm_body_id, :]  # (N, 13)
        palm_rot = palm_state[:, 3:7]  # wxyz
        palm_vel = palm_state[:, 7:13]
        palm_pos = apply_local_offset(palm_state[:, 0:3], palm_rot, robot.spec.palm_center_offset, (n,)) - env_origins
        # Observed as if the robot base stood at Play's position: zero unless cfg.robot_root_pos moves it.
        palm_pos_obs = palm_pos - (
            torch.as_tensor(cfg.robot_root_pos, device=palm_pos.device, dtype=palm_pos.dtype)
            - torch.as_tensor(PLAY_ROBOT_ROOT_POS, device=palm_pos.device, dtype=palm_pos.dtype)
        )

        ft_state = state.body_state_w[:, robot.fingertip_body_ids, :]  # (N, 5, 13)
        ft_pos_w = apply_local_offset(
            ft_state[:, :, 0:3], ft_state[:, :, 3:7], robot.spec.fingertip_offset, (n, NUM_FINGERTIPS)
        )

        obj_pos = view.object_pos_w - env_origins
        obj_rot = view.object_quat_w
        obj_linvel = view.object_lin_vel_w
        obj_angvel = view.object_ang_vel_w
        obj_vel = torch.cat([obj_linvel, obj_angvel], dim=-1)
        goal_pos = task.goal_pos_w - env_origins
        goal_rot = task.goal_quat

        tracker = self.tracker_on()
        point_mode = str(dr.point_dr_mode) != "off" or tracker
        kp_offsets = task.keypoint_offsets * self.dr.object_scale_multiplier.unsqueeze(1)
        obj_kp = keypoints_world(obj_pos, obj_rot, kp_offsets)
        goal_kp = keypoints_world(goal_pos, goal_rot, kp_offsets)
        noisy_obj_kp, noisy_obj_rot, noisy_obj_vel, obs_goal_pos, obs_goal_rot = self._observed_object_and_goal(
            state, view, episode_length_buf, start, obj_pos, obj_rot, obj_linvel, obj_angvel, goal_pos, goal_rot,
            kp_offsets,
        )
        if point_mode:
            # The tracker's assumed sheet size scales what the POLICY sees: its keypoint offsets and
            # object_scales (x, y). Read after the tracker stepped (inside _observed_object_and_goal).
            kp_offsets_obs = kp_offsets * self.tracker.size_xy() if tracker else kp_offsets
            noisy_goal_kp = keypoints_world(obs_goal_pos, obs_goal_rot, kp_offsets_obs)
        else:
            noisy_goal_kp = goal_kp

        object_scales = task.object_scale * self.dr.object_scale_multiplier

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
            "closest_keypoint_max_dist": task.closest_keypoint_max_dist.unsqueeze(-1),
            "closest_fingertip_dist": task.closest_fingertip_dist,
            "lifted_object": task.lifted_object.float().unsqueeze(-1),
            "progress": torch.log(episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
            "successes": torch.log(task.successes.float() + 1.0).unsqueeze(-1),
            "reward": (reward_buf * 0.01).unsqueeze(-1),
        }

        obs_noisy = dict(obs_clean)
        obs_noisy["object_rot"] = quat.convert_quat(noisy_obj_rot, to="xyzw")
        obs_noisy["object_vel"] = noisy_obj_vel
        obs_noisy["keypoints_rel_palm"] = noisy_obj_kp - palm_pos.unsqueeze(1)
        obs_noisy["keypoints_rel_goal"] = noisy_obj_kp - noisy_goal_kp
        if tracker:
            obs_noisy["object_scales"] = object_scales * self.tracker.size_xy().squeeze(1)
        if dr.joint_velocity_obs_noise_std > 0:
            obs_noisy["joint_vel"] = joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
        if dr.joint_pos_obs_noise_std > 0:
            # Noise in radians, carried through the [-1, 1] normalisation.
            span = robot.joint_upper_canon - robot.joint_lower_canon
            obs_noisy["joint_pos"] = joint_pos + torch.randn_like(joint_pos) * dr.joint_pos_obs_noise_std * 2.0 / span

        if point_mode and dr.use_obs_delay:
            self._delay_proprio_fields(start, obs_noisy, noisy_obj_kp, palm_pos - palm_pos_obs)

        critic = _stack(obs_clean, cfg.obs.state_list)
        policy = _stack(obs_noisy, cfg.obs.obs_list)
        if dr.use_obs_delay and not point_mode:
            policy = self.obs_delay.push_and_read(policy, start, self.dr.lag(self.dr.obs_delay_idx))

        clip = cfg.obs.clamp_abs_observations
        return {"policy": policy.clamp(-clip, clip), "critic": critic.clamp(-clip, clip)}

    # ------------------------------------------------------------------ divergence guard

    def guard_finite(self, obs: dict[str, torch.Tensor], state: SimState, view: FoldView, episode_length_buf) -> None:
        """Keep non-finite values out of the policy, and flag the env for reset.

        A NaN in the observation surfaces in the action sampler as ``normal expects all elements of
        std >= 0.0``, which names neither the env nor the quantity. This names the most upstream
        quantity that went bad -- the robot is checked too: most of the vector is robot state, and in
        a coupled solve the articulation can diverge while the sheet stays well behaved -- zeroes the
        observation (the env is thrown away at the next reset anyway; what matters is that the number
        entering the network is finite), and flags the env (``ClothTask.nonfinite_envs``, consumed by
        the dones). ``cloth.nan_policy: raise`` raises instead.
        """
        task = self.task
        quantities = {
            "particles": view.points_w,
            "velocities": view.points_vel_w,
            "quat": view.object_quat_w,
            "fold_targets": view.targets_w,
            "robot_joint_pos": state.joint_pos,
            "robot_joint_vel": state.joint_vel,
            "robot_body_pos_w": state.body_pos_w,
            "robot_body_quat_w": state.body_quat_w,
            "obs": obs["policy"],
        }
        offenders = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
            steps = episode_length_buf[bad].tolist()  # each offender's own progress
            msg = (
                f"cloth: non-finite {name} in envs {ids[:8]}{'...' if len(ids) > 8 else ''} "
                f"({len(ids)}/{self.num_envs}) at env-step {steps[:8]}{'...' if len(steps) > 8 else ''}{where}"
            )
            if self.cfg.cloth.nan_policy == "raise":
                raise RuntimeError(msg)
            print(f"[cloth] {msg} -- resetting those envs", flush=True)

        if not bool(offenders.any()):
            return
        for tensor in obs.values():
            torch.nan_to_num_(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        task.nonfinite_envs |= offenders
        task.nonfinite_count += int(offenders.sum())
        self.extras["nonfinite_envs"] = float(offenders.float().mean())
        self.extras["nonfinite_total"] = float(task.nonfinite_count)
