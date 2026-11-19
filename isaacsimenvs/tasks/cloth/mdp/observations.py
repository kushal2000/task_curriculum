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

Observation DR, two independent parts: the proprioception (``domain_randomization.proprio``: joint
noise, a delay line on every field that is not the sheet's) and the sheet (``keypoints``: its pose
delayed, noised and biased, :meth:`ObservationBuilder._observed_object`). The critic always reads the
clean values.
"""

from __future__ import annotations

import torch

from .actions import JointController
from .geometry import quat
from .randomization import DelayLine, DomainRandomization, perturb_quat
from .robot import NUM_FINGERTIPS, RobotLayout
from .state import FoldView, SimState
from .task import ClothTask, apply_local_offset, keypoints_world

__all__ = ["AUX_FIELDS", "FIELD_SIZES", "NUM_KEYPOINTS", "OBJECT_FIELDS", "PLAY_ROBOT_ROOT_POS", "ObservationBuilder", "obs_dim"]

NUM_KEYPOINTS = 4

#: Where the Play task puts the robot base; ``palm_pos`` is observed relative to it.
PLAY_ROBOT_ROOT_POS: tuple[float, float, float] = (0.0, 0.8, 0.0)

#: The policy fields that are the sheet's: built from the keypoints, or replaced by the point cloud.
#: Every other field is proprioception (``domain_randomization.proprio``).
OBJECT_FIELDS = ("object_rot", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales")
#: What a point-cloud student's auxiliary head regresses: the keypoint teacher's view of the sheet.
AUX_FIELDS = ("keypoints_rel_palm", "keypoints_rel_goal")

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
        dr_cfg = cfg.domain_randomization
        #: The proprioceptive fields, in policy order; ``palm_pos`` is carried along when the policy
        #: does not read it, so ``keypoints_rel_palm`` can be rebuilt from the delayed palm.
        self.proprio_fields = [f for f in cfg.obs.obs_list if f not in OBJECT_FIELDS]
        carried = self.proprio_fields + ([] if "palm_pos" in self.proprio_fields else ["palm_pos"])
        self.carried_fields = carried
        self.proprio_delay = DelayLine(num_envs, dr_cfg.proprio.delay_max, obs_dim(carried, robot.num_joints), device)
        self.keypoints_delay = DelayLine(num_envs, dr_cfg.keypoints.delay_max, 13, device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.proprio_delay.clear(env_ids)
        self.keypoints_delay.clear(env_ids)

    # ------------------------------------------------------------------ observation DR

    def _observed_object(self, start, obj_pos, obj_rot, obj_linvel, obj_angvel):
        """The sheet pose and velocity the keypoints are built from: delayed and noised every step
        (``keypoints.delay_max``, ``pos_noise_std``, ``rot_noise_deg``), then offset by the episode's
        bias (``pos_bias``, ``rot_bias_deg``). All neutral when off. Returns ``(pos, quat wxyz, vel (N, 6))``."""
        dr = self.cfg.domain_randomization.keypoints
        pos, rot = obj_pos, obj_rot
        vel = torch.cat([obj_linvel, obj_angvel], dim=-1)
        if dr.delay_max > 0 or dr.pos_noise_std > 0.0 or dr.rot_noise_deg > 0.0:
            state = torch.cat([obj_pos, obj_rot, vel], dim=-1)
            delayed = self.keypoints_delay.push_and_read(state, start, self.dr.keypoints_lag)
            pos = delayed[:, 0:3] + torch.randn_like(delayed[:, 0:3]) * dr.pos_noise_std
            rot = perturb_quat(delayed[:, 3:7], dr.rot_noise_deg)
            vel = delayed[:, 7:13]
        pos = pos + self.dr.object_bias_pos
        rot = quat.quat_mul(self.dr.object_bias_quat, rot)
        return pos, rot, vel

    def _delay_proprio(self, start, obs_noisy: dict, obj_kp: torch.Tensor, palm_frame_offset: torch.Tensor) -> None:
        """``proprio.delay_max``: the proprioceptive fields read ``proprio_lag`` steps back, in place.
        ``keypoints_rel_palm`` is rebuilt from the delayed palm, as on hardware: the camera's keypoints
        minus the palm the policy reads at the same step. ``palm_frame_offset`` turns the observed
        ``palm_pos`` (Play frame) back into the env frame the keypoints are in."""
        fields = self.carried_fields
        delayed = self.proprio_delay.push_and_read(_stack(obs_noisy, fields), start, self.dr.proprio_lag)
        begin = 0
        for f in fields:
            width = obs_noisy[f].reshape(self.num_envs, -1).shape[1]
            obs_noisy[f] = delayed[:, begin:begin + width].reshape(obs_noisy[f].shape)
            begin += width
        obs_noisy["keypoints_rel_palm"] = obj_kp - (obs_noisy["palm_pos"] + palm_frame_offset).unsqueeze(1)

    # ------------------------------------------------------------------ the observation

    def build(self, state: SimState, view: FoldView, episode_length_buf: torch.Tensor, reward_buf: torch.Tensor,
              start: torch.Tensor) -> dict[str, torch.Tensor]:
        """``{"policy": (N, obs_dim), "critic": (N, state_dim)}``, both clamped to
        ``obs.clamp_abs_observations``. ``start``: the envs on their episode's first step."""
        cfg, dr, task, robot = self.cfg, self.cfg.domain_randomization.proprio, self.task, self.robot
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

        kp_offsets = task.keypoint_offsets
        obj_kp = keypoints_world(obj_pos, obj_rot, kp_offsets)
        goal_kp = keypoints_world(goal_pos, goal_rot, kp_offsets)
        noisy_obj_pos, noisy_obj_rot, noisy_obj_vel = self._observed_object(start, obj_pos, obj_rot, obj_linvel, obj_angvel)
        noisy_obj_kp = keypoints_world(noisy_obj_pos, noisy_obj_rot, kp_offsets)

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
            "object_scales": task.object_scale,
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
        obs_noisy["keypoints_rel_goal"] = noisy_obj_kp - goal_kp
        if dr.joint_vel_noise_std > 0:
            obs_noisy["joint_vel"] = joint_vel + torch.randn_like(joint_vel) * dr.joint_vel_noise_std
        if dr.joint_pos_noise_std > 0:
            # Noise in radians, carried through the [-1, 1] normalisation.
            span = robot.joint_upper_canon - robot.joint_lower_canon
            obs_noisy["joint_pos"] = joint_pos + torch.randn_like(joint_pos) * dr.joint_pos_noise_std * 2.0 / span

        if dr.delay_max > 0:
            self._delay_proprio(start, obs_noisy, noisy_obj_kp, palm_pos - palm_pos_obs)

        critic = _stack(obs_clean, cfg.obs.state_list)
        policy = _stack(obs_noisy, cfg.obs.obs_list)

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
