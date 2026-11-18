"""Every cloth policy observes a fitted moving-half object in Play's actor layout.

The robot fields retain Play's joint order, normalization, xyzw quaternions and
palm reference frame. Object pose and velocity describe the moving half; four
ordered cuboid markers or physical planar landmarks describe its current and goal poses.
The yibo preset instead preserves the historical inset-marker and full-sheet
scale contract. Physical cloth scoring is independent of these representations.

Object-state DR delays this pose and its local markers together, then applies
pose noise and reconstructs the markers. Physical rewards and success continue
to use the task's cloth geometry, independently of these policy markers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaacsimenvs.tasks.cloth.mdp.geometry import rotations

from ..backends.robot import PLAY_ROBOT_ROOT_POS, ROBOT
from .actions import delayed
from .state import SimState, TaskView

NUM_FINGERTIPS = 5
NUM_KEYPOINTS = 4
# The pretrained Play actor's ordered cuboid markers, not planar cloth corners.
PLAY_KEYPOINT_CORNERS = ((1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1))


@dataclass(frozen=True)
class ObjectFeatures:
    """Policy-only object pose and markers; the physical TaskView stays unchanged."""

    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor
    keypoints_w: torch.Tensor
    targets_w: torch.Tensor


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
    # Point-cloud reward state for the critic (obs.num_points > 0): current and best matched distance.
    "point_dist": 1,
    "closest_point_dist": 1,
}
PER_JOINT_FIELDS = ("joint_pos", "joint_vel", "prev_action_targets")
# The top-face point cloud (obs.num_points = P): P x 3 each, world frame relative to the env origin.
PER_POINT_FIELDS = ("cloth_points", "goal_points")


def obs_dim(fields, num_joints: int, num_points: int = 0) -> int:
    fields = tuple(fields)
    needs_points = PER_POINT_FIELDS + ("point_dist", "closest_point_dist")
    if num_points <= 0 and any(f in needs_points for f in fields):
        raise ValueError(f"observation fields {needs_points} need obs.num_points > 0")
    return sum(num_joints if f in PER_JOINT_FIELDS else 3 * num_points if f in PER_POINT_FIELDS
               else FIELD_SIZES[f] for f in fields)


def stack(obs: dict[str, torch.Tensor], fields) -> torch.Tensor:
    return torch.cat([obs[f].reshape(obs[f].shape[0], -1) for f in fields], dim=-1)


class ObservationBuilder:
    def __init__(self, cfg, controller, task, *, robot_spec=ROBOT):
        self.cfg, self.controller, self.task = cfg, controller, task
        self.device, self.num_envs = task.device, task.num_envs
        self._palm_offset = torch.tensor(robot_spec.palm_center_offset, device=self.device)
        self._fingertip_offsets = torch.tensor(robot_spec.fingertip_offsets, device=self.device)
        self._palm_obs_shift = torch.tensor(cfg.robot_root_pos, device=self.device) - torch.tensor(
            PLAY_ROBOT_ROOT_POS, device=self.device
        )
        self.keypoint_scale = float(getattr(cfg.obs, "keypoint_scale", 1.5))
        self.keypoint_mode = getattr(cfg.obs, "keypoint_mode", "play")
        if self.keypoint_mode not in ("play", "planar", "yibo"):
            raise ValueError("obs.keypoint_mode must be 'play', 'planar' or 'yibo'")
        if not math.isfinite(self.keypoint_scale) or self.keypoint_scale <= 0:
            raise ValueError("obs.keypoint_scale must be finite and positive")
        self._yibo_object_scales = None
        if self.keypoint_mode == "yibo":
            thickness = cfg.obs.yibo_contact_thickness
            lift = float(cfg.obs.yibo_goal_lift)
            if not math.isfinite(lift):
                raise ValueError("yibo_goal_lift must be finite")
            dimensions = task._cloth_grid.dims.clone()
            if thickness is not None:
                thickness = float(thickness)
                if not math.isfinite(thickness) or thickness <= 0:
                    raise ValueError("yibo_contact_thickness must be finite and positive, or None")
                dimensions[:, 2] = thickness
            self._yibo_object_scales = dimensions / float(cfg.reward.object_base_size)
        dr = cfg.domain_randomization
        self._obs_queue = torch.zeros(
            self.num_envs, max(1, int(dr.obs_delay_max)), int(cfg.observation_space), device=self.device
        )
        self._object_state_queue = torch.zeros(
            self.num_envs, max(1, int(dr.object_state_delay_max)), 25, device=self.device
        )
        self._object_quat_previous = torch.zeros(self.num_envs, 4, device=self.device)
        self._object_quat_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def object_scales(self):
        """Selected policy dimensions normalized by reward.object_base_size."""
        return self._yibo_object_scales if self.keypoint_mode == "yibo" else self.task._object_scales

    def reset(self, ids):
        self._obs_queue[ids] = 0.0
        self._object_state_queue[ids] = 0.0
        self._object_quat_valid[ids] = False

    def palm_center_w(self, state: SimState) -> torch.Tensor:
        return state.palm_pos_w + rotations.apply(state.palm_quat_w, self._palm_offset.expand_as(state.palm_pos_w))

    def object_features(self, state: SimState, view: TaskView) -> ObjectFeatures:
        """Fit the object pose and read the selected current/goal landmarks."""
        if self.keypoint_mode == "yibo":
            from .geometry.yibo import yibo_object_features

            return ObjectFeatures(*yibo_object_features(self.task._cloth_grid, state, self.cfg.obs.yibo_goal_lift))
        from .geometry.moving_half import moving_half_pose

        pose = moving_half_pose(self.task._cloth_grid, state, view, self.task._unfold_env)
        # A point fit determines q only up to sign. Keep the clean policy quaternion
        # in the previous hemisphere: w>=0 alone flips its input near a 180° fold.
        # On reset, the stationary frame followed by a -Y quarter turn selects the
        # natural 0..180° fold branch, including an initially fully folded sheet.
        quarter_turn = pose.quat_w.new_tensor([2**-.5, 0., -2**-.5, 0.]).expand_as(pose.quat_w)
        initial = rotations.mul(rotations.from_matrix(view.frame_r), quarter_turn)
        reference = torch.where(self._object_quat_valid[:, None], self._object_quat_previous, initial)
        quat = torch.where((pose.quat_w * reference).sum(dim=-1, keepdim=True) < 0, -pose.quat_w, pose.quat_w)
        # object_features is also queried by the viewer; repeated reads of the same
        # clean pose are idempotent. Observation noise never enters this history.
        self._object_quat_previous.copy_(quat.detach())
        self._object_quat_valid.fill_(True)
        if self.keypoint_mode == "planar":
            # Use measured landmarks directly, including bending across several
            # plates. No cuboid reconstruction, thickness offset or 1.5 expansion.
            return ObjectFeatures(pose.pos_w, quat, pose.lin_vel_w, pose.ang_vel_w,
                                  view.corners_w, view.targets_w)
        dims = self.task._object_scales * float(self.cfg.reward.object_base_size)
        corners = torch.tensor(PLAY_KEYPOINT_CORNERS, dtype=dims.dtype, device=dims.device)
        offsets = corners[None] * dims[:, None] * (self.keypoint_scale / 2.0)
        current = pose.pos_w[:, None] + rotations.apply(quat[:, None], offsets)
        target = pose.goal_pos_w[:, None] + rotations.apply(pose.goal_quat_w[:, None], offsets)
        return ObjectFeatures(pose.pos_w, quat, pose.lin_vel_w, pose.ang_vel_w, current, target)

    def keypoints_w(self, view: TaskView, state: SimState) -> tuple[torch.Tensor, torch.Tensor]:
        """World-space policy markers; physical cloth landmarks remain in TaskView."""
        features = self.object_features(state, view)
        return features.keypoints_w, features.targets_w

    def build(self, state: SimState, view: TaskView, episode_length: torch.Tensor) -> dict[str, torch.Tensor]:
        """``{"policy": (N, obs dim), "critic": (N, state dim)}``, both clamped."""
        cfg, dr = self.cfg, self.cfg.domain_randomization
        n = self.num_envs
        origins = self.task.origins
        perm = self.controller._perm_lab_to_canon

        joint_pos = state.joint_pos[:, perm]
        joint_pos = (
            2.0
            * (joint_pos - self.controller._joint_lower_canon)
            / (self.controller._joint_upper_canon - self.controller._joint_lower_canon)
            - 1.0
        )
        joint_vel = state.joint_vel[:, perm]

        palm = self.palm_center_w(state) - origins
        palm_obs = palm - self._palm_obs_shift  # as if the base were at Play's position
        pads = state.fingertip_pos_w + rotations.apply(
            state.fingertip_quat_w, self._fingertip_offsets.expand_as(state.fingertip_pos_w)
        )
        fingertip_rel_palm = pads - origins.unsqueeze(1) - palm.unsqueeze(1)

        obj = self.object_features(state, view)
        pos, quat = obj.pos_w - origins, obj.quat_w
        kp, targets = obj.keypoints_w - origins.unsqueeze(1), obj.targets_w - origins.unsqueeze(1)
        vel = torch.cat([obj.lin_vel_w, obj.ang_vel_w], dim=-1)

        if dr.use_object_state_delay_noise:
            kp_local = rotations.apply(_conj(quat).unsqueeze(1), kp - pos.unsqueeze(1))  # (N, 4, 3)
            packed = torch.cat([pos, quat, vel, kp_local.reshape(n, -1)], dim=-1)
            self._object_state_queue, d = delayed(self._object_state_queue, packed, episode_length == 0)
            o_pos = d[:, 0:3] + torch.randn_like(d[:, 0:3]) * dr.object_state_xyz_noise_std
            o_rot = rotations.random_perturb(
                torch.nn.functional.normalize(d[:, 3:7], dim=-1), dr.object_state_rotation_noise_degrees
            )
            o_vel = d[:, 7:13]
            o_kp = o_pos.unsqueeze(1) + rotations.apply(o_rot.unsqueeze(1), d[:, 13:25].reshape(n, 4, 3))
        else:
            o_rot, o_vel, o_kp = quat, vel, kp

        clean = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "prev_action_targets": self.controller._prev_targets[:, perm],
            "palm_pos": palm_obs,
            "palm_rot": rotations.to_xyzw(state.palm_quat_w),
            "palm_vel": state.palm_vel_w,
            "object_rot": rotations.to_xyzw(quat),
            "object_vel": vel,
            "fingertip_pos_rel_palm": fingertip_rel_palm,
            "keypoints_rel_palm": kp - palm.unsqueeze(1),
            "keypoints_rel_goal": kp - targets,
            "object_scales": self.object_scales,
            "closest_keypoint_max_dist": self.task._closest_fold_err.unsqueeze(-1),
            "closest_fingertip_dist": self.task._closest_fingertip_dist,
            "lifted_object": torch.zeros(n, 1, device=self.device),
            "progress": torch.log(episode_length.float() / 10.0 + 1.0).unsqueeze(-1),
            "successes": torch.log(self.task._successes.float() + 1.0).unsqueeze(-1),
            "reward": (self.task.reward_buf * 0.01).unsqueeze(-1),
        }
        if state.surface_points_w is not None:
            # Exact (clean) for actor and critic alike; no delay or noise model yet. Same frame as
            # palm_pos (env frame, base shifted to Play's position), so palm and cloth line up.
            shift = (origins + self._palm_obs_shift).unsqueeze(1)
            clean["cloth_points"] = state.surface_points_w - shift
            clean["goal_points"] = self.task._goal_points - shift
            clean["point_dist"] = view.point_dist.unsqueeze(-1)
            clean["closest_point_dist"] = self.task._closest_point_dist.unsqueeze(-1)
        noisy = dict(clean)
        noisy["object_rot"] = rotations.to_xyzw(o_rot)
        noisy["object_vel"] = o_vel
        noisy["keypoints_rel_palm"] = o_kp - palm.unsqueeze(1)
        noisy["keypoints_rel_goal"] = o_kp - targets
        if dr.joint_velocity_obs_noise_std > 0:
            noisy["joint_vel"] = joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
        if dr.joint_pos_obs_noise_std > 0:
            span = self.controller._joint_upper_canon - self.controller._joint_lower_canon
            noisy["joint_pos"] = joint_pos + torch.randn_like(joint_pos) * dr.joint_pos_obs_noise_std * 2.0 / span

        policy = stack(noisy, cfg.obs.obs_list)
        critic = stack(clean, cfg.obs.state_list)
        if dr.use_obs_delay and dr.obs_delay_max > 0:
            self._obs_queue, policy = delayed(self._obs_queue, policy, episode_length == 0)
        clip = cfg.obs.clamp_abs_observations
        return {"policy": policy.clamp(-clip, clip), "critic": critic.clamp(-clip, clip)}

    def guard_finite(self, obs: dict[str, torch.Tensor], state: SimState, episode_length: torch.Tensor) -> None:
        """Keep non-finite values out of the policy: name the most upstream bad quantity, zero the
        obs of those envs and flag them (``_get_dones`` terminates them next)."""
        parts = {
            "cloth points": state.points_w,
            "joint_pos": state.joint_pos,
            "joint_vel": state.joint_vel,
            "palm": state.palm_pos_w,
            "policy obs": obs["policy"],
            "critic obs": obs["critic"],
        }
        bad_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name, t in parts.items():
            bad = ~torch.isfinite(t.reshape(t.shape[0], -1)).all(dim=-1)
            if not bool(bad.any()):
                continue
            ids = bad.nonzero(as_tuple=True)[0]
            msg = (
                f"cloth: non-finite {name} in envs {ids[:8].tolist()} ({ids.numel()}/{self.num_envs}) "
                f"at env-step {episode_length[ids[:8]].tolist()}"
            )
            if self.cfg.task.nan_policy == "raise":
                raise RuntimeError(msg)
            print(f"[cloth] {msg} -- resetting those envs", flush=True)
            bad_any |= bad
        if not bool(bad_any.any()):
            return
        for t in obs.values():
            torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
        self.task._nonfinite_envs |= bad_any
        self.task._nonfinite_count += int(bad_any.sum())


def _conj(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)
