"""The policy and critic observations: ``ObservationBuilder``, the assembler.

Named fields, concatenated in the order of ``cfg.obs.obs_list`` (policy) and ``cfg.obs.state_list``
(critic). The encoding is the one the Play prior was trained on, so every cloth checkpoint reads the
same vector. The fields come from the sensors in ``perception/``, each with its own DR:

* ``proprio.ProprioObservation`` -- joints, last targets, palm, fingertips; joint noise and the delay
  line of every policy field that is not the sheet's;
* ``keypoints.KeypointObservation`` -- the sheet as a rigid object and four keypoints on it (from the
  moving half's fitted pose and the fold goal's pose); its pose delayed, noised and biased;
* ``pointcloud_obs.PointCloudObservation`` (``obs.object: pointcloud``) -- replaces the sheet fields
  (``keypoints.OBJECT_FIELDS``): the policy reads ``[proprio fields | points | mask]``, and the clean
  keypoint vector is returned too, as ``teacher``.

The task fields (``closest_*``, ``lifted_object``, ``progress``, ``successes``, ``reward``) are built
here from the task. The critic always reads the clean values. The sensors draw their per-step noise
in a fixed order -- keypoints, then joints, then the cloud -- which is the random stream the trained
runs used.
"""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.cloth.perception.keypoints import FIELD_SIZES as KEYPOINT_FIELD_SIZES
from isaacsimenvs.tasks.cloth.perception.keypoints import OBJECT_FIELDS, KeypointObservation, rel_palm
from isaacsimenvs.tasks.cloth.perception.proprio import FIELD_SIZES as PROPRIO_FIELD_SIZES
from isaacsimenvs.tasks.cloth.perception.proprio import PER_JOINT_FIELDS, ProprioObservation

from .actions import JointController
from .randomization import DomainRandomization
from .robot import NUM_FINGERTIPS, RobotLayout
from .state import FoldView, SimState
from .task import ClothTask

__all__ = ["FIELD_SIZES", "ObservationBuilder", "obs_dim"]

#: The width of every field but the per-joint ones (``proprio.PER_JOINT_FIELDS``).
FIELD_SIZES: dict[str, int] = {
    **PROPRIO_FIELD_SIZES,
    **KEYPOINT_FIELD_SIZES,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS,
    "lifted_object": 1,
    "progress": 1,
    "successes": 1,
    "reward": 1,
}


def obs_dim(fields, num_joints: int) -> int:
    """Width of the vector the ordered ``fields`` make for a robot with ``num_joints`` joints."""
    sizes = {**FIELD_SIZES, **{f: num_joints for f in PER_JOINT_FIELDS}}
    return sum(sizes[f] for f in fields)


def _stack(obs: dict[str, torch.Tensor], fields) -> torch.Tensor:
    return torch.cat([obs[f].reshape(obs[f].shape[0], -1) for f in fields], dim=-1)


class ObservationBuilder:
    def __init__(
        self, cfg, robot: RobotLayout, controller: JointController, task: ClothTask, dr: DomainRandomization, *,
        num_envs: int, device, env_origins, extras: dict,
    ) -> None:
        self.cfg, self.robot, self.task, self.dr = cfg, robot, task, dr
        self.num_envs, self.device = num_envs, device
        self.env_origins = env_origins
        self.extras = extras
        #: The policy fields that are not the sheet's, in policy order.
        self.proprio_fields = [f for f in cfg.obs.obs_list if f not in OBJECT_FIELDS]
        #: The point cloud (``obs.object: pointcloud``), else None.
        self.pointcloud = None
        if cfg.obs.object == "pointcloud":
            from isaacsimenvs.tasks.cloth.perception.pointcloud_obs import PointCloudObservation

            self.pointcloud = PointCloudObservation(
                cfg, robot, num_envs=num_envs, device=device, proprio_delay_max=cfg.domain_randomization.proprio.delay_max
            )
            delayed = self.proprio_fields
        else:
            # palm_pos is carried along when the policy does not read it, so keypoints_rel_palm can
            # be rebuilt from the delayed palm.
            delayed = self.proprio_fields + ([] if "palm_pos" in self.proprio_fields else ["palm_pos"])
        self.proprio = ProprioObservation(
            cfg, robot, controller, dr, num_envs=num_envs, device=device,
            delayed_fields=delayed, width=obs_dim(delayed, robot.num_joints),
        )
        # Always built: with the point cloud it still gives the clean teacher vector.
        self.keypoints = KeypointObservation(cfg, dr, num_envs=num_envs, device=device)

    @property
    def pointcloud_layout(self) -> dict:
        """The point-cloud policy vector ``[proprio | points (M x D) | mask (M)]``:
        ``{n_proprio, num_points, point_dim}``."""
        out = {"n_proprio": obs_dim(self.proprio_fields, self.robot.num_joints),
               "num_points": self.pointcloud.num_points, "point_dim": self.pointcloud.point_dim}
        pc = self.cfg.obs.pointcloud
        if bool(getattr(pc, "split_goal", False)):
            # Two clouds, one PointNet each (pc_rl): the live group, then the goal group.
            setup = self.pointcloud.setup
            if setup.goal_mode not in ("goal_fixed", "goal_any") or setup.fk_points > 0:
                raise ValueError("obs.pointcloud.split_goal needs goal goal_fixed / goal_any and no fk points")
            out.update(split_goal=True, n_live=int(setup.n_points),
                       widths=list(pc.pointnet_widths), compile=bool(pc.compile_pointnet))
        return out

    def reset(self, env_ids: torch.Tensor) -> None:
        self.proprio.reset(env_ids)
        self.keypoints.reset(env_ids)

    # ------------------------------------------------------------------ the observation

    def _task_fields(self, episode_length_buf: torch.Tensor, reward_buf: torch.Tensor) -> dict[str, torch.Tensor]:
        """The fields the task gives, not a sensor (critic only, unless a policy lists them)."""
        task = self.task
        return {
            "closest_keypoint_max_dist": task.closest_keypoint_max_dist.unsqueeze(-1),
            "closest_fingertip_dist": task.closest_fingertip_dist,
            "lifted_object": task.lifted_object.float().unsqueeze(-1),
            "progress": torch.log(episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
            "successes": torch.log(task.successes.float() + 1.0).unsqueeze(-1),
            "reward": (reward_buf * 0.01).unsqueeze(-1),
        }

    def build(self, state: SimState, view: FoldView, episode_length_buf: torch.Tensor, reward_buf: torch.Tensor,
              start: torch.Tensor) -> dict[str, torch.Tensor]:
        """``{"policy": (N, obs_dim), "critic": (N, state_dim)}``, both clamped to
        ``obs.clamp_abs_observations``; with the point cloud also ``"teacher"``, the clean keypoint
        vector (what a keypoint policy would read, without DR). ``start``: the envs on their episode's
        first step."""
        cfg = self.cfg
        robot_now = self.proprio.observe(state, self.env_origins)
        sheet = self.keypoints.observe(view, self.task, self.env_origins, robot_now.palm_pos, start)
        obs_clean = {**robot_now.fields, **sheet.clean, **self._task_fields(episode_length_buf, reward_buf)}
        obs_noisy = {**obs_clean, **sheet.observed, **self.proprio.noised(robot_now.fields)}

        clip = cfg.obs.clamp_abs_observations
        critic = _stack(obs_clean, cfg.obs.state_list)
        if self.pointcloud is not None:
            return self._pointcloud_obs(state, view, episode_length_buf, start, obs_clean, obs_noisy, critic, clip)
        if cfg.domain_randomization.proprio.delay_max > 0:
            # As on hardware: the camera's keypoints minus the palm the policy reads at the same step.
            obs_noisy.update(self.proprio.delayed(obs_noisy, start))
            obs_noisy["keypoints_rel_palm"] = rel_palm(sheet.observed_keypoints, obs_noisy["palm_pos"] + robot_now.palm_frame_offset)
        policy = _stack(obs_noisy, cfg.obs.obs_list)
        return {"policy": policy.clamp(-clip, clip), "critic": critic.clamp(-clip, clip)}

    def _pointcloud_obs(self, state, view, episode_length_buf, start, obs_clean, obs_noisy, critic, clip):
        """``obs.object: pointcloud``: the policy reads ``[proprio | points | mask]``, the proprioception
        ``proprio_lag`` steps old and the cloud as the cameras deliver it; ``teacher`` is the clean
        keypoint vector (DAgger labels it, a student's auxiliary head regresses its sheet fields)."""
        teacher = _stack(obs_clean, self.cfg.obs.obs_list).clamp(-clip, clip)
        proprio = _stack(self.proprio.delayed(obs_noisy, start), self.proprio_fields).clamp(-clip, clip)
        proprio = torch.nan_to_num(proprio, nan=0.0, posinf=0.0, neginf=0.0)
        points, mask = self.pointcloud.observe(
            state, view, episode_length_buf, start, self.dr.proprio_lag, self.env_origins, self.task.geometry.goal_sheet
        )
        policy = torch.cat((proprio, points.flatten(1), mask.to(points.dtype)), dim=1)
        critic = critic.clamp(-clip, clip)
        grid_side = int(self.cfg.obs.pointcloud.critic_grid_side)
        if grid_side > 0:
            critic = torch.cat((critic, self._critic_grids(view, grid_side)), dim=1)
        return {"policy": policy, "critic": critic, "teacher": teacher}

    def _critic_grids(self, view: FoldView, grid_side: int) -> torch.Tensor:
        """``(N, 3 * grid_side**2 * 3)``: three clean ``grid_side x grid_side`` sample grids of the sheet
        surface, in the cloud's frame (``obs.pointcloud.critic_grid_side``): the sheet now, the
        designated goal carried on its stationary half, the frozen goal. Sample ``i`` is the same spot
        of the sheet in all three."""
        from isaacsimenvs.tasks.cloth.mdp.surface_match import dense_grid

        resolution = int(self.task.geometry.layout.resolution)
        cloud = self.pointcloud.cloud
        origins = self.env_origins.unsqueeze(1)
        sheets_w = (view.points_w, self.task.goal_now_w(view), self.task.goal_reference_w(view))
        in_cloud_frame = [
            ((dense_grid(sheet_w, resolution, grid_side) - origins - cloud.centre) * float(cloud.cfg.scale)).flatten(1)
            for sheet_w in sheets_w
        ]
        return torch.nan_to_num(torch.cat(in_cloud_frame, dim=1), nan=0.0, posinf=0.0, neginf=0.0)

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
