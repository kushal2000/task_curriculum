"""``PlayPCEnv``: Play's goal-pose reaching, observed as point clouds (``play_pc_env_cfg``).

Three observations per step:

* ``policy`` (the actor, what the real robot will have): ``[proprio | points | mask]``

  - proprio: the teacher's proprioceptive fields (every ``obs_list`` field but the tool's), with
    joint noise, one lag per episode (0..``proprio_delay_max``);
  - points ``(n_points + n_goal, 3)``: first the LIVE cloud -- what the two D415s see of the tool's
    whole surface (``tasks/cloth/perception/pointcloud.PointCloud``: occlusion by the robot and the
    tool itself, per-episode camera errors, depth noise, drops, a per-episode delay, the crop box over
    the table; ``tool_only``: the robot's own points dropped, as after segmentation), no labels;
    then the GOAL cloud -- the tool's whole surface at the goal pose, clean,
    redrawn every step (no correspondence with the live points);
  - mask: the valid slots.

* ``critic`` (privileged): the state fields without the tool's keypoint fields, then the tool's
  clean surface at its current pose and at the goal (``n_critic`` scored samples each).

* ``teacher``: the keypoint Play prior's clean ``obs_list`` vector, its goal turned to the
  symmetric equivalent nearest the tool (``object_surface.ShapeBank.canonical_goal_pose``), so a
  goal the student cannot tell from another is the same goal to the teacher.

Success and the keypoint reward term read the two-sided surface distance (``object_surface``)
through ``obs_utils.compute_intermediate_values``' ``_goal_distance`` hook.
"""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.cloth.mdp.delay import DelayLine
from isaacsimenvs.tasks.cloth.perception import robot_surface
from isaacsimenvs.tasks.cloth.perception.pointcloud import PointCloud, PointCloudSetup

from .play_env import PlayEnv
from .play_pc_env_cfg import PlayPCEnvCfg
from .utils.object_surface import EnvShapes, ShapeBank
from .utils.obs_utils import _episode_start, _keypoints_world, _stack_obs_dict, compute_obs_dim

__all__ = ["PlayPCEnv", "OBJECT_FIELDS", "AUX_FIELDS", "pc_layout"]

#: The tool's fields of the keypoint vector: what the clouds replace (as the cloth students').
OBJECT_FIELDS = ("object_rot", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales")
#: What a student's auxiliary head regresses: the teacher's view of the tool.
AUX_FIELDS = ("keypoints_rel_palm", "keypoints_rel_goal")
#: State fields the critic does not read (the tool's keypoint view; the clouds stand in).
_CRITIC_DROP = ("object_rot", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales")


def pc_layout(cfg, num_joints: int) -> dict:
    """The observation layout (no simulator needed)."""
    pc = cfg.pointcloud
    proprio = [f for f in cfg.obs.obs_list if f not in OBJECT_FIELDS]
    critic = [f for f in cfg.obs.state_list if f not in _CRITIC_DROP]
    n_p, m = compute_obs_dim(proprio, num_joints), int(pc.n_points) + int(pc.n_goal)
    return {
        "proprio_fields": proprio, "critic_fields": critic, "n_proprio": n_p,
        "n_live": int(pc.n_points), "n_goal": int(pc.n_goal), "num_points": m, "point_dim": 3,
        "policy_dim": n_p + 4 * m,
        "n_critic_state": compute_obs_dim(critic, num_joints), "n_critic_points": int(pc.n_critic),
        "critic_dim": compute_obs_dim(critic, num_joints) + 6 * int(pc.n_critic),
        "teacher_dim": compute_obs_dim(cfg.obs.obs_list, num_joints),
        "widths": [int(w) for w in pc.pointnet_widths], "compile": bool(pc.compile_pointnet),
    }


class _ToolCloud(PointCloud):
    """The camera cloud with the tool's surface in place of the sheet. ``observe``'s ``particles`` are
    the tool's candidates ``(N, K, 4)``: xyz and the area [m^2] each stands for."""

    def cloth_candidates(self, particles: torch.Tensor):
        pts, w = particles[..., :3], particles[..., 3]
        return pts, torch.zeros(w.shape, dtype=torch.bool, device=w.device), w


class PlayPCEnv(PlayEnv):
    cfg: PlayPCEnvCfg

    def __init__(self, cfg: PlayPCEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        self._pc_ready = False
        super().__init__(cfg, render_mode, **kwargs)
        pc, dev, n = self.cfg.pointcloud, self.device, self.num_envs
        self.layout = pc_layout(self.cfg, self.robot_spec.num_joints)
        self.cfg.observation_space = self.layout["policy_dim"]
        self.cfg.state_space = self.layout["critic_dim"]
        self._configure_gym_env_spaces()

        # The tools: surface samples, SDFs, symmetries (one-time, ~25 s for the 600-tool pool).
        self.shapes = ShapeBank(self._object_urdf_paths, dev, k_fixed=int(pc.n_score),
                                with_symmetries=bool(pc.teacher_canonical_goal))
        self._asset = self._object_asset_index_per_env
        # Each env keeps its tool for the run: its shape tensors gathered once (object_surface.EnvShapes).
        self.env_shapes = EnvShapes(self.shapes, self._asset)

        # The cameras and the crop box, from the robot base as on the real setup.
        root = torch.as_tensor(self.cfg.robot_root_pos, dtype=torch.float32)
        setup = PointCloudSetup(
            n_points=int(pc.n_points), n_cloth_cand=int(pc.n_object_cand), goal_mode="none", scale=float(pc.scale),
            box_y_near=float(root[1]) - float(pc.table_near_offset),
            cam_pos_noise=tuple(pc.cam_pos_noise), calib_xyz=tuple(pc.calib_xyz), calib_rot_deg=float(pc.calib_rot_deg),
            depth_noise_std_range=tuple(pc.depth_noise_std_range), drop_prob_range=tuple(pc.drop_prob_range),
            delay_max=int(pc.delay_max),
        )
        cams = torch.stack((torch.as_tensor(setup.cam_a_base), torch.as_tensor(setup.cam_b_base))) + root
        rots = torch.tensor([setup.cam_a_rot, setup.cam_b_rot], dtype=torch.float32).view(2, 3, 3)
        surf = robot_surface.load(self.cfg.assets.robot_urdf, list(self.robot.body_names))
        table_z = float(self.cfg.reset.table_reset_z) + float(self._table_half_height)
        self.cloud = _ToolCloud(setup, n, 2, cams, table_z, surf["body"], surf["offset"], surf["weight"], dev, cam_rots=rots)
        self.cloud.manipuland_only = bool(pc.tool_only)
        self.cloud.use_warp = bool(pc.use_warp)
        self.proprio_delay = DelayLine(n, int(pc.proprio_delay_max), self.layout["n_proprio"], dev)
        self._proprio_lag = torch.zeros(n, dtype=torch.long, device=dev)
        self._goal_dist_mean = torch.zeros(n, device=dev)
        self._pc_ready = True
        print(
            f"[play_pc] {setup.n_points} live points ({'tool only' if pc.tool_only else 'tool + robot'}; {setup.n_cloth_cand} tool + {self.cloud.n_robot} robot candidates, "
            f"crop box {self.cloud.box_lo.tolist()} .. {self.cloud.box_hi.tolist()}), {pc.n_goal} goal points, "
            f"critic {pc.n_critic}+{pc.n_critic} clean points, {len(self._object_urdf_paths)} tools, "
            f"policy {self.layout['policy_dim']} / critic {self.layout['critic_dim']} / teacher {self.layout['teacher_dim']}",
            flush=True,
        )

    # ------------------------------------------------------------------ scoring

    def _poses(self):
        org = self.scene.env_origins
        return (self.object.data.root_pos_w - org, self.object.data.root_quat_w,
                self.goal_viz.data.root_pos_w - org, self.goal_viz.data.root_quat_w)

    def _goal_distance(self) -> torch.Tensor:
        """Two-sided surface distance tool <-> goal [m] (max over points; the mean kept for logging)."""
        if not getattr(self, "_pc_ready", False):        # the base class's first reset
            return torch.full((self.num_envs,), 1.0, device=self.device)
        d_max, d_mean = self.env_shapes.surface_distance(*self._poses())
        self._goal_dist_mean = d_mean
        return d_max

    # ------------------------------------------------------------------ observations

    def _get_observations(self) -> dict[str, torch.Tensor]:
        base = super()._get_observations()                       # fills self._obs_fields
        if not getattr(self, "_pc_ready", False):
            return base
        clean, noisy = self._obs_fields
        lay, pc, n = self.layout, self.cfg.pointcloud, self.num_envs
        start = _episode_start(self)
        clip = float(self.cfg.obs.clamp_abs_observations)
        obj_pos, obj_quat, goal_pos, goal_quat = self._poses()
        org = self.scene.env_origins.unsqueeze(1)

        # actor: proprio, one lag per episode
        # Per-episode draws, masked (no ``nonzero`` host sync per step).
        self._proprio_lag = torch.where(start, torch.randint(0, int(pc.proprio_delay_max) + 1, (n,), device=self.device),
                                        self._proprio_lag)
        proprio = _stack_obs_dict(noisy, lay["proprio_fields"])
        proprio = self.proprio_delay.push_and_read(proprio, start, self._proprio_lag).clamp(-clip, clip)

        # actor: the camera cloud of the tool and the robot
        self.cloud.reset_mask(start)
        k = int(pc.n_object_cand)
        cand = self.env_shapes.points_w(obj_pos, obj_quat, "pool", k)
        area = (self.env_shapes.area / k).view(n, 1, 1).expand(-1, k, 1)
        body_pos = self.robot.data.body_pos_w - org
        body_quat = self.robot.data.body_quat_w
        live, live_mask = self.cloud.observe(self.episode_length_buf, torch.cat((cand, area), dim=-1), body_pos, body_quat)
        # actor: the goal cloud, clean and whole
        centre, scale = self.cloud.centre, float(pc.scale)
        goal_pts = self.env_shapes.points_w(goal_pos, goal_quat, "pool", int(pc.n_goal))
        self._last_goal_cloud = goal_pts
        goal = (goal_pts - centre) * scale
        points = torch.cat((live, goal), dim=1)
        mask = torch.cat((live_mask, torch.ones(n, int(pc.n_goal), dtype=torch.bool, device=self.device)), dim=1)
        policy = torch.cat((proprio, points.flatten(1), mask.to(points.dtype)), dim=1)

        out = {"policy": torch.nan_to_num(policy)}
        extra = tuple(pc.extra_obs)
        # critic: state without the keypoint view + the clean scored clouds
        if "critic" in extra:
            crit_state = _stack_obs_dict(clean, lay["critic_fields"]).clamp(-clip, clip)
            kc = int(pc.n_critic)           # the scored samples are in random order: the first kc are a uniform subset
            cur_c = (self.env_shapes.fixed_w_first(obj_pos, obj_quat, kc) - centre) * scale
            goal_c = (self.env_shapes.fixed_w_first(goal_pos, goal_quat, kc) - centre) * scale
            critic = torch.cat((crit_state, cur_c.flatten(1), goal_c.flatten(1)), dim=1)
            out["critic"] = torch.nan_to_num(critic)

        # teacher: the clean keypoint vector, goal canonicalised over the tool's symmetries
        if "teacher" in extra:
            teacher_fields = dict(clean)
            if bool(pc.teacher_canonical_goal):
                kp_offsets = self._keypoint_offsets * self._object_scale_multiplier.unsqueeze(1)
                c_pos, c_quat = self.env_shapes.canonical_goal_pose(obj_pos, obj_quat, goal_pos, goal_quat, kp_offsets)
                obj_kp = _keypoints_world(obj_pos, obj_quat, kp_offsets)
                teacher_fields["keypoints_rel_goal"] = obj_kp - _keypoints_world(c_pos, c_quat, kp_offsets)
            teacher = _stack_obs_dict(teacher_fields, self.cfg.obs.obs_list).clamp(-clip, clip)
            out["teacher"] = teacher
        return out
