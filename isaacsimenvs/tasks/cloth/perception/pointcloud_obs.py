"""``PointCloudObservation``: the point cloud a policy reads with ``obs.object: pointcloud``.

Built from the simulator's state every policy step (``perception/pointcloud.PointCloud``): the sheet's
particles, the robot's body poses (its link surfaces) and, for ``goal`` goal_fixed / goal_any, the
folded goal sheet. The episode's camera errors (``domain_randomization.pointcloud``) are drawn on the
episode's first observation. ``fk_points`` are built from the body poses the delayed proprioception
gives (``domain_randomization.proprio``), so they agree with the joint angles the policy reads.

``ObservationBuilder`` owns one.
"""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.cloth.mdp.delay import DelayLine
from isaacsimenvs.tasks.cloth.perception import robot_surface
from isaacsimenvs.tasks.cloth.perception.pointcloud import GOAL_MODES, PointCloud, PointCloudSetup

__all__ = ["PointCloudObservation", "point_cloud_setup", "point_cloud_layout"]


def point_cloud_setup(obs_pc, dr_pc) -> PointCloudSetup:
    """The cloud's setup from ``obs.pointcloud`` (what the run chose) and ``domain_randomization.pointcloud``."""
    if obs_pc.goal not in GOAL_MODES:
        raise ValueError(f"obs.pointcloud.goal must be one of {GOAL_MODES}, got {obs_pc.goal!r}")
    return PointCloudSetup(
        n_points=int(obs_pc.n_points), goal_mode=str(obs_pc.goal), ref_step=int(obs_pc.ref_step),
        fk_points=int(obs_pc.fk_points), chunk=int(obs_pc.chunk), cameras=str(obs_pc.cameras),
        # the camera errors
        cam_pos_noise=tuple(dr_pc.cam_pos_noise), calib_xyz=tuple(dr_pc.calib_xyz),
        calib_rot_deg=float(dr_pc.calib_rot_deg), depth_noise_std_range=tuple(dr_pc.depth_noise_std_range),
        drop_prob_range=tuple(dr_pc.drop_prob_range), delay_max=int(dr_pc.delay_max),
        # the real-cloud effects
        near_table_drop_episode_prob=float(dr_pc.near_table_drop_episode_prob),
        near_table_drop_height_range=tuple(dr_pc.near_table_drop_height_range),
        near_table_drop_prob_range=tuple(dr_pc.near_table_drop_prob_range),
        outlier_frac_range=tuple(dr_pc.outlier_frac_range), outlier_shift_range=tuple(dr_pc.outlier_shift_range),
        cloth_weight_range=tuple(dr_pc.cloth_weight_range),
    )


def point_cloud_layout(obs_pc) -> tuple[int, int]:
    """``(num_points, point_dim)`` of the cloud ``obs.pointcloud`` describes (no simulator needed)."""
    setup = PointCloudSetup(n_points=int(obs_pc.n_points), goal_mode=str(obs_pc.goal), fk_points=int(obs_pc.fk_points))
    return setup.num_points, setup.point_dim


class PointCloudObservation:
    """The cloud of every env, from the sim state. ``cloud`` keeps the latest one for viewers
    (``last_xyz``, ``last_mask``, ``last_label``, ``last_cam``, the episode's camera draws)."""

    def __init__(self, cfg, robot_layout, *, num_envs: int, device, proprio_delay_max: int, setup=None) -> None:
        """``cfg``: the env config (``obs.pointcloud``, ``domain_randomization.pointcloud``, the robot's
        base and URDF, the table). ``setup`` overrides the cloud's setup (a viewer's student)."""
        if cfg.cloth.fold_axis != "x":
            raise ValueError("the point cloud assumes cloth.fold_axis = x")
        setup = setup or point_cloud_setup(cfg.obs.pointcloud, cfg.domain_randomization.pointcloud)
        self.n, self.device = int(num_envs), device
        root = torch.as_tensor(cfg.robot_root_pos, dtype=torch.float32)
        cams = torch.stack((torch.as_tensor(setup.cam_a_base) + root, torch.as_tensor(setup.cam_b_base) + root))
        names = list(robot_layout.body_names)
        surf = robot_surface.load(cfg.assets.robot_urdf, names)
        fk_idx = robot_surface.fk_subset(surf, setup.fk_points) if setup.fk_points > 0 else None
        rots = None
        if setup.cam_a_rot is not None and setup.cam_b_rot is not None:
            rots = torch.tensor([setup.cam_a_rot, setup.cam_b_rot], dtype=torch.float32).view(2, 3, 3)
        table_z = float(cfg.reset.table_reset_z) + float(cfg.cloth.table_half_thickness)
        self.cloud = PointCloud(
            setup, self.n, int(cfg.cloth.resolution), cams, table_z,
            surf["body"], surf["offset"], surf["weight"], device, cam_rots=rots, fk_idx=fk_idx,
        )
        # The body poses the forward-kinematics points are built from: as old as the proprioception.
        self.body_delay = DelayLine(self.n, proprio_delay_max if setup.fk_points > 0 else 0, (len(names), 7), device)
        print(
            f"[pointcloud] {setup.n_points} live points from {setup.n_cloth_cand} cloth + {self.cloud.n_robot} robot "
            f"candidates ({len(surf['links'])} links), image {self.cloud.w}x{self.cloud.h}, goal {setup.goal_mode}, "
            f"fk {setup.fk_points}, labels {self.cloud.labels or 'none'}, cloud delay 0-{setup.delay_max}, cameras {setup.cameras}",
            flush=True,
        )

    @property
    def setup(self) -> PointCloudSetup:
        return self.cloud.cfg

    @property
    def num_points(self) -> int:
        return int(self.cloud.num_points)

    @property
    def point_dim(self) -> int:
        return int(self.cloud.point_dim)

    def observe(self, state, view, episode_length_buf: torch.Tensor, start: torch.Tensor, proprio_lag: torch.Tensor,
                env_origins: torch.Tensor, goal_sheet=None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(points (N, M, point_dim), mask (N, M))``. ``start``: the envs on their episode's first step
        (their camera errors are drawn now); ``proprio_lag`` the proprioception's lag (fk points);
        ``goal_sheet``: ``FoldGeometry.goal_sheet`` (``goal`` goal_fixed / goal_any)."""
        self.cloud.reset(start.nonzero(as_tuple=True)[0])
        origins = env_origins.unsqueeze(1)
        parts = view.points_w - origins
        body = torch.cat((state.body_pos_w - origins, state.body_quat_w), dim=-1)
        old = self.body_delay.push_and_read(body, start, proprio_lag)
        goal_now = goal_rest = None
        if self.setup.goal_mode in ("goal_fixed", "goal_any"):
            goal_now, goal_rest = goal_sheet(view.points_w)
            goal_now = goal_now - origins
        return self.cloud.observe(
            episode_length_buf, parts, body[..., :3], body[..., 3:], old[..., :3], old[..., 3:],
            goal_now=goal_now, goal_rest=goal_rest,
        )
