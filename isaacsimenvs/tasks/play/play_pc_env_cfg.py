"""Config of the point-cloud Play task (``play_pc_env.PlayPCEnv``): Play's goal-pose reaching, observed
as point clouds.

Defaults are the run: Play's ``PlayEnvCfg`` with

* the goal volume cut to what the two real D415s see over the table (a coverage check on the pool,
  2026-11-22: with Play's volume 19% of goal poses leave the tool entirely outside the cameras' crop
  box or view; over the table top and below 0.60 m, 2%);
* the object's pose DR and the whole-vector observation delay off -- the policy reads no pose: the
  cloud carries its own camera errors and delay (``pointcloud``), the proprioception its own delay
  (``pointcloud.proprio_delay_max``, one lag per episode);
* the action delay and every delay drawn once per episode (as on hardware), action noise 0.05, joint
  noise as the cloth students (0.005 rad, 0.1 rad/s), the prior's pushes (force 5, torque 1),
  object friction x0.7-1.5;
* success scored on the tool's surfaces (``object_surface``), not its keypoints. The tolerance
  curriculum is Play's (RL from scratch); a DAgger run holds it at the prior's final 1 cm
  (``distill_play_pc --success_tolerance``).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .play_env_cfg import AssetsCfg, DomainRandomizationCfg, PlayEnvCfg, ResetCfg

__all__ = ["PlayPCEnvCfg", "PointCloudCfg"]


@configclass
class PointCloudCfg:
    """What the actor, the critic and the teacher read (``play_pc_env``)."""

    n_points: int = 64
    """Live (camera) points per step."""
    tool_only: bool = True
    """The live cloud keeps only the tool's points (as after segmentation on the real setup); the robot
    still occludes the tool in the camera model, but its points are dropped. Its pose is in proprio."""
    use_warp: bool = True
    """The cameras' visibility test as fused Warp kernels (cloth/perception/visibility_warp.py)."""
    n_object_cand: int = 768
    """Tool surface candidates drawn per step before the camera model (with the robot's)."""
    n_goal: int = 64
    """Goal points: the tool's whole surface at the goal pose, clean, redrawn every step."""
    n_critic: int = 64
    """Points per cloud of the critic's clean current and goal clouds (a subset of the scored set)."""
    n_score: int = 256
    """Surface samples per tool the goal distance is scored on (object_surface.ShapeBank k_fixed)."""
    pointnet_widths: tuple[int, ...] = (32, 64, 128)
    """The per-point MLP of each PointNet (actor: live and goal). At 24576 envs x 128 points the
    training update is dominated by it: (64, 128, 256) costs ~1.7x as much."""
    compile_pointnet: bool = True
    """torch.compile the PointNets' per-point MLP + pool (student.PointNet): 1.5x faster update."""
    scale: float = 10.0
    """Every point is ``(xyz - crop box centre) * scale``."""
    # Camera errors, per episode, and the cloud's delay. The extrinsics are known (calibrated), so the
    # cameras sit where they are and each camera's cloud is not shifted or turned: no cam_pos_noise /
    # calib_* (the cloth students' version C has +-5 cm / +-1.5 cm / 2 deg, a cloud bias as large as
    # this task's 1.5 cm success threshold against a clean goal cloud). Depth noise, drops, delay stay.
    cam_pos_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    calib_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    calib_rot_deg: float = 0.0
    depth_noise_std_range: tuple[float, float] = (0.001, 0.004)
    drop_prob_range: tuple[float, float] = (0.0, 0.3)
    delay_max: int = 6
    proprio_delay_max: int = 3
    """The proprioception's lag, 0..max policy steps, one draw per episode."""
    table_near_offset: float = 0.31
    """The real table's near edge, this far in front of the robot base [m] (the crop box's near face)."""
    teacher_canonical_goal: bool = True
    """Give the keypoint teacher the symmetric equivalent of the goal nearest the tool (object_surface)."""


@configclass
class PCAssetsCfg(AssetsCfg):
    # The tools and the robot authored into every env (utils/author_tools.py, robots/usd.py) instead of
    # spawned from USD files: the same bodies (2048-env A/B: identical masses, inertias, limits, gains,
    # and identical states after 120 steps), scene setup 468 -> 60 s at 2048 envs.
    author_tool_prims: bool = True
    author_robot_prims: bool = True


@configclass
class PCResetCfg(ResetCfg):
    target_volume_mins: tuple[float, float, float] = (-0.25, 0.13, 0.35)
    target_volume_maxs: tuple[float, float, float] = (0.25, 0.49, 0.60)


@configclass
class PCDomainRandomizationCfg(DomainRandomizationCfg):
    use_obs_delay: bool = False
    use_object_state_delay_noise: bool = False
    action_delay_max: int = 3
    delay_mode: str = "per_episode"
    action_noise_std: float = 0.05
    joint_velocity_obs_noise_std: float = 0.1
    joint_pos_obs_noise_std: float = 0.005
    force_scale: float = 5.0
    torque_scale: float = 1.0
    object_friction_scale_range: tuple[float, float] = (0.7, 1.5)


@configclass
class PlayPCEnvCfg(PlayEnvCfg):
    assets: PCAssetsCfg = PCAssetsCfg()
    reset: PCResetCfg = PCResetCfg()
    domain_randomization: PCDomainRandomizationCfg = PCDomainRandomizationCfg()
    pointcloud: PointCloudCfg = PointCloudCfg()



@configclass
class PCRealAssetsCfg(PCAssetsCfg):
    robot_urdf: str = "assets/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf"
    """The real robot's joint limits (arm 5 deg inside the spec, finger PIP <= 100 deg), as the cloth tasks."""
    table_urdf: str = "assets/urdf/table_fold_pool.urdf"
    """The cloth tasks' table: a 0.6 x 0.4 x 0.3 m block."""


@configclass
class PCRealResetCfg(PCResetCfg):
    # The block's root 0.15 m below its top, so the top stays at Play's 0.28 (spawns, goal volume and the
    # fall check are unchanged); objects still start 0.10 m above it.
    table_reset_z: float = 0.13
    table_object_z_offset: float = 0.25


@configclass
class PlayPCRealEnvCfg(PlayPCEnvCfg):
    """``PlayPCEnvCfg`` on the cloth tasks' robot and table (Yibo's fold policies), exactly: the real joint
    limits, ``table_fold_pool.urdf`` and the robot placed relative to it as in cloth/config/task.yaml --
    base at (0, 0.51, 0.20) with the table root at (0, 0, 0.38): 0.51 m behind the table centre, 0.18 m
    below its root (top 0.33 m above the base, near edge 0.31 m in front). Here the table keeps Play's
    centre (0, 0.29) and top height (0.28), so the base goes to (0, 0.80, -0.05), the cameras with it."""

    assets: PCRealAssetsCfg = PCRealAssetsCfg()
    reset: PCRealResetCfg = PCRealResetCfg()
    robot_root_pos: tuple[float, float, float] = (0.0, 0.8, -0.05)
