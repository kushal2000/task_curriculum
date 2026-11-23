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
* success scored on the tool's surfaces (``object_surface``), not its keypoints, at the curriculum's
  final tolerance.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .play_env_cfg import DomainRandomizationCfg, PlayEnvCfg, ResetCfg, TerminationCfg

__all__ = ["PlayPCEnvCfg", "PointCloudCfg"]


@configclass
class PointCloudCfg:
    """What the actor, the critic and the teacher read (``play_pc_env``)."""

    n_points: int = 1024
    """Live (camera) points per step."""
    n_object_cand: int = 768
    """Tool surface candidates drawn per step before the camera model (with the robot's)."""
    n_goal: int = 256
    """Goal points: the tool's whole surface at the goal pose, clean, redrawn every step."""
    n_critic: int = 256
    """Points per cloud of the critic's clean current and goal clouds (the scored sample set)."""
    scale: float = 10.0
    """Every point is ``(xyz - crop box centre) * scale``."""
    # Camera errors, per episode (cloth dr/pointcloud.yaml, version C), and the cloud's delay.
    cam_pos_noise: tuple[float, float, float] = (0.05, 0.05, 0.05)
    calib_xyz: tuple[float, float, float] = (0.015, 0.015, 0.01)
    calib_rot_deg: float = 2.0
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
class PCResetCfg(ResetCfg):
    target_volume_mins: tuple[float, float, float] = (-0.25, 0.13, 0.35)
    target_volume_maxs: tuple[float, float, float] = (0.25, 0.49, 0.60)


@configclass
class PCTerminationCfg(TerminationCfg):
    # The prior reached the curriculum's floor; the student is held to the same, on the surfaces.
    eval_success_tolerance: float | None = 0.01


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
    reset: PCResetCfg = PCResetCfg()
    termination: PCTerminationCfg = PCTerminationCfg()
    domain_randomization: PCDomainRandomizationCfg = PCDomainRandomizationCfg()
    pointcloud: PointCloudCfg = PointCloudCfg()
