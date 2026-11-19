"""``ClothEnvCfg``: the fold task's configuration, shared by both backends.

These defaults are the task: a backend's config adds its own blocks and changes what differs for it
(``backends/rigid/config.py``, ``backends/vbd/config.py``). The robot comes from ``robot`` (a
``RobotSpec`` name) and ``assets.robot_urdf``, both set by a robot setup
(``robots/setups/<robot_setup>.yaml``); a run recipe (``config/runs/``) changes the rest.
"""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg

__all__ = [
    "ActionCfg",
    "AssetsCfg",
    "ClothEnvCfg",
    "DomainRandomizationCfg",
    "ObsCfg",
    "ResetCfg",
    "RewardCfg",
    "TerminationCfg",
]


@configclass
class AssetsCfg:
    """The robot and table URDFs, and their contact frictions."""

    robot_urdf: str = "assets/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf"
    table_urdf: str = "assets/urdf/table_narrow.urdf"
    robot_friction: float = 0.5
    finger_tip_friction: float = 1.5
    table_friction: float = 0.5


@configclass
class ObsCfg:
    """The policy (``obs_list``) and critic (``state_list``) fields; see ``mdp/observations.py``."""

    state_list: tuple[str, ...] = (
        "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot", "palm_vel",
        "object_rot", "object_vel", "fingertip_pos_rel_palm", "keypoints_rel_palm", "keypoints_rel_goal",
        "object_scales", "closest_keypoint_max_dist", "closest_fingertip_dist", "lifted_object",
        "progress", "successes", "reward",
    )
    obs_list: tuple[str, ...] = (
        "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot", "object_rot",
        "fingertip_pos_rel_palm", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales",
    )
    clamp_abs_observations: float = 10.0


@configclass
class ActionCfg:
    """Joint targets from the policy action; see ``mdp/actions.JointController``."""

    arm_moving_average: float = 0.1
    hand_moving_average: float = 0.1
    dof_speed_scale: float = 1.5


@configclass
class RewardCfg:
    """See ``mdp/task.py`` (``ClothTask.compute_reward``)."""

    keypoint_rew_scale: float = 1500.0
    """Fold progress: paid per metre of new best fold error."""
    reach_goal_bonus: float = 1000.0
    """Paid once, on the step the fold criterion is met."""
    distance_delta_rew_scale: float = 50.0
    """Fingertip approach: paid per metre of new best fingertip-to-sheet distance, until lifted."""
    fingertip_distance_from_pad: bool = True
    """Measure that distance from the pad point the policy observes (``RobotSpec.fingertip_offset``),
    not from the fingertip body's origin."""
    lifting_bonus_threshold: float = 0.15
    """The lift latch (the critic's ``lifted_object``): set once the sheet's centre has risen this
    far, minus 5 cm, above its start."""
    kuka_actions_penalty_scale: float = 0.03
    """L1 arm joint speed penalty."""
    hand_actions_penalty_scale: float = 0.003
    """L1 hand joint speed penalty."""
    object_base_size: float = 0.04
    """``object_scales`` is the sheet's bounding box divided by this."""


@configclass
class ResetCfg:
    """The episode's start; see ``ClothTask.sample_table`` / ``sample_sheet`` and ``JointController.sample_reset``."""

    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    fixed_start_pose: tuple[float, float, float, float, float, float, float] | None = None
    """``(x, y, z, qw, qx, qy, qz)`` in the env frame: pins the sheet's start (by its yaw)."""

    reset_dof_pos_random_interval_arm: float = 0.1
    reset_dof_pos_random_interval_fingers: float = 0.1
    reset_dof_vel_random_interval: float = 0.5

    table_reset_z: float = 0.38
    table_reset_z_range: float = 0.01
    table_object_z_offset: float = 0.25
    table_reset_xy_range_m: tuple[float, float] = (0.0, 0.0)
    table_reset_yaw_range_deg: float = 0.0


@configclass
class TerminationCfg:
    """See ``ClothTask.compute_dones``. The episode length is ``episode_length_s``."""

    success_steps: int = 1
    """Steps the fold criterion must hold before the episode ends on it and the bonus pays."""


@configclass
class DomainRandomizationCfg:
    """Observation, action and robot-dynamics DR; see ``mdp/randomization.py``. All off by default."""

    # Delay lines: policy observation, executed action, observed object state. A lag uniform in
    # [0, max - 1] policy steps, drawn every step ("per_step") or once per episode ("per_episode").
    use_obs_delay: bool = False
    obs_delay_max: int = 3
    use_action_delay: bool = False
    action_delay_max: int = 3
    delay_mode: str = "per_step"

    # Pose-level object DR: the observed sheet pose delayed and noised every step, plus a
    # per-episode bias (a calibration error; it does not average out).
    use_object_state_delay_noise: bool = False
    object_state_delay_max: int = 10
    object_state_xyz_noise_std: float = 0.0
    object_state_rotation_noise_degrees: float = 0.0
    object_pose_bias_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_pose_bias_rot_degrees: float = 0.0
    # Per-env factor on the observed sheet size (keypoint offsets and object_scales), per episode.
    object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0)

    # Per-step Gaussian noise: joint velocity and position observations, the executed action.
    joint_velocity_obs_noise_std: float = 0.0
    joint_pos_obs_noise_std: float = 0.0
    action_noise_std: float = 0.0

    # Robot dynamics, per episode: PD gain factors (arm and hand separately), and the hand's joint
    # friction (per joint) and torque limit (shared) factors -- the last two PhysX only.
    robot_stiffness_scale_range: tuple[float, float] = (1.0, 1.0)
    robot_damping_scale_range: tuple[float, float] = (1.0, 1.0)
    hand_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    hand_effort_limit_scale_range: tuple[float, float] = (1.0, 1.0)

    # Point-level object DR ("off" | "simple" | "realistic"; see mdp/point_dr.py): the tracked
    # corners and the goal share one lag; each corner gets its own noise and per-episode bias, all
    # of them a common per-episode bias; the corners are then rigidly re-fitted. Replaces the
    # pose-level object DR; use_obs_delay then delays the proprioceptive fields only.
    point_dr_mode: str = "off"
    point_delay_range: tuple[int, int] = (0, 0)
    point_base_delay_range: tuple[int, int] = (0, 0)
    point_delay_jitter: int = 0
    point_delay_max: int = 0
    point_hold_choices: tuple[int, ...] = (1,)
    point_noise_std: float = 0.0
    point_bias_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    point_common_bias_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _physx_sim_cfg() -> SimulationCfg:
    """60 Hz policy / 120 Hz physics, PhysX TGS. (The Newton backend replaces ``physics``.)"""
    return SimulationCfg(
        dt=0.00833333333,  # 1/120 to 11 digits, as every policy was trained (bit-identical replays)
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            gpu_max_rigid_patch_count=2**21,
        ),
    )


@configclass
class ClothEnvCfg(DirectRLEnvCfg):
    """The fold task, with the robot of ``robot``."""

    robot: str = "flexiv_wuji_right_human_like"
    """A key of ``isaacsimenvs.robots.ROBOT_SPECS``; ``assets.robot_urdf`` must be its URDF."""
    robot_setup: str = ""
    """A file stem in ``robots/setups/`` (robot, URDF, self-collision, base position), applied by the
    config loader (``tasks/loader``) onto these defaults, before the recipe. Empty: the task's."""
    robot_self_collision: bool = False
    """Robot self-collision, minus ``RobotSpec.self_collision_filter``."""
    robot_self_collision_scope: str = "all"
    """With self-collision on: "all", "hand" (palm and fingers only) or "tips" (fingertips of
    different fingers, thumb against palm); ``robots.usd.self_collision_keep``."""
    robot_root_pos: tuple[float, float, float] = (0.0, 0.8, 0.0)
    """Robot base in the env frame. ``palm_pos`` is observed as if it stood at the Play task's
    (0, 0.8, 0) (``observations.PLAY_ROBOT_ROOT_POS``)."""

    decimation: int = 2
    episode_length_s: float = 10.0
    # Set from the robot and the field lists in ClothEnv.__init__; placeholders until then.
    action_space: int = 29
    observation_space: int = 140
    state_space: int = 140

    sim: SimulationCfg = _physx_sim_cfg()
    viewer: ViewerCfg = ViewerCfg(eye=(0.5, -1.5, 1.2), lookat=(0.0, 0.4, 0.5), resolution=(640, 480))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=768, env_spacing=1.2, replicate_physics=False, clone_in_fabric=False
    )

    assets: AssetsCfg = AssetsCfg()
    obs: ObsCfg = ObsCfg()
    action: ActionCfg = ActionCfg()
    reward: RewardCfg = RewardCfg()
    reset: ResetCfg = ResetCfg()
    termination: TerminationCfg = TerminationCfg()
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()
    cloth: ClothCfg = ClothCfg()
