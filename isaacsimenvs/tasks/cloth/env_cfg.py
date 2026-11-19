"""``ClothEnvCfg``: the fold task's configuration, shared by both backends.

These defaults are the task: a backend's config adds its own blocks and changes what differs for it
(``backends/rigid/config.py``, ``backends/vbd/config.py``); a run recipe (``config/runs/``) changes the
rest. The robot is the real setup's: ``robots.FLEXIV_WUJI_RIGHT`` (the identified hand with
``hand_sysid``) with the real robot's joint limits (``assets.robot_urdf``), its base where the real
table puts it (``robot_root_pos``).
"""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg

__all__ = [
    "ActionCfg",
    "ActionDRCfg",
    "AssetsCfg",
    "ClothDRCfg",
    "ClothEnvCfg",
    "DomainRandomizationCfg",
    "ObsCfg",
    "ObservationDRCfg",
    "ResetCfg",
    "RobotDRCfg",
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


@configclass
class TerminationCfg:
    """See ``ClothTask.compute_dones``. The episode length is ``episode_length_s``."""

    success_steps: int = 1
    """Steps the fold criterion must hold before the episode ends on it and the bonus pays."""


@configclass
class ObservationDRCfg:
    """What the policy reads (the critic always reads the clean values); ``mdp/observations.py``.

    Delays are in policy steps (1/60 s): one lag per env and episode, uniform in ``0 .. *_delay_max``.
    Noise is drawn every step, biases once per episode."""

    delay_max: int = 0
    """Lag of the whole policy observation."""
    joint_pos_noise_std: float = 0.0
    """Gaussian noise on the joint positions [rad] (carried through the [-1, 1] normalisation)."""
    joint_vel_noise_std: float = 0.0
    """Gaussian noise on the joint velocities [rad/s]."""
    object_delay_max: int = 0
    """Lag of the observed sheet pose and velocity, on top of ``delay_max``."""
    object_pos_noise_std: float = 0.0
    """Gaussian noise on the observed sheet position [m]."""
    object_rot_noise_deg: float = 0.0
    """The observed sheet orientation turned about a random axis by up to this angle."""
    object_pos_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Per-episode offset of the observed sheet position, uniform in +- this per axis [m] (a
    calibration error: it does not average out)."""
    object_rot_bias_deg: float = 0.0
    """Per-episode turn of the observed sheet orientation, about a random axis, up to this angle."""


@configclass
class ActionDRCfg:
    """The action that is executed; ``mdp/actions.py``."""

    delay_max: int = 0
    """Lag in policy steps, one per env and episode, uniform in ``0 .. delay_max``."""
    noise_std: float = 0.0
    """Gaussian noise on every action every step (normalised units, clamped to [-1, 1])."""


@configclass
class RobotDRCfg:
    """The robot's dynamics, one draw per env and episode; ``mdp/randomization.py``."""

    stiffness_scale: tuple[float, float] = (1.0, 1.0)
    """Factor on the PD stiffness, drawn for the arm and the hand separately."""
    damping_scale: tuple[float, float] = (1.0, 1.0)
    """Factor on the PD damping, drawn for the arm and the hand separately."""
    hand_friction_scale: tuple[float, float] = (1.0, 1.0)
    """Factor on the hand's joint friction, per joint (PhysX only). Needs ``hand_sysid``: the
    HAND2_DRIVE hand has no joint friction to scale."""
    hand_effort_limit_scale: tuple[float, float] = (1.0, 1.0)
    """Factor on the hand's torque limit, one per env (PhysX only)."""


@configclass
class ClothDRCfg:
    """The sheet's physics: the rigid fold pool's plates (PhysX); ``backends/rigid/randomization.py``.
    The VBD sheet has none of these and refuses a range that is not neutral.

    Factors and offsets on the configured values (``rigid_cloth``, ``assets``), so the centre of each
    range is the configuration. Uniform draws unless marked log-uniform; the plates' terms are redrawn
    at every reset, the frictions once per env at startup."""

    slat_friction_scale: tuple[float, float] = (1.0, 1.0)
    """Factor on ``rigid_cloth.slat_friction``. Friction DR needs ``physx_chain.friction_model: cloth_min``."""
    robot_friction_scale: tuple[float, float] = (1.0, 1.0)
    """Factor on ``assets.finger_tip_friction``, the whole robot's friction under ``cloth_min``."""
    table_friction_scale: tuple[float, float] = (1.0, 1.0)
    """Factor on ``assets.table_friction`` under ``cloth_min``."""
    hinge_damping_scale: tuple[float, float] = (1.0, 1.0)
    """Log-uniform factor on ``rigid_cloth.joint_damping``. Keep the floor at 0.5 or above: the hinge
    is only stable at ``dt = 1/120`` because of this damping, and at a 0.3 floor one env in 32 went
    non-finite in a forced 180 deg fold of a two-plate chain (none at 0.5 or above)."""
    hinge_stiffness: tuple[float, float] = (0.0, 0.0)
    """Hinge spring [N m / rad]; ``(0, 0)`` keeps the hinge actuator velocity-only. For scale: a 10 g
    plate's weight at 25 mm is 2.5e-3 N m, and 5e-4 N m/rad returns 0.8e-3 N m at a 90 deg bend --
    a third of that, so still a limp sheet."""
    mass_scale: tuple[float, float] = (1.0, 1.0)
    """Factor on every plate's mass and inertia together (the mass distribution keeps its shape)."""
    contact_margin_offset: tuple[float, float] = (0.0, 0.0)
    """Added to ``rigid_cloth.contact_margin`` [m]: where the surface a fingertip meets sits. The
    rigid task's default is +-3 mm (``RigidClothEnvCfg``)."""


@configclass
class DomainRandomizationCfg:
    """Domain randomization, by what it acts on. Every range defaults to neutral (off); the DR blocks
    (``tasks/cloth/config/dr/``) turn a group on. Centre values (the hand model, frictions, contact
    shell) are configuration, not DR: ``hand_sysid``, ``assets``, ``rigid_cloth``."""

    observation: ObservationDRCfg = ObservationDRCfg()
    action: ActionDRCfg = ActionDRCfg()
    robot: RobotDRCfg = RobotDRCfg()
    cloth: ClothDRCfg = ClothDRCfg()


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
    """The fold task on the Flexiv + right Wuji, as the real robot is set up."""

    hand_sysid: bool = False
    """The real hand's identified joint dynamics (``robots.with_sysid``: armature, friction, passive
    spring, torque limit 1.0 A x Kt) instead of HAND2_DRIVE's. Configuration, not DR: the robot DR
    (``domain_randomization.robot``) scales around it."""
    robot_self_collision: bool = True
    """Robot self-collision, minus ``RobotSpec.self_collision_filter``."""
    robot_self_collision_scope: str = "all"
    """With self-collision on: "all", "hand" (palm and fingers only) or "tips" (fingertips of
    different fingers, thumb against palm); ``robots.usd.self_collision_keep``."""
    robot_root_pos: tuple[float, float, float] = (0.0, 0.51, 0.20)
    """Robot base in the env frame. The real table's top is 0.33 m above the robot frame (with the foam
    board on it) and its near side 0.31 m in front. The task's table is 0.4 m deep, centred on the env
    origin, its top at 0.53 m, so the base moves instead: y = 0.2 + 0.31, z = 0.53 - 0.33.
    ``palm_pos`` is observed as if it stood at the Play task's (0, 0.8, 0)
    (``observations.PLAY_ROBOT_ROOT_POS``), where the Play priors and the deploy code have it."""

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
