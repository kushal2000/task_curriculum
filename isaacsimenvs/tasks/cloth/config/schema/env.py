"""``ClothEnvCfg``: the fold task's configuration, shared by both backends.

The fields, what they mean and their checks; the values are ``config/task.yaml`` (a field with no
default here is MISSING until that file sets it). A backend's config adds its own blocks
(``rigid.py``, ``vbd.py``). The robot is the real setup's: ``robots.FLEXIV_WUJI_RIGHT`` (the
identified hand with ``hand_sysid``) with the real robot's joint limits (``assets.robot_urdf``), its
base where the real table puts it (``robot_root_pos``).
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .cloth import ClothCfg

__all__ = [
    "ActionCfg",
    "ActionDRCfg",
    "AssetsCfg",
    "ClothDRCfg",
    "ClothEnvCfg",
    "DomainRandomizationCfg",
    "KeypointsDRCfg",
    "ObsCfg",
    "PointCloudDRCfg",
    "PointCloudObsCfg",
    "ProprioDRCfg",
    "ResetCfg",
    "RobotDRCfg",
    "RewardCfg",
    "TerminationCfg",
]


@configclass
class AssetsCfg:
    """The robot and table URDFs, and their contact frictions."""

    robot_urdf: str = MISSING
    table_urdf: str = MISSING
    robot_friction: float = MISSING
    finger_tip_friction: float = MISSING
    table_friction: float = MISSING


@configclass
class PointCloudObsCfg:
    """The point cloud the policy reads with ``obs.object: pointcloud``: everything the two depth cameras
    see in a box over the table (cloth, hand, arm) as bare xyz; ``perception/pointcloud.py``."""

    n_points: int = MISSING
    """Live points per step (the fixed size; unused slots are masked)."""
    goal: str = MISSING
    """An extra group of points that says what to do: ``none``; ``label`` (the cloth at ``ref_step``
    with moving / stationary labels); ``goal_fixed`` (the folded goal sheet where it belongs) or
    ``goal_any`` (its shape only, at any yaw) -- these two need ``fold_pool.goal_task``."""
    ref_step: int = MISSING
    """The episode step the ``label`` / ``goal_fixed`` group is taken at (the sheet spawns ~4 cm up
    and lands by step ~12)."""
    fk_points: int = MISSING
    """Points on the forearm and hand from forward kinematics of the (delayed) joint state (0: none)."""
    chunk: int = 1024
    """Envs per batch when the cloud is built (memory, not the result)."""


@configclass
class ObsCfg:
    """The policy (``obs_list``) and critic (``state_list``) fields; see ``mdp/observations.py``."""

    state_list: tuple[str, ...] = MISSING
    obs_list: tuple[str, ...] = MISSING
    clamp_abs_observations: float = MISSING
    object: str = MISSING
    """How the policy sees the sheet. ``keypoints``: the ``obs_list`` fields as listed. ``pointcloud``:
    the ``obs_list`` fields that are not the sheet's (``observations.OBJECT_FIELDS``), then the cloud
    (``pointcloud``); the keypoint vector stays available as the clean ``teacher`` group."""
    pointcloud: PointCloudObsCfg = PointCloudObsCfg()


@configclass
class ActionCfg:
    """Joint targets from the policy action; see ``mdp/actions.JointController``."""

    arm_moving_average: float = MISSING
    hand_moving_average: float = MISSING
    dof_speed_scale: float = MISSING


@configclass
class RewardCfg:
    """See ``mdp/task.py`` (``ClothTask.compute_reward``)."""

    keypoint_rew_scale: float = MISSING
    """Fold progress: paid per metre of new best fold error."""
    reach_goal_bonus: float = MISSING
    """Paid once, on the step the fold criterion is met."""
    distance_delta_rew_scale: float = MISSING
    """Fingertip approach: paid per metre of new best fingertip-to-sheet distance, until lifted."""
    fingertip_distance_from_pad: bool = MISSING
    """Measure that distance from the pad point the policy observes (``RobotSpec.fingertip_offset``),
    not from the fingertip body's origin."""
    lifting_bonus_threshold: float = MISSING
    """The lift latch (the critic's ``lifted_object``): set once the sheet's centre has risen this
    far, minus 5 cm, above its start."""
    kuka_actions_penalty_scale: float = MISSING
    """L1 arm joint speed penalty."""
    hand_actions_penalty_scale: float = MISSING
    """L1 hand joint speed penalty."""
    object_base_size: float = MISSING
    """``object_scales`` is the sheet's bounding box divided by this."""


@configclass
class ResetCfg:
    """The episode's start; see ``ClothTask.sample_table`` / ``sample_sheet`` and ``JointController.sample_reset``."""

    reset_position_noise_x: float = MISSING
    reset_position_noise_y: float = MISSING
    reset_position_noise_z: float = MISSING
    fixed_start_pose: tuple[float, float, float, float, float, float, float] | None = MISSING
    """``(x, y, z, qw, qx, qy, qz)`` in the env frame: pins the sheet's start (by its yaw)."""

    reset_dof_pos_random_interval_arm: float = MISSING
    reset_dof_pos_random_interval_fingers: float = MISSING
    reset_dof_vel_random_interval: float = MISSING

    table_reset_z: float = MISSING
    table_reset_z_range: float = MISSING
    table_object_z_offset: float = MISSING


@configclass
class TerminationCfg:
    """See ``ClothTask.compute_dones``. The episode length is ``episode_length_s``."""

    success_steps: int = MISSING
    """Steps the fold criterion must hold before the episode ends on it and the bonus pays."""


@configclass
class ProprioDRCfg:
    """The robot's own state as the policy reads it (every field but the sheet's), whichever way the
    policy sees the sheet; ``mdp/observations.py``. The critic reads the clean values."""

    delay_max: int = 0
    """Lag in policy steps (1/60 s), one per env and episode, uniform in ``0 .. delay_max``. The
    sheet's own observation has its own lag (``keypoints`` / ``pointcloud``), independent of this one."""
    joint_pos_noise_std: float = 0.0
    """Gaussian noise on the joint positions [rad], every step (carried through the [-1, 1] normalisation)."""
    joint_vel_noise_std: float = 0.0
    """Gaussian noise on the joint velocities [rad/s], every step."""


@configclass
class KeypointsDRCfg:
    """The sheet as the keypoint policy reads it (``obs.object: keypoints``): its pose, from which the
    keypoints and its rotation are built."""

    delay_max: int = 0
    """Lag of the observed pose in policy steps, one per env and episode, uniform in ``0 .. delay_max``."""
    pos_noise_std: float = 0.0
    """Gaussian noise on the observed position [m], every step."""
    rot_noise_deg: float = 0.0
    """The observed orientation turned about a random axis by up to this angle, every step."""
    pos_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Per-episode offset of the observed position, uniform in +- this per axis [m] (a calibration
    error: it does not average out)."""
    rot_bias_deg: float = 0.0
    """Per-episode turn of the observed orientation, about a random axis, up to this angle."""


@configclass
class PointCloudDRCfg:
    """The sheet as the point-cloud policy sees it (``obs.object: pointcloud``): the two depth cameras'
    errors; ``perception/pointcloud.py``. Per-episode draws unless marked."""

    delay_max: int = 0
    """Lag of the whole cloud in policy steps, uniform in ``0 .. delay_max``."""
    cam_pos_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Each camera's position, uniform in +- this per axis [m] (what it can see from where)."""
    calib_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Rigid offset of each camera's cloud, uniform in +- this per axis [m]: the two clouds do not coincide."""
    calib_rot_deg: float = 0.0
    """Rigid turn of each camera's cloud about the box centre, per axis up to this angle."""
    depth_noise_std_range: tuple[float, float] = (0.0, 0.0)
    """The std of the per-point noise along the camera ray [m], drawn in this range; the noise every step."""
    drop_prob_range: tuple[float, float] = (0.0, 0.0)
    """The share of visible points dropped (holes, segmentation misses), drawn in this range; the drops every step."""


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
    rigid task's is +-3 mm (``config/task.yaml``)."""


@configclass
class DomainRandomizationCfg:
    """Domain randomization, by what it acts on. Every range defaults to neutral (off); the DR blocks
    (``tasks/cloth/config/dr/``) turn a group on (task.yaml sets the rigid task's contact shell). ``keypoints`` and ``pointcloud`` are the two ways the
    policy can see the sheet (``obs.object``); only the one in use may be set. Centre values (the hand model, frictions, contact
    shell) are configuration, not DR: ``hand_sysid``, ``assets``, ``rigid_cloth``."""

    proprio: ProprioDRCfg = ProprioDRCfg()
    keypoints: KeypointsDRCfg = KeypointsDRCfg()
    pointcloud: PointCloudDRCfg = PointCloudDRCfg()
    action: ActionDRCfg = ActionDRCfg()
    robot: RobotDRCfg = RobotDRCfg()
    cloth: ClothDRCfg = ClothDRCfg()


@configclass
class ClothEnvCfg(DirectRLEnvCfg):
    """The fold task on the Flexiv + right Wuji, as the real robot is set up."""

    hand_sysid: bool = MISSING
    """The real hand's identified joint dynamics (``robots.with_sysid``: armature, friction, passive
    spring, torque limit 1.0 A x Kt) instead of HAND2_DRIVE's. Configuration, not DR: the robot DR
    (``domain_randomization.robot``) scales around it."""
    robot_self_collision: bool = MISSING
    """Robot self-collision, minus ``RobotSpec.self_collision_filter``."""
    robot_self_collision_scope: str = MISSING
    """With self-collision on: "all", "hand" (palm and fingers only) or "tips" (fingertips of
    different fingers, thumb against palm); ``robots.usd.self_collision_keep``."""
    robot_root_pos: tuple[float, float, float] = MISSING
    """Robot base in the env frame. The real table's top is 0.33 m above the robot frame (with the foam
    board on it) and its near side 0.31 m in front. The task's table is 0.4 m deep, centred on the env
    origin, its top at 0.53 m, so the base moves instead: y = 0.2 + 0.31, z = 0.53 - 0.33.
    ``palm_pos`` is observed as if it stood at the Play task's (0, 0.8, 0)
    (``observations.PLAY_ROBOT_ROOT_POS``), where the Play priors and the deploy code have it."""

    decimation: int = MISSING
    episode_length_s: float = MISSING
    # Set from the robot and the field lists in ClothEnv.__init__; placeholders until then.
    action_space: int = 29
    observation_space: int = 140
    state_space: int = 140

    # Isaac Lab's defaults; task.yaml sets the time step and (rigid) the PhysX solver.
    sim: SimulationCfg = SimulationCfg()
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

    def check(self) -> None:
        """The config as a whole, after every override (a configclass's ``__post_init__`` runs on the
        defaults, before task.yaml, recipes or the command line). ``ClothEnv`` calls it."""
        self.cloth.check()
