"""``ClothEnvCfg``: the fold task's configuration, shared by both backends.

The fields, what they mean and their checks; the values are ``config/task.yaml`` (a field with no
default here is MISSING until that file sets it). A backend's config adds its own blocks
(``rigid_cloth.py``, ``vbd.py``). The robot is the real setup's: ``robots.FLEXIV_WUJI_RIGHT`` (the
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
    "GateCurriculumCfg",
    "KeypointsDRCfg",
    "ObsCfg",
    "PointCloudDRCfg",
    "PointCloudObsCfg",
    "ProprioDRCfg",
    "ResetCfg",
    "RobotDRCfg",
    "RewardCfg",
    "SurfaceSuccessCfg",
    "TerminationCfg",
]


@configclass
class AssetsCfg:
    """The robot and table URDFs, and their contact frictions.

    The three frictions are per-SHAPE coefficients (``shape_mu``), not the friction of a contact.
    A contact mixes the two sides' coefficients, by a rule that depends on who solves it:

    * VBD (cloth-rigid): ``sqrt(cloth.soft_contact_mu * shape_mu)`` (geometric mean), unless
      ``cloth.cloth_*_friction`` (cloth, table, fingertip, robot) set that contact directly.
    * MJWarp (rigid-rigid, e.g. hand-table): ``max(shape_mu_1, shape_mu_2)`` (MuJoCo, equal priority).
    * PhysX (the rigid sheet): ``physx_chain.friction_model`` decides which of the three are used
      and with which combine mode (``schema/rigid_cloth.py``).
    """

    robot_urdf: str = MISSING
    table_urdf: str = MISSING
    robot_friction: float = MISSING
    """Every robot link that is not a fingertip (arm, palm, proximal / middle phalanges). VBD: the
    cloth feels the palm and phalanges through it (``cloth.proxy_links``)."""
    finger_tip_friction: float = MISSING
    """The five distal links (``RobotSpec.fingertip_link_names``). VBD: cloth-fingertip friction
    ``sqrt(0.25 x 1.5)`` = 0.61 at the task values."""
    table_friction: float = MISSING
    """The table's shapes. VBD: cloth-table ``sqrt(0.25 x 0.5)`` = 0.35 at the task values, unless
    ``cloth.cloth_table_friction`` sets it directly; the hand against the table (MJWarp) always uses it."""


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
    cameras: str = "ab"
    """The cameras the cloud comes from: ``ab`` (both), ``a`` (left_camera_nov7) or ``b``. With one
    camera the ``goal_fixed`` group carries that camera's calibration error."""
    manipuland_only: bool = MISSING
    """The live cloud keeps only the cloth's points, as after segmenting the cloth out; the robot
    still hides the cloth behind it, but its own points are dropped. False: cloth, hand and arm."""
    n_cloth_cand: int = MISSING
    """Cloth candidates per step, before the camera model. With ``manipuland_only`` the visible ones
    are all the live cloud can hold, so keep it well above ``n_points``."""
    use_warp: bool = False
    """The cameras' visibility test as Warp kernels (CUDA): speed only, the same cloud up to depth
    ties; not checked against a student's recipe."""
    n_ref: int = MISSING
    """Points of the ``goal`` group (``label`` / ``goal_fixed`` / ``goal_any``)."""
    labels: bool = MISSING
    """One-hot group labels on the points when there is more than one group. False: bare xyz."""
    split_goal: bool = MISSING
    """The actor reads the live and the goal groups as two clouds, one PointNet each (pc_rl, as the
    point-cloud Play student); needs ``goal`` goal_fixed / goal_any and no fk points."""
    pointnet_widths: tuple[int, ...] = (64, 128, 256)
    """The per-point MLP's widths (pc_rl's PointNet; one per cloud with ``split_goal``)."""
    compile_pointnet: bool = False
    """torch.compile the PointNets' per-point MLP + pool (speed only)."""
    critic_grid_side: int = MISSING
    """0: the critic reads ``state_list`` only. ``G``: it also reads three clean ``G x G`` sample grids
    of the sheet surface, in the cloud's frame, in a fixed order (point ``i`` is the same spot of the
    sheet in all three): the sheet now, the designated goal sheet carried on its stationary half
    (the fold progress reward's goal) and the frozen goal sheet (``termination.surface.ref_step``).
    Needs ``fold_pool.goal_task``."""


@configclass
class ObsCfg:
    """The policy (``obs_list``) and critic (``state_list``) fields; see ``mdp/observations.py``."""

    state_list: tuple[str, ...] = MISSING
    obs_list: tuple[str, ...] = MISSING
    clamp_abs_observations: float = MISSING
    object: str = MISSING
    """How the policy sees the sheet. ``keypoints``: the ``obs_list`` fields as listed. ``pointcloud``:
    the ``obs_list`` fields that are not the sheet's (``perception/keypoints.OBJECT_FIELDS``), then the cloud
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
    slide_penalty_scale: float = MISSING
    """Paid per metre the stationary half's points travel in xy (their mean), once the sheet has
    landed (``termination.drift_settle_steps``); 0 turns it off. Compare ``keypoint_rew_scale``."""
    slide_deadband: float = MISSING
    """Travel slower than this [m/s] is not penalised (contact jitter)."""
    slide_step_cap: float = MISSING
    """At most this travel [m] is paid per policy step: a sheet thrown off the table would otherwise
    cost more than a fold pays."""


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
class SurfaceSuccessCfg:
    """The surface fold criterion (``termination.success: surface``; ``mdp/surface_match.py``)."""

    ref_step: int = MISSING
    """The goal sheet is frozen where it is at this episode step (the sheet spawns above the table
    and lands by step ~12): the same step as the policy's ``goal_fixed`` cloud."""
    tolerance_frac: float = MISSING
    """Folded when the aligned two-sided max distance is below this fraction of the sheet's side."""
    fine_side: int = MISSING
    """The distance is measured on the particle grid interpolated to ``fine_side x fine_side`` points."""
    uniform_start_yaws: int = MISSING
    """Evenly spaced start yaws of the alignment, besides the two major-axis ones."""
    icp_iters: int = MISSING
    """ICP rounds from each start yaw."""
    layered: bool = MISSING
    """Compare each ply only with the goal's same ply (``mdp/surface_layered.py``): the sheet's half on
    top with the goal's top ply, the half underneath with its bottom ply, after the same placement.
    Without it a top ply misaligned inside the bottom ply's outline still passes (its uncovered goal
    points match the bottom ply a ply gap below). Needs a half-fold goal (the stage-1 fold or
    ``fold_pool.goal_task.mode: half_fold``) and an odd ``cloth.resolution``."""


@configclass
class GateCurriculumCfg:
    """The in-place gate tightened by stages (``ClothTask`` module doc). Entry ``k`` of each list is
    stage ``k``; empty lists: off (the fixed ``termination.max_in_place_dist`` and
    ``reward.slide_penalty_scale`` apply)."""

    max_in_place_dist: list[float] = MISSING
    """Per stage, as ``termination.max_in_place_dist`` [m]."""
    slide_penalty_scale: list[float] = MISSING
    """Per stage, as ``reward.slide_penalty_scale`` [per m]."""
    advance_rate: float = MISSING
    """Next stage once this share of a window's finished episodes ended on a (gated) fold."""
    window_episodes: int = MISSING
    """Finished episodes per window (over all envs)."""
    start_stage: int = MISSING
    """Stage at startup: a resumed run passes the stage it reached (logged as ``gate_stage``)."""


@configclass
class TerminationCfg:
    """See ``ClothTask.compute_dones``. The episode length is ``episode_length_s``."""

    success_steps: int = MISSING
    """Steps the fold criterion must hold before the episode ends on it and the bonus pays."""
    success: str = MISSING
    """The fold criterion. ``fold_error``: the moved particles within ``cloth.keypoint_tolerance`` of
    the designated goal and the footprint halved (look-alikes with ``accept_lookalike``).
    ``surface``: the sheet's surface within ``surface.tolerance_frac`` of its side of the goal sheet's
    surface, both ways, after the best in-plane placement -- any fold that looks like the goal
    (needs ``fold_pool.goal_task``)."""
    surface: SurfaceSuccessCfg = SurfaceSuccessCfg()
    max_in_place_dist: float = MISSING
    """``success: surface`` only: a fold also needs the two-sided max distance to the frozen goal,
    NOT aligned, below this [m] (the in-place gate); ``.inf``: off."""
    drift_settle_steps: int = MISSING
    """Policy steps into the episode at which the stationary half's anchor is latched; its drift and
    yaw (logged) and the slide penalty start there."""
    gate_curriculum: GateCurriculumCfg = GateCurriculumCfg()

    def check(self) -> None:
        if self.success not in ("fold_error", "surface"):
            raise ValueError(f"termination.success must be fold_error or surface, got {self.success!r}")
        gc = self.gate_curriculum
        if len(gc.max_in_place_dist) != len(gc.slide_penalty_scale):
            raise ValueError("termination.gate_curriculum: max_in_place_dist and slide_penalty_scale need one length")
        if (len(gc.max_in_place_dist) or float(self.max_in_place_dist) < float("inf")) and self.success != "surface":
            raise ValueError("termination.max_in_place_dist / gate_curriculum need success: surface")


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

    # What the real cloud was measured to differ by (DR block ``pointcloud_real``).
    near_table_drop_episode_prob: float = 0.0
    """The share of episodes that lose robot points close over the table (real: the black hand and arm
    go with the black table). The cloth's points stay."""
    near_table_drop_height_range: tuple[float, float] = (0.0, 0.0)
    """In such an episode: the height [m] over the table top below which robot points are lost, drawn in this range."""
    near_table_drop_prob_range: tuple[float, float] = (0.0, 0.0)
    """In such an episode: the probability that a robot point below that height is lost, drawn in this range; the losses every step."""
    outlier_frac_range: tuple[float, float] = (0.0, 0.0)
    """Floating points: this share of the live cloud, drawn in the range, is robot points moved along
    their camera's ray; the points every step."""
    outlier_shift_range: tuple[float, float] = (0.02, 0.05)
    """How far a floating point moves [m], per point; towards or away from the camera, 50/50."""
    cloth_weight_range: tuple[float, float] = (1.0, 1.0)
    """Factor on the cloth's sampling weight (its share of the cloud against the robot's), drawn in this range."""


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
        self.termination.check()
        if self.termination.success == "surface" and self.termination.surface.layered and self.cloth.resolution % 2 != 1:
            raise ValueError(f"termination.surface.layered needs an odd cloth.resolution (a crease row), got {self.cloth.resolution}")
