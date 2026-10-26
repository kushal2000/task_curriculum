"""JointReachEnvCfg: reach a commanded 27-joint configuration on the Flexiv Rizon 4s + Wuji Hand 2.

A deployment test task. The policy sees its joint state and a goal joint configuration and drives the
robot there through the Play action pipeline (``play/utils/action_utils.py``), the same translation
from action to joint target that the cloth policies use. On hardware, a policy that tracks goals as it
does in sim shows the deploy loop (observation, action translation, drivers, timing) is right.

The scene is Play's (robot, table, an object parked out of reach); the table is the cloth tasks'
``table_fold_pool.urdf`` at the same height.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..play.play_env_cfg import (
    AssetsCfg,
    DomainRandomizationCfg,
    PlayFlexivWujiHumanLikeEnvCfg,
    TerminationCfg,
)


@configclass
class GoalCfg:
    """The goal bank and how goals are drawn from it."""

    # Bank of collision-free goal configurations, canonical joint order, radians. Built at env start
    # by JointReachEnv.build_goal_bank when the file does not exist, then reused.
    bank_file: str = "outputs/joint_reach/goal_bank_s0_n50000.pt"
    bank_size: int = 50_000
    bank_seed: int = 0
    # Arm joints are drawn uniformly inside their limits shrunk by this margin [deg]; hand joints over
    # their full (human-like) range.
    arm_limit_margin_deg: float = 5.0
    # Palm centre must lie over the table top, between these heights above it [m].
    palm_height_range: tuple[float, float] = (0.05, 0.50)
    # Table footprint margin for the palm-centre check [m] (negative shrinks the footprint).
    palm_footprint_margin: float = 0.0
    # Every robot body origin and fingertip pad must be this far above the table top when over its
    # footprint [m] (checked against the highest table the reset noise can place).
    body_table_clearance: float = 0.03
    # A configuration counts as collision-free if, held with zero joint gains for a few physics steps,
    # no joint moves more than this [rad] (gravity is off on the robot, so only contacts move it).
    collision_tol_rad: float = 2.0e-3
    collision_check_steps: int = 4
    # Give up building the bank after this many rounds of num_envs candidates.
    max_bank_rounds: int = 2000
    # The table's collision box [m] (assets.table_urdf), centred on the env origin in x and y.
    table_box_size: tuple[float, float, float] = (0.6, 0.4, 0.3)

    # Time each goal is held before the next one is drawn [s], uniform.
    hold_time_range_s: tuple[float, float] = (1.5, 3.0)
    # Probability that a new goal is the nearest of `near_candidates` random bank entries to the
    # current goal (small moves) instead of a uniformly random entry (large moves).
    near_prob: float = 0.5
    near_candidates: int = 32
    # Probability that an episode starts at a random bank configuration instead of the default pose.
    start_from_bank_prob: float = 0.5


@configclass
class ReachRewardCfg:
    """Per-step reward; errors are goal minus measured joint position [rad]."""

    arm_error_scale: float = 2.0       # x mean |arm error|
    hand_error_scale: float = 1.0      # x mean |hand error|
    arm_exp_weight: float = 0.5        # x exp(-mean |arm error| / arm_exp_width)
    arm_exp_width: float = 0.03
    hand_exp_weight: float = 0.5
    hand_exp_width: float = 0.06
    # Paid while every arm joint is within arm_tol_deg and every hand joint within hand_tol_deg.
    within_tol_bonus: float = 1.0
    arm_tol_deg: float = 1.0
    hand_tol_deg: float = 3.0
    action_rate_scale: float = 0.02    # x mean (a_t - a_{t-1})^2
    arm_action_scale: float = 0.01     # x mean a_arm^2 (arm actions are target velocities)
    joint_vel_scale: float = 5.0e-4    # x mean qdot^2
    # Paid while any robot body origin or fingertip pad over the table footprint is lower than
    # table_proximity above the table top.
    table_proximity_penalty: float = 1.0
    table_proximity: float = 0.01


@configclass
class GainDRCfg:
    """Per-episode log-uniform multipliers on the joint PD gains, arm and hand separately.

    The hand's nominal gains are the vendor MJCF servo's; Wuji's sim2real drive is ~3.2-3.7x stiffer
    and ~2.3-2.7x more damped, so the hand ranges reach past both.
    """

    enabled: bool = True
    arm_stiffness: tuple[float, float] = (0.5, 2.0)
    arm_damping: tuple[float, float] = (0.5, 2.0)
    hand_stiffness: tuple[float, float] = (0.5, 5.0)
    hand_damping: tuple[float, float] = (0.5, 4.0)
    # Eval only: pin the multipliers (arm k, hand k, arm d, hand d) instead of drawing; 0 = draw.
    fixed_scales: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@configclass
class JointReachEnvCfg(PlayFlexivWujiHumanLikeEnvCfg):
    """``Isaacsimenvs-JointReachFlexivWujiHumanLike-Direct-v0``."""

    assets: AssetsCfg = AssetsCfg(
        robot_urdf="assets/flexiv_wuji/rizon4s_left_wuji_human_like_joint_limits.urdf",
        table_urdf="assets/urdf/table_fold_pool.urdf",
        # One small object, parked out of reach; Play's scene needs one.
        handle_head_types=("eraser",),
        num_assets_per_type=1,
    )
    goal: GoalCfg = GoalCfg()
    reach_reward: ReachRewardCfg = ReachRewardCfg()
    gain_dr: GainDRCfg = GainDRCfg()
    termination: TerminationCfg = TerminationCfg(episode_length=600, max_consecutive_successes=0)
    # flat7DRABC's robot-side DR (A + B), none of the object terms. Gains are drawn by gain_dr.
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        use_obs_delay=True,
        obs_delay_max=3,
        use_action_delay=True,
        action_delay_max=3,
        use_object_state_delay_noise=False,
        joint_velocity_obs_noise_std=0.1,
        delay_mode="per_episode",
        action_noise_std=0.05,
        joint_pos_obs_noise_std=0.01,
        force_scale=0.0,
        torque_scale=0.0,
    )


__all__ = ["JointReachEnvCfg", "GoalCfg", "ReachRewardCfg", "GainDRCfg"]
