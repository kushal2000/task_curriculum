"""Settings for cloth actions, observations, resets, rewards, and fold goals."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from ..backends import robot as flexiv_wuji


@configclass
class AssetsCfg:
    robot_urdf: str = flexiv_wuji.URDF
    table_urdf: str = "assets/urdf/table_narrow.urdf"
    modify_asset_frictions: bool = True
    robot_friction: float = 0.5
    """Every robot shape except the fingertips (static = dynamic)."""
    finger_tip_friction: float = 1.5
    table_friction: float = 0.5


@configclass
class ActionCfg:
    """Play's action map: arm = joint-target velocity, hand = absolute targets, both EMA-smoothed."""

    arm_moving_average: float = 0.1
    hand_moving_average: float = 0.1
    dof_speed_scale: float = 1.5
    arm_effort_limit: list[float] = []
    """Per arm joint [N m]; empty keeps the robot USD's."""


@configclass
class ObsCfg:
    """Play's asymmetric layout (``observations.FIELD_SIZES``): the actor sees ``obs_list``, the
    critic ``state_list``. Kept so Play-pretrained weights load."""

    state_list: tuple[str, ...] = (
        "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot", "palm_vel",
        "object_rot", "object_vel", "fingertip_pos_rel_palm", "keypoints_rel_palm",
        "keypoints_rel_goal", "object_scales", "closest_keypoint_max_dist",
        "closest_fingertip_dist", "lifted_object", "progress", "successes", "reward",
    )
    obs_list: tuple[str, ...] = (
        "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot", "object_rot",
        "fingertip_pos_rel_palm", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales",
    )
    keypoint_mode: str = "play"
    """play: cuboid markers; planar: physical moving-half landmarks;
    yibo: the historical finetuned checkpoint's inset planar markers and pose contract."""
    yibo_contact_thickness: float | None = 0.016
    """Historical observed full-sheet thickness [m], including nominal contact shells.
    None reports each cloth's actual geometric thickness. Only affects yibo
    object_scales; it does not change physical collision geometry."""
    yibo_goal_lift: float = 0.0
    """Historical observation-only fold-target lift [m]; physical task scoring is unchanged."""
    keypoint_scale: float = 1.5
    """Expand virtual cuboid offsets in ``play`` mode; unused in ``planar`` and ``yibo`` modes."""
    clamp_abs_observations: float = 10.0


@configclass
class RewardCfg:
    keypoint_rew_scale: float = 1500.0
    """Fold progress: x (best fold error so far - fold error), clamped >= 0, per step [1/m]."""
    reach_goal_bonus: float = 1000.0
    """Paid once, on the step the fold criterion is entered."""
    distance_delta_rew_scale: float = 50.0
    """Fingertip approach: x the summed decrease of each pad's best distance to the cloth centre."""
    fingertip_distance_from_pad: bool = True
    kuka_actions_penalty_scale: float = 0.03
    """x the L1 arm joint velocity (Play's name; it is a velocity penalty)."""
    hand_actions_penalty_scale: float = 0.003
    object_base_size: float = 0.04
    """Policy size normalization [m]: moving-half dimensions [length/2, width, thickness]
    divided by this, on both rigid and VBD backends.
    Independent of obs.keypoint_scale, which expands virtual point offsets."""


@configclass
class ResetCfg:
    reset_dof_pos_random_interval_arm: float = 0.1
    reset_dof_pos_random_interval_fingers: float = 0.1
    reset_dof_vel_random_interval: float = 0.5
    table_reset_z: float = 0.38
    table_reset_z_range: float = 0.01
    """Table height noise, U(+-range) [m] per reset."""
    table_reset_xy: tuple[float, float] = (0.0, 0.0)
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    """Sheet centre noise U(+-noise) [m] around the table centre; the rigid env bounds it by the
    table instead (``rigid_cloth.place_x/y``) and centres the cloth when both are 0."""
    yaw_range_deg: list[float] = [0.0, 360.0]
    """Uniform cloth-heading range in degrees when random_yaw is enabled."""
    random_yaw: bool = True
    """Start heading U[0, 2 pi); False starts every cloth at yaw 0."""


@configclass
class TerminationCfg:
    success_steps: int = 1
    """Consecutive steps the fold criterion must hold to count as entered (bonus, termination)."""
    held_steps: int = 10
    """Steps it must hold for ``done_fold_held`` (reported, not rewarded)."""
    fall_height: float = 0.1
    """The cloth centre below this [m above the env origin] = fell off the table."""
    hand_far_distance: float = 1.5
    """Any fingertip farther than this from the cloth centre [m] ends the episode."""


@configclass
class DomainRandomizationCfg:
    """Play's observation/action DR (the terms the cloth runs use); delays are drawn per step."""

    use_obs_delay: bool = True
    obs_delay_max: int = 3
    use_action_delay: bool = True
    action_delay_max: int = 3
    use_object_state_delay_noise: bool = True
    object_state_delay_max: int = 10
    object_state_xyz_noise_std: float = 0.01
    object_state_rotation_noise_degrees: float = 5.0
    joint_velocity_obs_noise_std: float = 0.1
    joint_pos_obs_noise_std: float = 0.0
    action_noise_std: float = 0.0


@configclass
class ReleaseCfg:
    """Fold and release; robot_clearance completes immediately once all geometry is clear."""

    enabled: bool = False
    clearance: float = 0.03
    """Required surface separation [m] for robot_clearance, or hand-origin distance for origin."""
    latch_steps: int = 3
    fold_tolerance: float = 0.0
    """> 0: a release only latches with fold error under this [m]."""
    touch: str = "origin"
    """``robot_clearance``: every rigid robot collider must clear every cloth plate;
    ``origin``: legacy hand-origin distance; ``contact``: Newton solver contacts.
    Robot-clearance mode has no latch or persistence timer."""
    touch_gap: float = 0.002
    contact_bodies: str = ""
    """Regex of robot bodies for ``origin``; empty = fingertips + palm."""
    fold_entry_bonus: float = 1000.0
    release_bonus: float = 300.0
    release_dist_rew_scale: float = 3000.0
    persist_rew_per_step: float = 2.0
    persist_bonus: float = 1000.0
    persist_window_steps: int = 180
    persist_fail_penalty: float = -300.0
    persist_fail_terminates: bool = True
    persist_consecutive: bool = True


@configclass
class ClothTaskCfg:
    """The task: fold the +x half onto the -x half (crease along the cloth's y), or undo it."""

    mode: str = "fold"
    """``fold`` | ``unfold`` (start folded, target flat) | ``mixed`` (each episode draws one)."""
    keypoint_tolerance: float = 0.04
    """Success: every tracked corner within this of its target [m] ..."""
    max_folded_footprint: float = 0.65
    """... and the extent along the fold axis below this fraction of the length (fold)."""
    min_unfolded_footprint: float = 0.9
    """Unfold: the extent back above this fraction."""
    unfold_prob: float = 0.5
    unfold_start_deg: list[float] = [180.0, 180.0]
    """Unfold start: the + half turned up by U[lo, hi] deg."""
    unfold_on_table: bool = False
    """Unfold success also needs the cloth flat on the table (``on_table_max_*``)."""
    on_table_max_spread: float = 0.015
    on_table_max_lift: float = 0.015
    nan_policy: str = "reset"
    """``reset`` a diverged env (counted in ``done_nonfinite``) | ``raise``."""
    release: ReleaseCfg = ReleaseCfg()

    def check(self) -> None:
        if self.mode not in ("fold", "unfold", "mixed"):
            raise ValueError(f"task.mode must be fold, unfold or mixed, got {self.mode!r}")
        if self.nan_policy not in ("reset", "raise"):
            raise ValueError(f"task.nan_policy must be reset or raise, got {self.nan_policy!r}")
        if self.release.enabled and self.mode != "fold":
            raise ValueError("task.release is defined for mode=fold only")
        if self.release.touch not in ("origin", "contact", "robot_clearance"):
            raise ValueError(f"task.release.touch must be origin, contact or robot_clearance, got {self.release.touch!r}")
        if self.release.touch == "robot_clearance" and not (0 < self.release.clearance < float("inf")):
            raise ValueError("task.release.clearance must be finite and positive for whole-robot release")
        if self.mode == "mixed" and not 0.0 < float(self.unfold_prob) < 1.0:
            raise ValueError(f"task.unfold_prob must be in (0, 1) for task mixed, got {self.unfold_prob}")
        lo, hi = (float(v) for v in self.unfold_start_deg)
        if not 0.0 < lo <= hi <= 180.0:
            raise ValueError(f"task.unfold_start_deg must be 0 < lo <= hi <= 180, got {[lo, hi]}")
        if not 0.5 < float(self.min_unfolded_footprint) <= 1.0:
            raise ValueError(f"task.min_unfolded_footprint must be in (0.5, 1], got {self.min_unfolded_footprint}")
        if not (float(self.on_table_max_spread) > 0.0 and float(self.on_table_max_lift) > 0.0):
            raise ValueError("task.on_table_max_spread / on_table_max_lift must be > 0")
