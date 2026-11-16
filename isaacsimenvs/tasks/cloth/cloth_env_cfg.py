"""Config sections both cloth envs share; each backend's ``*_env_cfg.py`` adds sim + its cloth.

Section and key names follow Play's where the feature is the same (``action``, ``obs``, ``reward``,
``reset``, ``termination``, ``domain_randomization``, ``assets``), so an override means the same
here as there. Under Isaac Lab 3.0 import this after ``newton.compat.install_isaaclab3_compat()``.
"""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaacsimenvs.tasks.cloth.utils.mdp import flexiv_wuji

__all__ = [
    "ActionCfg", "AssetsCfg", "ClothEnvCfg", "DomainRandomizationCfg", "ClothCfg", "ObsCfg",
    "ReleaseCfg", "ResetCfg", "RewardCfg", "TerminationCfg",
]


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
    """``object_scales`` = cloth [length, width, thickness] / this."""


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
    """Fold, let go, and the fold must persist (off by default)."""

    enabled: bool = False
    clearance: float = 0.03
    """``touch=origin``: the hand counts as clear beyond this from every cloth point [m]."""
    latch_steps: int = 3
    fold_tolerance: float = 0.0
    """> 0: a release only latches with fold error under this [m]."""
    touch: str = "origin"
    """``origin`` (nearest hand body origin to any cloth point) | ``contact`` (Newton only: the
    cloth solver's own contacts)."""
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
class ClothCfg:
    """The task: fold the +x half onto the -x half (crease along the cloth's y), or undo it."""

    task: str = "fold"
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
        if self.task not in ("fold", "unfold", "mixed"):
            raise ValueError(f"cloth.task must be fold, unfold or mixed, got {self.task!r}")
        if self.nan_policy not in ("reset", "raise"):
            raise ValueError(f"cloth.nan_policy must be reset or raise, got {self.nan_policy!r}")
        if self.release.enabled and self.task != "fold":
            raise ValueError("cloth.release is defined for task=fold only")
        if self.release.touch not in ("origin", "contact"):
            raise ValueError(f"cloth.release.touch must be origin or contact, got {self.release.touch!r}")
        if self.task == "mixed" and not 0.0 < float(self.unfold_prob) < 1.0:
            raise ValueError(f"cloth.unfold_prob must be in (0, 1) for task mixed, got {self.unfold_prob}")
        lo, hi = (float(v) for v in self.unfold_start_deg)
        if not 0.0 < lo <= hi <= 180.0:
            raise ValueError(f"cloth.unfold_start_deg must be 0 < lo <= hi <= 180, got {[lo, hi]}")
        if not 0.5 < float(self.min_unfolded_footprint) <= 1.0:
            raise ValueError(f"cloth.min_unfolded_footprint must be in (0.5, 1], got {self.min_unfolded_footprint}")
        if not (float(self.on_table_max_spread) > 0.0 and float(self.on_table_max_lift) > 0.0):
            raise ValueError("cloth.on_table_max_spread / on_table_max_lift must be > 0")


@configclass
class ClothEnvCfg(DirectRLEnvCfg):
    """What both cloth envs share. Spaces are Play's for this robot (27 joints): 134 / 156."""

    robot_root_pos: tuple[float, float, float] = flexiv_wuji.ROOT_POS
    robot_self_collision: bool = True
    decimation: int = 2
    episode_length_s: float = 10.0
    action_space: int = flexiv_wuji.ROBOT.num_joints
    observation_space: int = 134
    state_space: int = 156
    viewer: ViewerCfg = ViewerCfg(eye=(0.5, -1.5, 1.2), lookat=(0.0, 0.4, 0.5), resolution=(640, 480))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=768, env_spacing=1.2, replicate_physics=False, clone_in_fabric=False
    )
    assets: AssetsCfg = AssetsCfg()
    action: ActionCfg = ActionCfg()
    obs: ObsCfg = ObsCfg()
    reward: RewardCfg = RewardCfg()
    reset: ResetCfg = ResetCfg()
    termination: TerminationCfg = TerminationCfg()
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()
    cloth: ClothCfg = ClothCfg()
