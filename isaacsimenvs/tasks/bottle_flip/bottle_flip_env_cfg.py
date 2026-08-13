"""Config for the bottle-flip Play variant.

Subclasses `PlayEnvCfg` rather than restating it, so the robot, action pipeline,
observation lists, domain randomisation and reset machinery are inherited unchanged.
That inheritance is load-bearing: `obs_list` / `state_list` and `action_space` stay
exactly as play2perfect defined them, which is the precondition for finetuning a
play2perfect checkpoint into this task with `--checkpoint_load_mode weights`.

Only three things are added — a bottle asset pool, flip-specific reward terms and
termination rules, and the shared `curriculum` section.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaacsimenvs.curriculum import CurriculumCfg
from isaacsimenvs.tasks.play.play_env_cfg import (
    AssetsCfg,
    PlayEnvCfg,
    RewardCfg,
    TerminationCfg,
)

__all__ = [
    "BottleFlipAssetsCfg",
    "BottleFlipRewardCfg",
    "BottleFlipTerminationCfg",
    "BottleFlipCfg",
    "BottleFlipEnvCfg",
]


@configclass
class BottleFlipAssetsCfg(AssetsCfg):
    """Bottle pool. Shape is baked per variant; fill level is set at runtime."""

    object_name: str = "bottle"

    num_bottle_variants: int = 32
    """Size of the USD pool. Isaac Lab hands env *i* variant ``i % num_bottle_variants``,
    so this is how many distinct bottle shapes appear across the batch."""

    bottle_height_range: tuple[float, float] = (0.18, 0.26)
    bottle_aspect_ratio_range: tuple[float, float] = (2.5, 4.0)
    """Height / diameter. Taller and thinner is harder to land."""

    bottle_wall_thickness: float = 0.002
    bottle_shell_density: float = 950.0
    bottle_liquid_density: float = 1000.0

    nominal_fill_fraction: float = 0.35
    """Fill baked into the URDF. Only what the bottle starts as before
    `runtime_fill` overwrites it — and the value actually used when `runtime_fill` is off."""

    fill_fraction_easy: float = 0.25
    fill_fraction_hard: float = 0.95
    """Fill level at difficulty 0 and 1. `easy` sits near the centre-of-mass minimum
    (~24% for the default bottle); CoM height — and so difficulty — rises monotonically
    from there. Do not set `easy` below ~0.2 expecting an easier task: an empty bottle's
    CoM climbs back toward its centre and gets *harder*, not easier."""

    runtime_fill: bool = True
    """Set mass / CoM / inertia per-env at every reset so fill level is a real per-episode
    curriculum knob. Costs one CPU round-trip per reset batch (PhysX exposes these only
    through host tensors). Turn off to pin every bottle at `nominal_fill_fraction`."""


@configclass
class BottleFlipRewardCfg(RewardCfg):
    """Play's reach/lift terms, plus the flip stages."""

    # Play's goal-reaching terms are inert here — the flip's own terms replace them —
    # so the keypoint reward is zeroed rather than left to fight the landing bonus.
    keypoint_rew_scale: float = 0.0
    reach_goal_bonus: float = 0.0

    spin_rew_scale: float = 400.0
    """Per *turn* of saturating progress toward `required_turns`."""

    upright_rew_scale: float = 5.0
    """Dense uprightness, paid only while airborne."""

    landing_bonus: float = 500.0
    success_bonus: float = 1500.0


@configclass
class BottleFlipTerminationCfg(TerminationCfg):
    """Landing criteria. The easy/hard pairs are what the difficulty scalar interpolates."""

    required_turns_easy: float = 0.0
    required_turns_hard: float = 1.0
    """Turns the bottle must complete before landing. 0 makes it a toss-and-land task;
    1 is a full flip. The uprightness requirement is what forces whole turns — this is
    only a floor."""

    upright_tolerance_deg_easy: float = 30.0
    upright_tolerance_deg_hard: float = 10.0

    settle_speed_easy: float = 0.30
    settle_speed_hard: float = 0.05
    """Linear speed [m/s] below which the bottle counts as settled."""

    settle_ang_speed_ratio: float = 10.0
    """Angular settle threshold as a multiple of `settle_speed`, in rad/s."""

    terminate_settle_steps: int = 15
    """Consecutive settled steps after which the outcome is decided and the episode ends,
    successful or not."""

    fall_below_table: float = 0.15
    """Terminate when the bottle drops this far below the table surface."""

    # Play's tolerance curriculum is replaced by the shared `curriculum` section.
    success_steps: int = 10
    max_consecutive_successes: int = 1


@configclass
class BottleFlipCfg:
    """Task geometry that is neither asset nor reward."""

    release_distance: float = 0.08
    """Minimum fingertip-to-bottle distance, once lifted, that counts as released [m]."""

    landing_z_margin: float = 0.02
    """Height above the resting position still treated as airborne [m]."""

    landing_xy_range: tuple[float, float] = (0.10, 0.10)
    """Half-width of the landing-target randomisation about the table centre [m]."""


@configclass
class BottleFlipEnvCfg(PlayEnvCfg):
    episode_length_s: float = 6.0

    assets: BottleFlipAssetsCfg = BottleFlipAssetsCfg()
    reward: BottleFlipRewardCfg = BottleFlipRewardCfg()
    termination: BottleFlipTerminationCfg = BottleFlipTerminationCfg()
    bottle_flip: BottleFlipCfg = BottleFlipCfg()

    curriculum: CurriculumCfg = CurriculumCfg(
        difficulty_dim=1,
        init_range=(0.0, 0.15),
        final_range=(0.0, 1.0),
        mode="adaptive",
        adapt_interval=3000,
        # A contact-rich task will not hit play's 0.7 success rate early on, and a
        # curriculum that never advances is just a fixed easy task with extra steps.
        adapt_success_threshold=0.5,
        adapt_step=0.05,
    )
