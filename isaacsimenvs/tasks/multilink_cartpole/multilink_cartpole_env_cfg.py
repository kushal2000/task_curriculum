"""Typed configuration for the multi-link cartpole balance task.

Section names mirror the keys in `cfg/task/MultiLinkCartpole.yaml` 1:1, so the YAML
overlay and the Hydra CLI (`env.geometry.n_max=5`) address the same fields.

Difficulty model, in one paragraph: every env spawns the same `n_max`-link pole, and a
per-env subset of the pole joints is *locked* at reset. A locked joint welds its two
links into one rigid segment, so an env with only joint 0 free is a classic single
inverted pendulum, while an env with every joint free is an `n_max`-link chain of the
same total length and mass. Difficulty is therefore purely "how many unactuated DOF must
one actuator stabilise", with the geometry held constant — and both the count and the
segment lengths are resampled per-env at every reset.
"""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaacsimenvs.curriculum import CurriculumCfg

__all__ = [
    "GeometryCfg",
    "ActionCfg",
    "ObsCfg",
    "RewardCfg",
    "ResetCfg",
    "TerminationCfg",
    "MultiLinkCartpoleEnvCfg",
]


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------


@configclass
class GeometryCfg:
    """Articulation shape, and how a pole joint is locked."""

    n_max: int = 4
    """Number of pole links baked into the URDF. Sets the obs width, so changing it
    breaks checkpoint transfer — keep it fixed across all stages of one experiment."""

    link_lengths: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    """Per-link length, base → tip. Must have `n_max` entries. The *total* is what the
    policy experiences at difficulty 0 (all joints but the first locked), so keeping the
    sum constant across stages keeps the easy end of the curriculum identical."""

    rail_length: float = 6.0
    cart_size: tuple[float, float, float] = (0.3, 0.2, 0.15)
    cart_mass: float = 1.0
    pole_radius: float = 0.02
    pole_density: float = 800.0
    """Link mass is derived from density x volume, so long links are correctly heavier."""

    pole_joint_damping: float = 0.0
    """Damping on a *free* pole joint. Non-zero makes the task easier by bleeding energy."""

    sim_position_iterations: int = 8
    sim_velocity_iterations: int = 1
    """Per-articulation solver iterations, written into the spawned USD's articulation
    root. A stiff locked joint at the base of a heavy chain needs more position
    iterations than an ordinary articulation before the lock stops creeping."""

    # --- joint locking ---
    lock_stiffness: float = 1.0e5
    lock_damping: float = 1.0e3
    """PD gains applied to a locked joint, holding it at 0."""

    lock_limit_rad: float = 1.0e-3
    """Half-width of the position limit clamped onto a locked joint."""

    use_joint_limit_lock: bool = True
    """Also narrow the locked joint's position limit, not just its PD gains. The limit is
    a solver constraint and is far more rigid than PD alone; set False to measure how
    much the constraint is actually contributing (see docs — locked joints are only an
    *approximation* of a rigid link, and this toggle is how that is quantified)."""


# ----------------------------------------------------------------------------
# action
# ----------------------------------------------------------------------------


@configclass
class ActionCfg:
    """One action: force on the cart. Independent of `n_max`."""

    action_scale: float = 100.0  # [N]


# ----------------------------------------------------------------------------
# obs
# ----------------------------------------------------------------------------


@configclass
class ObsCfg:
    """Padded, difficulty-conditioned observation.

    Layout, width `2 + 3 * n_max` (or `2 + 2 * n_max` without `include_segment_lengths`):

        [ cart_pos, cart_vel,
          seg_angle_0    .. seg_angle_{n_max-1},      # absolute, from vertical
          seg_ang_vel_0  .. seg_ang_vel_{n_max-1},
          seg_length_0   .. seg_length_{n_max-1} ]    # 0 for inactive slots

    Slots beyond an env's active segment count are zero-filled. The segment lengths are
    what let a single network act correctly across the whole curriculum: they tell the
    policy which morphology it is currently driving.
    """

    include_segment_lengths: bool = True
    clamp_abs_observations: float = 10.0
    normalize_cart_pos: bool = True
    """Divide cart position by half the rail length, so it lands in [-1, 1]."""


# ----------------------------------------------------------------------------
# reward
# ----------------------------------------------------------------------------


@configclass
class RewardCfg:
    """Generalised from Isaac Lab's `direct/cartpole`, with the pole-angle term replaced
    by normalised tip height so it means the same thing at every link count."""

    alive: float = 1.0
    terminated: float = -2.0
    upright: float = 2.0
    """Weight on normalised tip height in [-1, 1] (1.0 = perfectly vertical chain)."""
    cart_pos: float = -0.05
    cart_vel: float = -0.01
    pole_vel: float = -0.005
    action_rate: float = -0.001


# ----------------------------------------------------------------------------
# reset
# ----------------------------------------------------------------------------


@configclass
class ResetCfg:
    """Initial-state randomisation. The angle spread is difficulty-scaled."""

    cart_pos_range: tuple[float, float] = (-0.2, 0.2)
    cart_vel_range: tuple[float, float] = (-0.05, 0.05)

    pole_angle_range_easy: float = 0.02
    pole_angle_range_hard: float = 0.20
    """Half-width [rad] of the per-joint initial angle perturbation at difficulty 0 and
    difficulty 1. Interpolated per-env, so a harder env also starts further from upright."""

    pole_vel_range: float = 0.05


# ----------------------------------------------------------------------------
# termination
# ----------------------------------------------------------------------------


@configclass
class TerminationCfg:
    max_cart_pos: float = 2.8
    """Terminate when the cart leaves this |x| [m]. Kept inside `geometry.rail_length/2`
    so the episode ends before the prismatic joint slams into its limit."""

    fall_height_frac: float = 0.4
    """Terminate when normalised tip height drops below this. Replaces the stock env's
    per-joint angle check, which has no consistent meaning once links can fold."""

    success_height_frac: float = 0.85
    """A step counts as 'balanced' when normalised tip height is at least this."""

    success_min_upright_frac: float = 0.8
    """An episode succeeds if it timed out (never fell) *and* was balanced for at least
    this fraction of its steps. This is the signal the adaptive curriculum reads."""


# ----------------------------------------------------------------------------
# root
# ----------------------------------------------------------------------------


@configclass
class MultiLinkCartpoleEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 8.0

    action_space: int = 1
    observation_space: int = 14  # placeholder; recomputed in `MultiLinkCartpoleEnv.__init__`
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,  # TGS
            # A chain of unactuated revolute joints held by a stiff locked joint needs
            # more position iterations than the stock cartpole's default, or the lock
            # visibly creeps under load.
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=1,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        # Every env spawns the identical articulation — difficulty is a runtime joint
        # lock, not a geometry change — so the fast replication path stays available.
        replicate_physics=True,
        clone_in_fabric=True,
    )

    geometry: GeometryCfg = GeometryCfg()
    action: ActionCfg = ActionCfg()
    obs: ObsCfg = ObsCfg()
    reward: RewardCfg = RewardCfg()
    reset: ResetCfg = ResetCfg()
    termination: TerminationCfg = TerminationCfg()
    curriculum: CurriculumCfg = CurriculumCfg(
        difficulty_dim=1,
        init_range=(0.0, 0.15),
        final_range=(0.0, 1.0),
        mode="adaptive",
        adapt_interval=2000,
        adapt_success_threshold=0.7,
        adapt_step=0.05,
    )
