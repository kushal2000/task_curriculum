"""Configuration of the rigid backend: the fold-shape pool of hinged plates, on PhysX.

Isaac Sim 5.1 / Isaac Lab 2.3 (``.venv_isaacsim``). ``RigidClothEnvCfg`` is the task's
``ClothEnvCfg`` (whose ``cloth`` block is the sheet the plates stand in for) plus three blocks:
``rigid_cloth`` (plate physics and its randomisation), ``fold_pool`` (which shapes, how big, folded
starts) and ``physx_chain`` (how contacts are expressed in PhysX).
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from isaacsimenvs.tasks.cloth.mdp.geometry import fold_shapes as fs
from isaacsimenvs.tasks.cloth.env_cfg import ClothEnvCfg

__all__ = ["FoldPoolCfg", "PhysxChainCfg", "RigidClothCfg", "RigidClothEnvCfg", "RigidClothRandomizationCfg"]

#: The VBD cloth's own friction and the shapes' it meets (Newton mixes them by geometric mean).
_CLOTH_MU, _TABLE_MU, _FINGERTIP_MU = 0.25, 0.5, 1.5


@configclass
class PhysxChainCfg:
    """How the plates' contacts are expressed in PhysX terms."""

    slat_friction_combine: str = "min"
    """Friction combine mode on every slat (plate) collider.

    Newton reproduces the cloth's per-pair friction with MJWarp geom priority (slat 1, fingertip 2).
    PhysX has no priority; it has combine modes, and when two differ the higher one wins
    (average < min < multiply < max). Slats at ``min`` with ``rigid_cloth.friction`` (0.354) feel
    min(0.354, 0.5) = 0.354 against the table and the robot links; fingertips at ``max`` with
    ``rigid_cloth.fingertip_friction`` (0.612) win against the slats (0.612) and against the table
    (max(0.612, 0.5) = 0.612) -- the same four pairs Newton's priority rule produces."""

    fingertip_friction_combine: str = "max"
    """Friction combine mode on the fingertip colliders; see ``slat_friction_combine``."""

    friction_model: str = "priority"
    """How contact friction is set up.

    ``"priority"`` (default): the cloth-matched model above -- slats ``min`` at
    ``rigid_cloth.friction``, fingertips ``max`` at ``rigid_cloth.fingertip_friction``, everything
    else PhysX's default 0.5; fixed, friction DR refused.

    ``"cloth_min"``: slats ``min`` at ``rigid_cloth.friction``; the WHOLE robot at
    ``assets.finger_tip_friction`` and the table at ``assets.table_friction``, both average-mode
    (tensor-API written). Pairs: cloth-table = min(slat, table), cloth-robot = min(slat, robot),
    robot-table = (robot + table) / 2. With ``rigid_cloth.randomization`` on, each of the three is
    drawn per env ONCE at startup (``slat_friction_scale``, ``fingertip_friction_scale`` -- the robot
    here --, ``table_friction_scale``), quantised to ``friction_buckets`` values: PhysX caps a scene at
    64K materials and the slats need one USD material (with its combine mode) per bucket."""

    friction_buckets: int = 16
    """Distinct values per randomised friction under ``friction_model: cloth_min``."""

    friction_seed: int = 0
    """Seed of the per-env friction draw under ``friction_model: cloth_min``."""

    contact_offset_margin: float = 0.002
    """A slat's PhysX contact offset is its rest offset (the 7 mm ``rigid_cloth.contact_margin``) plus
    this [m]. PhysX starts generating contacts at the contact offset and holds them at the rest
    offset, so this is how early a contact is seen, not where the surface is."""

    solver_position_iterations: int = 8
    """Each plate articulation's TGS position iterations. 8, as the robot's."""

    solver_velocity_iterations: int = 0
    """Each plate articulation's TGS velocity iterations. 0, as the robot's."""

    hinge_armature: float = 1.0e-3
    """Hinge armature on PhysX [kg m^2], replacing ``rigid_cloth.joint_armature`` (1e-5); 0 keeps that.

    At 1e-5 -- about the plate's own inertia about its hinge -- PhysX's reduced-coordinate solve
    leaks a fingertip's squeeze into the hinge: pressing one plate flat on the table makes the OTHER
    plate jump 5-6 cm (Newton: no jump; the cloth: no lift). Policies trained from scratch on PhysX
    learned to fold with that jump and folded the cloth 3-6% of the time. At 1e-3 the untouched plate
    stays within 3 mm, passive physics (rest height, friction, hinge release, half fold) is
    unchanged, and stage 1 trained here folds the cloth as often as Newton-trained stage 1 (55% / 45%
    vs 52% / 43%). 3e-4 still leaks enough to halve that (24% pinned). See docs/reports/4_physx/physx_port.md."""

    slat_compliant_stiffness: float = 0.0
    """PhysX compliant-contact spring on every slat contact (table, robot, fingertips); 0 = rigid.

    Per unit mass when ``slat_compliant_acceleration`` [1/s^2], else [N/m]. MJWarp's contacts are
    soft constraints (solref/solimp) while PhysX's are rigid; this is the PhysX knob that softens a
    slat's contacts the same way, for calibrating how a fingertip pressing a plate behaves."""

    slat_compliant_damping: float = 0.0
    """Damping of the compliant contact above [1/s per unit mass, or N s/m]."""

    slat_compliant_acceleration: bool = True
    """Read ``slat_compliant_stiffness`` / ``_damping`` per unit mass (MuJoCo's solref is too)."""

    slat_max_depenetration_velocity: float = 0.0
    """Cap on how fast PhysX pushes a slat out of a penetration [m/s]; 0 = leave the converter's.

    A fingertip driven down onto a plate lying on the table squeezes it, and an uncapped
    depenetration shoots the plate out of the sandwich -- the "flick" a PhysX-trained policy learns
    to fold with, which neither a cloth nor the Newton chain allows."""

    gpu_patches_per_env: int = 48
    """PhysX GPU contact-patch buffer per env (``sim.physx.gpu_max_rigid_patch_count`` is raised to at
    least this times ``num_envs``). The pool needed ~16/env at 12288 envs before the hand touched
    anything; the default buffer (163840) overflowed there."""

    gpu_contacts_per_env: int = 256
    """Likewise for ``gpu_max_rigid_contact_count``."""

    gpu_pairs_per_env: int = 128
    """Likewise for ``gpu_found_lost_pairs_capacity`` and ``gpu_total_aggregate_pairs_capacity``."""

    size_quantum: float = 0.001
    """Per-env sheet sizes are quantised to this [m], and one USD per distinct size and shape is
    written. PhysX parses every env on its own (``replicate_physics: false``), so envs can
    simply reference differently sized sheets -- no per-world builder hook as on Newton."""


@configclass
class RigidClothRandomizationCfg:
    """Per-env physics randomisation of the chain, re-drawn for every env at every reset.

    The point is transfer, not robustness for its own sake: what is left between a chain and the
    cloth after the deterministic fixes is dynamics a hinged slab cannot reproduce exactly, so the
    policy is trained across a band of chain physics centred on the cloth-matched values rather
    than on one chain. Every range is a multiplier on (or an offset from) the matched value, so the
    centre of the distribution is the deterministic configuration.

    Uniform draws unless marked log-uniform. Off by default: the deterministic arm and the
    randomised arm differ in this block only.
    """

    enabled: bool = False
    slat_friction_scale: tuple[float, float] = (0.7, 1.4)
    """Multiplies ``slat_friction`` (slat vs table and non-fingertip links)."""
    fingertip_friction_scale: tuple[float, float] = (0.7, 1.4)
    """Multiplies ``fingertip_slat_friction``. Written on the fingertips, so with
    ``friction_priority`` it also moves that env's fingertip-table friction by the same factor.
    Under PhysX ``physx_chain.friction_model: cloth_min`` it multiplies the WHOLE robot's friction
    (``assets.finger_tip_friction``), drawn once per env at startup."""
    table_friction_scale: tuple[float, float] = (1.0, 1.0)
    """PhysX ``physx_chain.friction_model: cloth_min`` only: multiplies ``assets.table_friction``,
    drawn once per env at startup."""
    hinge_damping_scale: tuple[float, float] = (0.5, 3.0)
    """Log-uniform multiplier on ``joint_damping``. The floor is 0.5, not lower: the hinge is only
    stable at ``dt = 1/120`` because of this damping, and at a 0.3 floor one env in 32 went
    non-finite in a forced 180 deg fold of a two-plate chain (not reproduced in 128 at 0.3, none at 0.5 or above)."""
    hinge_stiffness: tuple[float, float] = (0.0, 5.0e-4)
    """Hinge spring [N m / rad]. For scale: a 10 g plate's weight at 25 mm is 2.5e-3 N m, and the
    top of the range returns 0.8e-3 N m at a 90 deg bend -- a third of that, so still a limp sheet."""
    contact_margin_offset: tuple[float, float] = (-0.003, 0.003)
    """Added to ``contact_margin`` [m]: where the surface a fingertip meets sits, +-3 mm."""
    mass_scale: tuple[float, float] = (0.7, 1.5)
    """Multiplies every slat's mass and inertia together (the shape of the mass distribution is
    kept)."""


@configclass
class RigidClothCfg:
    """The plates' physical parameters (every pool shape shares them)."""

    density: float = 2.0
    """Surface density [kg/m^2], PER AREA -- the same number and the same units as
    ``ClothCfg.density``, so the two manipulands have the same total mass (20 g for a 100 mm sheet).
    Split across a shape's plates in proportion to their areas (``assets.write_shape_usd``)."""

    joint_armature: float = 1.0e-5
    """Rotational inertia added to each hinge DOF [kg m^2]. **Load-bearing for stability.**

    A slat is 2-6 mm thick, so its inertia about a hinge axis is of order ``1e-8 kg m^2``. At the
    env's ``sim.dt = 1/120`` that mass matrix is too ill-conditioned to integrate: measured from a
    dead stop in FREE FALL (no contact possible), joint velocity reached 1.3e3-4.9e3 rad/s on the
    FIRST step and the hinges flew past their +-90 deg limits to 1648 deg. Every variant diverged
    within 2-7 steps and took the robot with it.

    Armature adds a diagonal term to the mass matrix, which is the standard MuJoCo remedy for
    exactly this, and the sweep in ``docs/reports/2_rigid_chain/rigid_cloth_approximation.md`` picks the smallest
    value that is stable -- smallest because armature is *fictitious* inertia: it makes the chain
    bend more sluggishly than its geometry says, and a cloth's bending inertia really is negligible.
    So this is the model's largest deliberate departure from the thing it approximates, alongside the
    fingertip friction.

    On PhysX ``physx_chain.hinge_armature`` replaces it."""

    joint_damping: float = 0.05
    """Hinge viscous damping [N m s / rad], applied as an Isaac Lab **actuator** gain.

    Not the URDF's ``<dynamics damping>``: that value does not survive URDF -> USD -> Newton (the
    model reports ``joint_damping = 0.0`` for every slat hinge however the URDF is written), so
    setting it there looks applied and is not. ``URDF_DEFAULTS["joint_damping"]`` is pinned at 0.0
    to keep the two from appearing to disagree, and this field is what actually damps.

    **Not calibrated against the cloth.** The sheet's bending damping is ``edge_kd = 1.0e-3`` per
    mesh edge and the map from that to a per-hinge value depends on how many edges a hinge stands
    in for. Chosen with ``joint_armature`` in the stability sweep instead."""

    joint_stiffness: float = 0.0
    """Hinge spring stiffness [N m / rad], applied as an Isaac Lab actuator gain against a zero
    target. Zero means a limp chain: it holds no shape of its own and drapes under gravity alone.

    A cloth is not limp -- it has bending rigidity ``D = E h^3 / (12 (1 - nu^2))`` -- and
    :func:`rigid_cloth.plate_joint_stiffness` converts a plate rigidity into the equivalent
    per-hinge spring. Left at zero because the cloth's own bending is itself nearly zero
    (``edge_ke = 8.0e-4``, and measurements said the response saturates by 0.05), so a limp chain is
    the closer match until someone runs the cantilever test on both."""

    slat_friction: float = -1.0
    """Friction a plate feels against the table and the robot links. Negative computes it (a float
    sentinel, not ``None``: the config checker refuses a float override of a ``None`` default):
    the cloth's effective value against the table, ``sqrt(0.25 * 0.5) = 0.354`` -- Newton's VBD
    mixes a particle's ``soft_contact_mu`` with a shape's by geometric mean."""

    fingertip_slat_friction: float = -1.0
    """Friction a plate feels against a fingertip. Negative means the cloth's value against the
    1.5 fingertip, ``sqrt(0.25 * 1.5) = 0.612``."""

    contact_margin: float = -1.0
    """Outward collision offset [m] on every plate (PhysX rest offset). Negative matches the cloth:
    ``cloth.thickness / 2 - plate thickness / 2``, resolved by the env against the pool's nominal
    plate thickness.

    The VBD sheet collides through particles of radius ``cloth.thickness / 2`` = 8 mm, so it rests
    with its mid-plane 8 mm above the table and a fingertip meets it 8 mm above that. A 2 mm plate
    with a 7 mm margin puts both where the cloth's are. Plate-plate contact would sum two margins,
    so those pairs are off whenever this is nonzero (articulation self-collision disabled)."""

    randomization: RigidClothRandomizationCfg = RigidClothRandomizationCfg()

    def __post_init__(self) -> None:
        if self.density <= 0.0:
            raise ValueError(f"density must be positive, got {self.density}")
        if self.joint_damping < 0.0:
            raise ValueError(f"joint_damping must be >= 0, got {self.joint_damping}")
        if self.joint_armature < 0.0:
            raise ValueError(f"joint_armature must be >= 0, got {self.joint_armature}")
        if self.joint_stiffness < 0.0:
            raise ValueError(f"joint_stiffness must be >= 0, got {self.joint_stiffness}")
        if self.slat_friction < 0.0 and self.slat_friction != -1.0:
            raise ValueError(f"slat_friction must be >= 0 (or -1: computed), got {self.slat_friction}")

    @property
    def friction(self) -> float:
        """The plate friction actually applied, resolving the sentinel."""
        if self.slat_friction >= 0.0:
            return float(self.slat_friction)
        return math.sqrt(_CLOTH_MU * _TABLE_MU)

    @property
    def fingertip_friction(self) -> float:
        """The plate-fingertip friction actually applied, resolving the sentinel."""
        if self.fingertip_slat_friction >= 0.0:
            return float(self.fingertip_slat_friction)
        return math.sqrt(_CLOTH_MU * _FINGERTIP_MU)


@configclass
class FoldPoolCfg:
    shapes: list[str] = list(fs.DEFAULT_SHAPES)
    """Pool members, by ``fold_shapes.POOL`` name. Each env carries ALL of them every episode and
    uses one; the rest are parked. Order is the order of the per-shape logs."""

    weights: list[float] | None = None
    """Relative draw probabilities, one per shape. ``None`` is uniform."""

    fold_lift_from_ply_gap: bool = True
    """Lift the fold target by the pool's ply gap (0 mm for mid-plane hinges) instead of the cloth's
    ``self_contact_radius`` (2 mm). The gap must agree across the pool: the target is built once for
    every env."""

    park_distance: float = 3.0
    """How far (+y, from the env origin) the parked shapes hover [m]. Only has to clear this env's
    own robot and table: worlds never collide, so another env's shapes are never in the way."""

    park_spacing: float = 0.3
    """Gap between consecutive parked shapes along +y [m]; a 100 mm sheet needs more than 0.1."""

    park_height: float = 2.0
    """Height of the parked shapes above the env origin [m]."""

    size_range: list[float] = [0.10, 0.10]
    """Sheet side length [m], drawn ONCE PER ENV at startup and fixed for the whole run.

    Not per episode: each env references a USD of its own size (``physx_chain.size_quantum``), built
    before the simulator starts. Every shape in an env shares its size. The task -- fold targets,
    keypoints, footprint, ``object_scales`` -- follows each env's size."""

    thickness_range: list[float] = [0.002, 0.002]
    """Plate (slab) thickness [m], drawn once per env like ``size_range``. NOT the thickness the
    hand feels: that is slab + 2 x ``rigid_cloth.contact_margin`` (2 + 2 x 7 = 16 mm by default),
    and the margin is already randomised PER EPISODE by ``rigid_cloth.randomization``."""

    size_seed: int = 0
    """Seed for the per-env size / thickness draws, so a run's sizes are reproducible."""

    reach_shift: list[float] = [0.0, 1.0]
    """Unit direction (env-frame x, y) toward the robot base. A sheet larger than ``cloth.size``
    has its reset centre moved this way by the growth of its half-diagonal, so its farthest point
    (at any yaw) stays where a ``cloth.size`` sheet's would be: the arm reaches only ~18 cm past the
    table centre on the far side. ``[0, 0]`` disables the shift."""

    one_shape_per_env: bool = False
    """Each env carries ONE pool shape, fixed for the run, instead of all of them.

    Assigned round-robin (env ``i`` -> shape ``i % S``; ``weights`` -> a seeded draw), so every SAPG
    block sees every shape. The shape distribution is then per env rather than per episode, which
    with thousands of envs is the same distribution. 1.6-1.8x faster end to end at 24576 envs: the
    four parked articulations per env are gone (PhysX steps every articulation, parked or not; a
    single-shape pool runs 107-117k env-steps/s against 62-66k for all five)."""

    init_fold_prob: float = 0.0
    """Share of episodes that start partly folded instead of flat; 0 = always flat.

    A folded start picks one of the active shape's SINGLE-STAGE named folds uniformly (the half fold
    included; ``box2-mid`` has only that one) and turns its hinges up by an angle drawn uniformly from
    ``init_fold_deg``. A single-stage fold is a rigid rotation about one crease line, so a shape with
    seams stays closed at any angle (checked at startup). Multi-stage folds are excluded: their first
    stage is a full 180 deg fold."""

    init_fold_deg: list[float] = [15.0, 120.0]
    """Range of the folded start's angle [deg], drawn uniformly per episode (per PART with
    ``init_fold_starts: other``). Positive = the moving plates lift off the table, so no start can
    press a plate into it. Up to 180 (fully folded) with ``other``; below 180 with ``single``,
    whose starts include the target fold itself."""

    init_fold_starts: str = "single"
    """Which folded starts.

    * ``single``: every single-stage named fold, the target half fold included, all its hinges at
      one angle (the 2026-10-06 runs).
    * ``other``: ``FoldShape.start_poses`` -- every legal combination of the NON-target single-stage
      folds (corners, strips, diagonals, bottom-up), each part at its own angle. A shape whose only
      fold is the target (``box2``) starts flat."""

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.init_fold_prob) <= 1.0:
            raise ValueError(f"fold_pool.init_fold_prob must be in [0, 1], got {self.init_fold_prob}")
        if self.init_fold_starts not in ("single", "other"):
            raise ValueError(f"fold_pool.init_fold_starts must be single or other, got {self.init_fold_starts!r}")
        lo, hi = (float(v) for v in self.init_fold_deg)
        top = 180.0 if self.init_fold_starts == "other" else 180.0 - 1e-6
        if not 0.0 <= lo <= hi <= top:
            raise ValueError(
                f"fold_pool.init_fold_deg must be 0 <= lo <= hi {'<=' if top == 180.0 else '<'} 180 "
                f"with init_fold_starts={self.init_fold_starts}, got {[lo, hi]}"
            )
        unknown = [n for n in self.shapes if n not in fs.POOL]
        if unknown:
            raise ValueError(f"unknown fold_pool.shapes {unknown}; known: {sorted(fs.POOL)}")
        if not self.shapes:
            raise ValueError("fold_pool.shapes is empty")
        if self.weights is not None:
            if len(self.weights) != len(self.shapes):
                raise ValueError(
                    f"fold_pool.weights has {len(self.weights)} entries for {len(self.shapes)} shapes"
                )
            if any(w < 0.0 for w in self.weights) or sum(self.weights) <= 0.0:
                raise ValueError("fold_pool.weights must be non-negative with a positive sum")
        for name in ("size_range", "thickness_range"):
            lo, hi = (float(v) for v in getattr(self, name))
            if not 0.0 < lo <= hi:
                raise ValueError(f"fold_pool.{name} must be 0 < lo <= hi, got {[lo, hi]}")
        if self.park_spacing <= 1.5 * float(self.size_range[1]):
            raise ValueError(
                f"park_spacing {self.park_spacing} m would let parked sheets of up to "
                f"{self.size_range[1]} m overlap"
            )


@configclass
class RigidClothEnvCfg(ClothEnvCfg):
    """The fold task on the fold-shape pool, on PhysX."""

    rigid_cloth: RigidClothCfg = RigidClothCfg()
    physx_chain: PhysxChainCfg = PhysxChainCfg()
    fold_pool: FoldPoolCfg = FoldPoolCfg()
