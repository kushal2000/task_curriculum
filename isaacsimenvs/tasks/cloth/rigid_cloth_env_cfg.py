"""RigidClothEnvCfg — the cloth-folding task with a hinged rigid chain in place of the VBD sheet.

Subclasses :class:`~isaacsimenvs.tasks.cloth.cloth_env_cfg.ClothEnvCfg` and keeps its ``cloth:``
block, because that block is where the *task* is defined: ``size``, ``resolution`` and ``fold_axis``
fix the particle grid the fold is scored on, and ``start_height`` / ``table_half_thickness`` fix
where the sheet spawns. Only the manipuland's **dynamics** change, so only the solver and the
manipuland's own parameters are new.

**The solver topology collapses.** ``ClothEnvCfg`` builds a coupled MJWarp + VBD solve with a proxy
mapping, because a particle sheet and a rigid robot cannot be integrated by one solver. A chain of
hinged boxes is rigid, so it is MJWarp's to begin with: :meth:`RigidClothEnv.build_physics_cfg`
falls back to ``PlayNewtonEnv``'s plain ``cfg.newton.build(...)``. Everything in ``cloth:`` from
``proxy_mode`` down -- substeps, VBD iterations, self-contact radii, AVBD ramping, contact buffers --
is **inert here**, and is left in place rather than deleted so the two configs stay diffable and so
``ClothCfg``'s calibration notes remain the reference for what is being approximated.

**What is NOT inherited and has to be re-derived.** Three of the cloth's physical parameters do not
transfer, and each has a field below rather than a silent reuse:

  * *friction*. Newton's VBD mixes the two materials as ``sqrt(mu_a * mu_b)``; MJWarp takes
    ``max(mu_a, mu_b)``. Copying ``soft_contact_mu = 0.25`` onto a rigid slat therefore makes it as
    grippy as the grippiest thing it touches instead of slipperier than everything -- 1.5 against a
    fingertip where the cloth feels 0.61. See :func:`rigid_cloth.cloth_effective_friction` and
    ``slat_friction`` below.
  * *bending stiffness*. The cloth's bending lives in ``edge_ke`` / ``edge_kd`` on every mesh edge; a
    chain's lives in the hinges. URDF cannot express a spring, so stiffness is an actuator setting
    (``joint_stiffness``) and damping is in the URDF (``joint_damping``).
  * *where the plies come to rest*. The cloth settles folded plies at ``self_contact_radius`` = 2 mm;
    a chain settles them at its own ply gap, which is 5.88 mm for the mid-hinge uniform chains. The
    fold target is built from that number, so ``fold_lift_from_ply_gap`` decides whether a perfect
    fold scores zero for this variant or carries the cloth's offset as a constant bias.
"""

from __future__ import annotations

# IMPORT ORDER IS LOAD-BEARING; see the note in `cloth_env_cfg`.
from isaacsimenvs.tasks.cloth.cloth_env_cfg import ClothEnvCfg  # isort: skip

from isaaclab.utils import configclass  # noqa: E402

from isaacsimenvs.tasks.cloth.utils import rigid_cloth as rc  # noqa: E402
from isaacsimenvs.tasks.cloth.utils import rigid_cloth_assets as rc_assets  # noqa: E402

__all__ = ["RigidClothCfg", "RigidClothEnvCfg", "RigidClothRandomizationCfg"]


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
    ``friction_priority`` it also moves that env's fingertip-table friction by the same factor."""
    hinge_damping_scale: tuple[float, float] = (0.5, 3.0)
    """Log-uniform multiplier on ``joint_damping``. The floor is 0.5, not lower: the hinge is only
    stable at ``dt = 1/120`` because of this damping, and at a 0.3 floor one env in 32 went
    non-finite in a forced 180 deg fold of ``box2-surface`` (not reproduced in 128 at 0.3, none at
    0.5 or above)."""
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
    """The chain itself: which of the five approximations, and its physical parameters."""

    # `density` and `joint_limit` below are imported from `rigid_cloth_assets.URDF_DEFAULTS` rather
    # than written here, because those two are the ONLY fields that enter the URDF text -- and the
    # URDF text is the USD cache key. The bake cannot import this module (it inherits PlayEnvCfg,
    # which needs Isaac Lab 3.0's SimulationCfg), so the shared definition has to live in the
    # kit-free assets module. Overriding either at run time is a different asset and needs its own
    # bake; it presents as a cache miss.
    #
    # `joint_damping` is NOT one of them, despite appearing in the URDF's `<dynamics>`: the importer
    # drops that tag (the Newton model reports 0.0 on every slat hinge), so the damping that acts is
    # the Isaac Lab actuator gain set here, and `write_chain_urdf_for_run` refuses to take it as a
    # parameter at all. That refusal is deliberate -- the bake and the env once wrote different
    # values of it into the same URDF, and every run died on a cache miss.

    variant: str = "box3-mid"
    """Which approximation to build; one of :data:`rigid_cloth.VARIANTS`.

    Defaults to ``box3-mid`` -- two plates and a 2 mm crease bar -- because it is the only variant
    that is simultaneously exact in the fold (zero in-plane residual), correct in the ply gap
    (2.00 mm measured against 2.00 mm predicted, matching the cloth's own 2 mm), bidirectional, and
    cheap (3 bodies, ~175 k steps/s in the MuJoCo probe). ``box-mid-odd`` is the faithful-shape
    choice at 17 bodies; ``box-surface`` is the one measured to diverge and is included to be
    measured, not to be trained on. See ``docs/results/rigid_cloth_approximation.md``."""

    density: float = rc_assets.URDF_DEFAULTS["density"]
    """Surface density [kg/m^2], PER AREA -- the same number and the same units as
    ``ClothCfg.density``, so the two manipulands have the same total mass (20 g for a 100 mm sheet).
    ``write_chain_urdf`` splits it across the slats in proportion to their widths, so
    ``box3-mid``'s thin bar is light and its plates are heavy."""

    joint_armature: float = 1.0e-5
    """Rotational inertia added to each hinge DOF [kg m^2]. **Load-bearing for stability.**

    A slat is 2-6 mm thick, so its inertia about a hinge axis is of order ``1e-8 kg m^2``. At the
    env's ``sim.dt = 1/120`` that mass matrix is too ill-conditioned to integrate: measured from a
    dead stop in FREE FALL (no contact possible), joint velocity reached 1.3e3-4.9e3 rad/s on the
    FIRST step and the hinges flew past their +-90 deg limits to 1648 deg. Every variant diverged
    within 2-7 steps and took the robot with it.

    Armature adds a diagonal term to the mass matrix, which is the standard MuJoCo remedy for
    exactly this, and the sweep in ``docs/results/rigid_cloth_approximation.md`` picks the smallest
    value that is stable -- smallest because armature is *fictitious* inertia: it makes the chain
    bend more sluggishly than its geometry says, and a cloth's bending inertia really is negligible.
    So this is the model's largest deliberate departure from the thing it approximates, alongside the
    fingertip friction.

    Note the MuJoCo probe never caught this: it ran at ``--dt 1/240``, half the env's step."""

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

    joint_limit: float | None = rc_assets.URDF_DEFAULTS["joint_limit"]
    """Per-hinge bend ceiling [rad]. ``None`` takes ``ChainSpec.joint_limit``, which for a mid-plane
    hinge is the angle at which slats ``i`` and ``i+2`` touch -- i.e. the chain's own thickness
    expressed as a kinematic limit, so the plies cannot interpenetrate even before contact is
    resolved."""

    slat_friction: float | None = None
    """Friction a slat feels against the table and the non-fingertip robot links. ``None``
    computes it: :func:`rigid_cloth.cloth_effective_friction` against the table, i.e.
    ``sqrt(0.25 * 0.5) = 0.354`` -- Newton's VBD mixes a particle's ``soft_contact_mu`` with a
    shape's by geometric mean (``_average_contact_material``), so that is what the sheet feels.

    Only honoured with ``friction_priority``: under MJWarp's equal-priority rule a contact takes the
    LARGER of its two coefficients, so a 0.354 slat on the 0.5 table used to feel 0.5, not 0.354."""

    fingertip_slat_friction: float = -1.0
    """Friction a slat feels against a fingertip. Negative means the cloth's value against the
    1.5 fingertip, ``sqrt(0.25 * 1.5) = 0.612``. Only honoured with ``friction_priority``.
    (A float sentinel, not ``None``: the config checker refuses a float override of a ``None``
    default, so ``None`` would make the field impossible to set from the command line.)"""

    friction_priority: bool = True
    """Resolve the chain's contact friction by MJWarp geom **priority** instead of by ``max``.

    MJWarp (``contact_params``) takes the higher-priority geom's friction outright, and the larger of
    the two only on a tie. So: slats at priority 1 carrying ``slat_friction``, fingertips at
    priority 2 carrying ``fingertip_slat_friction``. That reproduces the cloth's slat-table (0.354),
    slat-fingertip (0.612) and slat-link (0.354) contacts exactly, where the old ``max`` rule gave
    0.5 / 1.5 / 0.5.

    **The price is fingertip-TABLE friction**: the fingertip now wins against the table too, so that
    contact drops from 1.5 to 0.612. No single-geom assignment can hit all four pairs (a slat's
    friction appears in two pairs that need different values), and the fingertip-table pair is the
    one that does not touch the manipuland. False restores the old behaviour."""

    contact_margin: float = -1.0
    """Outward collision offset [m] written onto every slat shape (Newton ``shape_margin``).
    Negative matches the cloth: ``cloth.thickness / 2 - slat thickness / 2``.

    The VBD sheet collides through particles of radius ``cloth.thickness / 2`` = 8 mm, so it rests
    with its mid-plane 8 mm above the table (measured: z 0.538 on a 0.530 table) and a fingertip
    meets it 8 mm above that. A 2 mm slat rested at 0.531 and was met at 0.532 -- 14 mm lower. A
    policy that learns to press onto that surface presses 14 mm into the cloth. Margins are summed
    per contact and MJWarp holds a contact at its margin, so a 7 mm slat margin puts both the
    mid-plane and the contact surface where the cloth's are. Zero restores the old behaviour.

    Slat-slat contact would sum to twice the margin and hold a folded chain's plies 14 mm apart,
    so those pairs are removed from the collision pipeline whenever this is nonzero; a chain's
    joint limits already stop the plies at their geometric contact."""

    fold_lift_from_ply_gap: bool = True
    """Build the fold target at the CHAIN's ply gap instead of the cloth's ``self_contact_radius``.

    ``ClothEnv._init_fold_targets`` places the target one ``self_contact_radius`` (2 mm) above the
    stationary half, because that is where cloth-on-cloth contact settles. A chain settles where its
    own geometry says: 2.00 mm for ``box3-mid`` and the surface-hinge variants, 5.88 mm for the
    mid-hinge uniform ones. With this True a geometrically perfect fold scores ~0 for every variant,
    which makes the five comparable to each other; with it False they are all scored against the
    cloth's 2 mm and the mid-hinge chains carry a constant ~3.9 mm penalty -- 10% of the 0.04
    tolerance, spent before the policy acts."""

    randomization: RigidClothRandomizationCfg = RigidClothRandomizationCfg()

    def __post_init__(self) -> None:
        if self.variant not in rc.VARIANTS:
            raise ValueError(
                f"unknown rigid_cloth.variant {self.variant!r}; "
                f"known: {sorted(rc.VARIANTS)}"
            )
        if self.density <= 0.0:
            raise ValueError(f"density must be positive, got {self.density}")
        if self.joint_damping < 0.0:
            raise ValueError(f"joint_damping must be >= 0, got {self.joint_damping}")
        if self.joint_armature < 0.0:
            raise ValueError(f"joint_armature must be >= 0, got {self.joint_armature}")
        if self.joint_stiffness < 0.0:
            raise ValueError(f"joint_stiffness must be >= 0, got {self.joint_stiffness}")
        if self.slat_friction is not None and self.slat_friction < 0.0:
            raise ValueError(f"slat_friction must be >= 0, got {self.slat_friction}")

    # --- derived

    @property
    def spec(self) -> rc.ChainSpec:
        return rc.variant(self.variant)

    @property
    def friction(self) -> float:
        """The slat friction actually applied, resolving ``None``."""
        if self.slat_friction is not None:
            return float(self.slat_friction)
        return rc.cloth_effective_friction(rc.CLOTH_REFERENCE["shape_mu"]["table"])

    @property
    def fingertip_friction(self) -> float:
        """The slat-fingertip friction actually applied, resolving ``None``."""
        if self.fingertip_slat_friction >= 0.0:
            return float(self.fingertip_slat_friction)
        return rc.cloth_effective_friction(rc.CLOTH_REFERENCE["shape_mu"]["finger_tip"])

    def resolved_contact_margin(self, cloth_thickness: float) -> float:
        """The slat collision margin actually applied, resolving the sentinel against the cloth."""
        if self.contact_margin >= 0.0:
            return float(self.contact_margin)
        return max(0.5 * float(cloth_thickness) - 0.5 * self.spec.thickness, 0.0)

    @property
    def ply_gap(self) -> float:
        return rc.chain_ply_gap(self.spec)


@configclass
class RigidClothEnvCfg(ClothEnvCfg):
    """Play task with a hinged rigid chain approximating the folding cloth."""

    rigid_cloth: RigidClothCfg = RigidClothCfg()
