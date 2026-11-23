"""VbdClothCfg -- the VBD cloth sheet: the task's ``ClothCfg`` plus its material, contact and the
coupling that lets the hand feel it.

The ``cloth`` block of ``vbd.VbdClothEnvCfg``; values in ``config/task.yaml``. The Newton physics
built from it is ``backends/vbd/physics.py``.
"""

from __future__ import annotations

from dataclasses import MISSING

# The function, not the package attribute (see `cloth.py`).
from isaaclab.utils.configclass import configclass

from .cloth import ClothCfg

__all__ = ["VbdClothCfg"]


@configclass
class VbdClothCfg(ClothCfg):
    """The sheet as a VBD particle cloth on Newton, coupled to the robot.

    The material, contact and self-contact values are SIM1's calibrated cloth physics (arXiv
    2604.08544 Table 2, cross-checked against InternRobotics/SIM1 @ 973476b), adopted rather than
    measured here; each field says what it displaced. Two things are deliberately NOT copied:

    * the AVBD edge strain limit (xi = 0.05 and its ramping constants). Newton 1.5 has no per-edge
      strain constraint for cloth -- its AVBD is joints and contacts -- so that needs a solver port,
      not a config line. It does not run in SIM1's released env either: the limit is read from
      ``spring_control``, ``add_cloth_mesh`` hardcodes ``control=0.0``, and their env never passes
      ``add_springs``, so ``spring_count == 0`` and every AVBD kernel is guarded out.
    * Table 2's robot / table frictions (1.5 / 0.0), which their env overwrites to 1.0 on every shape
      right after building the table.
    """

    reset_offset_xy: list[float] = MISSING
    """Env-frame shift of the sheet's reset centre [m], added to every reset (pinned or random).

    For sheets larger than the 10 cm default: the arm reaches only ~18 cm past the table centre on
    the far side, so a big sheet is moved toward the robot (+y), as the fold pool does with its
    ``reach_shift`` -- by the growth of the half-diagonal, ``(size - 0.10) * sqrt(2) / 2``, which
    keeps the farthest point at any yaw where a 10 cm sheet's would be (0.0707 m for 20 cm)."""


    # --- material -------------------------------------------------------------------------
    density: float = MISSING
    """Surface density [kg/m^2]. **Per AREA, not per volume.**

    This reaches ``ModelBuilder.add_cloth_mesh(density=...)`` unchanged (via the ``newton:density``
    USD attribute, read in ``isaaclab_contrib.deformable.deformable_object``), and Newton documents
    that argument as "the density per-area of the mesh". The old 100.0 was written as though it
    were volumetric: it made the 0.1 x 0.1 m sheet weigh **1 kg**, a handkerchief with the mass of
    a litre of water, and it disagreed by 83x with the ``area_density = density * thickness``
    the env then computed for a rigid-object mass proxy of the sheet.

    2.0 kg/m^2 is SIM1 Table 2. Still heavy for cloth -- real cotton sheeting is 0.15-0.3 kg/m^2 --
    but it is their calibrated value and it is 50x lighter than what this ran before."""
    tri_ke: float = MISSING
    """In-plane stretch stiffness -- the Lame ``mu`` (shear modulus) of the membrane model.

    **1.0e2 is SIM1 Table 2, and also Newton's own ``default_tri_ke``.** Their "calibrated"
    elasticity is the library default; what they actually tuned away from default is bending.

    This was 1.0e4. The justification was the cable, where softening stretch was catastrophic
    (x0.01 peaked at 94.9 m/s) -- but a cable is a 1-D chain whose only in-plane mode IS stretch,
    and the sheet has an area term (``tri_ka``) carrying load the cable had no analogue for. The
    cable result is not evidence about a membrane. If the sheet oscillates or blows up, this is the
    first suspect and 1.0e3 is the obvious intermediate."""
    tri_ka: float = MISSING
    """In-plane shear/area stiffness -- the Lame ``lambda`` (area modulus). SIM1 Table 2.

    Note their own solver comment recommends ``lambda: 1000.0+`` to maintain area against
    penetration, and then their env ships 1e2. Table 2 agrees with the env, so 1e2 it is."""
    tri_kd: float = MISSING
    """In-plane damping. SIM1 Table 2 (was 1.0e-5 here).

    Newton's ``default_tri_kd`` is 10.0; both this and the previous value are far below it, which
    is right for a membrane that has to drape rather than settle."""
    edge_ke: float = MISSING
    """Bending stiffness. Low -- a cloth that will not bend cannot be folded. This is the opposite
    of the cable, where a stiff bend response was the thing ringing.

    **8.0e-4 is SIM1 Table 2.** Note their released env ships ``bending_ke = 1e-4``; Table 2 and
    the code disagree, and Table 2 is the stated calibration, so this follows Table 2.

    The previous 0.5 was measured here with a teleported fold: at 5.0 the crease springs open after
    **1** step; at 0.5 it holds **7**; a further 10x softer (0.05) gives the same 7. That measured
    saturation is the reason this change is expected to be inert -- 8.0e-4 is well past the knee,
    so it should behave like 0.05, which behaved like 0.5. It is adopted for alignment, not for an
    expected gain.

    One caveat on transferring it at all: the dihedral bending energy scales with edge length, and
    SIM1 tuned this on a scanned garment with ~mm edges against our 16.7 mm grid. The number is not
    dimensionally portable even though the saturation argument says it should not matter."""
    edge_kd: float = MISSING
    """Bending damping. SIM1 Table 2 (was 1.0e-2 here)."""

    # --- coupling (defaults carried from the cable result) --------------------------------
    proxy_mode: str = MISSING
    """See `docs/phase4_cable_env.md`. `lagged` syncs the hand's *begin* pose with its *end*
    velocities; the mismatch scales with hand speed and was the single change that took the cable
    from ~0 to 5 goals in an episode."""

    substeps: int = MISSING
    """Cloth substeps inside one coupled step, i.e. the VBD timestep is ``sim.dt / newton.num_substeps
    / substeps``.

    **The cloth dt is 1/1440 s, matching SIM1** (60 Hz with 24 substeps, 0.694 ms). task.yaml gets
    there as 1/120 / 6 / 2: the ns6 protocol (2026-11-12), six MJWarp substeps so the light Wuji
    fingers do not chatter against the sheet, two cloth substeps in each. Before that it was 1/120 /
    2 / 6 (the same cloth dt, a third of the MJWarp substeps), and before that 1/120 / 2 / 2 (1/480 s).

    **This directly contradicts a cable measurement**: on the cable there was a sharp optimum at 2,
    with 1 peaking at 37 m/s and 4/8/16 degrading monotonically, on the theory that fewer, larger
    substeps mean fewer proxy exchanges per step and the exchange is where the energy enters. That
    was never re-measured for the sheet, and the cloth is a different solver entry, but it is a
    direct conflict and not a gap -- if the sheet gains energy at reset or under contact, revert
    this first.

    Costs 3x the VBD work per policy step of the 1/480 s setting, on top of the iteration increase."""

    vbd_iterations: int = MISSING
    """VBD solver iterations per substep.

    **12 is SIM1's ``lift2`` value** (their ``acone`` task uses 16). Everything below was measured
    at 10, which is Newton's own default and was 8x cheaper than the 80 this inherited from the
    cable; 12 is a 20% increase on that and the cost model below predicts it directly. Measured at
    256 envs on pinned hardware, per-step cost is almost entirely this term:

        ms/step ~ 45.5 + 11.52 x iterations

    which fits every point (10 -> 160.7, 20 -> 275.9, 40 -> 489.0, 80 -> 933.7 measured). At 80
    iterations ~96% of the step is VBD. Dropping to 10 gives 5.80x throughput at 1024 envs
    (1077 -> 6253 env sps) with memory unchanged at 44.85 GB, and removes the ~20% per-step
    step-up previously seen above 128 envs -- that overhead was itself iteration work, so the scene
    now scales flat from 1 to 1024 envs.

    Quality is unchanged, on five seeds / 160 episodes each side:

        80 iters: 2/160 goals, best_fold_err 0.0775 +- 0.0028, falls 2-5
        10 iters: 3/160 goals, best_fold_err 0.0792 +- 0.0012, falls 2-4

    The error bars overlap heavily and nothing is monotonic in iteration count, so the claim is "10
    is not worse", not "10 is better". 80 was never validated for the cloth; 10 is the documented
    default of both `SolverVBD` (`solver_vbd.py:233`) and `isaaclab_contrib`'s VBD config, and
    Newton's own coupled-solver example and ADMM test use 10 and 5.

    That best_fold_err is flat across an 8x range also rules something out: the ~0.077 stopping
    point is not a solver-convergence limit.

    **10 is a COUPLING floor, not a cloth-physics one.** Newton's standalone cloth examples run 4-10
    (example_cloth_franka.py uses 5, example_cloth_twist.py 4), and iterations here serve two jobs at
    once: converging the sheet's internal dynamics, and resolving contact against the rigid proxy.
    What degrades at 5 is only the second -- the hand stops gripping, footprint stays near 1.0, the
    sheet is untouched rather than unstable -- so the extra iterations are buying contact, not cloth.

    A previous version of this note proposed `SolverVBD(rigid_contact_history=True)` as the targeted
    fix, and recorded it as tried-and-reverted because contact warm-starting packs contact ids into
    20 bits, capping buffered contacts at 2**20 = 1,048,576 globally while this scene budgets
    7,962,624 triangle pairs at 32 envs. **That blocker is real but it was aimed at the wrong
    lever.** In Newton 1.5 `rigid_contact_history` is body-BODY only -- it drives
    `snapshot_body_body_contact_history` and never touches body-particle contacts, so it could not
    have helped the sheet's grip even if it had fit.

    The knob that does reach cloth-vs-hand contact is `rigid_avbd_beta`; see that field."""
    coupler_iterations: int = MISSING
    """Raising this made the cable *worse* at every setting tried."""

    mass_scale: float = MISSING
    """KEEP AT 1.0. 0.05 was ~500x worse on the cable and NaN'd the robot."""

    collide_interval: int = MISSING

    proxy_relaxation: float = MISSING
    """Under-relaxation of the cloth -> hand feedback force: applied = r * new + (1 - r) * previous.
    1.0 is off (Newton's default). The exchange is explicit, so light fingers (Wuji armature
    2-5e-4) overshoot and chatter on contact; r < 1 damps that. With ``coupler_iterations`` 1 the
    Aitken mode is identical to fixed (its state resets every coupled step)."""

    proxy_relaxation_mode: str = MISSING
    """``fixed`` or ``aitken`` (adaptive per coupled step; needs ``coupler_iterations`` > 1)."""

    proxy_relaxation_min: float = MISSING
    """Aitken lower bound."""

    use_cuda_graph: bool = False
    """Capture the coupled MJWarp + VBD step as a CUDA graph. Off by default because every result so
    far was produced without it (inherited from the cable, whose reason -- entries stepping at
    different rates -- was never measured: the substep counts are fixed, so a graph records them).
    At small env counts the step is kernel-LAUNCH bound (decimation 2 x substeps 6 x 12 VBD
    iterations per policy step), which is what makes the live viewer's cloth ~4 steps/s."""

    soft_contact_ke: float = MISSING
    """Body-particle contact stiffness [N/m]. SIM1 Table 2 (was 8.0e3 here).

    The per-contact value is the AVERAGE of this and the rigid shape's own material stiffness
    (`NewtonShapeCfg.ke`, 2.5e3), so the effective ceiling moves 5250 -> 1500."""

    soft_contact_kd: float = MISSING
    """Body-particle contact damping [N*s/m]. SIM1 Table 2.

    NEW FIELD. `NewtonModelCfg` defaults it to 1.0e-2 and `build_physics` was passing only `ke` and
    `mu`, so this ran at the library default and was never stated."""

    soft_contact_mu: float = MISSING
    """Body-particle friction coefficient. SIM1 Table 2 (`self_contact_friction`; was 10.0 here).

    Effective per-contact friction is `sqrt(soft_contact_mu * shape_mu)`, so against the fingertips
    (`assets.finger_tip_friction` 1.5) this moves from `sqrt(10 x 1.5)` = 3.9 to `sqrt(0.25 x 1.5)`
    = 0.61. SIM1's own effective value is 0.5 -- they set every shape's mu to 1.0.

    Table 2's `robot_friction` 1.5 / `table_friction` 0.0 are NOT copied: their env overwrites
    every shape's mu to 1.0 immediately after building the table, so those two rows describe a
    configuration that does not run. Our per-shape frictions stay as the rigid-tool task sets
    them.

    The cloth's own mu, in every contact the cloth makes: cloth-rigid friction is
    ``sqrt(soft_contact_mu * shape mu)`` and cloth-cloth is ``soft_contact_mu``. One number, so it moves
    all of them together; the ``cloth_*_friction`` below set one contact directly."""

    cloth_cloth_friction: float | None = MISSING
    """The friction of cloth on cloth (a folded flap on the lower half). ``null``: ``soft_contact_mu``.

    Set, it becomes the cloth's mu, and the cloth solver's copy of the shapes' mu is scaled by
    ``soft_contact_mu / cloth_cloth_friction`` so cloth-hand and cloth-table stay as they were
    (``backends/vbd/friction.py``)."""

    cloth_table_friction: float | None = MISSING
    """The friction of cloth on the table. ``null``: ``sqrt(soft_contact_mu * assets.table_friction)``
    (0.35 at the task values). A slide-onset measurement gives tan(angle).

    Set, the table's mu in the cloth solver's copy is solved so the cloth-table contact comes out at
    exactly this value. The hand against the table (MJWarp) keeps ``assets.table_friction``."""

    cloth_fingertip_friction: float | None = MISSING
    """The friction of cloth on the five fingertips. ``null``: ``sqrt(soft_contact_mu *
    assets.finger_tip_friction)`` (0.61 at the task values). Set, as ``cloth_table_friction``; the
    fingertips against the table (MJWarp) keep ``assets.finger_tip_friction``."""

    cloth_robot_friction: float | None = MISSING
    """The friction of cloth on every other robot shape the cloth feels (palm and phalanges,
    ``proxy_links``). ``null``: ``sqrt(soft_contact_mu * assets.robot_friction)`` (0.35 at the task
    values). Set, as ``cloth_table_friction``; MJWarp keeps ``assets.robot_friction``."""

    rigid_contact_k_start: float = MISSING
    """Body-particle contact penalty SEED for AVBD ramping [N/m].

    **Inert unless `rigid_avbd_beta` > 0**, which is why it did nothing until now. Newton computes
    `rigid_contact_k_start_value = -1.0 if linear_beta == 0.0 else k_start` (solver_vbd.py:677) and
    the kernel reads `k_floor = avg_ke if k_start < 0.0 else min(k_start, avg_ke)`
    (rigid_vbd_kernels.py:3395) -- so with beta 0 the seed is discarded and every contact starts at
    full material stiffness."""

    rigid_avbd_beta: float = MISSING
    """AVBD penalty ramp rate for body-particle contacts [N/m per m of penetration].

    NEW FIELD, and **the one change here that is not from SIM1**: their solver (a fork of Newton
    0.1.3) has no contact ramping at all, so matching them exactly would mean leaving this at 0.
    Newton 1.5 grew AVBD natively -- `SolverVBD` is documented as "VBD for particles and Augmented
    VBD (AVBD) for rigid bodies", citing Giles et al. 2025 -- and
    `update_body_particle_contact_penalty` (rigid_vbd_kernels.py:4953) ramps exactly the
    cloth-vs-hand contacts this task depends on. It ships disabled (`rigid_avbd_beta = 0.0`).

    Enabling it is the Newton-native answer to the same failure SIM1's strain limit addresses:
    cloth going soft under fast gripper motion.

    Sizing, which is arithmetic and NOT a measurement. The kernel does
    `k += beta * penetration`, clamped to the per-contact ceiling (~1500, see `soft_contact_ke`),
    from a seed of `rigid_contact_k_start` = 100. To close that 1400 gap inside one substep's 12
    iterations needs `beta * penetration` ~ 117, i.e. beta ~ 1.2e6 at 0.1 mm penetration and
    ~1.2e5 at 1 mm. 5.0e5 sits mid-range: it reaches the ceiling within a substep for penetrations
    above ~0.24 mm and ramps more gradually below that.

    **This is the item most likely to go the wrong way.** With only 12 iterations, starting soft
    and ramping can leave contacts SOFTER than the current fixed-k behaviour, which would weaken
    the grip rather than strengthen it. A/B it against 0.0 before trusting it."""

    # --- self-contact ---------------------------------------------------------------------
    enable_self_contact: bool = MISSING
    """Whether the sheet collides with ITSELF.

    NEW FIELD, and the largest behavioural change in this config. `VBDSolverCfg` defaults
    `particle_enable_self_contact` to False and `build_physics` never set it, so until now the
    folded flap passed straight THROUGH the half it was supposed to land on. For a folding task
    that is not a tuning knob, it is a missing mechanic. SIM1 runs `handle_self_contact=True`."""

    self_contact_radius: float = MISSING
    """Distance at which cloth primitives start to repel each other [m]. SIM1 Table 2.

    **This is a geometric distance between mesh primitives, NOT scaled by `particle_radius`** --
    see `evaluate_self_contact_force_norm` (particle_vbd_kernels.py:821), which is driven by
    `collision_radius` alone.

    So copying SIM1's number does not reproduce SIM1's behaviour on our geometry. Their cloth is a
    scanned garment with ~mm triangles, where 2 mm is comparable to the mesh scale; our sheet is
    16 mm thick on a 16.7 mm grid, so a folded flap will settle 2 mm above the layer beneath it
    rather than ~16 mm -- the two layers visually interpenetrate by most of their thickness.

    That interacts with the fold targets: `cloth_env` builds them with `lift = 2 * particle_radius`
    = 16 mm, so the targets sit ~14 mm above where the flap can actually rest. That is inside the
    0.04 m `keypoint_tolerance`, so it biases the reward rather than making the fold impossible.
    Scaling this to ~`thickness` (with margin ~1.5x) is the consistent alternative and costs more
    in collision detection. Left at SIM1's value; flagged, not hidden."""

    self_contact_margin: float = MISSING
    """Detection margin for self-contact [m]. SIM1 Table 2, and 1.5x the radius, which is the ratio
    Newton's own error message asks for. Must be >= `self_contact_radius` or the solver raises."""

    self_contact_detection_interval: int = MISSING
    """How often self-contact detection runs. ``-1`` = once before initialisation, which is both
    the `VBDSolverCfg` default and what SIM1 uses."""

    conservative_bound_relaxation: float = MISSING
    """Relaxation factor for the penetration-free displacement bound (SIM1's beta, Table 2).

    Newton 1.5 defaults this to 0.85; 0.42 was the default in the Newton 0.1.3 that SIM1 forked, so
    their Table 2 row is arguably just that default restated. Kept anyway -- it is the value their
    calibration ran against, and a tighter bound truncates more aggressively, which is the safe
    direction on a sheet that has never had self-contact enabled before.

    Only has any effect when `enable_self_contact` is True."""

    proxy_links: str = MISSING
    """Which hand links the cloth can feel. `tips` (five distal phalanges) makes an enclosing grasp
    geometrically impossible; a pinch grasp is arguably enough to drag a cloth, but the asymmetry
    against the rigid baseline is not worth reintroducing."""

    proxy_table: bool = MISSING
    """The table must be in the proxy set or the sheet falls through it."""

    color: tuple[float, float, float] = (0.85, 0.35, 0.55)

    # --- buffers ---------------------------------------------------------------------------
    rigid_body_particle_contact_buffer_size: int = 65536
    """Body-to-particle soft-contact capacity **PER RIGID BODY**, not global.

    Newton's own comment on the argument is explicit: "Per-body soft-contact list capacity"
    (`newton/_src/solvers/vbd/solver_vbd.py:266`, default 256). So it must NOT scale with env count:
    the allocation is this value x body count, and body count already scales with envs. Multiplying
    it by `num_envs` as well makes the allocation quadratic -- at 512 envs that is 1,048,576 x 2,304
    bodies = 2.4e9 elements, which overflows Warp's signed-int32 array dimension and aborts before a
    single step.

    1024, inherited from the cable where the manipuland was bodies rather than particles, was far
    too small: a 32-env run died with a CUDA device-side assert, which is what an out-of-bounds
    write into a contact buffer looks like from the host."""
    rigid_body_contact_buffer_size: int = 16384
    """Body-to-body contact capacity per rigid body (Newton default 64). Also per-body: the failure
    message is literally "Per-body rigid contact buffer overflowed 87 > 64"."""
    per_particle_triangle_pairs: int = 2048
    """Triangle-pair budget per cloth particle. The budget is GLOBAL and must scale with env count
    *and* geometry: on the cable a correctly env-scaled 4.19M still overflowed once segments
    doubled.

    Raised from 512 after a 32-env run overflowed twice. That run then died with
    ``normal expects all elements of std >= 0.0`` from the policy's sampler -- dropped contacts let
    a sheet penetrate, which produced NaN in that env's observation. The policy error was the
    symptom; the overflow was the cause, and it is worth remembering that this class of failure
    surfaces far from its origin."""

    # Methods, not properties: configclass reads every attribute while it builds an instance, when
    # these fields are still MISSING (task.yaml has not been applied yet).
    def particle_radius(self) -> float:
        """Half the thickness: what the solver actually consumes."""
        return 0.5 * self.thickness

    def num_particles(self) -> int:
        return self.resolution**2

    def spacing(self) -> float:
        return self.size / (self.resolution - 1)

    def max_resolution_for_thickness(self) -> int:
        """Finest grid whose spacing still admits this thickness without self-overlap."""
        return max(2, int(self.size / self.thickness) + 1)

    def check_thickness(self) -> None:
        """Warn when particle DIAMETER exceeds grid spacing. No longer an error.

        **The original premise was wrong.** This raised on `thickness > spacing`, on the grounds
        that closer particles "self-collide at rest and the sheet crumples before anything
        touches it". They cannot: self-collision is driven by `self_contact_radius` via
        `evaluate_self_contact_force_norm`, and `particle_radius` appears in exactly three
        kernels, all of them body-contact. The two never meet.

        The crumpling this guard was written for is documented in `start_height` above as really
        being the spawn-height bug ("attributed to thickness, to grid resolution, and to cloth
        physics; it was this").

        Independent evidence: SIM1's shipping garment violates the old rule by 40% and simulates
        fine. Measured from their `short-shirt.usdc` -- 7,021 vertices, 13,767 triangles, median
        edge 11.40 mm -- against an 8 mm particle radius, i.e. a 16 mm diameter at 1.40x the
        median edge. Newton's own `example_cloth_franka.py` ships the same 8 mm radius.

        Kept as a WARNING because the ratio is still worth knowing: a diameter far above spacing
        means neighbouring particles present overlapping collision spheres to a finger, which
        changes grip in ways nobody here has measured. But it is not a correctness constraint, and
        as an error it forced this task from a 9x9 sheet to 7x7 for no reason.
        """
        if self.thickness > self.spacing() + 1e-9:
            import warnings

            warnings.warn(
                f"cloth thickness {self.thickness:.4f} m (2 x particle_radius) exceeds grid "
                f"spacing {self.spacing():.4f} m, ratio {self.thickness / self.spacing():.2f}x. This "
                f"is NOT an error -- particle_radius is a cloth-RIGID rest offset and cannot "
                f"cause self-collision, which uses self_contact_radius "
                f"({self.self_contact_radius:.4f} m, {self.self_contact_radius / self.spacing():.2f}x "
                f"spacing). For reference SIM1 ships 1.40x and Newton's example_cloth_franka.py "
                f"the same 8 mm radius. Flagged because a diameter well above spacing gives "
                f"neighbouring particles overlapping collision spheres, which changes grip.",
                stacklevel=2,
            )
