"""ClothEnvCfg — fold half a cloth onto the other half.

Subclasses :class:`PlayNewtonEnvCfg`, so the robot, table, action space and coupled-solver
machinery are exactly the ones measured on the cable task. Only the manipuland changes, from a
rigid tool to a VBD cloth sheet.

**The task.** A square sheet lies flat on the table. Its *far* half (``+x``) is tracked by a
handful of keypoints; the goal is to bring those keypoints onto the mirror image of their rest
positions across the fold line, i.e. to fold the far half back over the near half. Success is the
usual keypoint criterion: every tracked keypoint within tolerance, held for ``success_steps``.

**Why keypoints rather than a pose.** A cloth has no rigid frame. The cable task could fake one
from its first-to-last span, and that estimator degenerated as soon as the cable bent
(``docs/phase4_cable_env.md``). A sheet has no meaningful single axis at all, so the state *is* the
keypoint set -- there is nothing to reduce it to.

**Coupling defaults are inherited from the cable work, not guessed.** ``proxy_mode="staggered"``,
``cable_substeps=2``-equivalent substepping and the damping stack are what took the cable from 0 to
~14 goals per 64 envs; `lagged` coupling pairs the hand's begin-pose with its end-velocities, an
error that scales with hand speed and which shook the cable loose during transport. A cloth is more
compliant than a cable, so the same defect should hurt at least as much here.
"""

from __future__ import annotations

# IMPORT ORDER IS LOAD-BEARING, as in `cable_env_cfg`: importing `isaaclab_physx` binds
# `isaaclab.utils.configclass` to the same-named *submodule*, shadowing the decorator. Importing
# PlayNewtonEnvCfg first runs the compat shim that re-asserts it.
from isaacsimenvs.tasks.play_newton.play_newton_env_cfg import PlayNewtonEnvCfg  # isort: skip

from isaaclab.utils import configclass  # noqa: E402

__all__ = ["ClothCfg", "ClothEnvCfg"]


@configclass
class ClothCfg:
    """Cloth geometry, material, the fold definition, and the coupling that lets the hand feel it."""

    # --- geometry -------------------------------------------------------------------------
    size: float = 0.10
    """Side length of the square sheet [m]."""

    thickness: float = 0.016
    """Cloth thickness [m], as ``2 * particle_radius``.

    **A VBD cloth is a surface: thickness IS the particle collision radius**, there is no
    volumetric dimension. That makes it bounded by grid spacing -- particles closer together than
    their diameter self-collide at rest and the sheet crumples before anything touches it. The
    constraint is ``thickness <= size / (resolution - 1)``, checked in ``__post_init__``.

    Thicker is genuinely better for *contact*: the cable was ungraspable at 10 mm and liftable at
    30 mm, and thin geometry tunnels through fingers. But it is worse for *folding*, which needs
    bending. And 30 mm on a 100 mm sheet forces a 4x4 grid (spacing 33 mm), which is a stiff mat
    rather than cloth -- genuine 30 mm material is a volumetric soft body (``add_soft_mesh``, a
    tet mesh), not a cloth.

    **16 mm, adopted from SIM1 Table 2 (particle radius 0.008 m).** This is NOT a measurement on
    this task -- it is their calibrated value, taken wholesale along with the rest of their cloth
    material. It forces ``resolution=7``: spacing must be >= thickness, and 16 mm needs
    ``size / thickness + 1 = 7`` (spacing 16.7 mm). ``max_resolution_for_thickness`` returns
    exactly 7, so this pair sits on the guard boundary and any finer grid will raise.

    The previous value was 12 mm at ``resolution=9`` (spacing 12.5 mm), chosen as "thick enough for
    the hand to feel, thin enough to fold". That reasoning still stands and nothing measured
    displaced it; it was replaced to match SIM1, not to improve on it. If the fold regresses, this
    and ``resolution`` are the first pair to revert."""

    resolution: int = 7
    """Particles per side, so the sheet is ``resolution**2`` particles and
    ``2*(resolution-1)**2`` triangles. 7 gives 49 particles / 72 triangles.

    **Driven by ``thickness``, not chosen independently.** 16 mm particles cannot sit on a grid
    finer than 16.7 mm spacing without overlapping at rest, so matching SIM1's particle radius
    forces this down from 9 (81 particles / 128 triangles). The fold geometry survives the change
    unchanged -- ``corner_indices`` still returns the four far-half corners at the same
    coordinates, ``folded_targets`` is identical, and the odd resolution keeps a hinge row -- but
    the sheet is meaningfully coarser, and a 72-triangle sheet is close to the floor at which
    "cloth" stops being a useful description."""

    start_height: float = 0.30
    """Spawn height above ``reset.table_reset_z`` [m].

    **The table's collision surface is at ``table_reset_z + 0.150``, not + 0.045.** See
    ``table_half_thickness``. Everything below is measured with zero actions, so the robot never
    approaches the sheet and anything it does is spawn behaviour:

        0.150  -> spawn 0.5300, surface 0.5298: in contact. EJECTED upward at ~1.7 m/s, peaks at
                  0.68 by step 10, falls back, settles by step 30 with a 17 mm spread across a
                  12 mm sheet -- i.e. visibly wrinkled.
        0.300  -> spawn 0.6800, a genuine 15 cm drop. Falls at exactly g (predicted 0.0340 m over
                  the first 5 steps, measured 0.0349), lands at 0.5360 and settles PERFECTLY flat
                  (min = mean = max to four decimals).

    An earlier version of this docstring reported 0.150 as "settles dead flat", which was wrong: it
    was read after settling and never checked the spread. Every episode was therefore starting with
    the sheet thrown into the air and landing creased, and that was mis-attributed in turn to
    thickness, to grid resolution, and to cloth physics.

    Values below the surface are worse still, and were correctly diagnosed at the time: at 0.010 the
    sheet starts inside the table and reaches 51 m by step 80."""

    table_half_thickness: float = 0.150
    """Distance from ``reset.table_reset_z`` up to the table's COLLISION surface [m].

    **Measured, not read off the USD.** The visual box is 0.045 half-thick, and that number sat here
    for a long time -- but the sheet settles with its lowest particles at 0.5360 against a table root
    of 0.3800, putting the contact surface at 0.5298, i.e. 0.1498 above the root. Confirmed twice
    from different drop heights, and `Table/box` is the only non-robot collider in the scene, so
    that is what the cloth is resting on.

    Getting it wrong is not a cosmetic error. At 0.045 the guard below accepted
    ``start_height = 0.15``, which looked like a 10 cm drop and actually spawned the sheet 0.2 mm
    above the surface -- effectively in contact. Contact resolution then ejected it upward at
    ~1.7 m/s at the start of EVERY episode: it flew to 0.68, fell back, and settled visibly
    wrinkled (17 mm spread across a 12 mm sheet). With a real drop it falls at exactly g and lands
    perfectly flat (spread 0.0000). The crumpled spawn had been attributed to thickness, to grid
    resolution, and to cloth physics; it was this."""

    # --- material -------------------------------------------------------------------------
    density: float = 2.0
    """Surface density [kg/m^2]. **Per AREA, not per volume.**

    This reaches ``ModelBuilder.add_cloth_mesh(density=...)`` unchanged (via the ``newton:density``
    USD attribute, read in ``isaaclab_contrib.deformable.deformable_object``), and Newton documents
    that argument as "the density per-area of the mesh". The old 100.0 was written as though it
    were volumetric: it made the 0.1 x 0.1 m sheet weigh **1 kg**, a handkerchief with the mass of
    a litre of water, and it disagreed by 83x with the ``area_density = density * thickness``
    that ``cloth_env`` computes for the ``ClothAsRigidObject`` mass proxy.

    2.0 kg/m^2 is SIM1 Table 2. Still heavy for cloth -- real cotton sheeting is 0.15-0.3 kg/m^2 --
    but it is their calibrated value and it is 50x lighter than what this ran before."""
    tri_ke: float = 1.0e2
    """In-plane stretch stiffness -- the Lame ``mu`` (shear modulus) of the membrane model.

    **1.0e2 is SIM1 Table 2, and also Newton's own ``default_tri_ke``.** Their "calibrated"
    elasticity is the library default; what they actually tuned away from default is bending.

    This was 1.0e4. The justification was the cable, where softening stretch was catastrophic
    (x0.01 peaked at 94.9 m/s) -- but a cable is a 1-D chain whose only in-plane mode IS stretch,
    and the sheet has an area term (``tri_ka``) carrying load the cable had no analogue for. The
    cable result is not evidence about a membrane. If the sheet oscillates or blows up, this is the
    first suspect and 1.0e3 is the obvious intermediate."""
    tri_ka: float = 1.0e2
    """In-plane shear/area stiffness -- the Lame ``lambda`` (area modulus). SIM1 Table 2.

    Note their own solver comment recommends ``lambda: 1000.0+`` to maintain area against
    penetration, and then their env ships 1e2. Table 2 agrees with the env, so 1e2 it is."""
    tri_kd: float = 1.5e-6
    """In-plane damping. SIM1 Table 2 (was 1.0e-5 here).

    Newton's ``default_tri_kd`` is 10.0; both this and the previous value are far below it, which
    is right for a membrane that has to drape rather than settle."""
    edge_ke: float = 8.0e-4
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
    edge_kd: float = 1.0e-3
    """Bending damping. SIM1 Table 2 (was 1.0e-2 here)."""

    # --- the fold -------------------------------------------------------------------------
    fold_axis: str = "x"
    """Axis normal to the fold line, in cloth-local coordinates: the ``+`` half folds onto ``-``."""

    num_keypoints: int = 4
    """Tracked keypoints on the moving half. Four matches the task's existing keypoint machinery
    and is the minimum that pins a folded flap's position *and* orientation."""

    max_folded_footprint: float = 0.65
    """Largest sheet extent along the fold axis, as a fraction of ``size``, that still counts as
    folded.

    A second condition on success, independent of keypoint distance. A perfect fold gives ~0.5; an
    unfolded sheet gives 1.0. Keypoint proximity alone proved insufficient: a rigid slide scored
    "folds" because the targets were world-fixed, and after that was corrected a *crumpled* sheet
    still scored transiently by putting four keypoints near their targets. 0.65 leaves room for a
    real fold's drape while excluding anything close to flat."""

    keypoint_tolerance: float = 0.04
    """Per-keypoint distance for success [m]. Deliberately looser than the cable's 1.5 cm: a
    folded flap settles wherever the drape puts it, and demanding rigid-body precision from a
    sheet would measure the tolerance rather than the fold."""

    # --- coupling (defaults carried from the cable result) --------------------------------
    proxy_mode: str = "staggered"
    """See `docs/phase4_cable_env.md`. `lagged` syncs the hand's *begin* pose with its *end*
    velocities; the mismatch scales with hand speed and was the single change that took the cable
    from ~0 to 5 goals in an episode."""

    substeps: int = 6
    """Cloth substeps inside one coupled step, i.e. the VBD timestep.

    **6 makes the cloth dt 1/1440 s, matching SIM1.** Their loop is 60 Hz with 24 substeps; ours is
    ``sim.dt`` 1/120 / ``newton.num_substeps`` 2 / this, so 6 lands on the same 0.694 ms. It was 2
    (1/480 s, 3x larger).

    **This directly contradicts a cable measurement**: on the cable there was a sharp optimum at 2,
    with 1 peaking at 37 m/s and 4/8/16 degrading monotonically, on the theory that fewer, larger
    substeps mean fewer proxy exchanges per step and the exchange is where the energy enters. That
    was never re-measured for the sheet, and the cloth is a different solver entry, but it is a
    direct conflict and not a gap -- if the sheet gains energy at reset or under contact, revert
    this first.

    Costs 3x the VBD work per policy step, on top of the iteration increase."""

    vbd_iterations: int = 12
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
    coupler_iterations: int = 1
    """Raising this made the cable *worse* at every setting tried."""

    mass_scale: float = 1.0
    """KEEP AT 1.0. 0.05 was ~500x worse on the cable and NaN'd the robot."""

    collide_interval: int = 1

    soft_contact_ke: float = 5.0e2
    """Body-particle contact stiffness [N/m]. SIM1 Table 2 (was 8.0e3 here).

    The per-contact value is the AVERAGE of this and the rigid shape's own material stiffness
    (`NewtonShapeCfg.ke`, 2.5e3), so the effective ceiling moves 5250 -> 1500."""

    soft_contact_kd: float = 5.0e-3
    """Body-particle contact damping [N*s/m]. SIM1 Table 2.

    NEW FIELD. `NewtonModelCfg` defaults it to 1.0e-2 and `build_physics` was passing only `ke` and
    `mu`, so this ran at the library default and was never stated."""

    soft_contact_mu: float = 0.25
    """Body-particle friction coefficient. SIM1 Table 2 (`self_contact_friction`; was 10.0 here).

    Effective per-contact friction is `sqrt(soft_contact_mu * shape_mu)`, so against the fingertips
    (`assets.finger_tip_friction` 1.5) this moves from `sqrt(10 x 1.5)` = 3.9 to `sqrt(0.25 x 1.5)`
    = 0.61. SIM1's own effective value is 0.5 -- they set every shape's mu to 1.0.

    Table 2's `robot_friction` 1.5 / `table_friction` 0.0 are NOT copied: their env overwrites
    every shape's mu to 1.0 immediately after building the table, so those two rows describe a
    configuration that does not run. Our per-shape frictions stay as the rigid-tool task sets
    them."""

    rigid_contact_k_start: float = 1.0e2
    """Body-particle contact penalty SEED for AVBD ramping [N/m].

    **Inert unless `rigid_avbd_beta` > 0**, which is why it did nothing until now. Newton computes
    `rigid_contact_k_start_value = -1.0 if linear_beta == 0.0 else k_start` (solver_vbd.py:677) and
    the kernel reads `k_floor = avg_ke if k_start < 0.0 else min(k_start, avg_ke)`
    (rigid_vbd_kernels.py:3395) -- so with beta 0 the seed is discarded and every contact starts at
    full material stiffness."""

    rigid_avbd_beta: float = 5.0e5
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
    enable_self_contact: bool = True
    """Whether the sheet collides with ITSELF.

    NEW FIELD, and the largest behavioural change in this config. `VBDSolverCfg` defaults
    `particle_enable_self_contact` to False and `build_physics` never set it, so until now the
    folded flap passed straight THROUGH the half it was supposed to land on. For a folding task
    that is not a tuning knob, it is a missing mechanic. SIM1 runs `handle_self_contact=True`."""

    self_contact_radius: float = 0.002
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

    self_contact_margin: float = 0.003
    """Detection margin for self-contact [m]. SIM1 Table 2, and 1.5x the radius, which is the ratio
    Newton's own error message asks for. Must be >= `self_contact_radius` or the solver raises."""

    self_contact_detection_interval: int = -1
    """How often self-contact detection runs. ``-1`` = once before initialisation, which is both
    the `VBDSolverCfg` default and what SIM1 uses."""

    conservative_bound_relaxation: float = 0.42
    """Relaxation factor for the penetration-free displacement bound (SIM1's beta, Table 2).

    Newton 1.5 defaults this to 0.85; 0.42 was the default in the Newton 0.1.3 that SIM1 forked, so
    their Table 2 row is arguably just that default restated. Kept anyway -- it is the value their
    calibration ran against, and a tighter bound truncates more aggressively, which is the safe
    direction on a sheet that has never had self-contact enabled before.

    Only has any effect when `enable_self_contact` is True."""

    proxy_links: str = "hand"
    """Which hand links the cloth can feel. `tips` (five distal phalanges) makes an enclosing grasp
    geometrically impossible; a pinch grasp is arguably enough to drag a cloth, but the asymmetry
    against the rigid baseline is not worth reintroducing."""

    proxy_table: bool = True
    """The table must be in the proxy set or the sheet falls through it."""

    color: tuple[float, float, float] = (0.85, 0.35, 0.55)

    nan_policy: str = "raise"
    """What to do when any observed quantity goes non-finite: ``"raise"`` or ``"reset"``.

    ``raise`` is right for probes and evaluation -- a NaN there is a bug to diagnose, and the guard
    names which quantity went bad first (it is how the intermittent failure was localised to
    ``robot_joint_pos`` rather than to the cloth).

    ``reset`` is right for training. The offending envs have their observation zeroed so nothing
    non-finite can reach the network or the gradient, and are terminated so they reset through the
    normal path. Over a multi-hour run on 4 GPUs a single diverged env would otherwise kill
    everything; the failure has been seen roughly once per three 900-step evals."""

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

    @property
    def particle_radius(self) -> float:
        """Half the thickness: what the solver actually consumes."""
        return 0.5 * self.thickness

    @property
    def num_particles(self) -> int:
        return self.resolution**2

    @property
    def spacing(self) -> float:
        return self.size / (self.resolution - 1)

    @property
    def max_resolution_for_thickness(self) -> int:
        """Finest grid whose spacing still admits this thickness without self-overlap."""
        return max(2, int(self.size / self.thickness) + 1)

    def __post_init__(self) -> None:
        """Reject a sheet that would spawn inside the table, or a thickness the grid cannot hold.

        Both failures present as violent instability rather than as configuration errors, which is
        exactly how they were first diagnosed here -- as a solver problem.
        """
        if self.start_height <= self.table_half_thickness:
            raise ValueError(
                f"cloth start_height {self.start_height:.3f} m does not clear the table: "
                f"`reset.table_reset_z` is the table CENTRE and its surface is "
                f"{self.table_half_thickness:.3f} m higher. The sheet would spawn inside the "
                f"table and be ejected in one step. Use start_height > "
                f"{self.table_half_thickness:.3f}."
            )
        self._check_thickness()

    def _check_thickness(self) -> None:
        """Reject a thickness the grid cannot hold.

        Silent self-collision at rest looks exactly like a cloth that mysteriously crumples, and
        would be diagnosed as a solver problem rather than a configuration one.
        """
        if self.thickness > self.spacing + 1e-9:
            raise ValueError(
                f"cloth thickness {self.thickness:.3f} m exceeds grid spacing "
                f"{self.spacing:.3f} m (size {self.size:.3f} / {self.resolution - 1} gaps). "
                f"Particles would overlap at rest and the sheet would crumple before contact.\n"
                f"  Either reduce resolution to <= {self.max_resolution_for_thickness} "
                f"(spacing {self.size / max(1, self.max_resolution_for_thickness - 1):.3f} m), "
                f"or reduce thickness to <= {self.spacing:.3f} m.\n"
                f"  For genuine volumetric thickness, a cloth is the wrong primitive: use a soft "
                f"body (tet mesh / add_soft_mesh) instead."
            )


    def build_physics(self, newton_cfg, num_envs: int):
        """``NewtonCfg`` for the coupled MJWarp (robot) + VBD (cloth) scene.

        The one structural difference from the cable: the VBD entry owns **particles**, not bodies.
        A cloth is a particle system, so ``all_particles=True`` is what assigns the sheet to the
        deformable solver -- a body selector would match nothing and the cloth would be integrated
        by nobody, presenting as a sheet frozen in mid-air.
        """
        from isaaclab_contrib.coupling import (
            CouplerEntryCfg,
            CouplerProxyCfg,
            CouplerProxyMappingCfg,
        )
        from isaaclab_contrib.deformable import NewtonModelCfg, VBDSolverCfg
        from isaaclab_newton.physics import (
            MJWarpSolverCfg,
            NewtonCfg,
            NewtonCollisionPipelineCfg,
            NewtonShapeCfg,
        )

        rigid_bodies = [r"/World/envs/env_.*/Robot"]
        finger = r"/World/envs/env_.*/Robot/left_(thumb|index|middle|ring|pinky)_"
        proxy_bodies = [finger + ("(DP|MP|PP)" if self.proxy_links != "tips" else "(DP)")]
        if self.proxy_links == "hand":
            # The palm body is `iiwa14_link_7`; the URDF's `left_hand_C_MC` is merged away by the
            # importer and matches nothing.
            proxy_bodies.append(
                r"/World/envs/env_.*/Robot/(iiwa14_link_7|left_thumb_MC|left_pinky_MC)"
            )
        if self.proxy_table:
            rigid_bodies.append(r"/World/envs/env_.*/Table")
            proxy_bodies.append(r"/World/envs/env_.*/Table")

        budget = self.triangle_pair_budget(newton_cfg, num_envs)

        # No VBDSolverCfg subclass here, deliberately. `class_type` carries
        # "{DIR}.vbd_manager:NewtonVBDManager" and `{DIR}` is substituted against the *defining*
        # module, so subclassing inside this package makes it hunt for
        # `isaacsimenvs.tasks.cloth.vbd_manager`. The cable subclasses it only to add
        # `rigid_body_contact_buffer_size`, which is a body-BODY buffer; a cloth is particles, so
        # the buffer that matters (`rigid_body_particle_contact_buffer_size`) is already a field.
        vbd = VBDSolverCfg(
            iterations=self.vbd_iterations,
            rigid_body_particle_contact_buffer_size=self.rigid_body_particle_contact_buffer_size,
            rigid_contact_k_start=self.rigid_contact_k_start,
            # Self-contact. Without these the folded flap passes through the half it lands on --
            # `particle_enable_self_contact` defaults to False and this call never set it.
            particle_enable_self_contact=self.enable_self_contact,
            particle_self_contact_radius=self.self_contact_radius,
            particle_self_contact_margin=self.self_contact_margin,
            particle_collision_detection_interval=self.self_contact_detection_interval,
        )
        # `rigid_body_contact_buffer_size` is NOT a field on VBDSolverCfg (the cable subclasses it
        # for exactly this reason), but it IS in SolverVBD's signature, and the manager passes
        # through any cfg attribute the signature accepts. Setting it on the instance therefore
        # reaches the solver without a subclass -- and without a subclass, `{DIR}` in `class_type`
        # still resolves against isaaclab_contrib rather than this package.
        #
        # It defaults to 64. A 32-env cloth run hit "Per-body rigid contact buffer overflowed
        # 87 > 64" and then died with a CUDA device-side assert. Declaring the field in this config
        # without wiring it here left it inert: it read as configured and did nothing.
        vbd.rigid_body_contact_buffer_size = self.rigid_body_contact_buffer_size

        # Same pass-through for two more `SolverVBD` arguments that `VBDSolverCfg` does not
        # declare. `_filter_solver_kwargs` filters `solver_cfg.to_dict()` against the solver
        # signature, and isaaclab's `class_to_dict` walks `obj.__dict__`, so instance attributes
        # set here do reach the solver.
        #
        #   * `particle_conservative_bound_relaxation`: Newton 1.5 defaults 0.85; SIM1 ran 0.42.
        #   * `rigid_avbd_beta`: ships 0.0, which DISABLES AVBD contact ramping and makes
        #     `rigid_contact_k_start` above a no-op. See the field docstring -- this is the one
        #     setting here that could plausibly weaken the grip rather than strengthen it.
        vbd.particle_conservative_bound_relaxation = self.conservative_bound_relaxation
        vbd.rigid_avbd_beta = self.rigid_avbd_beta

        solver = CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        solver=newton_cfg.solver,
                        integrator=newton_cfg.integrator,
                        njmax=newton_cfg.njmax,
                        nconmax=newton_cfg.nconmax,
                        impratio=newton_cfg.impratio,
                        cone=newton_cfg.cone,
                        iterations=newton_cfg.iterations,
                        ls_iterations=newton_cfg.ls_iterations,
                        use_mujoco_contacts=newton_cfg.use_mujoco_contacts,
                        ccd_iterations=newton_cfg.ccd_iterations,
                        disable_sensors=newton_cfg.disable_sensors,
                    ),
                    bodies=rigid_bodies,
                ),
                CouplerEntryCfg(
                    name="cloth",
                    solver_cfg=vbd,
                    # Particles, not bodies -- see the docstring.
                    all_particles=True,
                    include_static_shapes=True,
                    substeps=self.substeps,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="cloth",
                    bodies=proxy_bodies,
                    mass_scale=self.mass_scale,
                    mode=self.proxy_mode,
                    collide_interval=self.collide_interval,
                    # The proxy builds its OWN pipeline and defaults max_triangle_pairs to a
                    # GLOBAL 1e6; the outer collision_cfg never reaches it. On the cable that
                    # silently dropped contacts above ~16 envs for hours.
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        rigid_contact_max=newton_cfg.rigid_contact_max,
                        max_triangle_pairs=budget,
                    ),
                )
            ],
            iterations=self.coupler_iterations,
            model_cfg=NewtonModelCfg(
                soft_contact_ke=self.soft_contact_ke,
                # `soft_contact_kd` was previously left at the `NewtonModelCfg` default (1.0e-2)
                # because this call only passed `ke` and `mu`.
                soft_contact_kd=self.soft_contact_kd,
                soft_contact_mu=self.soft_contact_mu,
            ),
        )

        return NewtonCfg(
            solver_cfg=solver,
            collision_cfg=NewtonCollisionPipelineCfg(
                rigid_contact_max=newton_cfg.rigid_contact_max,
                max_triangle_pairs=budget,
            ),
            default_shape_cfg=NewtonShapeCfg(),
            num_substeps=newton_cfg.num_substeps,
            collision_decimation=newton_cfg.collision_decimation,
            use_cuda_graph=False,
        )

    def triangle_pair_budget(self, newton_cfg, num_envs: int) -> int:
        """Global triangle-pair budget for this sheet at ``num_envs``.

        Scales with particle count as well as env count. The cable overflowed a correctly
        env-scaled 4.19M as soon as its segment count doubled, and a sheet has far more triangles
        than a cable -- so geometry has to be in this product, not just ``num_envs``.
        """
        base = newton_cfg.resolve_max_triangle_pairs(num_envs)
        per_env = self.num_particles * self.per_particle_triangle_pairs
        return int(max(base, num_envs * per_env) * 1.5)


@configclass
class ClothEnvCfg(PlayNewtonEnvCfg):
    """Play task with a folding cloth in place of the rigid tool."""

    cloth: ClothCfg = ClothCfg()
