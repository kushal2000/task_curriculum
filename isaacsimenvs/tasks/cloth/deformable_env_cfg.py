"""Config of the deformable (VBD) cloth env on Newton (``deformable_env.ClothEnv``), Isaac Lab 3.

The shared sections (``utils/cfg.ClothTaskEnvCfg``: robot, action, obs, reward, reset, termination,
DR, the fold) + ``newton`` (MJWarp for the robot, Newton's model / solver settings) + ``cloth``: the
VBD sheet and the coupled MJWarp + VBD solve. Real-fabric presets: ``utils/fabrics.py``.
"""

from __future__ import annotations

# IMPORT ORDER IS LOAD-BEARING: the compat shim re-asserts `isaaclab.utils.configclass` as a function
# after `isaaclab_physx` shadows it with its same-named submodule.
from isaacsimenvs.newton import compat  # isort: skip

compat.install_isaaclab3_compat()

from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from isaacsimenvs.tasks.cloth.utils.cfg import ClothTaskEnvCfg  # noqa: E402
from isaacsimenvs.tasks.cloth.utils.robot import ROBOT  # noqa: E402

__all__ = ["ClothCfg", "ClothEnvCfg", "NewtonSolverCfg"]


@configclass
class NewtonSolverCfg:
    """MJWarp / Newton solver settings.

    Values are the reference port's (github.com/kushal2000/isaac_newton @ beb9efb), which reached
    parity with the PhysX baseline on this exact task. **Almost none of them are library
    defaults**, and the defaults are not merely untuned -- several fail *silently*, dropping work
    rather than raising, which is why each carries its measured justification here.
    """

    # --- The decisive one -------------------------------------------------------------------
    max_triangle_pairs: int | None = None
    """Narrow-phase triangle-pair budget. ``None`` sizes it from the environment count.

    The library default is ``1_000_000`` and it is a **global** budget, not per-world. This robot
    carries 34 mesh collision prims with self-collision enabled, so demand grows linearly with
    environment count: it crosses 1e6 somewhere between 16 and 32 envs and reaches ~1.7M at 64.
    Past the cap the narrow phase **silently drops pairs** -- fingertip/tool contacts simply go
    missing and the grasp cannot be held.

    Measured in the reference port at 64 envs, changing only this value:
    ``1e6 -> 0.12 goals/episode``, ``8e6 -> 14.28``. A 114x effect from a number that appears in
    no config file.

    It also explains why that project spent a day misattributing the backend: every Newton run it
    had ever done used 64-1024 envs, i.e. was entirely inside the saturated regime, so there was
    no working configuration to A/B against and the solver looked uniformly broken rather than
    broken above a threshold.
    """

    per_env_triangle_pairs: int = 65_536
    """Headroom per env when sizing automatically -- about 2.5x the ~26.6k/env measured demand.

    The buffer is a GPU allocation so it is not free, but under-sizing fails silently and
    catastrophically while over-sizing costs memory, which is visible and diagnosable.
    """

    # --- Constraint budgets, same failure mode ------------------------------------------------
    njmax: int = 8192
    """Max constraint rows per world. The library default of 300 is sized for a Franka/Allegro
    lifting a simple object; this scene (22-DoF hand + tool + table) overflows it continuously
    (``nefc overflow - please increase njmax``). Overflow is not benign -- constraints are
    dropped and positions go wrong. The reference ran at 2400 for a long time, still overflowing
    to 2878, and raising it to 8192 moved a "matching" score from 15.97 to 18.61."""

    nconmax: int = 2048
    """Max contact points per world; same failure mode. The reference ran 800 while the solver's
    own message asked for 1081."""

    # --- Contact model, from the task authors' own MuJoCo scene -------------------------------
    cone: str = "elliptic"
    """Friction cone. MuJoCo's default is ``pyramidal``, which approximates the friction limit
    surface with a box and under combined slip lets a held tool rotate more readily. The
    SimToolReal authors' own MuJoCo model of this robot uses elliptic."""

    impratio: float = 10.0
    """Frictional-to-normal constraint impedance. The authors use 10, not MuJoCo's default 1; a
    higher ratio stiffens frictional constraints and stops a grasped tool creeping."""

    condim: int = 6
    """Contact dimensionality on the grasping shapes. The tool and fingertip assets already carry
    torsional and rolling friction coefficients, but MuJoCo ignores them below condim 4 and 6
    respectively. Set before the solver compiles -- MuJoCo Warp sizes its constraint arrays at
    compile time, so writing it afterwards produces NaNs.

    Kept on fidelity grounds (the authors' scene specifies it), **not** performance: measured
    cleanly post-fix it is neutral (19.375 +/- 2.134 at condim 6 against 20.359 +/- 1.918 at
    condim 3, z = 0.34), and an earlier claim that torsional friction was the missing physics was
    retracted."""

    # --- Pipeline -----------------------------------------------------------------------------
    use_mujoco_contacts: bool = False
    """Use Newton's own collision pipeline rather than MuJoCo's. Not a performance preference:
    under MuJoCo's pipeline ``separate_worlds=True`` builds the MuJoCo model from world 0 only, so
    per-env collision *meshes* are **silently ignored** -- every world simulates world 0's mesh
    while reporting its own mass and inertia. That is a physically nonexistent hybrid that passes
    a "do the environments differ?" check. Newton's pipeline reads per-world shape data correctly,
    and on a mesh scene is also better conditioned (0.43 mm penetration against 6.5-9.2 mm) and
    faster."""

    solver: str = "newton"
    integrator: str = "implicitfast"
    iterations: int = 100
    ls_iterations: int = 15
    ccd_iterations: int = 35
    update_data_interval: int = 2
    rigid_contact_max: int = 4_000_000
    num_substeps: int = 2
    collision_decimation: int = 0
    """Re-collide every N substeps; 0 = once per tick. Coupled to :attr:`num_substeps` -- raising
    substeps without also re-colliding more often leaves the solver integrating several steps
    against one stale collision update."""

    use_cuda_graph: bool = True
    disable_sensors: bool = True
    """Required for ``deterministic_mode="run_to_run"``, and safe here: every observation field is
    read through Isaac Lab's articulation and rigid-body state, never from MuJoCo sensors."""

    deterministic_mode: str = "not_guaranteed"

    def resolve_max_triangle_pairs(self, num_envs: int) -> int:
        """Triangle-pair budget for ``num_envs``, honouring an explicit override."""
        if self.max_triangle_pairs is not None:
            return int(self.max_triangle_pairs)
        return max(1_000_000, int(num_envs) * self.per_env_triangle_pairs)

    def build(self, num_envs: int):
        """Return an ``isaaclab_newton`` ``NewtonCfg``.

        ``num_envs`` is load-bearing, not incidental -- it sizes the triangle-pair budget.
        """
        from isaaclab_newton.physics import (
            MJWarpSolverCfg,
            NewtonCfg,
            NewtonCollisionPipelineCfg,
            NewtonShapeCfg,
        )

        return NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver=self.solver,
                integrator=self.integrator,
                njmax=self.njmax,
                nconmax=self.nconmax,
                impratio=self.impratio,
                cone=self.cone,
                update_data_interval=self.update_data_interval,
                iterations=self.iterations,
                ls_iterations=self.ls_iterations,
                use_mujoco_contacts=self.use_mujoco_contacts,
                ccd_iterations=self.ccd_iterations,
                disable_sensors=self.disable_sensors,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(
                rigid_contact_max=self.rigid_contact_max,
                max_triangle_pairs=self.resolve_max_triangle_pairs(num_envs),
            ),
            default_shape_cfg=NewtonShapeCfg(),
            num_substeps=self.num_substeps,
            collision_decimation=self.collision_decimation,
            use_cuda_graph=self.use_cuda_graph,
            deterministic_mode=self.deterministic_mode,
        )


@configclass
class ClothCfg:
    """The VBD sheet: geometry, material, plasticity, and the coupled MJWarp + VBD solve."""

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
    """Spawn height above the episode's table centre (``reset.table_reset_z`` + its noise) [m].

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

    plastic_yield_curvature: float = 0.0
    """Bending plasticity: yield curvature [1/m], 0 = off (purely elastic, as before).

    A hinge bent past ``kappa x hinge width`` (16.7 mm along the grid, 11.8 mm on the diagonals)
    keeps the excess as a new rest angle, so a pressed fold stays folded instead of springing back
    (``utils/plasticity.py``, after Narain et al. 2013). Reset to flat with the sheet.
    It is a mesh-level parameter: in the scripted fold-press-release test
    (``scripts/analysis/cloth_plasticity_probe.py``) 100 (95 deg per grid cell) holds a pressed
    towel or cotton fold without changing the free drape; napkin and denim need 50, chiffon and silk 25."""

    plastic_set_curvature: float = 0.0
    """Two-stage plasticity [1/m], 0 = off (single-stage, every result before 2026-11-10). A hinge
    gains set only while bent past ``plastic_set_curvature x hinge width`` (16.7 mm along the grid);
    gentler bends -- a light touch, a hanging drape -- stay elastic, and set can always shrink, so
    unfolding still irons a crease out (``utils/plasticity.py``)."""

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

    use_cuda_graph: bool = False
    """Capture the coupled MJWarp + VBD step as a CUDA graph. Off by default because every result so
    far was produced without it (inherited from the cable, whose reason -- entries stepping at
    different rates -- was never measured: the substep counts are fixed, so a graph records them).
    At small env counts the step is kernel-LAUNCH bound (decimation 2 x substeps 6 x 12 VBD
    iterations per policy step), which is what makes the live viewer's cloth ~4 steps/s."""

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

    self_contact_mu: float = 0.0
    """Cloth-on-cloth friction coefficient (ply on ply). 0 = ``soft_contact_mu`` (previous behaviour).

    Newton's VBD self-contact uses the raw ``model.soft_contact_mu``, while every cloth-vs-shape
    contact uses ``sqrt(soft_contact_mu * shape_mu)``. The fabric presets calibrate
    ``soft_contact_mu`` for cloth-on-table, which left ply-on-ply as the LOWEST friction in the
    scene (big napkin 0.265 against 0.36 table, 0.63 fingertips), so a pressed fold slid apart.

    When > 0: ``model.soft_contact_mu`` becomes this value and ``ClothEnv`` rescales every
    shape's friction IN THE CLOTH SOLVER'S VIEW ONLY by ``soft_contact_mu / self_contact_mu``, so
    each cloth-vs-shape coefficient stays ``sqrt(soft_contact_mu * shape_mu)``. The rigid solver
    (robot vs table) keeps the original shape frictions."""

    table_contact_mu: float = 0.0
    """Effective cloth-on-table friction coefficient. 0 = keep ``sqrt(soft_contact_mu *
    table_friction)`` (0.36 for the measured presets). Applied in the cloth solver's view only, on
    its BOX shapes (the table); the rigid solver's table friction is unchanged."""

    friction_epsilon: float = 0.0
    """``SolverVBD(friction_epsilon=...)`` [m/s]: relative tangential speed below which friction is
    smoothed into a stiff spring (stick). 0 = Newton's default (1e-2). Smaller = less creep under a
    sustained tangential load, and a stiffer friction Hessian."""

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

    full_surface_contact: bool = False
    """Cloth EDGES and TRIANGLE INTERIORS collide with the proxy shapes, not only its particles.

    Newton 1.5.2 ``CollisionPipeline(enable_rigid_soft_full_surface_contact=True)``: SDF-based
    edge/face soft contacts (Macklin et al. 2020 local optimisation), consumed by ``SolverVBD``.
    Targets the finger pierce (docs/reports/4_physx/cloth_finger_pierce.md): per-particle contact
    lets a fingertip push the particle under it aside and reach the table between particles.

    Every participating mesh shape (hand proxy links, a mesh table) needs a volume SDF, which
    ``ClothEnv`` provisions before finalize (``shape_force_sdf``). Rigid-rigid contacts run in
    Newton's pipeline too (``use_mujoco_contacts: false``), so those SDFs can change hand-table
    contact as well. Off by default, so earlier results keep their meaning."""

    full_surface_sdf_resolution: int = 0
    """``max_resolution`` of the provisioned SDFs (multiple of 8); 0 = Newton's default."""

    def proxy_body_patterns(self, robot_spec=None) -> list[str]:
        """Body-path regexes of the rigid shapes the cloth feels (the proxy set).

        ``robot_spec`` names the hand bodies (``utils/robot.py``'s robot when omitted).
        """
        robot_spec = robot_spec or ROBOT
        robot = r"/World/envs/env_.*/Robot/"
        segments = ("distal",) if self.proxy_links == "tips" else ("distal", "middle", "proximal")
        out = [robot + robot_spec.finger_links_regex(*segments)]
        if self.proxy_links == "hand":
            palm = (robot_spec.palm_body_name, *robot_spec.palm_side_link_names)
            out.append(robot + "(" + "|".join(palm) + ")")
        if self.proxy_table:
            out.append(r"/World/envs/env_.*/Table")
        return out

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
        if self.thickness > self.spacing + 1e-9:
            import warnings

            warnings.warn(
                f"cloth thickness {self.thickness:.4f} m (2 x particle_radius) exceeds grid "
                f"spacing {self.spacing:.4f} m, ratio {self.thickness / self.spacing:.2f}x. This "
                f"is NOT an error -- particle_radius is a cloth-RIGID rest offset and cannot "
                f"cause self-collision, which uses self_contact_radius "
                f"({self.self_contact_radius:.4f} m, {self.self_contact_radius / self.spacing:.2f}x "
                f"spacing). For reference SIM1 ships 1.40x and Newton's example_cloth_franka.py "
                f"the same 8 mm radius. Flagged because a diameter well above spacing gives "
                f"neighbouring particles overlapping collision spheres, which changes grip.",
                stacklevel=2,
            )


    def build_physics(self, newton_cfg, num_envs: int, robot_spec=None):
        """``NewtonCfg`` for the coupled MJWarp (robot) + VBD (cloth) scene.

        The one structural difference from the cable: the VBD entry owns **particles**, not bodies.
        A cloth is a particle system, so ``all_particles=True`` is what assigns the sheet to the
        deformable solver -- a body selector would match nothing and the cloth would be integrated
        by nobody, presenting as a sheet frozen in mid-air.

        ``robot_spec`` names the hand bodies the cloth couples to (``utils/robot.py``'s robot).
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

        robot_spec = robot_spec or ROBOT
        rigid_bodies = [r"/World/envs/env_.*/Robot"]
        proxy_bodies = self.proxy_body_patterns(robot_spec)
        if self.proxy_table:
            rigid_bodies.append(r"/World/envs/env_.*/Table")

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
        if self.friction_epsilon > 0.0:
            vbd.friction_epsilon = self.friction_epsilon

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
                        # Edge/face soft contacts -- see `full_surface_contact`.
                        enable_rigid_soft_full_surface_contact=self.full_surface_contact,
                    ),
                )
            ],
            iterations=self.coupler_iterations,
            model_cfg=NewtonModelCfg(
                soft_contact_ke=self.soft_contact_ke,
                # `soft_contact_kd` was previously left at the `NewtonModelCfg` default (1.0e-2)
                # because this call only passed `ke` and `mu`.
                soft_contact_kd=self.soft_contact_kd,
                # Self-contact reads this raw; cloth-vs-shape pairs are rescaled in the cloth view
                # (`deformable_env.ClothEnv._install_cloth_friction`) so they keep using `soft_contact_mu`.
                soft_contact_mu=self.self_contact_mu if self.self_contact_mu > 0.0 else self.soft_contact_mu,
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
            use_cuda_graph=bool(self.use_cuda_graph),
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
class ClothEnvCfg(ClothTaskEnvCfg):
    """Fold a VBD cloth sheet with the Flexiv + right Wuji, on a coupled MJWarp + VBD solve."""

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2, gravity=(0.0, 0.0, -9.81))
    """``sim.physics`` is set at construction from ``newton`` + ``cloth`` (``ClothCfg.build_physics``)."""
    newton: NewtonSolverCfg = NewtonSolverCfg()
    use_gravcomp: bool = True
    """The robot's gravity-free bodies as MuJoCo gravity compensation (global gravity stays -9.81)."""
    cloth: ClothCfg = ClothCfg()
