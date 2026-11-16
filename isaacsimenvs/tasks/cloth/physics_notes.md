# Cloth physics reference notes

Archived configuration rationale and measurements. These notes preserve historical
experiments, including superseded assumptions; current field descriptions live in
[deformable_cloth_env_cfg.py](deformable_cloth_env_cfg.py). In particular, thickness
exceeding grid spacing now warns rather than raising and does not imply self-collision.

## NewtonSolverCfg.overview

MJWarp / Newton solver settings.

Values are the reference port's (github.com/kushal2000/isaac_newton @ beb9efb), which reached
parity with the PhysX baseline on this exact task. **Almost none of them are library
defaults**, and the defaults are not merely untuned -- several fail *silently*, dropping work
rather than raising, which is why each carries its measured justification here.

## NewtonSolverCfg.max_triangle_pairs

Narrow-phase triangle-pair budget. ``None`` sizes it from the environment count.

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

## NewtonSolverCfg.per_env_triangle_pairs

Headroom per env when sizing automatically -- about 2.5x the ~26.6k/env measured demand.

The buffer is a GPU allocation so it is not free, but under-sizing fails silently and
catastrophically while over-sizing costs memory, which is visible and diagnosable.

## NewtonSolverCfg.njmax

Max constraint rows per world. The library default of 300 is sized for a Franka/Allegro
lifting a simple object; this scene (22-DoF hand + tool + table) overflows it continuously
(``nefc overflow - please increase njmax``). Overflow is not benign -- constraints are
dropped and positions go wrong. The reference ran at 2400 for a long time, still overflowing
to 2878, and raising it to 8192 moved a "matching" score from 15.97 to 18.61.

## NewtonSolverCfg.condim

Contact dimensionality on the grasping shapes. The tool and fingertip assets already carry
torsional and rolling friction coefficients, but MuJoCo ignores them below condim 4 and 6
respectively. Set before the solver compiles -- MuJoCo Warp sizes its constraint arrays at
compile time, so writing it afterwards produces NaNs.

Kept on fidelity grounds (the authors' scene specifies it), **not** performance: measured
cleanly post-fix it is neutral (19.375 +/- 2.134 at condim 6 against 20.359 +/- 1.918 at
condim 3, z = 0.34), and an earlier claim that torsional friction was the missing physics was
retracted.

## NewtonSolverCfg.use_mujoco_contacts

Use Newton's own collision pipeline rather than MuJoCo's. Not a performance preference:
under MuJoCo's pipeline ``separate_worlds=True`` builds the MuJoCo model from world 0 only, so
per-env collision *meshes* are **silently ignored** -- every world simulates world 0's mesh
while reporting its own mass and inertia. That is a physically nonexistent hybrid that passes
a "do the environments differ?" check. Newton's pipeline reads per-world shape data correctly,
and on a mesh scene is also better conditioned (0.43 mm penetration against 6.5-9.2 mm) and
faster.

## DeformableClothCfg.thickness

Cloth thickness [m], as ``2 * particle_radius``.

**A VBD cloth is a surface: thickness IS the particle collision radius**, there is no
volumetric dimension. That makes it bounded by grid spacing -- particles closer together than
their diameter self-collide at rest and the cloth crumples before anything touches it. The
constraint is ``thickness <= size / (resolution - 1)``, checked in ``__post_init__``.

Thicker is genuinely better for *contact*: the cable was ungraspable at 10 mm and liftable at
30 mm, and thin geometry tunnels through fingers. But it is worse for *folding*, which needs
bending. And 30 mm on a 100 mm cloth forces a 4x4 grid (spacing 33 mm), which is a stiff mat
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
and ``resolution`` are the first pair to revert.

## DeformableClothCfg.resolution

Particles per side, so the cloth is ``resolution**2`` particles and
``2*(resolution-1)**2`` triangles. 7 gives 49 particles / 72 triangles.

**Driven by ``thickness``, not chosen independently.** 16 mm particles cannot sit on a grid
finer than 16.7 mm spacing without overlapping at rest, so matching SIM1's particle radius
forces this down from 9 (81 particles / 128 triangles). The fold geometry survives the change
unchanged -- ``corner_indices`` still returns the four far-half corners at the same
coordinates, ``folded_targets`` is identical, and the odd resolution keeps a hinge row -- but
the cloth is meaningfully coarser, and a 72-triangle cloth is close to the floor at which
"cloth" stops being a useful description.

## DeformableClothCfg.start_height

Spawn height above the episode's table centre (``reset.table_reset_z`` + its noise) [m].

**The table's collision surface is at ``table_reset_z + 0.150``, not + 0.045.** See
``table_half_thickness``. Everything below is measured with zero actions, so the robot never
approaches the cloth and anything it does is spawn behaviour:

    0.150  -> spawn 0.5300, surface 0.5298: in contact. EJECTED upward at ~1.7 m/s, peaks at
              0.68 by step 10, falls back, settles by step 30 with a 17 mm spread across a
              12 mm cloth -- i.e. visibly wrinkled.
    0.300  -> spawn 0.6800, a genuine 15 cm drop. Falls at exactly g (predicted 0.0340 m over
              the first 5 steps, measured 0.0349), lands at 0.5360 and settles PERFECTLY flat
              (min = mean = max to four decimals).

An earlier version of this docstring reported 0.150 as "settles dead flat", which was wrong: it
was read after settling and never checked the spread. Every episode was therefore starting with
the cloth thrown into the air and landing creased, and that was mis-attributed in turn to
thickness, to grid resolution, and to cloth physics.

Values below the surface are worse still, and were correctly diagnosed at the time: at 0.010 the
cloth starts inside the table and reaches 51 m by step 80.

## DeformableClothCfg.density

Surface density [kg/m^2]. **Per AREA, not per volume.**

This reaches ``ModelBuilder.add_cloth_mesh(density=...)`` unchanged (via the ``newton:density``
USD attribute, read in ``isaaclab_contrib.deformable.deformable_object``), and Newton documents
that argument as "the density per-area of the mesh". The old 100.0 was written as though it
were volumetric: it made the 0.1 x 0.1 m cloth weigh **1 kg**, a handkerchief with the mass of
a litre of water, and it disagreed by 83x with the ``area_density = density * thickness``
that ``cloth_env`` computes for the ``ClothAsRigidObject`` mass proxy.

2.0 kg/m^2 is SIM1 Table 2. Still heavy for cloth -- real cotton sheeting is 0.15-0.3 kg/m^2 --
but it is their calibrated value and it is 50x lighter than what this ran before.

## DeformableClothCfg.tri_ke

In-plane stretch stiffness -- the Lame ``mu`` (shear modulus) of the membrane model.

**1.0e2 is SIM1 Table 2, and also Newton's own ``default_tri_ke``.** Their "calibrated"
elasticity is the library default; what they actually tuned away from default is bending.

This was 1.0e4. The justification was the cable, where softening stretch was catastrophic
(x0.01 peaked at 94.9 m/s) -- but a cable is a 1-D chain whose only in-plane mode IS stretch,
and the cloth has an area term (``tri_ka``) carrying load the cable had no analogue for. The
cable result is not evidence about a membrane. If the cloth oscillates or blows up, this is the
first suspect and 1.0e3 is the obvious intermediate.

## DeformableClothCfg.edge_ke

Bending stiffness. Low -- a cloth that will not bend cannot be folded. This is the opposite
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
dimensionally portable even though the saturation argument says it should not matter.

## DeformableClothCfg.plastic_yield_curvature

Bending plasticity: yield curvature [1/m], 0 = off (purely elastic, as before).

A hinge bent past ``kappa x hinge width`` (16.7 mm along the grid, 11.8 mm on the diagonals)
keeps the excess as a new rest angle, so a pressed fold stays folded instead of springing back
(``utils/plasticity.py``, after Narain et al. 2013). Reset to flat with the cloth.
It is a mesh-level parameter: in the scripted fold-press-release test
(``scripts/analysis/cloth_plasticity_probe.py``) 100 (95 deg per grid cell) holds a pressed
towel or cotton fold without changing the free drape; napkin and denim need 50, chiffon and silk 25.

## DeformableClothCfg.substeps

Cloth substeps inside one coupled step, i.e. the VBD timestep.

**6 makes the cloth dt 1/1440 s, matching SIM1.** Their loop is 60 Hz with 24 substeps; ours is
``sim.dt`` 1/120 / ``newton.num_substeps`` 2 / this, so 6 lands on the same 0.694 ms. It was 2
(1/480 s, 3x larger).

**This directly contradicts a cable measurement**: on the cable there was a sharp optimum at 2,
with 1 peaking at 37 m/s and 4/8/16 degrading monotonically, on the theory that fewer, larger
substeps mean fewer proxy exchanges per step and the exchange is where the energy enters. That
was never re-measured for the cloth, and the cloth is a different solver entry, but it is a
direct conflict and not a gap -- if the cloth gains energy at reset or under contact, revert
this first.

Costs 3x the VBD work per policy step, on top of the iteration increase.

## DeformableClothCfg.vbd_iterations

VBD solver iterations per substep.

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
once: converging the cloth's internal dynamics, and resolving contact against the rigid proxy.
What degrades at 5 is only the second -- the hand stops gripping, footprint stays near 1.0, the
cloth is untouched rather than unstable -- so the extra iterations are buying contact, not cloth.

A previous version of this note proposed `SolverVBD(rigid_contact_history=True)` as the targeted
fix, and recorded it as tried-and-reverted because contact warm-starting packs contact ids into
20 bits, capping buffered contacts at 2**20 = 1,048,576 globally while this scene budgets
7,962,624 triangle pairs at 32 envs. **That blocker is real but it was aimed at the wrong
lever.** In Newton 1.5 `rigid_contact_history` is body-BODY only -- it drives
`snapshot_body_body_contact_history` and never touches body-particle contacts, so it could not
have helped the cloth's grip even if it had fit.

The knob that does reach cloth-vs-hand contact is `rigid_avbd_beta`; see that field.

## DeformableClothCfg.use_cuda_graph

Capture the coupled MJWarp + VBD step as a CUDA graph. Off by default because every result so
far was produced without it (inherited from the cable, whose reason -- entries stepping at
different rates -- was never measured: the substep counts are fixed, so a graph records them).
At small env counts the step is kernel-LAUNCH bound (decimation 2 x substeps 6 x 12 VBD
iterations per policy step), which is what makes the live viewer's cloth ~4 steps/s.

## DeformableClothCfg.soft_contact_mu

Body-particle friction coefficient. SIM1 Table 2 (`self_contact_friction`; was 10.0 here).

Effective per-contact friction is `sqrt(soft_contact_mu * shape_mu)`, so against the fingertips
(`assets.finger_tip_friction` 1.5) this moves from `sqrt(10 x 1.5)` = 3.9 to `sqrt(0.25 x 1.5)`
= 0.61. SIM1's own effective value is 0.5 -- they set every shape's mu to 1.0.

Table 2's `robot_friction` 1.5 / `table_friction` 0.0 are NOT copied: their env overwrites
every shape's mu to 1.0 immediately after building the table, so those two rows describe a
configuration that does not run. Our per-shape frictions stay as the rigid-tool task sets
them.

## DeformableClothCfg.self_contact_mu

Cloth-on-cloth friction coefficient (ply on ply). 0 = ``soft_contact_mu`` (previous behaviour).

Newton's VBD self-contact uses the raw ``model.soft_contact_mu``, while every cloth-vs-shape
contact uses ``sqrt(soft_contact_mu * shape_mu)``. The fabric presets calibrate
``soft_contact_mu`` for cloth-on-table, which left ply-on-ply as the LOWEST friction in the
scene (big napkin 0.265 against 0.36 table, 0.63 fingertips), so a pressed fold slid apart.

When > 0: ``model.soft_contact_mu`` becomes this value and ``DeformableClothEnv`` rescales every
shape's friction IN THE CLOTH SOLVER'S VIEW ONLY by ``soft_contact_mu / self_contact_mu``, so
each cloth-vs-shape coefficient stays ``sqrt(soft_contact_mu * shape_mu)``. The rigid solver
(robot vs table) keeps the original shape frictions.

## DeformableClothCfg.rigid_contact_k_start

Body-particle contact penalty SEED for AVBD ramping [N/m].

**Inert unless `rigid_avbd_beta` > 0**, which is why it did nothing until now. Newton computes
`rigid_contact_k_start_value = -1.0 if linear_beta == 0.0 else k_start` (solver_vbd.py:677) and
the kernel reads `k_floor = avg_ke if k_start < 0.0 else min(k_start, avg_ke)`
(rigid_vbd_kernels.py:3395) -- so with beta 0 the seed is discarded and every contact starts at
full material stiffness.

## DeformableClothCfg.rigid_avbd_beta

AVBD penalty ramp rate for body-particle contacts [N/m per m of penetration].

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
the grip rather than strengthen it. A/B it against 0.0 before trusting it.

## DeformableClothCfg.enable_self_contact

Whether the cloth collides with ITSELF.

NEW FIELD, and the largest behavioural change in this config. `VBDSolverCfg` defaults
`particle_enable_self_contact` to False and `build_physics` never set it, so until now the
folded flap passed straight THROUGH the half it was supposed to land on. For a folding task
that is not a tuning knob, it is a missing mechanic. SIM1 runs `handle_self_contact=True`.

## DeformableClothCfg.self_contact_radius

Distance at which cloth primitives start to repel each other [m]. SIM1 Table 2.

**This is a geometric distance between mesh primitives, NOT scaled by `particle_radius`** --
see `evaluate_self_contact_force_norm` (particle_vbd_kernels.py:821), which is driven by
`collision_radius` alone.

So copying SIM1's number does not reproduce SIM1's behaviour on our geometry. Their cloth is a
scanned garment with ~mm triangles, where 2 mm is comparable to the mesh scale; our cloth is
16 mm thick on a 16.7 mm grid, so a folded flap will settle 2 mm above the layer beneath it
rather than ~16 mm -- the two layers visually interpenetrate by most of their thickness.

That interacts with the fold targets: `cloth_env` builds them with `lift = 2 * particle_radius`
= 16 mm, so the targets sit ~14 mm above where the flap can actually rest. That is inside the
0.04 m `keypoint_tolerance`, so it biases the reward rather than making the fold impossible.
Scaling this to ~`thickness` (with margin ~1.5x) is the consistent alternative and costs more
in collision detection. Left at SIM1's value; flagged, not hidden.

## DeformableClothCfg.conservative_bound_relaxation

Relaxation factor for the penetration-free displacement bound (SIM1's beta, Table 2).

Newton 1.5 defaults this to 0.85; 0.42 was the default in the Newton 0.1.3 that SIM1 forked, so
their Table 2 row is arguably just that default restated. Kept anyway -- it is the value their
calibration ran against, and a tighter bound truncates more aggressively, which is the safe
direction on a cloth that has never had self-contact enabled before.

Only has any effect when `enable_self_contact` is True.

## DeformableClothCfg.full_surface_contact

Cloth EDGES and TRIANGLE INTERIORS collide with the proxy shapes, not only its particles.

Newton 1.5.2 ``CollisionPipeline(enable_rigid_soft_full_surface_contact=True)``: SDF-based
edge/face soft contacts (Macklin et al. 2020 local optimisation), consumed by ``SolverVBD``.
Targets the finger pierce (docs/reports/4_physx/cloth_finger_pierce.md): per-particle contact
lets a fingertip push the particle under it aside and reach the table between particles.

Every participating mesh shape (hand proxy links, a mesh table) needs a volume SDF, which
``DeformableClothEnv`` provisions before finalize (``shape_force_sdf``). Rigid-rigid contacts run in
Newton's pipeline too (``use_mujoco_contacts: false``), so those SDFs can change hand-table
contact as well. Off by default, so earlier results keep their meaning.

## DeformableClothCfg.rigid_body_particle_contact_buffer_size

Body-to-particle soft-contact capacity **PER RIGID BODY**, not global.

Newton's own comment on the argument is explicit: "Per-body soft-contact list capacity"
(`newton/_src/solvers/vbd/solver_vbd.py:266`, default 256). So it must NOT scale with env count:
the allocation is this value x body count, and body count already scales with envs. Multiplying
it by `num_envs` as well makes the allocation quadratic -- at 512 envs that is 1,048,576 x 2,304
bodies = 2.4e9 elements, which overflows Warp's signed-int32 array dimension and aborts before a
single step.

1024, inherited from the cable where the manipuland was bodies rather than particles, was far
too small: a 32-env run died with a CUDA device-side assert, which is what an out-of-bounds
write into a contact buffer looks like from the host.

## DeformableClothCfg.per_particle_triangle_pairs

Triangle-pair budget per cloth particle. The budget is GLOBAL and must scale with env count
*and* geometry: on the cable a correctly env-scaled 4.19M still overflowed once segments
doubled.

Raised from 512 after a 32-env run overflowed twice. That run then died with
``normal expects all elements of std >= 0.0`` from the policy's sampler -- dropped contacts let
a cloth penetrate, which produced NaN in that env's observation. The policy error was the
symptom; the overflow was the cause, and it is worth remembering that this class of failure
surfaces far from its origin.

## DeformableClothCfg.check_thickness.overview

Warn when particle DIAMETER exceeds grid spacing. No longer an error.

**The original premise was wrong.** This raised on `thickness > spacing`, on the grounds
that closer particles "self-collide at rest and the cloth crumples before anything
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
as an error it forced this task from a 9x9 cloth to 7x7 for no reason.

## DeformableClothCfg.build_physics.overview

``NewtonCfg`` for the coupled MJWarp (robot) + VBD (cloth) scene.

The one structural difference from the cable: the VBD entry owns **particles**, not bodies.
A cloth is a particle system, so ``all_particles=True`` is what assigns the cloth to the
deformable solver -- a body selector would match nothing and the cloth would be integrated
by nobody, presenting as a cloth frozen in mid-air.

``robot_spec`` names the hand bodies the cloth couples to (``utils/mdp/flexiv_wuji.py``'s robot).

## DeformableClothCfg.triangle_pair_budget.overview

Global triangle-pair budget for this cloth at ``num_envs``.

Scales with particle count as well as env count. The cable overflowed a correctly
env-scaled 4.19M as soon as its segment count doubled, and a cloth has far more triangles
than a cable -- so geometry has to be in this product, not just ``num_envs``.
