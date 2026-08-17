# Phase 4 — cable goal reaching

**Status: built, running, and the cable now settles correctly.** With zero robot action it falls
under gravity and comes to rest on the table at z = 0.5351 (table top 0.53 + its 0.005 radius),
holding that value from step 20 through 200 with no drift and zero terminations.

The spawn-time ejection that blocked this is **fixed** — see below. It was our bug, not Newton's.

```bash
# the cable now settles; the policy has not been scored on it yet
scripts/newton_py -m isaacsimenvs.eval.episodes \
  --task Isaacsimenvs-Cable-Direct-v0 --num_envs 64 --single_variant
```

## What it is

`PlayEnv` → `PlayNewtonEnv` → **`CableEnv`**, following the `BottleFlipEnv` precedent one level
further down. The task is untouched: same action pipeline, same keypoint reward, same goal
sampler, same terminations, and the **observation stays 140-dim**, so the pretrained checkpoint
loads. Two things change — the manipuland is a `CableObject`, and the physics becomes a coupled
solve.

| file | role |
|---|---|
| `tasks/cable/cable_env.py` | the env; spawns the cable, installs the adapter |
| `tasks/cable/cable_env_cfg.py` | `CableCfg` (geometry, material, coupling) + the coupled `NewtonCfg` |
| `tasks/cable/utils/cable_adapter.py` | presents the cable as a rigid object |
| `cfg/task/Cable.yaml` | overlay |

### The adapter is the load-bearing idea

The task reads a rigid manipuland's pose; a cable has none. Rather than fork the task code,
`CableAsRigidObject` synthesises one — position from the segment centroid, orientation from a
right-handed frame whose +X follows the cable's span. The surface it has to cover was enumerated
from the task code, not guessed, and is small: five reads (`root_pos_w`, `root_quat_w`,
`root_lin_vel_w`, `root_ang_vel_w`, `default_mass`) and three writes (`write_root_pose_to_sim`,
`write_root_velocity_to_sim`, `set_external_force_and_torque`).

It also converts quaternions. `CableObject` speaks xyzw; the task speaks wxyz. For real Isaac Lab
assets `wrap_env_assets` arranges that, but the adapter is not an Isaac Lab asset, so those
patches never touch it and it converts in both directions itself.

### Coupling

MJWarp for the robot, VBD for the cable, bridged by `CouplerProxyCfg` with the five fingertip
`_DP` links proxied into the cable's entry. Settings carried from the reference investigation,
each of which is a correction rather than a preference:

* `cable_substeps = 4` — the cable entry was integrating the coupled contact at the coupled step
  rate, and a 0.2 g segment cannot absorb that impulse in one step.
* `mass_scale = 1.0` — **not** 0.05, which is ~500× worse and is itself the cause of robot NaNs.
* `rigid_body_contact_buffer_size = 4096` — the body-BODY buffer, which `VBDSolverCfg` does not
  expose (SolverVBD defaults it to 64) and which is the list that overflows for a cable, since a
  Newton cable is bodies and joints rather than particles.

## RESOLVED: the cable was ejected at spawn

The symptom, before the fix. Measured with zero robot action, env 0, first steps after reset:

| step | max‖v‖ | vx | vy | vz | centroid z |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.46 | **−2.13** | 0.00 | +0.17 | 0.6334 |
| 10 | 2.30 | −2.05 | −0.00 | −0.16 | 0.6857 |
| 19 | 2.65 | −2.03 | −0.00 | −1.62 | 0.5520 |

**At step 1 the cable already carries −2.13 m/s in x**, before gravity can contribute anything,
plus a small *upward* kick. Thereafter `vx` holds at ≈ −2.03 while `vz` decays at exactly
−0.163/step = −9.81 m/s²: pure ballistic flight. It never reaches the table because it is thrown
off the side first, and the task's `fall` termination then resets it on a 30-step loop.

### The one clue that discriminates: it scales with substeps

| `num_substeps` | solver dt | vx at step 1 |
|---:|---:|---:|
| 2 | 4.17 ms | −2.13 m/s |
| 8 | 1.04 ms | **−8.14 m/s** |

4× the substeps, 3.8× the impulse. **A force applied per substep instead of per tick** produces
exactly this, and the reference documented the same signature for a different bug ("if the
wrench were applied per substep, 8 substeps would mean 8× gravity, and would scale with substep
count exactly as observed").

Note the direction of the result: *more* substeps makes it worse. So this is not a stiff-constraint
integration-stability problem, which finer stepping would improve — even though the parameters
invite that reading. Isaac Lab's own `scripts/demos/cables.py` runs a cable at `num_substeps=8`,
`dt=0.01` and `iterations=20`, against our inherited `num_substeps=2` at `dt=1/120`; the moduli
formula we use is copied from that demo and matches exactly. The demo, however, runs the cable
**alone**, with no coupler.

### Ruled out by direct measurement

Each of these was a plausible hypothesis, tested, and killed:

| hypothesis | test | result |
|---|---|---|
| contact-buffer overflow | live `body_body_contact_overflow_max` every step; raw unfiltered stdout | `[0]` throughout, `prealloc=4096` live, **no warning printed anywhere** |
| reset velocity write failing | read velocity after reset, before stepping | exactly `0.0` — the write works |
| cable spawns intersecting the hand | reset height 0.63 vs 1.08 (clear of hand and table) | **identical** impulse |
| the table proxy | `proxy_table` true vs false | **identical** |
| gravity route | `use_gravcomp` true vs false | **identical** (−2.13 both) |
| reset pose inconsistent with rest state | compare spawned layout vs post-reset write | **identical**: n=12, spacing 0.025000, same x range |
| proxy re-collision frequency | `collide_interval` 1 vs 4 | **identical** |
| too coarse a timestep | `num_substeps` 2 vs 8 | **worse**, and scales — see above |

The cable also does not stretch (span holds at 0.275 throughout) while it does bend (z spread 0 →
0.24 in a few steps), so whatever acts on it is not a stretch-constraint violation.

### A wrong diagnosis, and how it survived three probes

This was first written up as "the cable falls through the table". That was wrong, and it is worth
recording why, because the failure was in the method rather than the arithmetic:

* Sampled at steps 20 / 100 / 200 against a 30-step reset cycle, the height looked like it settled
  at 0.524 m with the table top at 0.53 m — which read as contact.
* Free fall from 0.63 m takes ~20 steps at 60 Hz, which matched the observed reset period.
* The probe reported zero terminations, because it printed them only at the sampled steps and
  never landed on one.

Three consistent-looking signals, one wrong conclusion. The contradicting evidence was present the
whole time and under-weighted: **0.68 m of lateral travel in 20 steps.** Free fall does not move
sideways. Ten seconds of rendered video settled it in one pass.

Two lessons, both cheap: a quantity sampled sparsely from a cyclic process can look exactly like
equilibrium, so log terminations every step rather than at checkpoints; and when a vertical story
fits, check whether the horizontal one does too.

### What that means for `proxy_table`

The earlier claim that `proxy_table=True` restores cable–table contact is **unproven either way** —
the cable never reaches the table in either setting, so the comparison never tested it. The
underlying reasoning still stands and is worth keeping: a static shape belongs to exactly one
coupler entry, and this task's table is spawned `kinematic_enabled=True`, so it is a kinematic
*rigid body* that the cable entry's `include_static_shapes` never claims.

## Results: mass and shape are not the constraint

32 envs, one episode each, single shared object, DR off, `success_tolerance 0.01` — the same
harness and protocol as phases 1 and 3, so these are comparable to the **18.4 goals/episode**
rigid-tool baseline on this backend.

| config | goals/episode | lift | terminations |
|---|---:|---:|---|
| `cable_10mm` | 0.0 ± 0.0 | 0.0 | fall 21, timeout 11, cap 0 |
| `cable_30mm` | 0.0 ± 0.0 | 0.0 | fall 32, timeout 0, cap 0 |
| `cable_50mm` | 0.0 ± 0.0 | 0.0312 | fall 31, timeout 0, cap 0 |
| `rod_plain` | 15.7812 ± 2.5616 | 0.8438 | fall 7, timeout 21, cap 3 |

Two things are settled by this.

**Mass and cross-section are not what defeats the policy.** 10 mm to 50 mm spans **25x in mass**
(2.4 g to 58.9 g) and every one scores exactly 0.0. This independently reproduces the reference's
null result on "affordance as cross-section" — and unlike that attempt, `object_scale` is correct
here, so the policy genuinely saw the geometry change rather than being told it held a hammer
throughout. The 50 mm case lifted once in 32 envs (0.031), the only non-zero lift among the cables.

**Nor is the shape.** A plain 0.30 m capsule with no handle and no head scores **15.78 ± 2.56**,
close to the rigid-tool baseline. The policy does not need the handle+head affordance it was
trained on; it needs the object not to deform — or so it appears, subject to the control below.

## RESOLVED: deformability is the constraint

The control: the **same rigid rod, inside the same coupled MJWarp+VBD solve** the cables run
under, with a real cable parked on the ground so the VBD entry still owns a deformable. Only the
manipuland's deformability differs.

| manipuland | solver | result |
|---|---|---|
| cable 10 / 30 / 50 mm | coupled | 0.0 goals/episode; lift **0.000 / 0.000 / 0.031** |
| **rigid rod**, same mass and dimensions | **coupled** | **195 successes in 800 steps, 28/32 envs lifted** |

Progression, 32 envs: 25 successes / 25 lifted by step 200, 90 / 27 by 400, 148 / 28 by 600,
195 / 28 by 800 — steady, not a burst.

**So the coupling does not prevent grasping**, and the cable zeros mean what they appear to. Taken
with the two negatives above, the elimination is complete:

* **not mass** — 25x span (2.4 g to 58.9 g), all zeros
* **not cross-section** — 10 mm to 50 mm, all zeros
* **not the shape** — a plain capsule with no handle or head scores 15.78 +/- 2.56, near the 18.4
  rigid-tool baseline
* **not the coupled solver** — the same rod inside it is lifted by 28/32 envs
* **deformability** is what is left, and it is what the rod isolates

### Two caveats worth carrying

The coupled scene **diverges to NaN past ~1000 steps** even with the geometry correct, so the rod
result is 800 steps of a 32-env run rather than a completed `episodes.py` figure. The cable runs
survived 6000 steps; the difference is that their manipuland is reset every episode while the
parked cable is never reset and drifts. That instability is real and unexplained, and it is why
this is reported as successes-in-N-steps rather than goals/episode.

The rod also settles the *lift* question rather than the *placement* one. Cables never reach the
lifting stage at all, so nothing here says whether a lifted cable could be placed.

## Retracted: an earlier reading of this control

An earlier version of `rod_coupled` scored 0.0 and NaN'd, which was written up here as possibly
inverting the conclusion — "the coupling prevents grasping, deformability is exonerated". **That
was measuring a broken scene** and is withdrawn.

Three defects stacked, all invisible in the numbers and all visible in one rendered frame:

1. `_install_cable` overwrote `env.object` with the cable adapter *after* the rod was installed,
   so the "rigid rod" control was really the cable, with a spare rod on top of it;
2. the park never applied, because a cable's placement comes from a pose *write*, not its spawn
   position -- the same trap that made `cable_start_height` a no-op earlier;
3. the first corrected park put the cable below ground, and the second put it 1.0 m away with
   `env_spacing` at 1.2, i.e. inside the neighbouring environment, where that robot struck it as
   soon as the policy moved.

The tell was two objects on the table in the render. The numbers alone -- 0.0 goals, then a NaN --
looked exactly like a real physical finding.

## Superseded: the control that did not work

`rod_plain` runs a **plain MJWarp** solve, because a scene with no deformable gives the coupled
solver's VBD entry nothing to own. So `cable_30mm` vs `rod_plain` varies deformability **and the
solver together**, and the tidy reading — "the policy handles rigid, fails on deformable" — is not
established by it.

`rod_coupled` was built to close that: the same rigid rod, inside the *same* coupled MJWarp+VBD
solve, with a real cable parked out of the way so the VBD entry still owns a deformable. It is
**not usable yet**:

* at 8 envs / 1200 steps it completes and scores **0.0 goals, 0.0 lift** — which, taken at face
  value, would mean the *coupling* prevents grasping and deformability is exonerated;
* at 32 envs / 6000 steps it diverges to NaN (`RuntimeError: normal expects all elements of
  std >= 0.0`).

Those two facts cannot be separated yet: a scene that NaNs at full scale may well have been
heading there at reduced scale, in which case its 0.0 measures its own instability rather than the
coupling. **So the headline question — deformability or coupling — is open**, and any claim that
the cable results demonstrate deformability is unsupported until this control runs clean.

There is a plausible mechanism for the coupling degrading manipulation, which makes it worth
ruling out properly rather than assuming: the proxy maps the fingertips into the VBD entry as
virtual bodies, and `mass_scale` exists precisely because that alters their effective inertia
there. The reference's `mass_scale=0.05` retraction was about exactly this going wrong.

## The rigid-rod control, and a bug it exposed

`env.cable.rigid_rod=true` swaps the cable for a single rigid capsule of the same length,
thickness and mass — the zero-DOF control for "is it deformability, or is it the shape?".

A cable cannot be made rigid by reducing segments: `CableCfg` rejects fewer than three control
points (`segments=1` raises `CableCfg requires at least three positions, got 2`), and the minimum
of 2 segments is actively pathological — measured peak speed **39.3 m/s against 4.8 m/s** for the
12-segment cable at the same thickness, because all bending concentrates at a single midpoint
hinge, which is exactly where a descending hand folds it. The reference saw the same thing and its
`thick2` runs NaN'd.

**The bug: the rod was frozen, and it looked fine at reset.** Registered at the `Cable` prim path,
it was assigned to the coupler's **VBD entry** — which does not integrate rigid bodies. It never
fell, never responded to a 0.5 m/s push, and reported the same pose for 90 steps:

```
before   step 10 [0.6, 0.0, 0.63]   step 100 [0.6, 0.0, 0.63]     <- never moves
after    step 10 [0.6, 0.0, 0.5385] step 100 [0.6217, 0.0, 0.5449] <- falls, then is pushed
```

Every check made *at reset* passed — pose, orientation, keypoints, `object_scale` all correct and
identical to the cable's. A frozen value is indistinguishable from a correct one at t=0. The test
that caught it was applying a known velocity and asking whether the observation followed.

Fixed by routing the rod to the plain MJWarp solve, since a scene with no deformable has nothing
for a VBD entry to own.

**Confound, stated rather than hidden**: the rod therefore runs a plain MJWarp solve while the
cable runs the coupled MJWarp+VBD one, so `cable_30mm` vs `rod_30mm_rigid` isolates deformability
*and* the coupling together. A rod the policy still cannot grasp would rule deformability out; a
rod it can grasp would need the coupling ruled out separately before crediting deformability.

## Does damping unlock goals? No -- and an ejection guard the harness was missing

Scoring the damped configurations (32 envs, hammer, single variant) answers the obvious follow-up
to the tuning above: stabilising the cable does not make it graspable.

| config | goals/ep | lift (guarded) | ejected | terminations |
|---|---:|---:|---:|---|
| 30 mm, 12 seg, damped | 0.0 | 3/32 | 0 | fall 30, hand_far 2 |
| 30 mm, 2 seg, damped | 0.0 | 4/32 | 0 | fall 32 |

**Goals remain exactly zero**, with the identical failure mode, while peak speed improved ~6x. The
one alternative reading of the earlier cable results -- that they failed because the simulation was
poor rather than because cables are hard to grasp -- is now tested and rejected.

### The guard, and why the lift numbers needed one

`lift_fraction` was reading `_lifted_object`, which `reward_utils.lifting_reward` computes as
`(z_lift > threshold) | prev_lifted` -- **latched, with no upper bound**. An ejected cable therefore
sets it permanently True and reports as a lift. The plan called for a workspace-bound check and the
harness never had one; `episodes.py` now records per-env `max_obj_z` and classifies anything above
`REACH_M = 1.40` (iiwa14 extended plus hand) as ejected, reporting guarded and unguarded fractions
separately.

Applied to 2 segments -- the config with known velocity excursions, the one most likely to be
flattered by the bug -- it reported 0 ejected, with the four lifts at 0.815 / 0.781 / 0.768 /
0.763 m against a 0.63 m table. That looked like 13-19 cm of real clearance, and it was briefly
recorded here as such.

**Watching the video showed it is not.** The hand contacts the cable at the table edge, the cable
tips **up on end**, and then falls to the floor. It is a pivot, not a grasp.

The arithmetic confirms it and should have been the tell without the video. A 0.30 m cable lying
flat has its centre at the 0.63 m table height; pivoted upright its centre sits at
0.63 - 0.015 + 0.15 = **0.765 m**. The four values cluster on that within ~2 cm -- the signature of
one fixed geometry, where four independent grasps would scatter. And `lifting_bonus_threshold` is
0.15, precisely the rise a pivot produces, so the lift flag fires on a cable standing on end.

So `REACH` closed one hole and not the class of hole: ejection was never the only way to inflate
this statistic. **No cable has been grasped in any configuration measured.** The 12-segment env that
reached 1.221 m did not reproduce on a re-run and remains unexplained.

Videos: `videos/cable_lift_2seg.mp4` (pivot then fall), `videos/cable_lift_12seg.mp4` (the cable
never leaves the table while the hand works above it).

`termination_reasons` counts *reasons*, not episodes: the `done_*` flags are not mutually exclusive,
so an env tripping two on its final step is counted twice and the totals can exceed n. This is why
one run read `fall 30 + hand_far 3 = 33` on 32 envs.

### What the height distribution actually shows

The informative statistic was one not being looked for. **28 of 32 envs (12 seg) have `max_obj_z`
pinned at exactly the 0.63 m table height** -- the cable was never displaced upward at all -- yet
every episode terminated in `fall`. The policy pushes the cable sideways off the table without ever
getting underneath it.

That is a sharper failure than "cables do not work", and it is consistent with the one live
untested hypothesis: a cable lying flat offers no clearance to close fingers around, at any
diameter. It predicts a raised-support arrangement would be graspable, which remains the next
experiment worth running.

Do not read the 4/32 vs 3/32 difference between segment counts as an effect. At n=32 that is well
inside noise (Fisher exact ~0.35), the same 12-segment config drew 1/32 on one run and 3/32 on the
next, and per the correction above neither figure counts grasps in the first place.

## Tuning the 2-segment cable: damping + iterations, and why single draws misled

The 2-segment cable is intermittently unstable — peak speed ~67 m/s against 4.8 for 12 segments.
Two knobs were never being set at all: `CableMaterialCfg` exposes stiffness but **no damping**,
while `ModelBuilder.add_joint_cable` takes eight parameters (stretch/shear/bend/twist, stiffness
*and* damping), each defaulting to 0.0. So the cable ran completely undamped.
`patches.install_cable_damping` writes `joint_target_kd` before finalize, since the USD path
cannot carry it.

**Three runs per config**, peak speed, 4 envs x 250 steps:

| config | runs | mean | worst |
|---|---|---:|---:|
| baseline | 75.3 / 61.7 / 64.5 | 67 | 75 |
| `angular_damping=1.0` | 29.0 / 4.48 / 7.16 | 13.5 | 29 |
| `vbd_iterations=40` | 5.27 / 78.3 / 21.3 | 35 | 78 |
| **`angular_damping=1.0` + `vbd_iterations=20`** | 5.01 / 11.08 / 4.48 | **6.9** | **11.1** |

**Recommended: `angular_damping=1.0` with `vbd_iterations=20`.** Best mean and best worst case, and
cheaper than 40 iterations. Worst case is the figure that matters for a config whose failure is
intermittent, and only the combination has no excursion into the baseline range.

Not defaults, deliberately: damping changes the cable's physics (newton#2557 — high VBD cable
damping visibly alters a catenary's shape), so applying it to the 12-segment configuration would
invalidate the results recorded above without re-validation.

### What did not work

* `proxy_mode="staggered"` instead of `lagged` — 36.3, inside the baseline band.
* `coupler_iterations` 1 -> 4 — **109**, worse than baseline. Re-exchanging state between two
  only-approximately-converged entries amplifies rather than settles. Note the contrast with the
  *inner* VBD iterations, which do help.
* `cable_substeps` 4 -> 8 — 22.4, mild, and the most expensive of the three.
* `angular_damping=10.0` — 54.7, no better than none. The response is U-shaped: damping stiff
  relative to the timestep overshoots and injects energy itself.
* `linear_damping` on top of angular — no gain. The instability is a bending mode at a single
  hinge; span deviation was already 0.0000, so stretch was never the problem.

### A methodology note, learned the hard way here

The first pass ran one draw per config and produced a clean story: damping 0.1 -> 1.0 -> 10.0
tracing a U, and iterations 10 -> 20 -> 40 falling monotonically 39-99 -> 9.98 -> 3.57. Both curves
were reported, and the monotonic one was used to argue that iterations was the better lever.

Repeats destroyed that. `vbd_iterations=40` re-ran at 78.3 — indistinguishable from broken — and
its apparent monotonicity was three single samples from a distribution wide enough to produce it by
chance. The baseline itself had drawn 39.3 once and 61-75 on three later runs.

The underlying finding is that these knobs shift the *probability* of an excursion rather than
removing its cause, which fits the geometry: one hinge carrying the entire bend response, at a
timestep where a stiff hinge is marginal. **No solver tuning found here makes 2 segments genuinely
sound**, and the 12-segment cable remains the configuration to prefer.

## The hand could only ever feel the cable with five fingertips

The coupling is proxy-based: the cable is touched **only** by robot links named in
`CouplerEntryCfg.proxies`, and that list was
`left_(thumb|index|middle|ring|pinky)_DP` -- the five distal phalanges. The palm and the proximal
and medial phalanges did not exist as far as the VBD entry was concerned, while the rigid-rod
control collides with the whole hand through MJWarp. An enclosing grasp was therefore
*geometrically impossible* for a cable, and that asymmetry sat underneath every cable measurement
as a precondition rather than a variable.

`CableCfg.proxy_links` now selects `tips` | `fingers` | `hand`, and **defaults to `hand`** (full
phalanges, palm, thumb/pinky metacarpals). Anything meant to characterise the cable rather than the
coupling should stay there.

Two traps found while wiring it. The URDF link `left_hand_C_MC` is merged away by the importer and
matches **no** Newton body -- the palm body is `iiwa14_link_7` (`scene_utils.PALM_BODY_NAME`).
Naming the URDF link proxied nothing, and because the surrounding alternation still matched the
thumb/pinky metacarpals, the coupler's own empty-regex guard (`coupler.py:337`, which *does* raise
on a regex matching zero bodies) never fired. And `bend_stiffness_scale=2000` fails config
validation: the field is `float`, so the CLI needs `2000.0`.

### The positive control, on the fixed scene

| object | goals/ep | lift | terminations |
|---|---:|---:|---|
| **rigid rod** (0.30 m x 30 mm, matched mass) | **9.05 +/- 1.70** (n=21, 11 censored) | **28/32** | fall 6, timeout 15 |
| cable, same everything | 0.0 | 3/32 | fall 29 |

Same coupled solver, same whole-hand proxies, same harness, same dimensions. This is what makes the
cable zeros interpretable: the scene is sound, the proxies work, the goal is reachable, and the
cable's 0.0 is a property of the cable. It also sets the target -- 0 -> 9 goals/episode, with lift
3/32 -> 28/32 as the leading indicator, since goals follow lift.

## Open defect: the pose estimator degenerates exactly when the cable is held

`CableAsRigidObject.root_quat_w` derives the manipuland's orientation from **one vector** -- first
segment to last (`_frame_from_span(pos[:, -1] - pos[:, 0])`). That is well-conditioned only while
the cable stays straight.

| segments | worst span deviation | % of rest | lifted |
|---:|---:|---:|---:|
| 12 | 0.0030 m | **1.1%** | 4/8 |
| 24 | 0.0467 m | **16.2%** | **8/8** |

A 24-segment cable lifts in **every** env and droops 16% in the hand -- so the first-to-last vector
stops describing it precisely when it is being manipulated. Success needs keypoints (built from
position *and orientation*) within 1.5 cm of the goal for 10 consecutive steps, so a degraded
orientation makes scoring impossible however good the grasp is.

That resolves the 24-segment paradox: highest lift measured anywhere (73-80% at 64 envs, 8/8 here),
the **calmest** dynamics of any config (max 1.94 vs 2.84 for 12 segments), and **zero goals**. The
earlier explanation in this document -- "more segments means more exchange traffic, so transport
degrades" -- is **wrong**; its excursions are smaller, not larger.

It also qualifies the headline result: goals come from the 12-segment config partly because it
stays straight enough for a one-axis frame to work, not necessarily because it is grasped better.
Lift says the opposite.

**Fix implemented, and it did not help.** `CableCfg.pose_axis = "pca"` takes the principal axis of
the segment cloud (sign-matched to first->last so the frame cannot flip). At 24 segments it scores
**0 goals at 0.797 lift** -- unchanged. So the span collapse is real and the estimator genuinely
does degenerate, but that was **not** the operative cause, and the prediction stated here before
running it was wrong.

What that leaves open: 24 segments lifts 80%, has the calmest dynamics measured, now has a
bend-tolerant orientation estimate, and still never scores. Untested candidates, both about the
*position/scale* half of the keypoints rather than orientation:

* `object_scale` comes from nominal cable dimensions, so keypoint offsets are sized for a straight
  cable. On a drooping one they may sit outside the physical body regardless of frame.
* `root_pos_w` is the centroid of the segments, which moves away from where a goal pose expects the
  object's centre as the cable curls.

Default stays `span`: every result in this document was measured with it, and `pca` changes both
the policy's observation and the pose the success test scores.

## Solved by the coupling, not the cable: `proxy_mode=staggered`

**Four of five seeds reach 5 goals in a single episode.** Flat table, **no clearance**, zero
contact-buffer overflows, zero ejected, zero censored.

| seed | total goals | max/episode | lift | falls |
|---|---:|---:|---:|---:|
| 0 | 16 | **5** | 34/64 | 14 |
| 100 | **21** | **5** | 33/64 | 15 |
| 200 | 11 | **5** | 32/64 | 13 |
| 300 | 7 | 2 | 36/64 | 14 |
| 400 | 13 | **5** | 38/64 | 13 |

Totals 7-21 (mean ~14) against the best `lagged` config's 4-5, and falls collapse from ~46/64 to a
consistent **13-15/64** with ~50 episodes surviving to timeout. `collide_interval=2` on top gives
19 goals, max 4 -- the other knob that changes exchange *frequency* rather than cable behaviour,
also strongly positive, which is the same mechanism showing up twice.

    proxy_mode=staggered   cable_substeps=2      linear_damping=10
    angular_damping=20     vbd_iterations=80     bend_stiffness_scale=10.0

These are now the defaults, in **both** `CableCfg` and `cfg/task/Cable.yaml`. The YAML shadows the
dataclass, so editing `CableCfg` alone silently changes nothing: the first attempt did exactly that
and scored 0 goals where the sweep scored 16. Verified end to end with no overrides -- 13 goals,
max 4 per episode, lift 38/64, 13 falls.

Against the best `lagged` configuration's 4-5 total goals and max 2-3 per episode, this is 3x on
totals and the first episode to score 5 -- far outside the run-to-run spread that made every other
comparison here unrankable.

### What the two modes actually do

From Newton's own source (`newton/_src/solvers/coupled/solver_coupled_proxy.py`, `Proxy.mode`):

> `"lagged"` syncs source **begin** poses and end velocities, then prepares proxies to avoid
> double-counting lagged feedback. `"staggered"` syncs source **end** poses and end velocities
> directly.

Under `lagged` the cable sees the hand's pose from the *start* of the step paired with its *end*
velocities -- a mismatched pair whose error scales with how fast the hand is moving. That predicts
every symptom measured here: worst during **transport** (fastest hand motion), worse with **more
segments** (more mismatched proxy contacts), unaffected by **VBD iterations** and harmful past 80
(converging harder against stale boundary data), and improved by **fewer substeps** (each substep
re-applies the stale pose).

There is **no public discussion of these modes** -- no issue, no doc beyond a one-line
"Proxy transfer mode passed to Newton's coupled-proxy solver", and `lagged` is the default with
nothing flagging it as a poor fit for fast manipulation.

Untested and indicated by the same docstring: **`proxy_relaxation`** with `proxy_relaxation_mode`
`"aitken"`, which adapts the relaxation factor from consecutive feedback residuals within a step.
That is the textbook remedy for residual partitioned-coupling error, and is the obvious next lever.

### Velocity: the gap halved, not closed

4 envs, identical stack, mode the only difference:

| config | p90 | vs rod | p99 | max |
|---|---:|---:|---:|---:|
| rigid rod (reference) | 0.288 | 1.0x | 0.83 | 1.31 |
| **staggered (defaults)** | 0.548 | **1.9x** | 1.40 | 2.84 |
| lagged (same stack) | 0.843 | 2.9x | 2.48 | 3.50 |
| stock | 1.118 | 3.9x | 2.48 | 3.22 |

`staggered` cuts p90 by 35% and p99 by 43% with everything else fixed. Note the *shape*: the median
rises slightly (0.223 -> 0.251) while the upper tail falls sharply -- removing spurious impulses,
not damping motion. Angular damping did the opposite (lower median, worse tail), which is why it
never helped.

**The cable still moves ~1.9x the rigid baseline at p90.** The coupling is much healthier, not
settled.

**The change is to how the two solvers exchange state, not to the cable.** That is exactly what the
stability work predicted: VBD iterations did nothing (and hurt past 80), substeps had an optimum at
*fewer* than default, and coupler iterations made things worse -- three independent signs that
energy entered through the MJWarp<->VBD exchange rather than the VBD solve. `staggered` reorders
that exchange.

**This was dismissed early on.** `proxy_mode=staggered` measured 36.3 m/s in the very first
stability sweep and was written off. That run was on the broken proxy contact pipeline
(silently dropping contacts) with an unstable base config, so the number meant nothing. It only
became testable after the pipeline was fixed and the base stabilised -- a reminder that a negative
result taken on a broken scene is not a negative result.

Related: `collide_interval=2` also helps (9 goals, max 4, at seed 0), and it is the other knob that
changes exchange frequency rather than cable behaviour.

Staggered + 24 segments lifts **53/64 = 83%**, the highest recorded, though it scored 0 in that
draw -- worth repeats given the size of the grip advantage.

## Where the cable ends up (lagged coupling): 2-5 goals per 64 envs, max 2-3 per episode

Measured on a clean sim -- flat table, no clearance, overflow guard active, contacts intact.
Best configuration found (`p_bd05`):

    cable_substeps=2  linear_damping=10  angular_damping=20
    vbd_iterations=80  bend_stiffness_scale=10.0   (bend x0.5)

| config | seeds | total goals | max/episode | lift |
|---|---|---|---|---|
| bend x0.5 | 0, 400, 500 | 4, 5, 4 | 3, 2, 2 | 0.27-0.36 |
| bend x0.1 | 0, 100, 300 | 5, 2, 3 | 3, 1, 2 | 0.14-0.25 |
| 24 segments | 3 configs | **0, 0, 0** | 0 | **0.56-0.59** |

**Run-to-run spread exceeds the difference between most configurations.** The same config drew 5,
2 and 3 total goals across seeds, so no single 64-env run can rank two parameter settings -- which
invalidates every one-run comparison made earlier in this document, including several reported as
leads. Bend x0.5 beating bend x0.1 is the one config-level claim here backed by repeats.

**The 24-segment result is the sharpest open question.** Three independent configurations lift
~57% -- more than double anything else -- and score **exactly zero**. A finer cable conforms to the
fingers far better, gets picked up, and never arrives. Whatever breaks transport gets *worse* with
segment count, which matches the proxy-exchange diagnosis: more segments means more joints, more
proxy contacts, and more exchange traffic per step.

So the two axes trade against each other and neither reaches 5 in one episode: 12 segments scores
2-5 totals at ~20% lift, 24 segments lifts ~57% and scores nothing. Totals rise because more envs
score once or twice, never because one episode scores repeatedly.

## The instability, and the two knobs that actually fix it

39 configurations screened at 4 envs x 300 steps (`scripts/cluster/sbatch_velgrid.sh`), against a
**matched 4-env rigid rod**: mean 0.161, p50 0.112, p90 0.359, p99 0.898, max 1.31, lifting 4/4.

| config | mean | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| **rigid rod (reference)** | 0.161 | 0.112 | **0.359** | 0.898 | **1.31** |
| **`cable_substeps=2` + `linear_damping=10`** | 0.290 | 0.226 | **0.620** | 1.217 | **1.84** |
| bend x0.1 + `linear_damping=10` | 0.333 | 0.254 | 0.686 | 1.707 | 2.93 |
| all three + 40 mm | 0.306 | 0.197 | 0.701 | 2.023 | 2.83 |
| stock (damping 1, 20 iters) | 0.437 | 0.269 | 1.118 | 2.484 | 3.22 |
| `stretch_stiffness_scale` x0.01 | 5.777 | 3.171 | 15.873 | 32.782 | **94.89** |

**Substeps 2 with linear damping 10 reaches 1.7x the rigid p90 and 1.4x its max**, against stock's
3.1x and 2.5x -- the first cable configuration in the same regime as a rigid object the policy
succeeds with.

### What the grid rules out, and what that implies

* **Angular damping is not the lever.** 1 -> 200 at fixed iterations moves p50 by a third and leaves
  p90 flat, while making the tail worse (damping 50 gives the *best* median at 0.189 and a p99 of
  15.2). Judging by median alone picks it; judging by max alone misses that its ordinary motion is
  calm. Both are why percentiles are reported here.
* **Solver iterations do nothing at all.** 20 / 40 / 80 / 160 -> p50 0.279 / 0.347 / 0.263 / 0.284.
  The cable is *not* under-resolved, so throwing solver budget at it cannot help.
* **Substeps have a sharp optimum at 2**, degrading upward (4 -> 8 -> 16: p50 0.269 -> 0.358 ->
  0.912) and failing catastrophically at 1 (max 37 m/s). Fewer, larger substeps mean fewer proxy
  exchanges per step.
* **Coupler iterations make it worse** (max 17.3 at 2), reproducing the very first sweep's finding.

Flat in iterations, optimal at *fewer* substeps, worse with more coupler iterations: the energy
enters through the **proxy exchange between the MJWarp and VBD entries**, not through an
under-converged VBD solve. That is why a year of solver-budget knobs did nothing.

* **Softening stretch stiffness is catastrophic** (x0.01 -> 94.9 m/s peak) while **softening bend
  helps** (x0.1 is among the best). Opposite directions: a stiff bend response rings at this
  timestep, and stretch stiffness is what stops the chain oscillating along its length. Every
  earlier stiffness sweep moved both the same way, and harder.
* Segment count optimum is the default 12 (6 -> 0.524, 12 -> 0.269, 24 -> 0.498, 48 -> 1.13).
* Density and thickness do nothing for stability.

### Method note

Screening wants **many configs at few envs**; goal counts want the opposite. 39 configurations at
4 envs took less wall-clock than six at 64. But max speed scales with env count -- stock measured
3.22 at 4 envs and 8.61 at 8 -- so cross-env-count comparisons are invalid, which is why the rigid
reference was re-run at 4 envs before any of the above was concluded.

## Throughput: linear to 2048 envs, then memory-bound

`isaacsimenvs/eval/throughput.py`, one job per env count under `scripts/cluster/throughput/`.
RTX 6000 Ada, sim only (zero actions), warm-up excluded, CUDA synchronised, **overflow asserted**.

| envs | ms/step | env-steps/s | GPU GB | CPU GB |
|---:|---:|---:|---:|---:|
| 1 | 402 | 2.5 | 2.56 | 2.15 |
| 4 | 401 | 10.0 | 2.56 | 2.11 |
| 16 | 401 | 39.9 | 2.59 | 2.13 |
| 64 | 408 | 157 | - | - |
| 512 | 455 | 1,125 | 11.10 | 2.78 |
| **2048** | **408** | **5,019** | 37.98 | 5.93 |
| 4096 | - | **OOM** | >49 | - |

**Per-step cost is constant from 1 to 2048 envs**, so throughput is exactly linear
(2.45 policy-steps/s x n) and the GPU never saturates on compute. The ceiling is **memory**:
~2.5 GB fixed plus ~17 MB/env, which predicts the observed OOM at 4096 and a hard limit near
2,700 envs (~6,600 env-steps/s) on a 49 GB card.

**Overflow is fatal in this benchmark, deliberately.** Dropped contacts are work the solver skips,
so an overflowing scene reports *higher* throughput while being physically wrong. Zero overflows
were recorded at any env count.

### Two measurement traps hit here

* **`torch.cuda.max_memory_allocated` reported 0.01 GB** for a run using 38 GB -- Newton and Warp
  allocate outside PyTorch's allocator. The GPU figure now comes from `nvidia-smi` for this PID.
* **Mixed cluster hardware looked like an env-count effect.** An unpinned sweep put a third of its
  runs on an RTX A6000 (~630 ms/step) and the rest on RTX 6000 Ada (~402-455 ms), producing an
  apparent "n=4 is slow" and "1024 is slower than 2048". Grouping by hostname showed each node
  constant to ~2% across all env counts. The sbatch files now pin
  `--gres=gpu:nvidia_rtx_6000_ada_generation:1`.

## Read the metric carefully: episodes are not fixed-length

`termination_utils.py:51` zeroes `episode_length_buf` on every goal, so the 600-step deadline
restarts each time a goal is scored. Measured episode lengths make this unmistakable:

    0 goals -> length maxes at exactly 600
    1 goal  -> 748, 785, 867, 948, 1158
    2 goals -> 799, 942
    7 goals -> 1826

**Scoring buys more time, so episode length is a function of the goal count.** "goals/episode" is
therefore a poor label: the quantity is *consecutive successes before a 600-step dry spell*, not a
rate over a fixed window, and a per-episode count is inflated by the very thing it measures.

Normalising by simulated steps changes one conclusion and not the other:

| config | goals | max/ep | steps | **goals / 1k steps** |
|---|---:|---:|---:|---:|
| pca + staggered (s400) | 16 | 6 | 33,450 | **0.478** |
| staggered + span (s0) | 16 | 5 | 34,289 | 0.467 |
| pca + staggered (s0) | 16 | 7 | 35,117 | 0.456 |
| lagged, best stack | 5 | 3 | 17,793 | 0.281 |
| lagged, plain | 2 | 1 | 27,055 | 0.074 |

* **`staggered` over `lagged` survives normalisation** -- 0.47 vs 0.074, a 6x rate difference. That
  result is real.
* **`pca` over `span` does not.** 0.456-0.478 against 0.467 is indistinguishable. PCA episodes
  reach higher per-episode counts largely *because they run longer*. The "ceiling 5 -> 7" framing
  recorded above overstates it: the per-episode maximum does improve, but scoring *rate* does not.

Comparisons against the rigid rod (9.05) remain valid, since it was measured under the identical
protocol.

## The reference: what a *working* manipulation looks like

Peak speed was used as a stability metric here for a long time against a badly chosen anchor --
4.81 m/s, which was itself a *cable* measurement. The right reference is a rigid manipuland that the
policy actually succeeds with. `isaacsimenvs/eval/velocity_stats` measures the distribution, 16 envs
x 600 steps:

| object | mean | p50 | p90 | p99 | **max** | goals |
|---|---:|---:|---:|---:|---:|---:|
| rigid rod, coupled cable scene | 0.124 | 0.082 | 0.276 | 0.654 | **1.68** | 75 |
| rigid tool, `PlayNewton` | 0.126 | 0.080 | 0.289 | 0.654 | **1.31** | 65 |

**A successful manipulation peaks near 1.3-1.7 m/s with p99 at 0.65.** Against that:

| cable config | peak |
|---|---:|
| damping 20 + 120 iters (calmest found) | 3.92 |
| damping 5 + 60 iters | 5.40 |
| stock (damping 1, 20 iters) | 8.61 |
| damping 5 + 60 iters + 8 substeps | 15.5 |
| `rigid_contact_k_start=1e4` | 39.9 |

**Every cable configuration measured is violently unstable by this standard** -- the calmest is still
2.3x the rigid baseline's *maximum* and ~6x its p99. This reframes the whole sweep below: those runs
were not measuring how well a policy grasps a cable, they were measuring a solver that never
integrates the cable calmly, so most of the negative results say little about deformability.

Percentiles matter, not just the max: a successful rollout legitimately contains fast motion
(carry, score, new goal sampled far away, accelerate). It is the *whole distribution* sitting an
order of magnitude high that identifies a solver problem.

## Best result: 6 goals on a cable (64 envs, supports + AVBD contact ramp)

`docs/results/n64_k3.json` -- 64 envs, 60 mm support gap, `rigid_contact_k_start=1.0e3`:
**6 goals across 5 distinct envs**, one episode scoring 2, lift 44/64, zero ejected, zero censored.

Two things got there, and it is worth keeping them separate.

**Clearance made goals possible at all** (see below): 0 goals in 22 unsupported configurations,
goals in every supported one.

**The AVBD contact ramp is the only physics parameter that separated from noise.** Same envs, same
supports; the ramp is the only difference:

| run | `rigid_contact_k_start` | goals/ep | goals |
|---|---:|---:|---:|
| `n64_plain` | 1.0e2 (default) | 0.0312 | 2 / 64 |
| `n64_k3` | 1.0e3 | **0.0938** | **6 / 64** |
| `b_k1e4` | 1.0e4 | **0.1250** | 4 / 32 |

A 3-4x rate increase over default, reproduced at two settings and two env counts. It matters
because it is a *different contact path* from `soft_contact_ke`: the cable is a chain of rigid
bodies (`mesh_edge_body_N`), so hand-vs-cable is a body-body contact governed by the VBD solver's
augmented-Lagrangian constraints, not by the proxy's soft contact. Sweeping `soft_contact_ke` over
three decades did nothing; one decade of the right parameter tripled the rate.

Found from a Newton release note, not from the config: `VBDSolverCfg` exposes
`rigid_contact_k_start` but `CableCfg` never set it.

**What did not separate from noise**, all at 32 envs with supports (goal counts 0-3, mutually
indistinguishable): support gap 20/40/60/80 mm, flat-topped vs round supports, wide vs narrow
stance, friction x4, density x10, bend stiffness x100. Lift sits at 19-22/32 across all of them.

**The open gap.** Lift is ~68% but goals/episode is ~0.09, and the rigid rod scores 9.05. Envs pick
the cable up and lose it before reaching the goal pose, so *transport*, not grasp acquisition, is
what remains. `max_steps=6000` does not help, because episodes end in `fall` (~70%) rather than
being truncated.

## Clearance is the answer: the first goals scored on a cable

**A cable lying flush on a table gives the fingers nowhere to close.** That is the whole thing.
`CableCfg.supports` rests it on two low rails with the middle span open, and the same policy --
unchanged, same checkpoint -- starts picking it up.

| config | goals/ep | goals | lift |
|---|---:|---:|---:|
| supports, 40 mm gap | **0.0938 +/- 0.0515** | 3 | 19/32 |
| supports + `soft_contact_ke` x10 | 0.0312 | 1 | 19/32 |
| supports + bend x100 | 0.0 | 0 | 17/32 |
| supports, 60 mm gap | 0.0312 | 1 | **22/32** |
| supports + contact + bend | 0.0625 | 2 | 16/32 |
| *(22 configs without supports)* | **0.0** | **0** | 0-5/32 |

**7 goals across the supported configs against 0 in every unsupported one**, and lift 16-22/32
against 0-5/32. Do not rank the supported configs by goal count -- 0 to 3 events is Poisson noise.
Lift is the robust statistic, and the dose-response is the evidence: 40 mm -> 19/32, 60 mm -> 22/32.

Verified rather than assumed, on the 40 mm run: three *distinct* envs scored (5, 15, 27), not one
env repeating; **zero ejected**, guarded and unguarded lift identical; and peak object height
0.889 m, i.e. 26 cm above the 0.63 m table -- clear of the 0.765 m on-end pivot that faked lifts
earlier in this document.

`videos/cable_supports.mp4` shows the rig, `videos/cable_supports_grasp.mp4` shows the hand lifting
and carrying the cable. Neither shows a scored goal: re-running does not reproduce a specific
episode under GPU nondeterminism, so the goals live in the measurement, not the footage.

### Why every other knob failed

The 22-configuration table below is worth keeping precisely because it is negative. Stiffness,
density, contact stiffness, friction, solver iterations and substeps all govern **how the cable
behaves**. None of them changes **the geometry of the grasp**, and that was the binding constraint
the whole time. The monotonic stiffening trend -- falls 29 -> 13, timeouts 0 -> 18, lift -> 0 -- was
the clearest signal that the object was not the problem: an infinitely stiff cable the hand still
cannot pick up is a statement about where the fingers can go, not about compliance.

### The full sweep: 22 configurations, zero goals

All at 32 envs, whole-hand proxies, damping 1.0 + 20 VBD iterations unless stated. Rigid-rod
control on the same scene: **9.05 goals, 28/32 lift**.

| config | falls | timeouts | lift | goals |
|---|---:|---:|---:|---:|
| `proxy_links=tips` | 30 | 0 | 0/32 | 0 |
| `proxy_links=hand` | 29 | 0 | 3/32 | 0 |
| `vbd_iterations=30`, `substeps=8` | 32 | 0 | 2/32 | 0 |
| bend x100 | 27 | 1 | 2/32 | 0 |
| bend x100 + stretch x10 | 22 | 7 | 0/32 | 0 |
| 2 seg + bend x100 | 25 | 0 | 5/32 | 0 |
| bend x1000 + stretch x100 | 13 | 18 | 0/32 | 0 |
| density x10 | 27 | 1 | 2/32 | 0 |
| density x20 | 26 | 5 | 0/32 | 0 |
| density x20 + bend x100 | 23 | 4 | 0/32 | 0 |
| `soft_contact_ke` x10 | 28 | 3 | 5/32 | 0 |
| `soft_contact_ke` x100 | 32 | 0 | 3/32 | 0 |
| `soft_contact_ke` x1000 | 29 | 0 | 2/32 | 0 |
| `ke` x100 + friction x4 | 31 | 0 | 1/32 | 0 |
| `ke` x100 + density x20 | 26 | 4 | 4/32 | 0 |
| `ke` x100 + bend x100 | 29 | 1 | 5/32 | 0 |
| everything combined | 31 | 1 | 1/32 | 0 |

**Best cable lift anywhere is 5/32 against the rod's 28/32, and no configuration scores a single
goal.**

Three conclusions the table supports:

*Stiffening does not converge to the rigid rod.* Falls drop monotonically (29 -> 27 -> 22 -> 13) and
timeouts rise (0 -> 1 -> 7 -> 18) as stiffness increases, while lift goes to **zero**. The limit of
"make the cable rigid" is an inert object the hand cannot move -- not a graspable one. That is a
statement about the interface: a rigid rod contacts the hand through MJWarp's rigid solver, while a
cable of any stiffness reaches it only through the proxy's *soft* contact.

*Mass is not the constraint.* 100 -> 1000 -> 2000 kg/m^3 gives 3/32 -> 2/32 -> 0/32.

*Contact stiffness peaks and then declines.* x10 gave 5/32 and the first non-zero timeouts, which
looked like the start of a trend; x100 and x1000 lost the timeouts entirely and dropped lift to
3/32 and 2/32. It was the top of a curve read from one draw -- the same single-draw error as the
damping sweep earlier in this document.

*The levers do not stack.* Everything-combined scores 1/32, worse than several of its parts.

### What moved, and what did not

All at 32 envs, damping 1.0 + 20 VBD iterations, whole-hand proxies:

| config | goals | lift | timeouts |
|---|---:|---:|---:|
| `proxy_links=tips` | 0.0 | 0/32 (1 ejected) | 0 |
| `proxy_links=hand` | 0.0 | 3/32 | 0 |
| `vbd_iterations=30`, `substeps=8` | 0.0 | 2/32 | 0 |
| **`soft_contact_ke` x10** | 0.0 | **5/32** | **3** |

Whole-hand proxying did *not* measurably help: 3/32 is inside the 1-4/32 range fingertip-only runs
produce on repeats. It is kept because it is physically right, not because it scored.

Raw solver fidelity did nothing at all. The only change that altered the *failure mode* was contact
stiffness: three episodes surviving the full 3000 steps, where every other run had zero. A timeout
means the cable stayed on the table with the hand engaged. `soft_contact_ke = 8.0e3` against a 30 mm
cable may simply be too soft to transmit a grasp -- fingers sink instead of gripping -- which would
present as "the policy cannot grasp cables" while being a contact-model artifact.

## Rendering the goal

The goal was invisible in every render. It was first worked around with an immediate-mode overlay,
which was the wrong instinct: **the marker is already in the scene.** `_colorize` finds and colours
its shapes (`goal: 8`), and they import with `shape_flags == 0` -- neither VISIBLE nor
COLLIDE_SHAPES. The viewer draws collide-flagged shapes (`show_collision=True`) and visible ones, so
a shape with neither is silently skipped. Robot shapes are `flags=6`: collide set, visible clear.

`_reveal_goal_viz` sets VISIBLE on those shapes and nothing else. Collision is already off, which is
exactly what a goal marker wants -- it must not push the manipuland around -- and the task already
writes the marker's pose every step, so it tracks the goal in every world with no re-derivation in
the render path. What is on screen therefore cannot drift from what the env believes the goal is,
which a stand-in overlay cannot guarantee. This is the default; `--no_goal` turns it off.

`goal_viz` spawns from the same procedural handle+head USD as the manipuland, so the cable env
initially drew a *hammer-shaped* ghost for a goal a cable has to reach. `CableEnv._reshape_goal_marker`
replaces that geometry with a capsule of `cable.length` x `cable.thickness`, authored before Newton
imports it, with `collisionEnabled` false -- the marker must never touch the manipuland, and
revealing it needs only the VISIBLE flag, not collision.

Two things to know if this is ever touched again. The rigid body sits on a **child**
(`.../GoalViz/object_root`), not on the `GoalViz` root, so deactivating the root's children removes
the body the physics view resolves and asset construction fails with `Expected 1 prims ... found 0`;
the marker has to be rebuilt *under* that body. And the capsule runs along object-frame **+X**, the
same axis `object_scale` stretches for the cable.

Scoring is unaffected, as expected -- success keypoints come from `object_scale`, not the marker
mesh.

`--goal_keypoints` additionally draws what the marker cannot: success is the *max* over per-keypoint
distances held for `success_steps`, so the quantity actually thresholded is those segment lengths.
Green goal keypoints and wireframe quad, amber object keypoints, lines for the error.

Validated on `Isaacsimenvs-PlayNewton-Direct-v0`, where the policy scores: the green marker lands on
the held tool and the keypoint error lines collapse to nearly nothing. On the cable the marker hangs
in the air off the table while the cable never leaves the surface -- the goal requires lift and
carry, and the grasp that would start it never happens.

Gotcha for the overlay path: `log_lines`/`log_points` colours must be per-element `wp.array`s, not a
single tuple -- the GL backend calls `.numpy()` on whatever it gets and a tuple raises in
`_update_vbo`.

## The orphaned rigid tool was live the whole time

Spotted in a render, twice: a hammer lying on the floor beside the table in every cable video.
`_neutralise_rigid_object` exists precisely to prevent that -- its docstring says a stray rigid body
"can collide with the cable and with the hand, which would quietly corrupt every measurement taken
here" -- and it was printing **`0 colliders off`**. The loop walked `[prim, *prim.GetChildren()]`,
one level, while the colliders sit two levels down at `/Object/<mesh>/collisions`. The kinematic pin
does not survive the Newton import either. So the tool stayed fully dynamic and collidable through
every cable measurement recorded above.

Fixed by walking the subtree with `Usd.PrimRange` (now `1 prims, 2 colliders off, 1 bodies pinned`),
and by **raising** if the tool is ever found with no collider disabled -- the failure mode was a
guard that silently did nothing, so it should be loud rather than absent.

### Does it change the results? No.

| config | tool live | tool inert |
|---|---|---|
| 30 mm, 12 seg | 0.0 goals, lift 3/32, fall 30 | 0.0 goals, lift 1/32 (1 ejected), fall 31 |
| 30 mm, 2 seg | 0.0 goals, lift 4/32, fall 32 | 0.0 goals, lift 1/32, fall 30 |

**Goals stay at exactly 0.0 and terminations stay at ~32 `fall` in every condition**, so the primary
conclusion is unaffected. Lift fell toward 1/32 in both configs. Each move alone is inside noise at
n=32, but the same direction in two independent configs is weakly suggestive that the falling tool
was helping tip the cable on end -- consistent with the pivot reading above. Treat as a mechanism
worth remembering, not an established effect.

The lesson worth keeping: the bug was visible in every video for as long as videos have existed, and
was read past because attention was on the cable. `0 colliders off` was printed on every run too.

## Also open: the orphaned rigid tool

`setup_scene` still spawns the procedural handle+head tool. Once `env.object` points at the cable,
nothing resets or reads that prim, so it falls once and lies on the floor — plainly visible in the
render, and completely invisible in the observation. `_neutralise_rigid_object` disables its
colliders (which works) and tries to pin it kinematic (which does **not** take). Suppressing the
spawn is probably the right fix, but the asset pipeline, the USD cache and the `object_scales`
observation are all built around that object, so it is a larger change than it looks.

### Staged bring-up: the cable and its contacts are fine; the coupled solve is not

`isaacsimenvs/eval/cable_stages.py` builds the scene up one element at a time, using the task's own
`CableCfg` so a difference between a stage and the env is the *scene* rather than the cable.

| stage | scene | at rest after 120 steps | step-1 speed |
|---|---|---|---|
| 1 | cable + ground plane | z = 0.0050 (= its radius), **max‖v‖ = 0.000** | 0.082 (= g·dt) |
| 2 | + the task's table | z = 0.5350 (top 0.53 + radius), **max‖v‖ = 0.000** | 0.082 |
| 3 | full env: robot + coupler | — | **4.46** |

Both clean stages were run at the *env's own* solver settings (`substeps=2`, `iterations=10`), not
just the demo's, and behave identically either way. Step 1 shows exactly `−9.81 · dt`, and the
cable settles to a dead stop.

**So the following are exonerated**: the cable geometry and moduli, the VBD solver settings, the
timestep, gravity, and cable–table contact. What remains is the coupled solve — the MJWarp entry,
the robot, or the proxy mapping.

This also settles the earlier `proxy_table` confusion. In stage 2 the table is a plain *static*
collider and the cable rests on it perfectly. The task's table is spawned `kinematic_enabled=True`,
a kinematic *rigid body*, which is why the env needs it proxied at all — the reasoning was right,
but it was never the thing breaking.

## The cause: VBD's angular rest reference

`newton-physics/newton#3847` (open) states it directly:

> Newton currently uses the cable body poses used to initialize `Model.body_q` as **VBD's angular
> rest reference**.

So a segment orientation written after construction is interpreted as a bend/twist *deviation from
the spawn pose*, not as an absolute placement. The adapter's reset wrote the task's quaternion
straight through:

```
spawn / VBD rest reference : (0, 0.7071, 0, 0.7071)   90 deg about Y, aligned along the cable
what the reset wrote       : (0, 0,      1, 0     )   180 deg about Z
max |quat difference|      : 1.0000
```

Twelve segments each carrying a half-turn of twist against a 1.02e9 bend modulus is an enormous
restoring torque, released on the first step after every reset.

**The fix** is one line of intent in `cable_adapter.write_root_pose_to_sim`: compose the requested
rotation with each segment's cached spawn orientation (`_quat_mul(requested, rest)`) instead of
replacing it. Related upstream work: `#3856` adds explicit rest/initial separation, and its
changelog note — "remove post-construction state mutation" — describes exactly what we were doing.

### Every measured property fits, in hindsight

| observation | explanation |
|---|---|
| position-independent (identical at z = 0.63 and 1.08) | an orientation error carries no position dependence |
| scales linearly with substep count | a restoring torque re-applied every substep |
| bends but never stretches (span pinned at 0.275) | orientations wrong, positions right |
| stages 1-3 all clean | **none of them write segment poses** — the bug needs a reset to fire |

That last row is why the staged bring-up was decisive rather than merely reassuring: it isolated
the fault to the one thing the stages did not exercise.

### After the fix

| | before | after |
|---|---:|---:|
| vx at step 1 | −2.13 m/s | **0.00** |
| z spread at step 8 | 0.227 | **0.0000** |
| vz at step 8 | erratic | −1.311 = −8 · g · dt |
| settles at | never (30-step reset cycle) | **z = 0.5351, stable to step 200** |
| terminations | `fall` every 30 steps | **none** |

### Superseded: split the coupler from the robot

The remaining suspects are "the coupler mechanism" and "the robot inside it". One more stage
separates them: **cable + table + coupler, with no robot in the scene**, proxying only the table.
If that explodes, the coupler is at fault independent of the robot; if it is clean, the robot
entry or the fingertip proxies are.

Note an empty `proxies` list is *not* a valid control — it silently freezes the whole simulation,
so the mapping has to contain something.

Two properties of the impulse should constrain whatever is found: it is **position-independent**
(identical with the cable at z = 0.63 and z = 1.08, clear of both hand and table, so it is not
contact) and it **scales with substep count**.

### Superseded plan

Every test above varies one knob inside the full coupled scene, and all of them came back
identical — which is itself informative: the impulse does not depend on anything about the
*contact* configuration. The remaining difference between this and Isaac Lab's working cable demo
is the coupler itself.

So build up instead of narrowing down, which is how the reference investigation got traction
(`tools/cable/stage1_cable_only.py` → `stage2_cable_table.py` → `stage3_coupled.py`):

1. **Cable alone**, no robot, no coupler, on the demo's settings (`num_substeps=8`, `dt=0.01`,
   `iterations=20`). Confirm it hangs and falls sanely. If it explodes here, the parameters are
   wrong and nothing else matters.
2. **Cable + table**, still no coupler.
3. **Add the coupler with the robot**, and see at which step the impulse appears.

Between 1 and 3 lies the answer, and each stage is a few minutes. Continuing to A/B knobs inside
the full scene has now produced seven consecutive null results.

Also outstanding, independent of this: finish neutralising the orphaned rigid tool. And do not run
the policy until the cable settles — a goals/episode number measured on a cable that ejects itself
would be meaningless, and worse, plausible.
