# SIM1 physics adoption: what the fold criterion was actually measuring

Covers the work on branch `2026-08-25_update_cloth_physics_params` after commit 8f3833a. Three
threads, each of which turned out to be measuring something other than what it claimed: the two
contact offsets, the fold-target geometry, and the eval harness's own success statistics.

Every number below is measured. Where a claim is inference, it says so.

---

## 1. Two contact offsets, not one "thickness"

Newton has **no cloth thickness parameter**. `Model` has no `thickness` field, and the VBD paper
(Chen et al. 2024, arXiv 2403.06321) contains zero occurrences of "thickness", "offset" or a
collision radius -- its entire contact model is `E_c = ½k_c d²` with `d = max(0, (x_b - x_a)·n̂)`,
i.e. force begins at literal interpenetration. A VBD cloth is a zero-thickness surface by
construction.

What Newton adds on top are **two independent rest offsets**, which never meet:

| | parameter | scope | kernels |
|---|---|---|---|
| cloth <-> rigid | `particle_radius` | per particle, on `Model` | `accumulate_particle_body_contact_force_and_hessian`, `..._no_self_contact` |
| cloth <-> cloth | `self_contact_radius` | one scalar, on the solver | `evaluate_self_contact_force_norm`, `evaluate_vertex_triangle_collision_force_hessian` |

`particle_radius` appears in exactly three kernels, all body-contact. Self-collision takes
`collision_radius` and never sees it.

**Measured (probe, 4 envs):** the sheet rests **8.0 mm** above the table (= `particle_radius`,
exactly -- gravity compresses the penalty contact by 2.7 um, unmeasurable) and folded plies rest
**2.5 mm** apart (= `self_contact_radius`). Each offset tracks its own parameter and nothing else.

They are the PhysX rest/contact-offset construction: the body-particle kernel computes
`penetration = -(dot(n, p - cp) - radius - margin)`, summing radius and the shape's margin exactly
as PhysX sums restOffset and contactOffset. PhysX's own defaults (`PxShape.h`) are
`contactOffset = 0.02f * PxTolerancesScale::length`, `restOffset = 0.0f`.

### The 4:1 asymmetry is a convention, not a bug

An earlier draft of this file called 8 mm / 2 mm an incoherent material description -- "16 mm thick
to the table, 4 mm to itself". Geometrically that is true (a slab of thickness t implies
`particle_radius = t/2` and `self_contact_radius = t`, a 1:2 ratio, and ours is 4:1 the other way).
But it is also what **Newton's own reference example ships**:

    example_cloth_franka.py (scene is in CENTIMETRES: gravity -981, "scale=100, URDF is in meters")
      cloth_particle_radius        0.8 cm = 8 mm
      cloth_body_contact_margin    0.8 cm = 8 mm
      particle_self_contact_radius 0.2 cm = 2 mm
      particle_self_contact_margin 0.2 cm = 2 mm
      self_contact_friction        0.25

SIM1 copied it verbatim, `self_contact_friction = 0.25` included. The two parameters serve
different masters and are pulled in opposite directions: radius wants to be **large** (grip,
anti-tunnelling against a fast robot), self-contact radius wants to be **small** (tight ply
stacking, no self-repulsion). They are not describing one material.

### Making them "coherent" upward does not work

Probe, `self_contact_radius` raised to 16 mm on the 16.7 mm grid: the sheet never settles. It falls
off the table during the drop, `footprint` reads 1.072 (stretched beyond rest size), and during
release `z_stat` climbs 0.51 -> 0.79 -> 1.15 -> **1.61 m and still rising**. Energy injection, not
drape. The self-contact band is nearly one full cell wide, and
`conservative_bound_relaxation = 0.42` scales the displacement bound by `min(collision_radius, ...)`,
so a large radius simultaneously widens the contact band and loosens the truncation that keeps VBD
stable. **`self_contact_radius` is bounded above by grid spacing.**

---

## 2. The fold target used the wrong offset

`_init_fold_targets` lifts each fold target by `2 * particle_radius` (16 mm) -- a cloth<->**rigid**
offset -- when ply separation is set by `self_contact_radius` (2 mm), a cloth<->**cloth** offset.
The physics delivers 2.5 mm; the criterion demanded 16 mm.

Probe, teleported-then-released fold, 4 envs, settled and stable for 180 steps:

| config | radius | scr | target lift | settled fold_err | table gap | folded |
|---|---|---|---|---|---|---|
| current | 8 mm | 2 mm | 16 mm | **0.0160** | 8.0 mm | 4/4 |
| thin | 1 mm | 2 mm | 2 mm | 0.0020 | 1.2 mm | 4/4 |
| **target fix** | **8 mm** | **2 mm** | **2 mm** | **0.0020** | **8.0 mm** | **4/4** |
| thick | 8 mm | 16 mm | 16 mm | unstable | -- | 0/4 |

**The target fix dominates.** Same fold accuracy as going thin, same grip as now, 8x better than
what is running, and it touches nothing the robot interacts with -- so the cable's tunnelling
result ("ungraspable at 10 mm, liftable at 30 mm") never comes into play. One line:
`lift = self_contact_radius` instead of `2 * particle_radius`.

The residual 2.0 mm floor is a separate defect: `corner_indices` uses `mid = resolution // 2`,
which for odd resolutions is the **hinge row**, not the first row past the crease as its own
comment claims. Two of the four tracked keypoints therefore sit on the fold axis and are asked to
rise one lift while the crease stays down (measured hinge lift: 0.2 mm). With keypoints entirely on
the moving half the corner set is rigid under a fold, Kabsch recovers exactly 180 deg instead of
162.3 deg, and a perfect fold scores 0.

Also good to know the fold itself is fine: it **holds** at `edge_ke = 8.0e-4`, footprint steady at
0.501 for 180 steps. That value was adopted from SIM1, never measured here; it is now measured.

---

## 3. `_check_thickness` guards a failure mode that cannot occur

The guard requires `2 * particle_radius <= grid spacing`, on the stated grounds that closer
particles "self-collide at rest". They cannot: self-collision never reads `particle_radius`. The
`start_height` docstring in the same file records that the crumpling this guard was written for was
really the spawn-height bug ("attributed to thickness, to grid resolution, and to cloth physics; it
was this").

Independent evidence: **SIM1's shipping mesh violates it by 40%.** Measured from
`short-shirt.usdc` (7,021 vertices / 13,767 triangles, median edge **11.40 mm**), their 16 mm
particle diameter is 1.40x the median edge. Their simulation works.

That mesh measurement also corrects a claim made repeatedly during this work: SIM1's garment does
**not** have ~mm edges. At 11.40 mm it is essentially our res-9 grid (12.5 mm) and finer than our
current res-7 (16.7 mm). Their `edge_ke` and `tri_ke` therefore transfer directly -- the
dimensional-transfer worry was unfounded. (`long-shirt.usdc` is authored at a different scale,
1.11 mm edges, and would need a different `CLOTH_SCALE`.)

Consequence: dropping 9x9 -> 7x7 to accommodate 8 mm particles was **unnecessary**, and it moved us
away from SIM1's resolution rather than toward it. Not verified by experiment yet -- the probe takes
`--resolution`, so `--resolution 9 --thickness 0.016` tests it directly.

---

## 4. The eval harness reports two statistics that cannot mean what they say

Both affect every cloth eval ever run with `sbatch_cloth_eval.sh`, baseline included.

**`folds (held)` is structurally 0.00%.** `_get_dones` returns `terminated | entered`, and `entered`
fires at `cfg.termination.success_steps`, which is 1. The episode ends on the first folded step, so
`_fold_hold` can never reach `HELD_FOLD_STEPS = 10`. Fixed by
`env.termination.success_steps=10`, now forwardable through the launcher.

**`best_fold_err` cannot observe a fold.** It is a correct per-step running minimum, but it is
sampled *after* `env.step()` returns, and `DirectRLEnv.step()` calls `_reset_idx()` internally. For
the env that just folded, the sheet is already flat again. This is why the `success_steps=1` run
reported `within tolerance 0/160` while 12 episodes terminated with reason `fold` -- a flat
contradiction that is entirely an artifact. Raising `success_steps` to 10 fixes this too
(`within tolerance` goes 0/160 -> 8/160 with no change to the policy).

---

## 5. Result: the policy has learned to fold, at 2.5%

8-GPU run 629607, snapshot `eval_snapshot_2259.pth` (~epoch 250), 5 seeds x 32 envs x 900 steps.

| | entered | held | best_fold_err | falls |
|---|---|---|---|---|
| pretrained, `success_steps=1` | 0/160 | 0/160 (artifact) | -- | -- |
| pretrained, `success_steps=10` | **0/160** | **0/160** | **0.1013 +- 0.0000** | 0 |
| trained, `success_steps=1` | 12/160 (7.50%) | 0/160 (artifact) | 0.0793 (artifact) | 6 |
| trained, `success_steps=10` | 4/160 (2.50%) | **4/160 (2.50%)** | 0.0826 +- 0.0028 | 8 |

* **Genuine learning.** The control produces `best_fold_err = 0.1013` with standard deviation
  **0.0000** across 160 episodes -- exactly the flat-sheet value. The pretrained policy does not
  move the cloth at all, ever. Against that, 4 held folds and 0.0826 is unambiguous.
* **2.5%, not 16%.** Two-thirds of the training-log successes are transient and do not survive a
  10-step hold. `entered == held` for the survivors, so the ones that count are genuinely settled.
* **It also learned to knock the sheet off the table**: 8 falls vs the control's 0.

Do NOT compare against `cloth_nohold_s*`. That is the old physics (9x9, 12 mm, no self-contact) and
is superseded. Comparing the trained policy's 0.0826 against its 0.0824 suggested "no improvement";
against the correct same-physics control it is a large one. The same-physics control is
`cloth_baseline_held10_s*`.

Note the pretrained policy scored 3/160 and 0.0824 under the *old* physics but 0/160 and dead-flat
under the new. The physics change made the prior policy stop engaging the cloth entirely, so the
finetune started from further back than the old baseline implied.

---

## Open, not done

1. `_init_fold_targets`: lift from `self_contact_radius`, not `2 * particle_radius`. Measured 8x.
2. `corner_indices`: `mid = resolution // 2 + 1` for odd resolutions. Removes the 2.0 mm floor.
3. `_check_thickness`: guards an impossible failure; blocks returning to 9x9.
4. Note (1)-(3) change the scoring criterion, so they invalidate comparison against runs 629081 /
   629607. They belong on a branch for the next run, not mid-flight.

Reproduce section 1-2 with `scripts/analysis/cloth_fold_probe.py`.
