# Press-test experiments (2026-11-18)

Goal: find what makes a rigid-plate cloth penetration-free against a pressing hand, ideally at realistic
density (nominal 0.1 kg/m^2), without a policy in the loop.

Tool: `isaacsimenvs/scripts/physics_test/press_test.py` -- training scene from a run config, robot colliders
off, a force-controlled probe sphere (r = 8 mm) lowered onto a site and pushed down (ramp 0.5 s, hold 2 s).
A site counts as **through** if the probe's lowest point gets deeper than the panel thickness below the
panel's top face; "held max" is the worst steady depth (median of the last 0.5 s) among sites that held.

Config: `debug_outputs/train_logs/nov/2026-11-17_cloth_curriculum_prior_20to2mm_742942/.hydra/config.yaml`.
Jobs via `experiments/physics_test.sh` (one GPU each); results under `experiments/nov18/press_test/<exp>/`.

## Standard case (unless stated)
single_hinge, 10 x 10 cm sheet (two 5 x 10 cm panels -- the smallest panels this design gets), 20 mm thick,
nominal material (bending 5e-5, viscosity 5e-5, density 0.1, frictions 0.4 / 0.8 / 0.8), probe 0.5 kg
(calibrated against policy rollouts: fails where the policy penetrated, holds where it did not),
forces 20 N and 50 N, sites = both panel centres + the hinge midpoint.

## What we knew before these experiments
- Probe-to-panel mass ratio decides it: on strip_4 (5 N, 50 g probe) density 0.5 held, 0.3 failed
  -> holds up to a ratio of ~10, fails from ~17 (matches PhysX guidance: avoid > 10 for a light body
  between two heavy ones).
- Of the UWLab settings only position iterations matter: 8 -> 36 made density 2.0 hold at 20 N on the
  standard case; friction-correlation / friction-offset / bounce changed nothing. At density 0.1 nothing
  held. (`uwlab_ablation/`)
- Policy runs: halving dt (1/240) reduced but did not remove penetration at density 0.1 (strip_4).

---

## E1 -- Does a finer physics step make nominal density hold?
**Question:** at density 0.1 the contact fails; does a smaller dt (more solver passes per second, less
motion per step) fix it?
**Setup:** standard case, density 0.1, 36 position iterations (`solver_position_iterations=36`,
`max_position_iteration_count=36`), dt = 1/120 (trained), 1/240, 1/480, 1/960 s (`env.sim.dt`).
Verified from the JSONs that `physics_dt` really changed (0.00833 -> 0.00104 s).
**Command:** `press_test --design single_hinge --size 0.1 0.1 --density 0.1 --forces 20 50 --probe_mass 0.5 <iters> env.sim.dt=<dt>`
**Results** (`e1_dt/`):

| run | 20 N | 50 N |
|---|---|---|
| dt 1/120 | 2/3 through, 1 slid | 2/3 through, 1 slid |
| dt 1/240 | 3/3 through | 2/3 through, 1 slid |
| dt 1/480 | 3/3 through | 2/3 through, 1 slid |
| dt 1/960 | 1/3 through, 2 slid (held 6.8 mm) | 2/3 through, 1 slid |

Both panel centres went through at every dt and force (depth 73-89 mm, i.e. the probe ended under the
panel); the hinge site mostly "slid" (probe left the panel outline after sinking 10-20 mm).
**Takeaway:** an 8x finer step does not help at all. The failure is not "too much motion per step"; it is
the per-step solve itself failing for this mass ratio (0.5 kg probe vs a 0.5 g panel = 1000:1), which a
smaller dt does not change. Drop dt as a lever.

## E2 -- Is it really the mass ratio? (probe mass sweep at fixed force)
**Question:** at the same force on the same 0.5 g panel, does the outcome depend on how heavy the pusher is?
**Setup:** standard case, density 0.1 (panel 0.5 g), 36 iterations, probe mass 0.5 g / 5 g / 50 g / 500 g
(probe:panel = 1:1, 10:1, 100:1, 1000:1), forces 20 and 50 N.
**Command:** `press_test ... --density 0.1 --forces 20 50 --probe_mass <m> <iters>`
**Results** (`e2_probe_mass/`; per site: THR = through, else max depth during the press):

| probe (ratio) | 20 N | 50 N |
|---|---|---|
| 0.5 g (1:1) | 0/3 through -- max depth 0.0 / 0.0 / 0.0 mm | 0/3 through -- max depth 4.5 / 0.6 / 1.3 mm |
| 5 g (10:1) | 1/3 through (hinge); centres sank 18-19 mm | 2/3 through (centres) |
| 50 g (100:1) | 3/3 through | 3/3 through |
| 500 g (1000:1) | 2/3 through, hinge slid after 20 mm | 2/3 through, hinge slid after 15 mm |

(The light probes are marked "slid" because the weak x/y spring cannot hold a 0.5-5 g sphere under
20-50 N; the max depths show they never sank in.)
**Takeaway:** confirmed -- with the same force on the same panel, a 1:1 pusher barely dents it (<= 4.5 mm),
10:1 sinks in and partly goes through, >= 100:1 always goes through. Force alone is not the problem; the
pusher-to-panel mass ratio is. The real hand (finger links + armature + arm behind them) is a heavy pusher,
so the fix has to raise the panel's effective mass in the solver (density, inertia/armature, or contact
mass scaling) or shrink the pusher's.

## E3 -- Does hinge armature (joint inertia without weight) help?
**Question:** Yibo's recipe sets `hinge_armature = 1e-3`. Does extra joint inertia let a light panel hold?
**Setup:** standard case, density 0.1, 36 iterations, `env.rigid_cloth.hinge_armature` = 0 / 1e-4 / 1e-3 /
1e-2 kg m^2 (applied through the training randomization path; `rigid_check` read it back -- no mismatch).
**Results** (`e3_armature/`):

| armature | 20 N | 50 N |
|---|---|---|
| 0 | 2/3 through (centres), hinge slid | 2/3 through (centres), hinge slid |
| 1e-4 | 3/3 through | 3/3 through |
| 1e-3 | 3/3 through | 3/3 through |
| 1e-2 | 3/3 through | 3/3 through |

Panel centres went through identically at every armature (73-74 mm, probe under the panel).
**Takeaway:** no help -- if anything the hinge site gets worse. Armature only adds inertia to the hinge's
rotation; pressing a panel into the table is a translation of the panel, which armature does not touch.
Not a lever for penetration (it may still matter for fold dynamics).

## E4 -- Does a thick contact shell (contact / rest offset) help?
**Question:** Yibo's recipe uses contact offset 12 mm and rest offset 4-10 mm, and the Yibo-values policy
eval penetrated far less. Is the shell the reason?
**Setup:** standard case, density 0.1, 36 iterations; cloth (rest, contact) offset = (0, 2 mm) trained,
(0, 12 mm), (3, 6 mm), (7, 12 mm) Yibo (`env.rigid_cloth.rest_offset`, `env.rigid_cloth.contact_offset`).
Depth is measured against the panel's geometric box, so a resting probe sits above it by the rest offset.
**Results** (`e4_shell/`):

| cloth offsets | 20 N | 50 N |
|---|---|---|
| rest 0 / contact 2 mm (trained) | 2/3 through, hinge slid | 2/3 through, hinge slid |
| rest 0 / contact 12 mm | 2/3 through, hinge slid | 2/3 through, hinge slid |
| rest 3 / contact 6 mm | 2/3 through, hinge slid | 3/3 through |
| rest 7 / contact 12 mm (Yibo) | 2/3 through, hinge slid | 3/3 through |

Panel centres went through in every case (72-78 mm).
**Takeaway:** the shell does nothing for a light panel -- it only moves where the contact starts. So the
much lower penetration of the Yibo-values policy eval came from its density (2.0), not its shell. Not a
lever on its own.

## E5 -- Density ladder: how heavy must the panel be?
**Question:** with 36 iterations, at what density does the standard case stop going through?
**Setup:** standard case, 36 iterations, density 0.1 / 0.2 / 0.5 / 1 / 2 / 5 / 10 kg/m^2
(each 5 x 10 cm panel weighs density x 5 g).
**Results** (`e5_density/`; "held" = steady depth at the panel centres):

| density (panel mass, probe:panel) | 20 N | 50 N |
|---|---|---|
| 0.1 (0.5 g, 1000:1) | 2/3 through | 2/3 through |
| 0.2 (1 g, 500:1) | 3/3 through | 3/3 through |
| 0.5 (2.5 g, 200:1) | 3/3 through | 2/3 through |
| **1.0 (5 g, 100:1)** | **0/3 through, held 0.3 mm** | 2/3 through |
| 2.0 (10 g, 50:1) | 0/3 through, held 0.0 mm | 2/3 through (centre+, hinge) |
| **5.0 (25 g, 20:1)** | 0/3 through, held 0.0 mm | **0/3 through, held 0.2 mm** |
| 10.0 (50 g, 10:1) | 0/3 through, held 0.0 mm | 0/3 through, held 0.0 mm |

**Takeaway:** a sharp threshold that moves with force: 20 N needs density >= 1.0 (5 g panel), 50 N needs
>= 5.0 (25 g). Both boundaries sit at roughly the same push-per-panel-mass, F / m_panel ~ 2,000-4,000 m/s^2
(20 N / 5 g = 4,000 holds, 20 N / 2.5 g = 8,000 fails; 50 N / 25 g = 2,000 holds, 50 N / 10 g = 5,000
fails). So with 36 iterations the contact holds only while the panel would not be accelerated by more than
~200-400 g's. Realistic cloth (0.1 -> 0.5 g panels here) is 10-50x lighter than that at 20 N.

## E6 -- How do solver iterations move the density threshold?
**Question:** E5's threshold was with 36 iterations. Does 8 (trained) or 192 (Factory's value) shift it?
**Setup:** standard case, position iterations 8 / 36 / 192 (cloth request and scene max both set),
density 0.2 / 1.0 / 2.0 (36-iteration rows are from E5).
**Results** (`e6_iterations/`; panel centres unless noted):

| iterations | density 0.2 (1 g) | density 1.0 (5 g) | density 2.0 (10 g) |
|---|---|---|---|
| 8 (trained) | 20 N: through; 50 N: through | 20 N: centres through; 50 N: through | 20 N: through; 50 N: through |
| 36 | 20 N: through; 50 N: through | **20 N: holds (0.3 mm)**; 50 N: through | 20 N: holds (0.0); 50 N: through |
| 192 | **20 N: holds (0.6-3.5 mm)**; 50 N: through | 20 N: holds (0.1); **50 N: holds (0.5 mm)** | 20 N: holds; 50 N: centres hold, hinge site through (23 mm) |

**Takeaway:** iterations trade directly against density. Going 36 -> 192 (5.3x) lowers the density needed
by ~5x at both forces (20 N: 1.0 -> 0.2; 50 N: 5.0 -> 1.0), i.e. the tolerable push-per-panel-mass grows
roughly in proportion to the iteration count -- the classic Gauss-Seidel convergence behaviour for a light
body between heavy ones. The trained 8 iterations fail even at density 2.0. Extrapolating, realistic
density 0.1 would need ~400 iterations at 20 N and ~2,000 at 50 N; PhysX caps solver iterations at 255, so
iterations alone cannot reach nominal density at 50 N. The hinge site stays the weakest point (it fails
first at 192 / density 2.0 / 50 N), probably because the probe lands on the crease between two panels.

## E7 -- Do the thresholds hold at the curriculum's 2 mm thickness?
**Question:** E6 used 20 mm panels. The curriculum targets 2 mm; does a thin panel need more?
**Setup:** standard case but `--thickness 0.002`, 192 iterations, density 0.2 / 1.0 / 5.0.
"Through" now only needs > 2 mm of depth, so it is a much stricter bar.
**Results** (`e7_thin/`):

| density | 20 N | 50 N |
|---|---|---|
| 0.2 (1 g) | 2/3 through (one centre held at 1.7 mm max) | 3/3 through (depths 2.0-4.1 mm) |
| 1.0 (5 g) | 0/3 through (held 0.2 mm) | 1/3 through (centre+ reached 3.2 mm; centre- held 0.3 mm) |
| 5.0 (25 g) | 0/3 through (0.0 mm) | 0/3 through (0.0 mm) |

(20 mm with 192 iterations, E6: density 0.2 held 20 N, density 1.0 held 50 N.)
**Takeaway:** thin panels need a bit more margin -- each threshold moves up roughly one density step
(20 N: 0.2 -> ~1.0; 50 N: 1.0 -> ~2-5) -- mainly because a 2 mm panel leaves almost no room: a few mm of
sinking that a 20 mm panel absorbs counts as through. The mass-ratio picture is unchanged.

---

## Summary (E1-E7)
- **The cause is the pusher-to-panel mass ratio, not force or step size alone** (E2). A heavy pusher on a
  light panel fails; a 1:1 pusher barely dents it at the same force.
- **Levers that do NOT help at realistic density:** finer dt down to 1/960 (E1), hinge armature up to 1e-2
  (E3), a thick contact/rest-offset shell (E4), and (earlier) the UWLab friction/bounce settings.
- **Levers that DO help:** panel mass (density) and solver position iterations, and they trade off almost
  1:1 (E5, E6). Threshold for the smallest single_hinge panel (5 x 10 cm), 20 mm thick:

  | iterations | holds 20 N from | holds 50 N from |
  |---|---|---|
  | 8 (trained) | > 2.0 (not reached) | > 2.0 (not reached) |
  | 36 | density 1.0 | density 5.0 |
  | 192 | density 0.2 | density 1.0 |

  At 2 mm add roughly one density step (E7).
- **Implication:** nominal density 0.1 cannot be made penetration-free at hand-scale forces by solver
  tuning alone -- it would need ~400 iterations at 20 N and ~2,000 at 50 N, beyond PhysX's cap of 255.
  Options: (a) a density floor of ~1-2 kg/m^2 with 192 iterations (heavier than real cloth, about Yibo's
  2.0); (b) lower the robot's effective pushing mass/force in the solver; (c) penalize or terminate on
  penetration during training so the policy cannot exploit it; (d) the deformable (VBD) backend.
- **Open questions:** cost of 192 iterations in training throughput; whether larger panels (bigger sheets,
  single_hinge at 30 cm) relax the threshold as the mass scaling predicts; whether hinge sites need their
  own treatment (they fail first in several runs).

## E8 -- Policy check: does density 5.0 remove penetration in real rollouts?
**Question:** the press test held 50 N at density 5.0 with 36 iterations (E5). Does the trained policy
then stop penetrating, and what happens to its success?
**Setup:** `eval_cloth.py`, best checkpoint (epoch 5491), 1024 episodes (first per env), single_hinge,
material DR off with density 5.0 (all else nominal), 20 mm, position iterations 8 / 36 / 192. Penetration =
deepest robot-cloth contact per episode (PhysX separation).
**Results** (`../eval/density5/`):

| iterations | fold | release | penetration median / p90 / max | > 2 mm | > 10 mm | through (> 20 mm) |
|---|---|---|---|---|---|---|
| 8 (trained) | 0.754 | 0.753 | 0.9 / 3.0 / 29.7 mm | 20% | 1% | 1% |
| 36 | 0.745 | 0.745 | 0.5 / 1.0 / 25.0 mm | 2% | 1% | 1% |
| 192 | 0.736 | 0.734 | 0.5 / 1.1 / 36.6 mm | 3% | 1% | 1% |
| (density 0.1, 8 it, for reference) | 0.830 | 0.673 | 11.6 / 24.8 / 55.1 mm | 99% | 57% | 33% |

**Takeaway:** density 5.0 + 36 iterations makes rollouts nearly penetration-free (98% of episodes within
2 mm); 192 iterations add nothing and cost ~2x wall time. A ~1% tail still goes > 20 mm at every
iteration count -- worth inspecting. Release success rises (0.67 -> 0.75) because the heavy folded half
no longer springs open (fold = release); fold success drops a little (0.83 -> 0.75), consistent with a
policy trained on light cloth handling heavy cloth. Price: 5 kg/m^2 is 10-50x heavier than real fabric.

## E9 -- Press thresholds at 2 mm thickness
**Question:** E5/E6 found the density thresholds at 20 mm. What holds at the 2 mm the base recipe uses?
**Setup:** standard case but `--thickness 0.002`; 36 iterations at density 1 / 2 / 5 / 10, and 192 iterations
at density 2. "Through" = deeper than 2 mm (the probe reached the table through the panel).
**Results** (`e9_2mm/`; centre sites / hinge site):

| iterations, density | 20 N | 50 N |
|---|---|---|
| 36, 1.0 | centres hold (0.4 mm); hinge through | centres 1.9 mm (all but through); hinge through |
| 36, 2.0 | holds (0.0 mm) | centre- 1.9 mm, centre+ 0.5 mm; hinge through |
| 36, **5.0** | **holds (<= 0.1 mm)** | **holds (<= 0.1 mm)** |
| 36, 10.0 | holds | holds |
| 192, 2.0 | holds (<= 0.2 mm) | holds (<= 0.6 mm) |

**Takeaway:** at 2 mm, with 36 iterations, 20 N needs density ~2 and 50 N ~5 -- about the same as 20 mm
(1.0 / 5.0), one step stricter at 20 N. But the margin is thin: where it fails, the probe sinks the full
1.9-2.0 mm, i.e. it touches the table through the panel. The hinge site fails first. 192 iterations hold
50 N already at density 2.

## E10 -- Policy check at 2 mm (density 5, 36 iterations)
**Setup:** as E8 (best checkpoint, 1024 single_hinge episodes, nominal material, density 5.0, 36 iterations)
but `thickness=0.002`.
**Results** (`../eval/density5_2mm/it36.json`): penetration per episode median 0.47 / p90 0.97 / max 23.5 mm;
episodes > 1 mm 9%, > 2 mm (through the panel) **1.3%**, > 10 mm 0.5%. Fold 0.11, release 0.10.
**Takeaway:** density 5 + 36 iterations keeps rollouts nearly penetration-free at 2 mm as well (98.7% never
go through). The low success is expected: this checkpoint trained at 20 mm, and 2 mm changes the folded
geometry (ply gap / fold targets) it learned.
