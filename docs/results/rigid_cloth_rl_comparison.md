# Rigid chains vs the VBD cloth: RL training, speed, and transfer

Companion to `rigid_cloth_approximation.md`, which established the **geometry** of five hinged-slat
approximations without a simulator. This one puts all five into the real env, trains a policy on each,
and asks the only question that decides whether they are useful:

> A policy trained on a rigid chain — does it fold the actual cloth?

Three numbers per approximation: how fast it trains, how well it learns its own manipuland, and how
much of that survives the move to the VBD sheet.

## Setup

| | |
|---|---|
| arms | `box3-mid`, `box-mid-odd`, `cyl-mid-odd`, `box2-surface`, `box-surface`, and **`vbd`** — the unmodified `Isaacsimenvs-Cloth-Direct-v0` — as the control |
| seeds | 1, 2, 3 per arm (`agent.params.seed`), 18 runs on 18 RTX 5090s |
| scale | 768 envs, horizon 32, minibatch 24576, SAPG, 1 GPU per run |
| budget | 4069 epochs = **1.0e8 env-steps**, the same budget as `cloth_prior_vs_scratch.md` |
| prior | `pretrained_policy/model.pth`, `--checkpoint_load_mode weights`, on every arm |
| launcher | `scripts/cluster/bos14_rigid_cloth_train.sh` (array job 155046) |

**768 envs is set by the control's memory ceiling, not by choice.** The VBD cloth OOMs on a 32 GB
RTX 5090 at both 1536 and 1152 envs (it asks for another 3.7 GB); the published 1536/rank runs used
96 GB RTX PRO 6000s. Since a speed comparison across different env counts or different GPUs measures
the hardware, every arm gets the same 768 envs on the same card.

### What is held identical

`RigidClothEnv` subclasses `ClothEnv` and overrides **six** members, five of which are setup or
physics: `build_physics_cfg` (plain MJWarp instead of coupled MJWarp + VBD), `_install_cloth` (an
articulation instead of an authored deformable mesh), `__init__`, `_reset_idx` (it calls `super()`
and then re-draws the physics DR) and `pre_clone_scene_hook` (same value as the parent's). The
reward, the progress ratchet, the success criterion, the observation layout and the finite-value
guard are inherited — the same code on the same quantities.

**The sixth changes what is scored, for two of the five arms.** `_init_fold_targets` writes this
chain's own ply gap into `cloth.self_contact_radius` before calling the parent, so the fold target
sits where *this* manipuland's plies actually come to rest. For `box3-mid`, `box2-surface` and
`box-surface` that is 2.00 mm — the cloth's own value, so the scoring is byte-identical. For the two
mid-hinge uniform chains, `box-mid-odd` and `cyl-mid-odd`, it is 5.88 mm, and their fold error,
fold criterion, keypoint reward and termination all differ from the cloth's by that 3.88 mm. It is
deliberate (a perfect fold should score ~0 on every variant, and `ClothEnv` itself got this wrong
once, costing the VBD sheet 14 mm at a perfect fold) and it is one flag away from off
(`rigid_cloth.fold_lift_from_ply_gap=false`). But it means the round-1 and round-3 rows for those two
arms are not measured against the cloth's criterion, and no claim about them should say they are.
`RigidCloth.yaml` is a copy of `Cloth.yaml` with one block added (`rigid_cloth`) and exactly one
differing shared key: `newton.per_env_triangle_pairs`, the contact buffer, raised 65536 -> 524288
because a folded 16-slat chain demands eight times what the sheet does (see *The contact buffer was
too small*). It sizes an allocation; it does not change what is scored.

That was the design goal from the start: if the two envs scored the fold even slightly differently, a
difference in learning curves would be unattributable, and the experiment would quietly measure the
reward change instead of the physics change.

## Sim speed

Zero actions, so this is the physics alone, with no policy and no PPO update.

| manipuland | env-steps/s @ 64 envs | vs cloth |
|---|---|---|
| `box2-surface` | 6493 | **24.0x** |
| `box3-mid` | 6473 | **24.0x** |
| `cyl-mid-odd` | 6128 | 22.7x |
| `box-surface` | 6032 | 22.3x |
| `box-mid-odd` | 5913 | 21.9x |
| **VBD cloth** | **270** | 1.0x |

The cloth reaches 3222 env-steps/s at 768 envs, so most of that 24x is the cloth's poor scaling at
small env counts rather than a constant factor. The honest figure is the training one below.

The speed-up is structural, not a tuning difference: `ClothEnv` builds a **coupled MJWarp + VBD**
solve with a staggered proxy mapping, because a particle sheet cannot share a solver with a rigid
robot. A hinged chain is rigid, so it joins the robot in MJWarp and the whole coupling layer — proxy
bodies, substepping, contact buffers, AVBD ramping — disappears.

## Training speed

**Measure this against wall-clock, not against rl_games' `fps total`.** That counter overstates a
fast env by up to 2x: over a 90 s window the VBD control's reported 3001 fps was a realised 3003,
while `box3-mid`'s reported 22254 was a realised 10923. Whatever it excludes (logging, resets, the
gap between epochs) is a fixed cost, so it is a larger share of a faster env's epoch — and it turns a
true 3.6x into a claimed 7.4x. The table below is `frames / wall_s` from each job's own timing.

**And measure it late, not early.** The ratio collapses as the policy learns to touch the sheet:

| arm | bodies | early (idle) | under manipulation | vs VBD |
|---|---|---|---|---|
| `box3-mid` | 3 | 7.4x | 10976 | **3.8x** |
| `box2-surface` | 2 | 6.1x | 9460 | **3.3x** |
| `box-surface` | 16 | 4.7x | 4361 | 1.5x |
| `cyl-mid-odd` | 17 | 4.5x | 3744 | 1.3x |
| `box-mid-odd` | 17 | 3.9x | 3635 | 1.3x |
| **`vbd`** | — | 1.0x | 2905 | 1.0x |

The early figures were taken while the policy barely contacted the sheet. Once it grasps and drags,
the 16- and 17-body chains are **only 1.3-1.5x faster than the VBD cloth** — not worth the
approximation error — while the 2- and 3-body chains keep a real 3.3-3.8x. Body count is the cost
driver, as the MuJoCo probe predicted, and it decides which variant is worth using at all.

## Learning: the training curves say nothing folds, and the training curves are wrong

The first thing the curves say is that `successes` and the **fold** are different events, and only
the second one is the task:

| arm (seed 1) | `successes` first -> last decile | `fold_rate` | `fold_err_mean` (m) |
|---|---|---|---|
| `box2-surface` s3 | 0.002 -> **0.932** | 0.000 -> 0.006 | 0.0999 -> 0.0954 |
| `box3-mid` | 0.003 -> 0.519 | 0.000 -> 0.002 | 0.0999 -> 0.0962 |
| `box-mid-odd` | 0.002 -> 0.218 | 0.000 -> 0.002 | 0.0997 -> 0.0908 |
| **`vbd` (control)** | 0.038 -> **0.503** | 0.000 -> 0.001 | 0.0989 -> 0.0912 |

`successes` climbs to 0.5-0.93 while `fold_err_mean` stays within 5 mm of the 0.100 that doing
nothing scores. The policies are satisfying the **inherited rigid-tool pose criterion**, which
`episodes.py` documents as the long-standing ambiguity on this task, and not folding anything. The
evaluation protocol pins `--success_tolerance 0.01` precisely so that criterion cannot fire; training
does not pin it, so it is what gets optimised.

**The control behaves the same way**, which is the important part: this is a property of the budget
and scale, not of the rigid approximation. All six arms are learning the same wrong thing equally, so
the comparison between them stays valid.

### …except `fold_rate` is not measuring what the section title assumed

**That heading was wrong, and the evaluation says so.** `fold_rate` never exceeds 0.009 for any arm,
yet evaluated deterministically at a pinned tolerance:

| arm | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| `box3-mid` | 100% | 100% | 100% |
| `box2-surface` | 100% | 100% | 100% |
| **`vbd` (control)** | **94%** (30/32) | **0%** | **0%** |

`fold_rate` is computed on the *stochastic* policy under the tolerance *curriculum*; the evaluation
runs the deterministic mean action against a tolerance pinned at 0.01. The two differ by two orders
of magnitude, in the same direction, for every arm. **Never read a training-time `fold_rate` on this
task as evidence that a policy cannot fold** — read it as a statement about the exploration noise.

What the control's three seeds show instead is a **variance** result, and it is the opposite of what
the rigid approximation was expected to cost. `vbd` s2 and s3 are inert: 0/32 at every SAPG block
(0, 10, 30, 50 — so it is not the block-selection trap), 32/32 timeouts, zero falls, and
`fold_err` exactly 0.1000, i.e. the hand never disturbs the sheet at all. Their *training* reward is
if anything higher than s1's (1036 and 992 against 1026), because the inherited rigid-tool pose
criterion pays without touching the cloth.

So at this budget the cloth learns its own task in **one seed out of three**, while `box3-mid` and
`box2-surface` learn theirs in **three out of three**. As a testbed the chain is not merely faster,
it is *reproducible* — which is worth more than the speed-up for anything that needs to compare runs.

`cloth_prior_vs_scratch.md` reached 93% at the same 1e8-step budget but at **6144 total envs**
(4 ranks x 1536) against 768 here, and showed the same seed spread (93%/18%/13%). Round B repeats
everything at 1536 envs on 96 GB RTX PRO 6000s (array 155121).

## Transfer: it scales with body count, and that is the whole trade-off

*Full matrix in flight; the first complete pair, 32 episodes each, `--sapg_expl_coef 0`,
`--success_tolerance 0.01` so only a real fold counts.*

| policy | tested on | held folds | best_fold_err (m) | falls | mean ep len |
|---|---|---|---|---|---|
| `box3-mid` s1 (1e8 steps on the chain) | **`box3-mid`** | **32/32 (100%)** | 0.0026 | 0 (0%) | 81 |
| `box3-mid` s1 (1e8 steps on the chain) | VBD cloth | **5/32 (16%)** | 0.0679 | **24 (75%)** | 399 |
| reference (cloth-trained, ~7.5e8 steps) | `box3-mid` | **6/32 (19%)** | 0.0769 | **23 (72%)** | 266 |
| reference (cloth-trained, ~7.5e8 steps) | **VBD cloth** | **32/32 (100%)** | 0.0141 | 0 (0%) | 57 |

Two things follow, and the second is the answer to the question this whole exercise asked.

**The chains are learnable, and the training curves say otherwise.** `box3-mid` folds its own chain in
**every** episode, in 81 steps, with zero falls — while its training-time `fold_rate` never rose above
0.009. That scalar is computed on the stochastic policy under the tolerance *curriculum*; the
evaluation runs the deterministic mean action against a pinned tolerance. Do not read `fold_rate`
during training as "it cannot fold".

**Transfer fails in both directions** — 100% -> 16% chain-to-cloth, 100% -> 19% cloth-to-chain, with
~75% and ~72% of episodes ending with the sheet knocked off the table. **A rigid chain slides and
tips as a unit where a cloth absorbs the same push by deforming locally**, and a policy tuned to
either behaviour shoves the other one off the table.

### But "19%" was a property of `box3-mid`, not of rigid chains

Running the cloth policy against **all five** variants instead of one turns that single number into a
monotone trend in body count:

| tested on | bodies | reference (cloth-trained) | `vbd` s1 (matched budget) |
|---|---|---|---|
| `box-mid-odd` | 17 | **20/32 (63%)** | **17/32 (53%)** |
| `cyl-mid-odd` | 17 | **16/32 (50%)** | **14/32 (44%)** |
| `box-surface` | 16 | 7/32 (22%) | 6/32 (19%) |
| `box2-surface` | 2 | 14/32 (44%) | 1/32 (3%) |
| `box3-mid` | 3 | 6/32 (19%) | 1/32 (3%) |
| **VBD cloth** | — | **32/32 (100%)** | **30/32 (94%)** |

Two independently trained cloth policies — one at ~7.5e8 steps, one at 1e8 — **agree on the top two
and disagree below them**: both put `box-mid-odd` first and `cyl-mid-odd` second, but the reference
scores `box2-surface` at 44% where `vbd` s1 scores it at 3%. So the defensible claim is narrow: *the
two 17-slat mid-attached chains are the most cloth-like of the five, by both measures.* The ordering
of the remaining three is not reproducible across policies and should not be quoted. `fold_err`
agrees on the part that replicates (0.0400 on `box-mid-odd` against 0.0769 on `box3-mid`).

**A 17-slat chain reproduces roughly half to two-thirds of the cloth policy's behaviour; a 3-slat
chain reproduces almost none of it.**

### The complete matrix, and the single axis it collapses to

All 18 policies x 3 seeds, 32 episodes each, coef 0, tolerance 0.01. Every cell is a held fold.

| variant | bodies | **speed** | own manipuland (s1/s2/s3) | **rigid -> cloth** | **cloth -> rigid** |
|---|---|---|---|---|---|
| `box2-surface` | 2 | **3.2x** | 100 / 100 / 100 | 0 / 3 / 0% | 44% (ref), 3% (`vbd` s1) |
| `box3-mid` | 3 | **2.4x** | 100 / 100 / 100 | 19 / 0 / 0% | 19% (ref), 3% (`vbd` s1) |
| `box-surface` | 16 | 1.5x | 16 / 44 / 0 | 0 / 12 / -% | 22% (ref), 19% (`vbd` s1) |
| `cyl-mid-odd` | 17 | 1.4x | 100 / 97 / 34 | 28 / 25 / 12% | 50% (ref), 44% (`vbd` s1) |
| `box-mid-odd` | 17 | **1.2x** | 59 / 100 / 12 | 16 / 31 / 12% | 62% (ref), 53% (`vbd` s1) |
| **`vbd` cloth** | — | 1.0x | **94 / 0 / 0** | — | — |

Read down any column and the same ordering appears, because **there is really only one free
parameter: how close the manipuland is to a cloth.**

- **Speed** falls monotonically with body count, 3.2x -> 1.2x.
- **Transfer** rises with body count, ~0% -> ~30% outward and 19% -> 62% inward.
- **Trainability falls with it too**, and this is the part that was not anticipated: the 2- and
  3-body chains hit 100% on all three seeds, the 17-body chains scatter over 12-100%, and the cloth
  itself manages 94/0/0. The chains inherit the cloth's *difficulty* at the same rate they inherit
  its *behaviour*.

So there is no setting that is simultaneously fast, transferable and reproducible; those three are
the same axis seen from different ends. Pick `box3-mid` or `box2-surface` for throughput and
seed-to-seed reproducibility on work that will not move to cloth; pick `box-mid-odd` when cloth-like
behaviour is the point and 1.2x plus a wide seed spread is an acceptable price.

The earlier conclusion that the approximation "does not work" was drawn from `box3-mid` alone and is
too strong: it does not work *for the variant that happened to be fastest to run*. `box-surface` is
off-trend in both directions (16 bodies, but 22% inward and 0-44% own) — it is the one variant whose
slats are surface-attached rather than mid-attached, so body count is necessary and not sufficient.

So as a *drop-in substitute* for the VBD sheet — train on the chain, deploy on the cloth — this
approximation does not work at 1e8 steps. What it is good for is a separate question: it is 3.3-3.8x
faster (for the low-body-count variants), fully learnable, and identical in scoring, which makes it a
usable testbed for reward, curriculum and infrastructure work that would otherwise pay the cloth's
solver cost.

### The 100% is measured from a pinned start pose, and it does not survive randomisation

Every number above runs under `disable_randomization`, which `episodes.py` applies unless
`--randomize` is passed: the manipuland starts dead-centre, axis-aligned, with no reset noise and no
domain randomisation. That is the right default for a backend-vs-backend comparison, but it is not
the training distribution, and the gap between them is not small. Same checkpoints, same 32 episodes,
same tolerance, `--randomize` the only change:

| policy | tested on | pinned start | randomised start |
|---|---|---|---|
| `box3-mid` s1 | `box3-mid` | 100% | **81%** (26/32) |
| `box2-surface` s1 | `box2-surface` | 100% | **34%** (11/32) |
| `box3-mid` s1 | VBD cloth | 11% | 9% (3/32) |
| `box2-surface` s1 | VBD cloth | 0% | 3% (1/32) |
| reference (cloth-trained) | **VBD cloth** | 100% | **97%** (31/32) |
| reference (cloth-trained) | `box3-mid` | 19% | 28% (9/32) |

The cloth policy loses three points; the chain policies lose nineteen and sixty-six. So the
near-zero seed variance the chains show is partly an artefact of always being handed the same
initial condition — **the rigid arms are not merely non-transferable, they are individually more
brittle than the cloth policy they were meant to stand in for.** The transfer column is unaffected:
0-11% pinned, 3-9% randomised.

Reproduce with `RANDOMIZE=1 LABEL_SUFFIX=__rand MANIFEST=<subset>.tsv sbatch ...bos14_rigid_cloth_eval.sh`.

### On film

`scripts/cluster/bos14_rigid_cloth_render.sh` films the same four pairs under the evaluation's own
settings. 900 steps, one env:

| clip | folds | min fold error (m) | criterion 0.040 |
|---|---|---|---|
| `box3-mid` s1 on its own chain | **11** | 0.0022 | met, every episode |
| `box2-surface` s1 on its own chain | **11** | 0.0257 | met, every episode |
| `box3-mid` s1 on the cloth | **0** | 0.0730 | never approached |
| `box2-surface` s1 on the cloth | **0** | 0.0996 | sheet barely moves |

Eleven folds in 900 steps is 900/81, i.e. the clip reproduces the evaluation's episode length exactly.
The failure is legible: the hand still runs the reach-press-drag it learned, and the cloth absorbs it
locally instead of hinging.

## The contact buffer was too small for 16-17 slats, and that invalidated three arms

Caught by `contact_guard`, which refuses to report an evaluation measured on dropped contacts. Once
it fired, counting the same warning in the TRAINING logs (where nothing guards) split the six arms
cleanly:

| arm | bodies | overflow warnings per run | peak demand (768 envs) | per env |
|---|---|---|---|---|
| `box-mid-odd` | 17 | 137,832 - 177,125 | 91,036,614 | 118,537 |
| `cyl-mid-odd` | 17 | 138,902 - 198,470 | 91,615,819 | 119,291 |
| `box-surface` | 16 | 43,742 - 207,998 | 85,518,038 | 111,351 |
| `box3-mid` | 3 | **0** | — | — |
| `box2-surface` | 2 | **0** | — | — |
| **VBD cloth** | — | **0** | — | — |

`per_env_triangle_pairs = 65536` is sized for the VBD sheet (~2.5x its measured ~26.6k/env). A 16-17
slat chain with self-collision needs ~119k, so those three arms ran their **entire training** with the
buffer overflowing against a budget of 50,331,648.

This is the worst kind of error for this particular experiment, because dropping contacts makes the
simulation both **wrong and faster** — it flatters precisely the number the exercise exists to
measure. The 1.3-1.5x figures first reported for these three arms were measured on a degraded scene.

Raised to 262144 (~2.2x the measured peak; the peak is itself a lower bound, since an overflowing
buffer truncates its own count). Verified at 768 envs on `box-mid-odd`, the worst case: **0
overflows**, driven fold 0.00000, 0 divergences, and 31,996 env-steps/s against the cloth's 3,222 --
**9.9x sim-only**. The nine contaminated runs were deleted and re-launched; `box3-mid`,
`box2-surface` and the control are unaffected and were kept.

**262144 was still too small for the evaluation**, and for a reason worth stating separately: the
training peak was measured on runs that never fold. A policy that folds every episode stacks the
chain into two plies face to face, which is the configuration that maximises triangle pairs -- the
guard caught ~285k/env at 32 envs, 2.4x the training peak. So the overflow appears exactly when the
measurement *succeeds*: under-sizing corrupts the successful half of the transfer matrix and leaves
the failing half clean. `RigidCloth.yaml` therefore ships 524288, ~1.8x the observed eval peak.

## The clip said 0 folds while the table said 100%, and the clip was wrong

Worth its own section because it cost three rounds of debugging a policy that was working, and
because the failure mode — *a measurement that reports zero rather than erroring* — is the same one
the contact buffer had.

`render_newton.py` captioned every clip `~0 goals`, including a `box3-mid` clip that folds its chain
eleven times. The counter is `inner._successes`, and `_get_dones` increments it and Isaac Lab's
auto-reset zeroes it **inside the same `env.step()` call**, so by the time the render loop can read it
the value is back to 0. The existing "accumulate positive deltas instead of end-minus-start" patch
does not help: there is no step at which the delta is positive.

Three wrong diagnoses came first, each plausible and each disproved by an experiment that should have
been run before the explanation was believed:

- *reset randomisation* — the render passed `--randomize_reset` while `episodes.py` pins the start
  pose. A real mismatch, and fixing it changed nothing.
- *env count* — 8 on film against 32 in the evaluation. Running `episodes.py` at 8 gave 8/8 folds at
  step 81, identical to 32.
- *the priming zero-action tick* — removed from the render as "the difference", when `episodes.py`
  does exactly the same thing. Reverted.

What settled it was `RC_TRACE=<n>`, now in **both** modules, printing the same
`(obs, action, fold error)` checksum line per step. The two paths agree to 1e-6 from step 0, which
means the rollouts are identical and the disagreement was never in the physics. **A render that
disagrees with an evaluation is a reporting bug until that trace says otherwise.**

The counter is now `terminated & was_folded`, sampled before the step — the fold is scored as "the
episode ended while the sheet was folded", which is what the termination means. Counting a run of
`HELD_FOLD_STEPS` folded observations instead is off by one and never fires: the env scores the state
the step *produces*, reaches ten and terminates, while the loop has seen nine and the reset wipes the
tenth. The HUD carries the same number, so the video and the table cannot disagree silently again.

Two framing fixes went in alongside: the cloth launcher's camera sits 2.3 m from a 10 cm manipuland
(the sheet rendered as a few dozen pixels — the fold was not visible at all, which is half of why the
caption was believed), and the chain's slats had no colour rule and drew as one blob. Camera at
~0.4 m; slats take the cable hue sweep, so which slat folded onto which is readable.

## Bringing the env up: five bugs, all of them silent

Recorded because every one of them produced a plausible-looking number rather than an error, and
because the same class of failure will recur in anything that emulates one manipuland with another.

**1. The bake imported the env config, and exited 0 having baked nothing.** `bake_rigid_cloth`
imported `RigidClothCfg`, which inherits `PlayEnvCfg`, which builds a `SimulationCfg(physics=...)`
that only Isaac Lab 3.0 accepts. The bake runs under `.venv_isaacsim`, where that raises `TypeError` —
and Kit's shutdown path calls `os._exit(0)`, so the job reported success. Fixed by moving the three
URDF parameters into `rigid_cloth_assets.URDF_DEFAULTS`, which the kit-free assets module owns, and
by making the bake job count the chains it actually baked instead of trusting the exit status.

**2. The articulation had no body names yet.** `ChainAsClothMesh` resolved slat index to body index at
construction, but the chain is created inside `pre_clone_scene_hook`, where an Isaac Lab
`Articulation` has no `_data` and `body_names` raises. Resolution is now deferred to first use.

**3. Every variant diverged on step 1.** From a dead stop in **free fall**, with no contact possible,
joint velocity reached 1.3e3–4.9e3 rad/s on the first integration step and the hinges flew past their
±90° limits to 1648°; all five took the robot non-finite within 2–7 steps. A 2 mm slat's inertia about
a hinge is ~1e-8 kg·m², which is too ill-conditioned to integrate at the env's `dt = 1/120`, and the
URDF's `<dynamics damping>` does not survive the importer (the Newton model reports `joint_damping =
0.0` however the URDF is written). Fixed with an Isaac Lab actuator that is now created
**unconditionally** — it used to exist only when `joint_stiffness > 0`, which is why the default
configuration ran with an undamped, inertia-free hinge. `joint_damping = 0.05` is what actually
stabilises it; `joint_armature = 1e-5` is kept as the documented remedy for the conditioning itself.
The MuJoCo probe never saw this because it ran at `--dt 1/240`, half the env's step.

**4. The quaternion convention.** This one cost the most -- and the first diagnosis of it was wrong.

> **Corrected in round 2.** What follows originally concluded that reads are xyzw and *writes* are
> wxyz. Writes are xyzw too, exactly as documented; see "Round 2" below. What looked like a wxyz
> write was a quaternion reordered by `patches.install_pose_write_conversion` and then decoded in the
> wrong order by the adapter. The read-side finding below stands.

Isaac Lab 3.0's reads are xyzw. Reading xyzw as wxyz turns
the identity into a half-turn about z, which mirrored every grid vertex inside its slat: **a 100 mm
sheet reported a 47 mm cloud** (76.5 mm for the 17-slat chains — both reproduced to 0.1 mm by working
the mirror through by hand). Nothing raised, because a 47 mm cloud is a perfectly believable *crumpled
sheet*. The hunt went through spawn height, armature, damping and self-collision before the
convention. Writing xyzw, symmetrically, mirrored the whole chain in x.

The resting value is genuinely ambiguous — `(0, 0, 0, 1)` is the xyzw identity *and* the wxyz
half-turn about z — so the adapter now settles it by **measurement**: at the known flat rest pose only
one layout reproduces the analytic grid. It picks xyzw at 0.00 mm against wxyz's 126 mm, and prints
which it chose.

**5. Writes are silently dropped, selectively.** `write_joint_position_to_sim_index` — the current,
documented entry point — is accepted and does nothing: hinges driven to 90° stayed at 0.03°. The
deprecated `write_joint_state_to_sim` works, but only with `env_ids=None`; handed an explicit full
`arange` it also does nothing (0.07° over 120 steps). (This paragraph originally also said
`write_root_pose_to_sim` drops the orientation. It does not; the adapter did -- see "Round 2".)

A dropped joint write is the dangerous one: `_write_flat_joints` is how a reset un-folds the chain, so
a chain that ended an episode folded would **start the next one folded**, already at the goal, on an
env the task believes it reset.

### The guard that would have caught all of them

`RigidClothEnv._calibrate_and_verify_placement` now runs at construction and refuses to start if the
emulated cloud is not the flat sheet. It checks, cheapest first: the quaternion layout (against the
analytic grid), the placement centroid (calibrated, then re-measured), the rest pose (every hinge
within 1° of zero), the cloud's shape (every vertex within 1 mm of `grid_mesh`), and a **partial**
joint write (bend half the envs, confirm they bent and the rest did not).

The point is that none of these can be checked against the simulator, only against ground truth the
simulator does not supply. A displaced, mirrored, or never-reset sheet still folds, still scores, and
still trains — against a fold target built from itself — so nothing downstream reports it.

Verified state, all five variants:

| variant | flat cloud | rest `fold_err` | best fold driven | peak hinge | divergences |
|---|---|---|---|---|---|
| `box3-mid` | 100.0 x 100.0 mm | 0.10002 | **0.00002** | 90.10° | 0 |
| `box-mid-odd` | 100.0 x 100.0 mm | 0.10017 | **0.00000** | 90.06° | 0 |
| `cyl-mid-odd` | 100.0 x 100.0 mm | 0.10018 | **0.00008** | 90.07° | 0 |
| `box2-surface` | 100.0 x 100.0 mm | 0.10002 | **0.00003** | 180.50° | 0 |
| `box-surface` | 100.0 x 100.0 mm | 0.10002 | **0.00000** | 180.07° | 0 |
| **VBD cloth** | 100.0 x 100.0 mm | 0.10002 | — | — | 0 |

Flat cloud within **0 µm** of the analytic grid for every variant, and the rest score matches the
cloth's own 0.10002. A kinematically driven fold scores 0.00000–0.00008 m against a 0.04 m tolerance.

## Known differences from the cloth

Stated rather than buried; two of the three make the rigid task *easier*, which is the direction that
flatters the approximation.

*Round 1 as trained. The first two are fixed for round 2 (below); the old text is kept because it
describes what the round-1 policies saw -- with one correction to each.*

* **Friction was wrong in both contacts.** A slat felt **1.5** against a fingertip (cloth: 0.61) and,
  contrary to what this bullet first said, **0.5** against the table, not 0.354: under MJWarp's
  equal-priority `max` rule the table's 0.5 wins. Measured by a slide test in round 2: effective
  table friction 0.53 on the chain against the cloth's 0.35. **Grip was easier and sliding harder.**
* **The chain never turned at reset.** Not because `write_root_pose_to_sim` drops orientation (it
  does not) but because the adapter decoded an already-reordered quaternion and then wrote a wxyz one
  into an xyzw API; the net effect was a *roll* about the chain's long axis instead of a yaw. Round-1
  chains trained axis-aligned while the cloth's heading is uniform. **An easier task.**
* **The contact surface was 14 mm low** (found in round 2). The cloth collides through 8 mm
  particles; a 2 mm slat does not.
* **Hinge armature is fictitious inertia.** `joint_armature = 1e-5 kg·m^2` is ~1000x a slat's real
  hinge inertia. A cloth's bending inertia really is negligible, so this makes the chain bend more
  sluggishly than its geometry says. Needed for stability at `dt = 1/120`; the honest description is
  that it is the model's largest deliberate physical departure.

## Round 2: the chain matched to the cloth, with and without physics randomisation

Round 1's transfer failure was measured on a chain that differed from the cloth in three ways that
matter to a policy -- where it starts, where it can be touched, and how it slides and grips -- so a
0% transfer said little about hinged chains as such. Round 2 fixes all three, **measures each against
the VBD cloth under one protocol before training**, and retrains `box3-mid` and `box2-surface` twice:
once with the fixes only (`fix`), once adding per-env physics randomisation (`fixdr`).

### The fixes, and what they measured

`scripts/analysis/rigid_cloth_match_probe.py` (launcher `bos14_rigid_cloth_match.sh`, 32 envs, the
same script on the cloth) -- heading from the particle grid, rest height as the per-env median vertex
height of the flat sheets, table friction from a 0.4 m/s slide, `mu = v^2 / (2 g d)`:

| | VBD cloth (target) | box3-mid round 1 | box3-mid fixed | box2-surface round 1 | box2-surface fixed |
|---|---|---|---|---|---|
| heading vs the cloth for the same reset draw | -- | never turned | within 2.1° | never turned | within 3.0° |
| spread of reset headings (uniform: 104°) | 99.6° | 0° | 109° | 0° | 86° |
| grid rests above the table | 7.5 mm | 2.9 mm | 5.9 mm | 4.0 mm | 9.1 mm |
| surface a fingertip meets | ~15.5 mm | ~3.9 mm | ~13.9 mm | ~4.0 mm | ~16.1 mm |
| effective table friction (slide) | 0.348 | 0.526 | 0.366 | ~0.5 | 0.369 |
| slat-fingertip friction (resolved) | 0.612 | 1.5 | 0.612 | 1.5 | 0.612 |
| driven fold error | -- | 0.00002 | 0.00004 | 0.00003 | 0.00003 |

**Heading.** `patches.install_pose_write_conversion` wraps `env.object.write_root_pose_to_sim` and
reorders the task's wxyz quaternion to xyzw. Both adapters then decode it as wxyz. For the random
draws of training that is harmless (an isotropic Gaussian is Haar-uniform under any relabelling), so
the cloth's heading was always uniform; the chain adapter, however, then built a wxyz yaw and wrote it
into the xyzw API, which is a roll about the chain's long axis. `scripts/analysis/rigid_cloth_yaw_probe.py`
settled the convention from slat *positions* alone: an xyzw write turns the chain to exactly the
requested 0 / 45 / 90 / -120°. Fix: both adapters share one decode (`cloth_adapter.task_yaw`) and the
chain writes xyzw; the chain now starts wherever the cloth would for the same draw, including the
pinned evaluation start (which that decode puts at 180° for both). A wrong heading is now fatal at env
construction instead of a warning.

**Contact height.** `rigid_cloth.contact_margin` (default: `cloth.thickness/2 - slat/2` = 7 mm) is
written onto every slat's Newton `shape_margin`; margins sum per contact and MJWarp holds a contact at
its margin, so the chain rests and is touched where the cloth is, to within 1.6 mm (from 11.5 mm).
Slat-slat pairs are removed from the explicit contact list (a folded chain's plies would otherwise
hover 14 mm apart); the joint limits already stop them at their geometric contact, and the driven fold
still scores 0.00004.

**Friction.** MJWarp takes the higher-*priority* geom's coefficient and `max` only on a tie. Slats at
priority 1 with 0.354 and fingertips at priority 2 with 0.612 reproduce the cloth's slat-table,
slat-link and slat-fingertip contacts exactly. **Cost:** fingertip-*table* friction falls from 1.5 to
0.612 in the chain env (the fingertip wins that contact too); no one-geom-per-slat assignment can hit
all four pairs.

### Physics randomisation (`fixdr`)

`rigid_cloth.randomization`, re-drawn per env at every reset, centred on the matched values: slat
friction x[0.7, 1.4], fingertip friction x[0.7, 1.4], hinge damping x[0.5, 3.0] (log-uniform), hinge
stiffness [0, 5e-4] N m/rad, contact margin +-3 mm, slat mass and inertia x[0.7, 1.5]. Per-env
readback of what MJWarp actually simulates matched every drawn value exactly (friction, margin, mass,
damping gain, stiffness gain), and the per-env slide friction correlates 1.00 with the drawn one.
Two implementation facts that decided whether this was usable:

* The public `notify_model_changed` for inertial and joint properties costs **34 ms** (it recomputes
  MuJoCo's global constants) against an 11 ms step; the solver's own copy kernels cost 0.02 / 0.06 ms
  and are what the re-draw calls. `body_invweight0` therefore stays at the nominal chain's value.
* A hinge built with zero stiffness is a MuJoCo *velocity* actuator with no position gain, so a
  randomised stiffness would be ignored; with stiffness DR on the hinge is built with 1e-9 N m/rad to
  get the position+velocity actuator. The damping floor is 0.5x: at 0.3x one env in 32 diverged in a
  forced 180° fold.

### Runs

`scripts/cluster/bos14_rigid_cloth_retrain.sh`, array 155349: 2 arms x 2 groups x 3 seeds, 12 RTX
5090s in parallel, 1536 envs, 2035 epochs = 1.0e8 env-steps, pretrained prior, everything else as the
round-1 1536-env arm. `per_env_triangle_pairs` is 65536 here (262144 does not fit a 32 GB card at
1536 envs; these two chains never overflowed 65536). Evaluation: `bos14_rigid_cloth_eval_rc2.sh`,
every policy on its own chain (nominal physics) and on the cloth, pinned and `--randomize`, 2 eval
seeds x 32 episodes, protocol as round 1. Collated by `rigid_cloth_collect_eval.py --round rc2
--round 1 --round 1-e1536 --cutoff 4cm`, which reproduces the table below exactly.

### Results: transfer to the cloth went from ~0% to ~50%, and randomisation added nothing measurable

Held folds / episodes (32 envs x 900 steps, eval seeds 1 and 2 pooled; "falls" = sheet off the
table). Rows in bold are round 2, pooled over 3 training seeds:

| policy | own chain, pinned | own chain, randomised | cloth, pinned | cloth, randomised |
|---|---|---|---|---|
| **box3-mid fix** (3 seeds) | 192/192 (100%), falls 0% | 177/192 (92%), falls 6% | 100/192 (52%), falls 31% | 82/192 (43%), falls 28% |
| **box3-mid fixdr** (3 seeds) | 191/192 (99%), falls 0% | 169/192 (88%), falls 10% | 104/192 (54%), falls 33% | 75/192 (39%), falls 34% |
| **box2-surface fix** (3 seeds) | 190/192 (99%), falls 1% | 171/192 (89%), falls 8% | 91/192 (47%), falls 29% | 71/192 (37%), falls 28% |
| **box2-surface fixdr** (3 seeds) | 133/192 (69%), falls 3% | 106/192 (55%), falls 27% | 85/192 (44%), falls 24% | 39/192 (20%), falls 40% |
| box3-mid round 1 (768 envs) | 192/192 (100%), falls 0% | 26/32 (81%), falls 19% | 7/192 (4%), falls 85% | 3/32 (9%), falls 72% |
| box2-surface round 1 (768 envs) | 192/192 (100%), falls 0% | 11/32 (34%), falls 28% | 1/192 (1%), falls 95% | 1/32 (3%), falls 56% |
| box2-surface round 1 (1536 envs, 2 seeds) | 128/128 (100%), falls 0% | 58/64 (91%), falls 8% | 0/128 (0%), falls 77% | 4/64 (6%), falls 56% |
| reference, cloth-trained, on cloth | 32/32 (100%), falls 0% | 31/32 (97%), falls 0% | (same) | (same) |
| reference on box3-mid, OLD chain | 6/32 (19%), falls 72% | 9/32 (28%), falls 53% |  |  |
| reference on box2-surface, OLD chain | 26/64 (41%), falls 44% | 9/32 (28%), falls 47% |  |  |
| reference on box3-mid, FIXED chain | 39/64 (61%), falls 39% | 24/64 (38%), falls 59% |  |  |
| reference on box2-surface, FIXED chain | 29/64 (45%), falls 48% | 10/64 (16%), falls 73% |  |  |

Per training seed, held folds on the **cloth, pinned** (of 64):

| | s1 | s2 | s3 |
|---|---|---|---|
| box3-mid fix | 31 | 29 | 40 |
| box3-mid fixdr | 25 | 50 | 29 |
| box2-surface fix | 37 | 21 | 33 |
| box2-surface fixdr | 50 | 29 | 6 (failed run) |

**The three fixes are the whole effect.** Zero-shot on the cloth, box3-mid went from 4% to 52%
pinned (9% to 43% randomised) and box2-surface from 0-1% to 47% (3-6% to 37%). Falls on the cloth
dropped from 77-95% to ~30%. Own-chain success stayed at 99-100%, so the policies did not buy
transfer by getting worse at their own task. The gap is now roughly symmetric: chain policies score
~50% on the cloth, and the cloth-trained reference scores 45-61% on the fixed chains.

**Physics randomisation did not help at these ranges.** box3-mid `fixdr` equals `fix` within noise
(54 vs 52% pinned, 39 vs 43% randomised). Seed-to-seed spread (29-50 of 64) is larger than any group
difference, and one eval cell is +-6 points at p = 0.5, so nothing below ~15 points is a result.

**box2-surface fixdr s3 is a failed training run, not a DR result.** Its best reward rose from 14 to
48 over all 2035 epochs, while every other run passed 800 (the pretrained prior was lost in the first
epochs and never recovered); it scores 5/64 on its own chain. Without it, box2-surface `fixdr` is
128/128 own pinned, 103/128 (80%) own randomised, **79/128 (62%) cloth pinned**, 38/128 (30%) cloth
randomised: better than `fix` pinned, worse randomised, both inside the seed spread.

**What is left.** About half the cloth episodes still fail, with ~30% of them sheet-off-table falls
versus 0% for the reference on the cloth. What the chain still does not model: the cloth bends
*within* a slat, shears, and drapes over the table edge. The chain's fingertip-table friction is also
0.612, not the cloth env's 1.5 (see Friction above).

### Re-scored at 1.5 cm: the ~50% cloth transfer is mostly loose folds

The fold test's 4 cm keypoint tolerance is 40% of the 10 cm sheet and 80% of the 5 cm flap: the
crease-side corners pass while the sheet is still flat, and a rigid flap passes at ~130 of 180
degrees. A settled perfect fold scores ~2 mm. The chain policies' held cloth folds had a median best
error of 3.1 cm (the reference: 1.3 cm). So every policy was re-evaluated with
`env.cloth.keypoint_tolerance=0.015` (~160 degrees), everything else unchanged -- a real rerun, not a
re-scoring of the logs, since a 4 cm episode ends at the first loose fold
(arrays 155645 + 155699; `rigid_cloth_collect_eval.py --round rc2 --format rate
--exclude_seed box2-surface:fixdr:3`):

| policy | tol | own chain, pinned | own chain, random | cloth, pinned | cloth, random |
|---|---|---|---|---|---|
| box3-mid | 4cm | 100% (192/192) | 92% (177/192) | 52% (100/192) | 43% (82/192) |
| box3-mid | 1.5cm | 98% (188/192) | 88% (169/192) | 14% (27/192) | 11% (22/192) |
| box3-midDR | 4cm | 99% (191/192) | 88% (169/192) | 54% (104/192) | 39% (75/192) |
| box3-midDR | 1.5cm | 85% (163/192) | 72% (138/192) | 21% (41/192) | 10% (20/192) |
| box2-surface | 4cm | 99% (190/192) | 89% (171/192) | 47% (91/192) | 37% (71/192) |
| box2-surface | 1.5cm | 72% (138/192) | 74% (142/192) | 17% (33/192) | 12% (24/192) |
| box2-surfaceDR | 4cm | 69% (133/192) | 55% (106/192) | 44% (85/192) | 20% (39/192) |
| box2-surfaceDR | 1.5cm | 67% (128/192) | 46% (89/192) | 23% (44/192) | 9% (18/192) |
| box2-surfaceDR excl. failed s3 | 4cm | 100% (128/128) | 80% (103/128) | 62% (79/128) | 30% (38/128) |
| box2-surfaceDR excl. failed s3 | 1.5cm | 99% (127/128) | 70% (89/128) | 34% (44/128) | 14% (18/128) |
| reference (cloth-trained); chain cols = box3-mid / box2-surface | 4cm | 61% (39/64) / 45% (29/64) | 38% (24/64) / 16% (10/64) | 100% (32/32) | 97% (31/32) |
| reference (cloth-trained); chain cols = box3-mid / box2-surface | 1.5cm | 53% (34/64) / 20% (13/64) | 30% (19/64) / 16% (10/64) | 84% (54/64) | 47% (30/64) |

At 1.5 cm the chains transfer 14-23% pinned and 9-12% from random starts, against the reference's
84% / 47%. The reference itself is strict-limited from random starts (97% -> 47%), and on the fixed
chains it holds 53% / 20% pinned. Best single seeds on the cloth at 1.5 cm, pinned: box3-midDR s2
48%, box2-surfaceDR s1 45%, box2-surface s1 42%; none exceeds 19% from random starts. DR cost
own-chain precision (box3-mid 98% -> 85% pinned at 1.5 cm) without a matching gain from random starts.

**Wall-clock.** 1.0e8 env-steps took 2.6-4.9 h per run on one RTX 5090 (5.7k-10.5k env-steps/s,
peak 16.6-17.4 GB). The spread is per seed, not per group: DR cost no measurable throughput. Runs
that start manipulating early step slower. Four eval tasks on bos14-node-104/105 hit a CUDA-init
failure and were rerun as array 155551.

## Round 3: the same fixes on the three many-slat chains, and the trade-off reverses

Round 2 fixed two chains of 2 and 3 slats. Round 3 applies the identical three fixes to the three
that were left -- `cyl-mid-odd` (17 bodies), `box-mid-odd` (17), `box-surface` (16) -- because round
1 had found these to be *the most cloth-like of the five* when the cloth policy was driven against
them (63%/50%/22%, the "scales with body count" trend above). If body count is what makes a chain
cloth-like, these three should transfer best.

**They transfer worst.** `scripts/cluster/bos14_rigid_cloth_retrain_rc3.sh`, array 0-17: 3 arms x
(`fix`, `fixdr`) x 3 seeds, held identical to round 2 -- pretrained prior, SAPG, 1536 envs, horizon
32, 2035 epochs = 1.0e8 env-steps -- except for the two things the arms force. `per_env_triangle_pairs`
is 524288, not round 2's 65536, because a folded 16-slat chain stacks two plies of eight slats face
to face (see *The contact buffer*); 524288 pairs x 1536 envs does not fit a 32 GB RTX 5090, so these
ran on 96 GB RTX PRO 6000s and **their wall-clock is comparable only with itself, not with round 2**.
Evaluation and collation as round 2, at both cutoffs. Zero non-finite episodes in every eval cell,
`box-surface` included, so the scores are readable as scores.

| policy | tol | own chain, pinned | own chain, randomised | cloth, pinned | cloth, randomised |
|---|---|---|---|---|---|
| **cyl-mid-odd fix** (3 seeds) | 4cm | 192/192 (100%), falls 0% | 180/192 (94%), falls 6% | 66/192 (34%), falls 44% | 59/192 (31%), falls 38% |
| **cyl-mid-odd fix** (3 seeds) | 1.5cm | 97/192 (51%), falls 36% | 71/192 (37%), falls 47% | 3/192 (2%), falls 63% | 7/192 (4%), falls 46% |
| **cyl-mid-odd fixdr** (3 seeds) | 4cm | 192/192 (100%), falls 0% | 180/192 (94%), falls 6% | 65/192 (34%), falls 44% | 49/192 (26%), falls 47% |
| **cyl-mid-odd fixdr** (3 seeds) | 1.5cm | 93/192 (48%), falls 29% | 70/192 (36%), falls 30% | 13/192 (7%), falls 66% | 11/192 (6%), falls 53% |
| **box-mid-odd fix** (3 seeds) | 4cm | 192/192 (100%), falls 0% | 180/192 (94%), falls 6% | 46/192 (24%), falls 46% | 67/192 (35%), falls 32% |
| **box-mid-odd fix** (3 seeds) | 1.5cm | 94/192 (49%), falls 30% | 62/192 (32%), falls 31% | 6/192 (3%), falls 48% | 11/192 (6%), falls 43% |
| **box-mid-odd fixdr** (3 seeds) | 4cm | 192/192 (100%), falls 0% | 179/192 (93%), falls 6% | 15/192 (8%), falls 33% | 61/192 (32%), falls 36% |
| **box-mid-odd fixdr** (3 seeds) | 1.5cm | 85/192 (44%), falls 24% | 77/192 (40%), falls 24% | 4/192 (2%), falls 38% | 11/192 (6%), falls 42% |
| **box-surface fix** (3 seeds) | 4cm | 192/192 (100%), falls 0% | 170/192 (89%), falls 9% | 29/192 (15%), falls 51% | 64/192 (33%), falls 34% |
| **box-surface fix** (3 seeds) | 1.5cm | 56/192 (29%), falls 41% | 72/192 (38%), falls 28% | 5/192 (3%), falls 64% | 12/192 (6%), falls 36% |
| **box-surface fixdr** (3 seeds) | 4cm | 192/192 (100%), falls 0% | 171/192 (89%), falls 9% | 31/192 (16%), falls 48% | 57/192 (30%), falls 34% |
| **box-surface fixdr** (3 seeds) | 1.5cm | 23/192 (12%), falls 11% | 77/192 (40%), falls 17% | 1/192 (1%), falls 57% | 9/192 (5%), falls 49% |
| **box3-mid fix** (3 seeds) | 4cm | 192/192 (100%), falls 0% | 177/192 (92%), falls 6% | 100/192 (52%), falls 31% | 82/192 (43%), falls 28% |
| **box3-mid fix** (3 seeds) | 1.5cm | 188/192 (98%), falls 2% | 169/192 (88%), falls 10% | 27/192 (14%), falls 53% | 22/192 (11%), falls 32% |
| **box3-mid fixdr** (3 seeds) | 4cm | 191/192 (99%), falls 0% | 169/192 (88%), falls 10% | 104/192 (54%), falls 33% | 75/192 (39%), falls 34% |
| **box3-mid fixdr** (3 seeds) | 1.5cm | 163/192 (85%), falls 9% | 138/192 (72%), falls 20% | 41/192 (21%), falls 54% | 20/192 (10%), falls 47% |
| **box2-surface fix** (3 seeds) | 4cm | 190/192 (99%), falls 1% | 171/192 (89%), falls 8% | 91/192 (47%), falls 29% | 71/192 (37%), falls 28% |
| **box2-surface fix** (3 seeds) | 1.5cm | 138/192 (72%), falls 28% | 142/192 (74%), falls 20% | 33/192 (17%), falls 42% | 24/192 (12%), falls 45% |
| **box2-surface fixdr** (3 seeds) | 4cm | 133/192 (69%), falls 3% | 106/192 (55%), falls 27% | 85/192 (44%), falls 24% | 39/192 (20%), falls 40% |
| **box2-surface fixdr** (3 seeds) | 1.5cm | 128/192 (67%), falls 4% | 89/192 (46%), falls 33% | 44/192 (23%), falls 34% | 18/192 (9%), falls 43% |
| reference, cloth-trained, on cloth | 4cm | 32/32 (100%), falls 0% | 31/32 (97%), falls 0% | (same) | (same) |
| reference, cloth-trained, on cloth | 1.5cm | 54/64 (84%), falls 8% | 30/64 (47%), falls 17% | (same) | (same) |
| reference on cyl-mid-odd, OLD chain | 4cm | 39/64 (61%), falls 39% | -- |  |  |
| reference on box-mid-odd, OLD chain | 4cm | 39/64 (61%), falls 38% | -- |  |  |
| reference on box-surface, OLD chain | 4cm | 19/64 (30%), falls 58% | -- |  |  |
| reference on box3-mid, OLD chain | 4cm | 6/32 (19%), falls 72% | 9/32 (28%), falls 53% |  |  |
| reference on box2-surface, OLD chain | 4cm | 26/64 (41%), falls 44% | 9/32 (28%), falls 47% |  |  |
| reference on cyl-mid-odd, FIXED chain | 4cm | 35/64 (55%), falls 45% | 24/64 (38%), falls 61% |  |  |
| reference on cyl-mid-odd, FIXED chain | 1.5cm | 1/64 (2%), falls 80% | 5/64 (8%), falls 77% |  |  |
| reference on box-mid-odd, FIXED chain | 4cm | 43/64 (67%), falls 30% | 24/64 (38%), falls 61% |  |  |
| reference on box-mid-odd, FIXED chain | 1.5cm | 6/64 (9%), falls 69% | 3/64 (5%), falls 78% |  |  |
| reference on box-surface, FIXED chain | 4cm | 25/64 (39%), falls 55% | 12/64 (19%), falls 77% |  |  |
| reference on box-surface, FIXED chain | 1.5cm | 0/64 (0%), falls 70% | 1/64 (2%), falls 83% |  |  |
| reference on box3-mid, FIXED chain | 4cm | 39/64 (61%), falls 39% | 24/64 (38%), falls 59% |  |  |
| reference on box3-mid, FIXED chain | 1.5cm | 34/64 (53%), falls 44% | 19/64 (30%), falls 62% |  |  |
| reference on box2-surface, FIXED chain | 4cm | 29/64 (45%), falls 48% | 10/64 (16%), falls 73% |  |  |

**More bodies cost transfer, at both cutoffs.** At 4 cm every arm folds its own chain in 100% of
pinned episodes, and all six round-3 cells (8-34%) sit below round 2's four (44-54%). At 1.5 cm the
gap becomes categorical: round 2's `box3-mid` holds 85-98% on its own chain, while the three
many-slat arms hold 12-51% -- **they cannot fold themselves tightly**, so there is no precise fold
left to transfer, and their cloth numbers are 1-7% against round 2's 14-23%.

**The fold geometry does not explain it, and that is worth stating because it looks like it should.**
A perfect fold of `box-mid-odd` or `cyl-mid-odd` leaves its plies 5.88 mm apart against `box3-mid`'s
2.00 mm (`rc.chain_fold_residual`, whose in-plane shortfall is 0 for all five variants). But
`RigidClothEnv._init_fold_targets` builds the fold target from *this chain's* ply gap by default
(`fold_lift_from_ply_gap`), so a geometrically perfect fold scores ~0 mm on every variant and the
5.88 mm is compensated, not spent. The many-slat chains fail the 1.5 cm cutoff on their own
manipuland with the generous target already granted them.

What is left is a hypothesis, not a measurement: a 17-slat chain reaches a tight fold only when eight
passive joints are each driven to their limit, while `box3-mid` needs two, and the arm is what has to
drag them there. More passive degrees of freedom means more ways to be loosely folded. `box-surface`
is a separate story -- its ply gap is 2.00 mm and it still scores worst of all six, and it is the one
variant the approximation study declined to recommend (14 joints on their limits at rest, 64 degrees
of limit violation in a drape, "unstable"). Neither reading was isolated by an experiment.

**A caveat on reading the two halves of the table against each other.** Because the target is built
per manipuland, the own-chain columns for `box-mid-odd` and `cyl-mid-odd` are scored at a 5.88 mm
lift while their cloth columns are scored at the cloth's 2.00 mm. Each number is right for its own
manipuland -- that is the point of the override -- but the own-chain-to-cloth *drop* for those two
arms carries a 3.88 mm change of criterion that the other four arms do not have. Scoring every
variant against the cloth's 2 mm instead is one flag away (`rigid_cloth.fold_lift_from_ply_gap=false`)
and has not been run.

**The direction of the comparison matters, and that is the result.** The cloth policy handles the
17-slat chains best (reference on the FIXED chains: 67% `box-mid-odd`, 55% `cyl-mid-odd`, 39%
`box-surface` -- round 1's ordering, replicated after the fixes). Policies *trained* on those same
chains transfer to the cloth worst. "Most cloth-like" is therefore not one property: a chain can be
easy for a cloth policy to fold and still be a bad thing to learn on. The useful approximation is the
coarse one.

**Two cautions on these numbers.** Per-seed spread at 1.5 cm is larger than every effect in the
table -- `box-mid-odd fix` scores 100%, 20%, 27% on its own chain across three seeds, and
`cyl-mid-odd fix` 38%, 16%, 98% -- so no 1.5 cm row should be read as a point estimate. And for four
of the six round-3 cells the *randomised* cloth start beats the pinned one at 4 cm (`box-mid-odd
fixdr`: 8% pinned, 32% randomised; both `cyl-mid-odd` arms are the exceptions), which is backwards
and is the opposite of round 2's behaviour. The pinned
start is one specific pose, and a known quaternion relabelling puts it at 180 degrees
(`cloth_adapter.task_yaw`); randomisation averages over poses. That makes the pinned column a sample
of one configuration rather than the easier setting it is elsewhere -- it has not been chased down.

**Physics randomisation again adds nothing.** `fixdr` matches `fix` within the seed spread on every
arm and cutoff, as in round 2. Three rounds have now failed to measure a benefit at these ranges.

## Reproduce

```bash
# 1. bake the five chains into the USD cache (needs Isaac Sim + a GPU)
sbatch scripts/cluster/bos14_rigid_cloth_bake.sh

# 2. verify the env: geometry, stability, driven fold, sim throughput
sbatch --array=0-4 scripts/cluster/bos14_rigid_cloth_probe.sh
TASK=Isaacsimenvs-Cloth-Direct-v0 NUM_ENVS=768 sbatch --array=0-0 \
    scripts/cluster/bos14_rigid_cloth_probe.sh

# 3. train all six arms x 3 seeds
sbatch --array=0-17 scripts/cluster/bos14_rigid_cloth_train.sh

# 4. cross-evaluate: every policy on its own manipuland AND on the cloth
.venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_eval_manifest.py --job <train_job> \
    --reference outputs/.../nn/0_pretrained_s1_resume_150911.pth
sbatch --array=0-<N> scripts/cluster/bos14_rigid_cloth_eval.sh

# 5. collate
.venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_collect.py --job <train_job>

# 6. film the same pairs under the evaluation's own settings (four clips)
sbatch --exclude=bos14-node-092 --array=0-3 scripts/cluster/bos14_rigid_cloth_render.sh

# 7. when a clip and the table disagree, diff the two rollouts rather than reading both files
sbatch --exclude=bos14-node-092 scripts/cluster/bos14_rigid_cloth_trace.sh

# round 2: measure the matched chain against the cloth, retrain, evaluate, collate
sbatch --array=0-6 scripts/cluster/bos14_rigid_cloth_match.sh
sbatch --array=0-11 scripts/cluster/bos14_rigid_cloth_retrain.sh
sbatch --dependency=afterany:<train_job> --array=0-47 --export=ALL,TRAIN_JOB=<train_job> \
    scripts/cluster/bos14_rigid_cloth_eval_rc2.sh
.venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_collect_eval.py \
    --round rc2 --round 1 --round 1-e1536 --cutoff 4cm

# the same at both fold cutoffs, which is a second set of evaluation RUNS rather than a re-scoring
EXTRA_ARGS=env.cloth.keypoint_tolerance=0.015 OUT_PREFIX=docs/results/rc_eval_tol15_ \
    sbatch --array=0-53 --export=ALL,TRAIN_JOB=<train_job> \
    scripts/cluster/bos14_rigid_cloth_eval_rc2.sh
.venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_collect_eval.py \
    --round rc2 --format rate --exclude_seed box2-surface:fixdr:3

# round 3: the same three fixes on the three many-slat chains, on 96 GB cards
sbatch --array=0-17 scripts/cluster/bos14_rigid_cloth_retrain_rc3.sh
sbatch --array=0-77 --export=ALL,TRAIN_JOB=<rc3_job> scripts/cluster/bos14_rigid_cloth_eval_rc3.sh
.venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_collect_eval.py --round rc3 --round rc2
```
