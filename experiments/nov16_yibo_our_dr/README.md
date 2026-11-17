# Yibo interface with our cloth DR

Completed evaluation findings: [transfer analysis](analysis/README.md),
[rigid ablations](analysis/rigid.md), and [VBD fabrics](analysis/fabrics.md).

## Active training and viewer verification

W&B project: [rigidClothDR](https://wandb.ai/kk837/rigidClothDR).
The initial memory-adjusted runs were
[fine-tune 710318](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_finetune_20261116_194243)
and [scratch 710319](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_scratch_20261116_194220),
each with 12,288 environments and six SAPG blocks. Fine-tune failed after epoch
127 and scratch after epoch 238: a singular angular-velocity least-squares fit
in one environment raised from the batched solve and stopped the whole run.
Neither failure was a GPU out-of-memory error.

Recovery uses the finite full fine-tune checkpoint at epoch 122 (47,972,352
frames) and scratch periodic checkpoint at epoch 230 (90,439,680 frames), restoring
model, critic, optimizer, normalization statistics and training counters. Physics
state is not checkpointed, so episodes and recurrent rollout state restart.
The estimator now scales its equations and handles failed rows independently;
invalid task frames terminate their own environments even when the actor omits
angular velocity. Physics and DR settings are unchanged.

Recovery submissions are [fine-tune 710887](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_finetune_20261116_203228)
and [scratch 710889](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_scratch_20261116_203026),
using commit `58067d5`. The regression suite passed 402 tests with two skips;
CPU/CUDA degeneracy checks and the existing Play parity tolerances also passed.
Both replacements reached optimizer updates after restore: fine-tune epoch 123
and scratch epoch 240, with a new full scratch checkpoint saved. This verifies
recovery startup, not long-run stability.

Previous jobs [fine-tune 709852](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_finetune_20261116_192340)
and [scratch 709853](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_scratch_20261116_192301)
passed initialization and reached optimizer updates, but scratch exhausted GPU
memory at epoch 28. Neither had saved a checkpoint; they were restarted from
their intended initializations after the memory-budget fix. The interrupted
W&B runs are marked failed with their replacement job IDs.

Rigid captures now export closed plate prisms with the actual 2 mm thickness,
with one interactive viewer per design. A finished
[viewer integration check](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_viewer_check_flushed_20261116_192913)
uploaded 15 HTML files across all five design keys; the main scratch run's first
five viewers were also verified through the server API.

Startup materials are shared across environments, avoiding PhysX's 64K material
limit at 24,576 environments while preserving independent material DR writes.
The trainer also flushes W&B before its forced Kit shutdown so short runs do not
lose queued metrics or media.

Both recipes inherit `isaacsimenvs/scripts/configs/runs/yibo_our_dr.yaml`, sharing the Yibo observation/robot
interface, our current rigid designs and material/reset/sensor/action DR, and the
physical fold-then-release objective.

Additional `finetune_thickness_2_20mm.yaml` and `scratch_thickness_2_20mm.yaml`
inherit the corresponding experiment and change only physical thickness to
`[0.002, 0.020]` m. Construction uses 16 evenly spaced levels (768 environments
per level), fixed for each environment. The Yibo observation reports that actual
thickness. The prior-initialized variant uses the original checkpoint's weights
and normalization with fresh training state; the other starts from scratch.
Both retain the fixed 33 cm tabletop height, five rigid designs, density/bending/
friction/sensor DR, 12,288 environments and fold-then-release rewards.
Submitted as prior-initialized job `711271` and scratch job `711273`, using
commit `a22a4b5`, W&B project `rigidClothDR` and group
`yibo_our_dr_thickness_2_20mm`. Their run names are
`rigidClothDR_2to20mm_finetune` and `rigidClothDR_2to20mm_scratch` (with timestamps).

- `finetune.yaml` initializes from
  `pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth`, loading network
  weights and normalization statistics. It starts a fresh optimizer, epoch/frame
  counters and rollout; it does not resume the old training run.
- `scratch.yaml` starts with a newly initialized network and no checkpoint.

From the repository root, submit either recipe using the existing training launcher:

```bash
experiments/nov16_yibo_our_dr/submit_finetune.sh
experiments/nov16_yibo_our_dr/submit_scratch.sh
```

The submission scripts pass their YAML paths to `--experiment`; additional training
arguments and Hydra overrides can be appended to either command.
Both enable Weights & Biases in project `rigidClothDR`, group `yibo_our_dr`, with
run names `rigidClothDR_finetune` and `rigidClothDR_scratch`. Periodic interactive
viewers are enabled by `experiments/train.sh` and uploaded to those runs; `VIEWER=0`
disables viewer capture. The shared launcher keeps `WANDB_CONSOLE=off` so Wandb
logs metrics and viewers while Slurm captures the process console.

Checkpoint initialization is selected by the recipe; no additional `--checkpoint`
flag is needed. An explicit `--checkpoint` overrides the recipe and keeps its usual
resume behavior unless `--checkpoint_load_mode weights` is also supplied.

To restart an existing run from a full periodic checkpoint, append
`--checkpoint <run>/nn/periodic_<name>_ep_<N>.pth --checkpoint_load_mode resume`.
Without that override the two scripts retain their original initialization choices.

The shared recipe now uses 12,288 environments on one GPU with six SAPG blocks of
2,048 environments. Actor and critic minibatches remain 24,576 transitions;
with a 32-step horizon, the 393,216-transition rollout divides into 16 minibatches.
The wrappers default `PYTORCH_CUDA_ALLOC_CONF` to `expandable_segments:True`
unless the caller has selected another allocator configuration.

Full periodic checkpoints are saved every 10 epochs; best-model saving starts
after epoch 1, and the inherited weight snapshots remain every 100 epochs.
The smaller batch changes memory use, preserving the cloth physics, DR,
observations, reset distribution and table setup.

Changing the number of environments requires changing
`agent.params.config.expl_coef_block_size` to retain six blocks for the checkpoint.
Submitting these commands launches training; creating the recipes does not submit jobs.

## Controlled rigid-cloth evaluation

```bash
experiments/nov16_yibo_our_dr/submit_rigid_eval.sh
```

This reserves one GPU and evaluates 64 first episodes per case in separate simulator
processes. Starting from `yibo_our_dr`, the cases restore one factor group at a time:
sensor/action delay and noise off, 16 mm observed thickness, density 2 kg/m², historical
hinge gains/armature, smaller dimensions, or historical contacts. The final `yibo`
case is a reference with several differences restored together. Contact-only evaluation
retains the hybrid's density and gain DR; it does not accidentally freeze those draws.

Results are written to `debug_outputs/yibo_our_dr_analysis/rigid_<job>/`, including
resolved configs, initial states, physical pre-reset fold/release outcomes, motion
diagnostics, PhysX readbacks and incrementally updated `summary.md`/`summary.json`.
The runner explicitly resets the requested seed after loading the policy, so these
controlled baselines supersede earlier diagnostics whose loader reseeded to 42.
Use `ABLATION_CASES='hybrid delays_off noise_off'` or `ABLATION_SEED=1` before the
submission command for additional cases or a repeat. Single-seed outcomes remain
preliminary; the script reports paired rescues/losses alongside the headline rates.

## VBD base with branch geometry and resets

`vbd_base_branch.yaml` restores `yiboc_wuji@2b82faf`'s 10 cm sheet,
15 cm drop above the tabletop, +/-10 cm XY reset noise and +/-1 cm table-height
variation. The evaluator retains the Flexiv-Wuji checkpoint, robot/table assets,
base material and Yibo actor encoding. It scores 32 first episodes over 10 seconds;
whole-robot release is not evaluated.

Completed job `711188`: **0/32 folds, 12 falls and 20 timeouts**, with finite
states and no buffer overflows. See [the comparison](analysis/vbd_branch_base.md).

```bash
experiments/submit.sh --job-name=yibo_vbd_branch_base \
  experiments/nov16_yibo_our_dr/submit_vbd_eval.sh \
  --material base --experiment experiments/nov16_yibo_our_dr/vbd_base_branch.yaml
```

## VBD contact convergence

`vbd_iteration_sweep.sh` compares 6, 12, 40 and 80 VBD iterations per cloth substep
with the same branch-value base cloth, Flexiv-Wuji checkpoint, seed, resets and
32 first 10-second episodes. Material, timestep, six cloth substeps, two coupled
substeps and one coupler iteration stay fixed. The native VBD count is checked
against the config at startup.

```bash
experiments/submit.sh --array=0-3 experiments/nov16_yibo_our_dr/vbd_iteration_sweep.sh
```

Submitted array `717322` on commit `407aae9`: tasks 0/1/2 use 12/40/80 iterations
on RTX 6000 Ada GPUs. The 40-iteration instrumentation smoke completed 12
environments × 12 steps with finite diagnostics and no overflow. Its maximum
robot record-envelope overlap was 11.23 mm versus 3.23 mm contact-plane crossing;
table record-envelope overlap was 0.138 mm, with zero table contact-plane crossing
or independent table-box node penetration. This short check does not establish
folding performance or convergence with respect to iteration count.

The additional 6-iteration case is job `718953_3` (commit `aa688fb`). Completed
closed-loop results so far, each with 32 first episodes:

| Iterations | Folds | Falls | Worst robot contact-plane crossing | Table-node penetration |
|---|---:|---:|---:|---:|
| 6 | 0 | 8 | 17.49 mm | 0 |
| 12 | 0 | 11 | 16.89 mm | 0 |
| 40 | 1 | 11 | 18.52 mm | 0 |
| 80 | 0 | 9 | 30.89 mm | 0 |

These runs have otherwise identical resolved configs and exactly matching initial
joint positions, cloth centers and table positions. Policy actions then diverge
with the simulated states; this is not a fixed-action convergence test. The robot
metric uses cached contact planes, not exact current hand-mesh signed distance.
None of these preliminary results establishes an optimal iteration count.

Results land in `results/vbd_iterations_<array>/iterations_<6|12|40|80>/base.json`.
Opt-in diagnostics distinguish collision-envelope overlap (including the 8 mm
particle radius) from crossing a cached robot/table contact plane. Contact counts
are recorded: zero depth with zero records does not establish collision coverage.
An independent signed-distance query against the actual native table box measures
cloth-node penetration, including nodes without contact records. These policy-rate
measurements can miss between-step and triangle-interior crossings, and do not
measure cloth self-intersections. Higher iteration counts are a convergence test;
fold counts alone do not establish that penetration was fixed.

## Repeatable solver stability test

`benchmark_vbd_stability.py` restores one saved native initial state and action
tape in a fresh process per candidate. Its 12 worlds contain four flat-drop/hold,
four prescribed manipulation and four folded-release probes. The robot uses its
normal finite-stiffness PD control. A folded sheet is allowed to reopen; folding
success is not the stability criterion. There are no policy-dependent actions or
automatic episode resets in these probes.

The matrix in `vbd_stability.yaml` tests `(VBD iterations, cloth substeps, coupling
passes)`:

`(6,6,1), (12,6,1), (40,6,1), (80,6,1), (24,3,1), (6,12,1), (12,6,2)`.

Material, robot, contact coefficients and the 120 Hz outer physics timestep are
fixed. The two alternative cloth-substep cases preserve the baseline's total
iteration work. Native solver readbacks verify the actual settings; every physics
call checks its measured clock advance. Diagnostics sample each 120 Hz physics
boundary and separate synchronized physics time from diagnostic overhead.

Measured outputs include node/table penetration, robot contact-plane crossing and
coverage, stretch/compression, velocity/acceleration spikes, cloth kinetic energy,
tracking error and last-third peak speed. The initial drop is excluded from that
late-speed metric. Comparisons require matching initial-state, action and physical
protocol hashes. Different horizons, missing contact coverage, nonfinite states
and contact-buffer overflows cannot produce a false optimum.

From the repository root, prepare an immutable fixture before submitting all
seven candidates. Set a new output directory for each material/seed experiment:

```bash
export VBD_STABILITY_RUN=experiments/nov16_yibo_our_dr/results/vbd_stability_trial
experiments/submit.sh --job-name=vbd_stability_fixture \
  experiments/nov16_yibo_our_dr/vbd_stability_job.sh --prepare-fixture
# Replace <fixture-job> with the returned job ID.
experiments/submit.sh --array=0-6 --dependency=afterok:<fixture-job> \
  experiments/nov16_yibo_our_dr/vbd_stability_job.sh

.venv/isaaclab3/bin/python experiments/nov16_yibo_our_dr/analyze_vbd_stability.py \
  "$VBD_STABILITY_RUN" --limits experiments/nov16_yibo_our_dr/vbd_stability.yaml \
  --out-dir "$VBD_STABILITY_RUN/report"
```

Pass the same `--material` and `--seed` to both preparation and replay jobs for
additional fabrics/seeds. The report provides per-probe quality/runtime Pareto
frontiers and selects the fastest tested candidate passing all declared probe
limits only after the full matrix completes. The YAML acceptance limits are
explicit provisional targets, not calibrated real-cloth truth; if none passes,
the report recommends none. Rerunning analysis with revised limits requires no
simulation changes. Short smoke replays can use `--steps 12`; their shorter
horizon keeps them separate from the full five-second benchmark.

Development array `722263` and report job `722339` were stopped when the first
full candidate showed no hand-cloth contact in its scripted probe. Those records
cannot establish contact quality. The revised fixture in
`results/vbd_stability_contact_20261116_native/` initializes scripted worlds from
checkpoint pose row 18447 and centers the cloth beneath its fingertip cluster;
the checkpoint SHA and row are stored in the fixture metadata. Material and solver
physics are unchanged. A probe without observed contact in every scripted world
fails validation rather than receiving a perfect zero-penetration score.

Native 6/12-iteration checks on that same fixture each ran 120 control steps:
240 physics boundaries advanced exactly two seconds, state restore errors were
zero, and all four scripted worlds registered robot contact (peak counts 59–77).
Both checks stayed finite with no overflow. Earlier development smokes without
verified clock metadata are also excluded from solver selection.

The validated contact matrix is submitted as array `722887` on commit `e88aaa7`;
report job `722923` runs after all candidates finish, including failed cases.
It writes `results/vbd_stability_contact_20261116_native/report/stability.md`
and `stability.json`. The analysis/contact regression suite passes 31 tests.
