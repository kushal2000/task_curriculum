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

`vbd_iteration_sweep.sh` compares 12, 40 and 80 VBD iterations per cloth substep
with the same branch-value base cloth, Flexiv-Wuji checkpoint, seed, resets and
32 first 10-second episodes. Material, timestep, six cloth substeps, two coupled
substeps and one coupler iteration stay fixed. The native VBD count is checked
against the config at startup.

```bash
experiments/submit.sh --array=0-2 experiments/nov16_yibo_our_dr/vbd_iteration_sweep.sh
```

Results land in `results/vbd_iterations_<array>/iterations_<12|40|80>/base.json`.
Opt-in diagnostics distinguish collision-envelope overlap (including the 8 mm
particle radius) from crossing a cached robot/table contact plane. Contact counts
are recorded: zero depth with zero records does not establish collision coverage.
An independent signed-distance query against the actual native table box measures
cloth-node penetration, including nodes without contact records. These policy-rate
measurements can miss between-step and triangle-interior crossings, and do not
measure cloth self-intersections. Higher iteration counts are a convergence test;
fold counts alone do not establish that penetration was fixed.
