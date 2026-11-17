# Yibo interface with our cloth DR

Completed evaluation findings: [transfer analysis](analysis/README.md),
[rigid ablations](analysis/rigid.md), and [VBD fabrics](analysis/fabrics.md).

## Active training and viewer verification

W&B project: [rigidClothDR](https://wandb.ai/kk837/rigidClothDR).
Current jobs are [fine-tune 709852](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_finetune_20261116_192340)
and [scratch 709853](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_scratch_20261116_192301).
Both passed large-scene initialization and reached optimizer updates.

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

- `finetune.yaml` initializes from
  `pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth`, loading network
  weights and normalization statistics. It starts a fresh optimizer, epoch/frame
  counters and rollout; it does not resume the old training run.
- `scratch.yaml` starts with a newly initialized network and no checkpoint.

From the repository root, submit either recipe using the existing training launcher:

```bash
experiments/yibo_our_dr/submit_finetune.sh
experiments/yibo_our_dr/submit_scratch.sh
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

The inherited default is 24,576 environments on one GPU with six SAPG blocks of
4,096 environments. Changing the number of environments requires changing
`agent.params.config.expl_coef_block_size` to retain six blocks for the checkpoint.
Submitting these commands launches training; creating the recipes does not submit jobs.

## Controlled rigid-cloth evaluation

```bash
experiments/yibo_our_dr/submit_rigid_eval.sh
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
