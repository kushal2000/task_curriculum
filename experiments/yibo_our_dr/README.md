# Yibo interface with our cloth DR

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
