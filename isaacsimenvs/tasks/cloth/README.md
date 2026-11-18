# Cloth folding tasks

## Environments

| Task id (`Isaacsimenvs-<name>-Direct-v0`) | Simulator | Object | Code |
|---|---|---|---|
| `Cloth` | Newton | VBD particle cloth | `cloth_env.py`, `cloth_env_cfg.py`, `cloth_cfg.py` |
| `FoldPoolPhysx` | PhysX | pool of 7 hinged-plate fold shapes | `physx_envs.py`, `physx_env_cfg.py`, `fold_pool_cfg.py` |


## Training

All launchers are Slurm batch scripts. Run them from the repo root on bos14. Array index = seed - 1.

### On PhysX (rigid plates)

`scripts/cluster/bos14_physx_cloth_train.sh`: 1 GPU per seed.

```bash
REPO=$PWD EXPERIMENT=flat7RW_base TASK=pool ROBOT=FlexivWujiRightHLRL RUN_TAG=flat7RW_base WANDB=1 \
PRIOR=pretrained_policy/flexiv_right_wuji/nov8_dr_friction_0p5.pth \
sbatch --array=0-2 scripts/cluster/bos14_physx_cloth_train.sh
```

- `TASK=pool` -> `FoldPoolPhysx<ROBOT>`; `TASK=chain VARIANT=box3-mid` -> `RigidClothPhysx<ROBOT>`.
- With `EXPERIMENT` set, the recipe supplies env count, epochs and minibatch. The launcher's
  `ENVS` / `EPOCHS` are ignored; override them through `EXTRA_ARGS`.
- `LOAD_MODE=weights` (default) loads model + normalizers only. `LOAD_MODE=resume` also loads the
  optimizer and epoch counter; `max_epochs` is then the total.
- `PRIORS="<s1> <s2> <s3>"` gives each seed its own start checkpoint.
- Output: `outputs/<date>/<time>/0_px<pool|chain>_<RUN_TAG>_s<seed>_<job>/nn/`. Files:
  `snap_*_ep_<N>.pth` (every `snapshot_every` epochs), `last_*`, `best_*`.

### On VBD cloth (Newton)

`scripts/cluster/bos14_cloth_vbd_train.sh`: `NPS` nodes per seed (default 4, multi-node SAPG).

```bash
REPO=$PWD EXPERIMENT=vbdRW_sharpa ROBOT=FlexivWujiRightSharpaGain RUN_TAG=vbdRW_sharpa \
PRIOR=pretrained_policy/flexiv_right_wuji/nov8_dr_friction_0p5.pth \
EXTRA_ARGS=env.robot_self_collision=false \
sbatch --array=0-11 scripts/cluster/bos14_cloth_vbd_train.sh     # 3 seeds x 4 nodes
```

- Defaults: 1488 envs per rank, 3750 epochs.
- Resume: `CKPTS="<s1> <s2> <s3>" LOAD_MODE=resume`. Set `RDZV_TIMEOUT=7200` if array tasks may
  start far apart.
- Robot self-collision must be off for the Flexiv. With it on, MJWarp is ~26x slower.
- Output name: `0_pxvbd_<RUN_TAG>_s<seed>_<job>`.

## Evaluation

The metric is held folds on the VBD cloth, 32 episodes x 2 eval seeds per cell, in 4 cells:

- start: pinned (one fixed start state) or rand (`--randomize`)
- keypoint cutoff: 4 cm or 1.5 cm

The key metrics are 1.5 cm pinned and 1.5 cm rand.

### One checkpoint list

`scripts/cluster/bos14_px_eval_trend.sh` takes a list file, one checkpoint per line:

```
<tag> <seed> <checkpoint .pth>
<tag> <seed> <run name> <best | last | ep:N>      # resolved when the task runs
```

```bash
LIST=my_list.txt ROBOT=FlexivWujiRightHLRL \
EVAL_EXTRA="env.cloth.start_height=0.20 env.robot_self_collision=false env.newton.njmax=32768" \
sbatch --array=0-$(( 2 * $(wc -l < my_list.txt) - 1 )) scripts/cluster/bos14_px_eval_trend.sh
```

- Each array task is one line x one start: index = 2 x line + start.
- Each task runs `isaacsimenvs.eval.episodes` on `Isaacsimenvs-Cloth<ROBOT>-Direct-v0` for both
  cutoffs and both eval seeds.
- Output: `docs/results/px_matrix/<tag>_s<seed>__on_vbd__<pinned|rand>__<4cm|1p5cm>_s<es>.json`.
- `EVAL_SEEDS="3 4 5 6 7 8"` re-tests a picked checkpoint on fresh seeds.
- `EVAL_EXTRA` must match the training geometry. Example: the right-Wuji tasks need
  `start_height=0.20`.

