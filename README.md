# task_curriculum

Cloth folding and unfolding with a Flexiv Rizon 4s + right Wuji Hand 2: pretrain on rigid objects,
train on rigid cloth, then move to deformable cloth.

## Setup

```bash
docs/installation/install_isaacsim.sh      # .venv/isaacsim   (PhysX: Play, rigid cloth)
docs/installation/install_isaaclab3.sh     # .venv/isaaclab3  (Newton: deformable cloth)
```

Details, and the Newton USD-cache step, in [`docs/installation/`](docs/installation/README.md).
The Play prior goes in `pretrained_policy/flexiv_wuji/{model.pth,config.yaml}` (gitignored).

## The three experiments

Each is one config in `experiments/configs/`: Flexiv + right Wuji (the identified hand), every
randomization on, no epoch limit. Lighter variants are the overrides listed in each config's header.

```bash
# 1. Play pretraining (rigid objects)
experiments/submit.sh --time=7-00:00:00 --cpus-per-task=4 --mem=80G experiments/train.sh \
    --task Isaacsimenvs-PlayFlexivWujiRightReal-Direct-v0 --experiment play_flexiv_wuji_full_dr \
    --checkpoint pretrained_policy/flexiv_wuji/model.pth --checkpoint_load_mode weights \
    agent.params.config.name=0_play

# 2. Rigid cloth: fold + unfold, from the Play prior
experiments/submit.sh --time=7-00:00:00 experiments/train.sh \
    --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 --experiment rigid_cloth_flexiv_wuji_full_dr \
    --checkpoint pretrained_policy/flexiv_wuji/model.pth --checkpoint_load_mode weights --single_variant \
    agent.params.config.name=0_rigid_cloth

# 3. Deformable cloth (VBD sheet), from a rigid-cloth checkpoint
experiments/submit.sh --time=2-00:00:00 experiments/train.sh \
    --task Isaacsimenvs-ClothFlexivWujiRightReal-Direct-v0 --experiment deformable_cloth_flexiv_wuji_full_dr \
    --checkpoint <rigid-cloth run>/nn/0_rigid_cloth.pth --checkpoint_load_mode weights --single_variant \
    agent.params.config.name=0_deformable_cloth
```

Each run gets one folder, `debug_outputs/train_logs/<month>/<date>_<name>_<job id>/`: checkpoints
(`nn/<name>.pth` is the best), config, wandb and the Slurm log. Run names must start with `0_`.
Kit exits 0 even when training crashed, so check the log.

Evaluate a checkpoint: `experiments/submit.sh experiments/eval.sh --experiment_dir experiments/<day> --task <id> --checkpoint <pth>`.

Tests: `.venv/isaaclab3/bin/python -m pytest`.
