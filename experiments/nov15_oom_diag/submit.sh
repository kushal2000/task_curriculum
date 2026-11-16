#!/usr/bin/env bash
# Why does GPU memory grow until OOM at 24576 envs? (rigid_cloth_prior 658160 / scratch 658161 died at
# epochs 29 / 21: 39 GiB at the start of training, then up to 48 GiB.) Four 4-hour runs, identical
# except for one change each; experiments/<this>/README.md reads their gpumem logs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

run() {  # name, extra sbatch options, extra train args
    experiments/submit.sh --job-name="oom_$1" --time=04:00:00 $2 experiments/train.sh \
        --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 \
        --experiment rigid_cloth_flexiv_wuji_full_dr --single_variant \
        --checkpoint pretrained_policy/flexiv_wuji/model.pth --checkpoint_load_mode weights \
        --wandb_activate --wandb_project flexiv_wuji_cleanup --wandb_group nov15_oom_diag --wandb_name "oom_$1" \
        agent.params.config.name="0_oom_$1" $3
}
run asis      ""                                                          "--capture_viewer_per_shape"
run noviewer  "--export=ALL,VIEWER=0"                                     ""
run foldonly  ""                                                          "--capture_viewer_per_shape env.task.mode=fold env.fold_pool.init_fold_prob=0.0"
run expandseg "--export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" "--capture_viewer_per_shape"
