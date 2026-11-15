#!/usr/bin/env bash
# Rigid cloth (fold + unfold), Flexiv + right Wuji (identified hand): from the Play prior, 2 GPUs x 12288 envs (24576 total).
# Run from anywhere: experiments/nov15_rigid_cloth_init/submit_prior_2gpu.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

experiments/submit.sh --job-name=rigid_cloth_prior_2gpu --time=7-00:00:00 --gres=gpu:nvidia_rtx_6000_ada_generation:2 --cpus-per-task=16 --mem=200G experiments/train.sh \
    --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 \
    --experiment rigid_cloth_flexiv_wuji_full_dr --single_variant --distributed \
    --checkpoint pretrained_policy/flexiv_wuji/model.pth --checkpoint_load_mode weights \
    --capture_viewer_per_shape \
    --wandb_activate --wandb_project flexiv_wuji_cleanup --wandb_group nov15_rigid_cloth_init \
    --wandb_name rigid_cloth_prior_2gpu \
    env.scene.num_envs=12288 agent.params.config.expl_coef_block_size=2048 \
    agent.params.config.name=0_rigid_cloth_prior_2gpu
