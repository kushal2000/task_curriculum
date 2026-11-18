#!/usr/bin/env bash
# Rigid cloth (fold + unfold), Flexiv + right Wuji (identified hand): from the Play prior, 18432 envs on 1 GPU (24576 ran out of GPU memory).
# Run from anywhere: experiments/nov15/rigid_cloth_init/submit_prior_18k.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

experiments/submit.sh --job-name=rigid_cloth_prior_18k --time=7-00:00:00  experiments/train.sh \
    --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 \
    --experiment rigid_cloth_flexiv_wuji_full_dr --single_variant  \
    --checkpoint pretrained_policy/flexiv_wuji/model.pth --checkpoint_load_mode weights \
    --capture_viewer_per_shape \
    --wandb_activate --wandb_project flexiv_wuji_cleanup --wandb_group nov15_rigid_cloth_init \
    --wandb_name rigid_cloth_prior_18k \
    env.scene.num_envs=18432 agent.params.config.expl_coef_block_size=3072 \
    agent.params.config.name=0_rigid_cloth_prior_18k
