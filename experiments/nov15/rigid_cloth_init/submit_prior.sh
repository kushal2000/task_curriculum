#!/usr/bin/env bash
# Rigid cloth (fold + unfold), Flexiv + right Wuji (identified hand), from the right-Wuji Play prior (pretrained_policy/flexiv_wuji, run 41213's best checkpoint).
# Run from the repo root: experiments/nov15/rigid_cloth_init/submit_prior.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

experiments/submit.sh --job-name=rigid_cloth_prior --time=7-00:00:00 experiments/train.sh \
    --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 \
    --experiment rigid_cloth_flexiv_wuji_full_dr --single_variant \
    --checkpoint pretrained_policy/flexiv_wuji/model.pth --checkpoint_load_mode weights \
    --capture_viewer_per_shape \
    --wandb_activate --wandb_project flexiv_wuji_cleanup --wandb_group nov15_rigid_cloth_init \
    --wandb_name rigid_cloth_prior \
    agent.params.config.name=0_rigid_cloth_prior
