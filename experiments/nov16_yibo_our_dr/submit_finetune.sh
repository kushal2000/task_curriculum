#!/usr/bin/env bash
# Train with our cloth DR, initialized from Yibo's checkpoint with fresh training state.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec experiments/submit.sh --job-name=yibo_our_dr_finetune experiments/train.sh \
    --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
    --experiment experiments/nov16_yibo_our_dr/finetune.yaml \
    --capture_viewer_per_design \
    --wandb_activate --wandb_project rigidClothDR --wandb_group yibo_our_dr \
    --wandb_name rigidClothDR_finetune "$@"
