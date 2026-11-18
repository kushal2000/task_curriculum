#!/usr/bin/env bash
# Train the same environment and cloth DR from scratch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec experiments/submit.sh --job-name=yibo_our_dr_scratch experiments/train.sh \
    --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
    --experiment experiments/nov16/yibo_our_dr/scratch.yaml \
    --capture_viewer_per_design \
    --wandb_activate --wandb_project rigidClothDR --wandb_group yibo_our_dr \
    --wandb_name rigidClothDR_scratch "$@"
