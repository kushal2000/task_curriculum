#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec experiments/submit.sh --job-name=yibo_prior_finetune experiments/train.sh \
  --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
  --experiment experiments/nov17_yibo_prior_finetune/prior.yaml \
  --capture_viewer_per_design \
  --wandb_activate --wandb_project rigidClothDR --wandb_group nov17_yibo_prior_finetune \
  --wandb_name rigidClothDR_yibo_prior_finetune "$@"
