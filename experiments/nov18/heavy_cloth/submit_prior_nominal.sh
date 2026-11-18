#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec experiments/submit.sh --job-name=cloth_heavy_prior_nominal experiments/train.sh \
  --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
  --experiment experiments/nov18/heavy_cloth/prior_nominal.yaml \
  --capture_viewer_per_design \
  --wandb_activate --wandb_project rigidClothHeavy --wandb_group nov18_heavy_cloth \
  --wandb_name heavy_prior_nominal_single_hinge "$@"
