#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec experiments/submit.sh --job-name=cloth_curr_scratch experiments/train.sh \
  --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
  --experiment experiments/nov17_cloth_resets/scratch.yaml \
  --capture_viewer_per_design \
  --wandb_activate --wandb_project rigidClothDR --wandb_group nov17_cloth_resets \
  --wandb_name rigidClothDR_scratch_20to2mm_curriculum_resets "$@"
