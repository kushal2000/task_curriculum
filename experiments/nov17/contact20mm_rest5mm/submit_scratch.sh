#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec experiments/submit.sh --job-name=cloth_c20_r5_scratch experiments/train.sh \
  --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
  --experiment experiments/nov17/contact20mm_rest5mm/scratch.yaml \
  --capture_viewer_per_design \
  --wandb_activate --wandb_project rigidClothDR --wandb_group nov17_contact20mm_rest5mm \
  --wandb_name rigidClothDR_scratch_contact20mm_rest5mm_thickness2mm_resets "$@"
