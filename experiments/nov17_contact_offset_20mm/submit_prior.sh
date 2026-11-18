#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec experiments/submit.sh --job-name=cloth_contact20mm_prior experiments/train.sh \
  --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
  --experiment experiments/nov17_contact_offset_20mm/prior.yaml \
  --capture_viewer_per_design \
  --wandb_activate --wandb_project rigidClothDR --wandb_group nov17_contact_offset_20mm \
  --wandb_name rigidClothDR_prior_contact20mm_thickness2mm_resets "$@"
