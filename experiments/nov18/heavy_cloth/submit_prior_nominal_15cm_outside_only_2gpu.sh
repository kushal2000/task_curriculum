#!/usr/bin/env bash
# 2 GPUs x 12288 envs (24576 total), one torchrun rank per GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec experiments/submit.sh --job-name=cloth_heavy_15cm_outside_2gpu --time=7-00:00:00 \
  --gres=gpu:nvidia_rtx_6000_ada_generation:2 --cpus-per-task=16 --mem=200G experiments/train.sh \
  --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
  --experiment experiments/nov18/heavy_cloth/prior_nominal_15cm_outside_only.yaml \
  --distributed --capture_viewer_per_design \
  --wandb_activate --wandb_project rigidClothHeavy --wandb_group nov18_heavy_cloth \
  --wandb_name heavy_prior_nominal_15cm_outside_only_2gpu \
  env.scene.num_envs=12288 "$@"
