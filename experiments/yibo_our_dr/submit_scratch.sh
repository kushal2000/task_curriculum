#!/usr/bin/env bash
# Train the same environment and cloth DR from scratch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

exec experiments/submit.sh --job-name=yibo_our_dr_scratch experiments/train.sh \
    --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
    --experiment experiments/yibo_our_dr/scratch.yaml \
    --capture_viewer_per_design \
    --wandb_activate --wandb_project rigidClothDR --wandb_group yibo_our_dr \
    --wandb_name rigidClothDR_scratch "$@"
