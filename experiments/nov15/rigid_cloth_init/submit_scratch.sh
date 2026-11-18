#!/usr/bin/env bash
# Rigid cloth (fold + unfold), Flexiv + right Wuji (identified hand), from scratch (no checkpoint).
# Run from the repo root: experiments/nov15/rigid_cloth_init/submit_scratch.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

experiments/submit.sh --job-name=rigid_cloth_scratch --time=7-00:00:00 experiments/train.sh \
    --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 \
    --experiment rigid_cloth_flexiv_wuji_full_dr --single_variant \
    --capture_viewer_per_shape \
    --wandb_activate --wandb_project flexiv_wuji_cleanup --wandb_group nov15_rigid_cloth_init \
    --wandb_name rigid_cloth_scratch \
    agent.params.config.name=0_rigid_cloth_scratch
