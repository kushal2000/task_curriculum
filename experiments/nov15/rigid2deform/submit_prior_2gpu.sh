#!/usr/bin/env bash
# Rigid2Deform single-hinge rigid sheet (fold only, 4096 designs), Flexiv + right Wuji (identified hand):
# from the Play prior, 2 GPUs x 12288 envs (24576 total; SAPG blocks of 2048 = half the designs each).
# Run from anywhere: experiments/nov15/rigid2deform/submit_prior_2gpu.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

experiments/submit.sh --job-name=rigid_body_prior_2gpu --time=7-00:00:00 --gres=gpu:nvidia_rtx_6000_ada_generation:2 --cpus-per-task=16 --mem=200G experiments/train.sh \
    --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
    --experiment rigid_body_cloth --single_variant --distributed \
    --checkpoint pretrained_policy/flexiv_wuji/model.pth --checkpoint_load_mode weights \
    --wandb_activate --wandb_project flexiv_wuji_cleanup --wandb_group nov15_rigid2deform \
    --wandb_name rigid_body_prior_2gpu \
    env.scene.num_envs=12288 agent.params.config.expl_coef_block_size=2048 \
    agent.params.config.name=0_rigid_body_prior_2gpu
