#!/bin/bash
# Cloth-fold simulation throughput at 32 environments.
#
#   sbatch scripts/cluster/throughput_cloth/tput_0032.sbatch
#
# One file per environment count, so the env count, step budget and resource request are visible in
# the artifact that produced the number rather than in a submit-time shell variable.
#
# GPU TYPE IS PINNED. portal has mixed hardware, and an unpinned cable sweep landed a third of its
# runs on an RTX A6000 (~630 ms/step) against RTX 6000 Ada (~402-455 ms) -- which reads exactly like
# an env-count effect until you check the hostname. Per-step cost is constant in env count on a
# given node.
#
# Contact-buffer overflow ABORTS the run (isaacsimenvs.newton.contact_guard). That is deliberate:
# dropped contacts are work not done, so an overflowing scene reports *higher* throughput while
# also being physically wrong. A missing JSON here means the buffers were too small, not that the
# machine was busy.

#SBATCH --job-name=ctput_32
#SBATCH --partition=portal
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail
cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs docs/results

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

echo "[ctput] num_envs=32 host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"

scripts/newton_py -m isaacsimenvs.eval.throughput \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --num_envs 32 \
    --steps 200 \
    --warmup 60 \
    --num_assets_per_type 1 \
    --out docs/results/ctput_32.json \
    'env.assets.handle_head_types=[hammer]'

echo "[ctput] num_envs=32 exit=$?"
