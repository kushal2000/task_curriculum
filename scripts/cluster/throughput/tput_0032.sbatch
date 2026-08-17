#!/bin/bash
# Cable-task simulation throughput at 32 environments.
#
#   sbatch scripts/cluster/throughput/tput_0032.sbatch
#
# One file per environment count so each run is a fixed, readable artifact: the env count, the
# step budget and the resource request are all visible in the file that produced the number,
# rather than living in a shell variable at submit time.
#
# One GPU per job -- two throughput benchmarks sharing a device measure contention, not throughput.
#
# The GPU type is PINNED. portal has mixed hardware: an unpinned sweep landed a third of its runs
# on an RTX A6000 (~630 ms/step) and the rest on RTX 6000 Ada (~402-455 ms), which looks exactly
# like an env-count effect if you do not check the hostname. It is not: per-step cost is constant
# in env count on any given node.

#SBATCH --job-name=tput_32
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

echo "[tput] num_envs=32 host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"

scripts/newton_py -m isaacsimenvs.eval.throughput \
    --num_envs 32 \
    --steps 200 \
    --warmup 60 \
    --num_assets_per_type 1 \
    --out docs/results/tput_32.json \
    'env.assets.handle_head_types=[hammer]' \
    env.cable.thickness=0.03

echo "[tput] num_envs=32 exit=$?"
