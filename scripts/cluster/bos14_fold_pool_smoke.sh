#!/bin/bash
# Smoke test for the stage-2 fold-shape pool: build the env, run its per-shape startup proofs, train a
# few epochs so resets draw shapes and the unused ones are parked every step.
#
#   sbatch scripts/cluster/bos14_fold_pool_smoke.sh
#   ENVS=768 EPOCHS=45 sbatch scripts/cluster/bos14_fold_pool_smoke.sh     # resets + throughput (~11k steps/s)
#
# gpu_a100 (RTX PRO 6000): the mesh contact buffer OOMs a 32 GB 5090 at 768 envs; see the train script.
#
# Passing means: "[fold_pool] <shape>: placed within ..." for all five shapes, no "Triangle pair
# buffer overflowed", and rl_games epochs completing. It does NOT test the contact buffer at a fold --
# a few epochs never fold (see the rigid-cloth notes: overflow appears only once folding succeeds).

#SBATCH --job-name=fp_smoke
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --exclude=bos14-node-092,bos14-node-104,bos14-node-105
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs

ENVS="${ENVS:-96}"
EPOCHS="${EPOCHS:-5}"
BLOCK=$((ENVS / 6))             # num_blocks is baked into the checkpoint at 6
MINIBATCH=$((ENVS * 32))        # one minibatch per epoch at this scale
NAME="0_fp_smoke_${SLURM_JOB_ID:-local}"

export PYTHONUNBUFFERED=1
echo "[fp_smoke] envs=$ENVS epochs=$EPOCHS host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

START=$(date +%s)
scripts/newton_py -m isaacsimenvs.train \
    --task Isaacsimenvs-FoldPool-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    --checkpoint pretrained_policy/model.pth \
    --checkpoint_load_mode weights \
    --single_variant \
    env.cloth.nan_policy=reset \
    "env.scene.num_envs=${ENVS}" \
    agent.params.config.multi_gpu=False \
    "agent.params.config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.central_value_config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.expl_coef_block_size=${BLOCK}" \
    "agent.params.config.max_epochs=${EPOCHS}" \
    "agent.params.config.name=${NAME}" \
    agent.params.seed=1
RC=$?
echo "[fp_smoke] exit=$RC wall_s=$(( $(date +%s) - START ))"
