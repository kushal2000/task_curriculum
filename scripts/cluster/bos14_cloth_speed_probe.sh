#!/bin/bash
# Single-GPU training throughput + memory for the cloth SAPG run on bos14 (1 RTX PRO 6000 per node).
#
#   sbatch --array=0-4 scripts/cluster/bos14_cloth_speed_probe.sh
#
# Runs the REAL train.py for a few epochs rather than eval/throughput.py, so the figure includes the
# PPO update and the number is what a 24 h run will actually get. No wandb, no viewer capture.
#
# Each array index is one configuration. num_blocks must stay 6 (the checkpoint's sigma is (6, 29)),
# so envs is always 6 x block. Minibatch is held at 24576 so the per-optimizer-step batch is the same
# everywhere; bigger env counts just take more steps per epoch. Index 4 is the from-scratch arm, to
# check that code path runs before committing GPUs to it.
#
# GPU memory is sampled every 10 s into slurm_logs/<job>_<task>.gpumem.csv.

#SBATCH --job-name=cloth_speed
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum

#          envs  block  init
CONFIGS=(
    "768   128   pretrained"
    "1536  256   pretrained"
    "2304  384   pretrained"
    "3072  512   pretrained"
    "1536  256   scratch"
)
read -r ENVS BLOCK INIT <<<"${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MINIBATCH=24576
EPOCHS="${EPOCHS:-30}"
NAME="0_speed_${ENVS}_${INIT}_${SLURM_ARRAY_JOB_ID}"

export PYTHONUNBUFFERED=1

CKPT_ARGS=()
if [ "$INIT" = "pretrained" ]; then
    CKPT_ARGS=(--checkpoint pretrained_policy/model.pth --checkpoint_load_mode weights)
fi

MEMLOG="slurm_logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpumem.csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader -l 10 >"$MEMLOG" &
SMI_PID=$!

echo "[speed] envs=$ENVS block=$BLOCK minibatch=$MINIBATCH init=$INIT epochs=$EPOCHS host=$(hostname)"
echo "[speed] commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
START=$(date +%s)

scripts/newton_py -m isaacsimenvs.train \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    "${CKPT_ARGS[@]}" \
    --single_variant \
    env.cloth.nan_policy=reset \
    "env.scene.num_envs=${ENVS}" \
    agent.params.config.multi_gpu=False \
    "agent.params.config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.central_value_config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.expl_coef_block_size=${BLOCK}" \
    "agent.params.config.max_epochs=${EPOCHS}" \
    "agent.params.config.name=${NAME}"
RC=$?

kill "$SMI_PID" 2>/dev/null
echo "[speed] exit=$RC wall_s=$(( $(date +%s) - START )) peak_gpu_mib=$(cut -d, -f2 "$MEMLOG" | tr -dc '0-9\n' | sort -n | tail -1)"
