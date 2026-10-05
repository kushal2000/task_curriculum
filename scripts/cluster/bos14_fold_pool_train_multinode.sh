#!/bin/bash
# Stage 2 fold pool, ONE run on several GPUs: one bos14 array task per node, one GPU each.
#
#   SEED=1 PRIOR=<run>.pth RUN_TAG=box3 sbatch --array=0-3 scripts/cluster/bos14_fold_pool_train_multinode.sh
#
# The array size is the node count. Launch mechanics are bos14_cloth_prior_vs_scratch.sh's (no
# srun, no ssh: task 0 publishes its IP in a shared file, torchrun STATIC rendezvous, gloo on the
# node's own interface); everything else is bos14_fold_pool_train.sh's, per rank.
#
# Each rank runs ENVS envs (default 1536, the single-GPU run's count), so N nodes collect N times
# the data per epoch. The optimisation is kept the single-GPU run's: rl_games all-reduces the
# gradient over ranks, so one optimizer step sees N x MINIBATCH samples, and MINIBATCH defaults to
# 24576 / N -- the same 24576 per step, N times as many steps per epoch. EPOCHS defaults to the
# single-GPU 1e8 env-steps divided by N.
#
# rl_games prints `fps total` on rank 0 already multiplied by the world size: the run's rate.
#
# MEASURED (2026-10-05, 40 epochs from the folding box3 s3 policy, env-steps/s, epochs 11-40):
#
#                        1 GPU    2 nodes   4 nodes
#   MINIBATCH=24576/N    5,070    7,620     9,410     same optimisation as one GPU
#   MINIBATCH=24576        -      9,290    15,670     N x the samples per optimizer step
#
# Simulation scales ~linearly (4 nodes step 21-22k); what does not is the update. Every minibatch
# all-reduces the gradient over gloo on ethernet, and 24576/N makes N times as many of them: on 4
# nodes the update took ~12 s of a ~21 s epoch. MINIBATCH=24576 keeps the all-reduce count at the
# single-GPU run's and is stage 1's multi-node cloth setting (4 x 24576 = 98304 per step).

#SBATCH --job-name=fp_mn
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs

NNODES=$SLURM_ARRAY_TASK_COUNT
NODE_RANK=$SLURM_ARRAY_TASK_ID
JOB=$SLURM_ARRAY_JOB_ID
SEED="${SEED:-1}"
ENVS="${ENVS:-1536}"
EPOCHS="${EPOCHS:-$(( (2035 + NNODES - 1) / NNODES ))}"
MINIBATCH="${MINIBATCH:-$(( 24576 / NNODES ))}"
BLOCK=$((ENVS / 6))             # num_blocks is baked into the checkpoint at 6
PRIOR="${PRIOR:-pretrained_policy/model.pth}"
NAME="0_fp_${RUN_TAG:-pool}_s${SEED}_${JOB}"
RDZV_FILE="slurm_logs/.rdzv_${JOB}"
WANDB_ARGS=()
if [ "${WANDB:-0}" = 1 ]; then
    WANDB_ARGS=(--wandb_activate --wandb_project fold_pool --wandb_group "${RUN_TAG:-pool}_${JOB}" --wandb_name "$NAME")
fi
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
export PYTHONUNBUFFERED=1

MY_IP="$(hostname -I | awk '{print $1}')"
IFACE="$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ "^"ip"/" {print $2; exit}')"
export GLOO_SOCKET_IFNAME="$IFACE" TP_SOCKET_IFNAME="$IFACE"

if [ "$NODE_RANK" = "0" ]; then
    echo "$MY_IP" >"$RDZV_FILE"
    HEAD_IP="$MY_IP"
else
    for _ in $(seq 3600); do [ -s "$RDZV_FILE" ] && break; sleep 1; done
    HEAD_IP="$(cat "$RDZV_FILE" 2>/dev/null)"
    [ -n "$HEAD_IP" ] || { echo "[fp_mn] no rendezvous file after 60 min"; exit 1; }
fi

echo "[fp_mn] name=$NAME node=$NODE_RANK/$NNODES host=$(hostname -s) ip=$MY_IP head=$HEAD_IP"
echo "[fp_mn] envs/rank=$ENVS minibatch/rank=$MINIBATCH epochs=$EPOCHS env-steps=$((NNODES * ENVS * 32 * EPOCHS))"
echo "[fp_mn] prior=$PRIOR wandb=${WANDB:-0} extra=${EXTRA_ARGS:-none}"
echo "[fp_mn] commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"

START=$(date +%s)
scripts/newton_py -m torch.distributed.run \
    --nnodes="$NNODES" --nproc_per_node=1 --node_rank="$NODE_RANK" \
    --master_addr="$HEAD_IP" --master_port=$((29500 + JOB % 1000)) \
    -m isaacsimenvs.train \
    --task Isaacsimenvs-FoldPool-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    --checkpoint "$PRIOR" \
    --checkpoint_load_mode weights \
    --distributed \
    --single_variant \
    "${WANDB_ARGS[@]}" \
    env.cloth.nan_policy=reset \
    "env.scene.num_envs=${ENVS}" \
    "agent.params.config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.central_value_config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.expl_coef_block_size=${BLOCK}" \
    "agent.params.config.max_epochs=${EPOCHS}" \
    "agent.params.config.name=${NAME}" \
    "agent.params.seed=${SEED}" \
    "${EXTRA[@]}"
echo "[fp_mn] node=$NODE_RANK exit=$? wall_s=$(( $(date +%s) - START ))"
