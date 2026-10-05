#!/bin/bash
# Stage 2: train ONE policy on the half fold across the whole fold-shape pool, a shape drawn per env
# per episode (Isaacsimenvs-FoldPool-Direct-v0).
#
#   sbatch --array=0-2 scripts/cluster/bos14_fold_pool_train.sh          # seeds 1, 2, 3
#   SHAPES='[box2-mid,cross4-mid]' sbatch --array=0 scripts/cluster/bos14_fold_pool_train.sh
#   PRIOR=<run>.pth RUN_TAG=box3 sbatch --array=0-2 scripts/cluster/bos14_fold_pool_train.sh
#
# PRIOR is the starting weights (default: the generic tool-use prior); RUN_TAG names the run
# 0_fp_<RUN_TAG>_s<seed>_<job> so the eval launcher can find it (pass it the same RUN_TAG).
# WANDB=1 also logs to wandb (project fold_pool); EXTRA_ARGS are appended hydra overrides.
#
# For more than one GPU per run see bos14_fold_pool_train_multinode.sh.
#
# More envs on the one GPU buys little: 1536 / 2304 / 3072 / 3840 envs ran 5,070 / 5,240 / 5,920 /
# 6,240 env-steps/s (folding policy; peak 69 / 74 / 78 / 83 GB). Above ~2048 envs the triangle-pair
# budget must be held at 1536 x 524288 in total (EXTRA_ARGS=env.newton.per_env_triangle_pairs=
# $((805306368 / ENVS))): Newton's contact-reduction hashtable is 7 x the next power of two of a
# quarter of it, and at 2304 x 524288 that exceeds Warp's int32 array limit at startup. No
# overflow warning in those 40-epoch runs.
#
# Same learner, prior, reward, horizon and 1e8 env-step budget as the stage-1 rigid-cloth runs
# (bos14_rigid_cloth_train.sh), so a pool policy is comparable to a single-chain one.
#
# WHY gpu_a100 (RTX PRO 6000, 96 GB) AND NOT gpu_5090
#
# Newton sizes its mesh contact buffer as `per_env_triangle_pairs` x envs x 12 bytes, and the pool's
# plates are convex MESHES. At the inherited 524288 that is 4.8 GB at 768 envs, which OOMed a 32 GB
# RTX 5090 at startup (job 156598); 1536 envs needs 9.7 GB of it. 524288 is kept deliberately: the
# stage-1 notes record that the buffer's real peak appears only once a policy folds, and was
# under-read every time it was sized from a less successful one. Size it down only from a
# measurement taken with a policy that folds.

#SBATCH --job-name=fp_train
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs

I="${FP_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
SEED=$(( I + 1 ))
ENVS="${ENVS:-1536}"
EPOCHS="${EPOCHS:-2035}"        # 2035 x 1536 x 32 = 1.00e8 env-steps
MINIBATCH="${MINIBATCH:-24576}"
BLOCK=$((ENVS / 6))             # num_blocks is baked into the checkpoint at 6
JOB_TAG="${FP_JOB_ID:-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
PRIOR="${PRIOR:-pretrained_policy/model.pth}"
NAME="0_fp_${RUN_TAG:-pool}_s${SEED}_${JOB_TAG}"
SHAPE_ARGS=()
if [ -n "${SHAPES:-}" ]; then
    SHAPE_ARGS=("env.fold_pool.shapes=${SHAPES}")
fi
WANDB_ARGS=()
if [ "${WANDB:-0}" = 1 ]; then
    WANDB_ARGS=(--wandb_activate --wandb_project fold_pool --wandb_group "${RUN_TAG:-pool}_${JOB_TAG}" --wandb_name "$NAME")
fi
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"

export PYTHONUNBUFFERED=1
echo "[fp_train] seed=$SEED envs=$ENVS epochs=$EPOCHS shapes=${SHAPES:-all} name=$NAME"
echo "[fp_train] prior=$PRIOR wandb=${WANDB:-0} extra=${EXTRA_ARGS:-none}"
echo "[fp_train] host=$(hostname -s) commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MEMLOG="slurm_logs/${SLURM_JOB_NAME:-fp_train}_${JOB_TAG}_${I}.gpumem.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 >"$MEMLOG" &
SMI_PID=$!

START=$(date +%s)
scripts/newton_py -m isaacsimenvs.train \
    --task Isaacsimenvs-FoldPool-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    --checkpoint "$PRIOR" \
    --checkpoint_load_mode weights \
    --single_variant \
    "${WANDB_ARGS[@]}" \
    "${SHAPE_ARGS[@]}" \
    env.cloth.nan_policy=reset \
    "env.scene.num_envs=${ENVS}" \
    agent.params.config.multi_gpu=False \
    "agent.params.config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.central_value_config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.expl_coef_block_size=${BLOCK}" \
    "agent.params.config.max_epochs=${EPOCHS}" \
    "agent.params.config.name=${NAME}" \
    "agent.params.seed=${SEED}" \
    "${EXTRA[@]}"
RC=$?

kill "$SMI_PID" 2>/dev/null
WALL=$(( $(date +%s) - START ))
echo "[fp_train] seed=$SEED exit=$RC wall_s=$WALL peak_gpu_mib=$(cut -d, -f2 "$MEMLOG" | tr -dc '0-9\n' | sort -n | tail -1)"
find outputs -type d -name "$NAME" -newermt "-25 hours" | tee "slurm_logs/${NAME}.ckptdir"
