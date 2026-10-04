#!/bin/bash
# Retrain box3-mid and box2-surface on the cloth-matched chain, with and without physics DR.
#
#   sbatch --array=0-11 scripts/cluster/bos14_rigid_cloth_retrain.sh
#
# Index = group * 6 + arm * 3 + (seed - 1):
#
#   group 0  "fix"    the three fixes only: reset heading applied (the adapter), contact surface at
#                     the cloth's height (`contact_margin`), cloth friction per contact pair
#                     (`friction_priority`). Everything the first round lacked, nothing else.
#   group 1  "fixdr"  the same, plus per-env physics randomisation re-drawn at every reset:
#                     slat and fingertip friction, hinge damping and stiffness, contact height, mass.
#
#   arm 0 box3-mid, arm 1 box2-surface; seeds 1-3.
#
# Everything else is held to the first round's 1536-env arm (array 155121) so the two rounds are
# comparable: pretrained prior, SAPG, 1536 envs, horizon 32, minibatch 24576, 2035 epochs = 1.0e8
# env-steps. Only the GPU differs -- an RTX 5090 here, because 12 of them are free and a rigid chain
# at 1536 envs peaked at 17 GB -- so wall-clock is not comparable with 155121, only with itself.

#SBATCH --job-name=rc2_train
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs

ARMS=(box3-mid box2-surface)
GROUPS_=(fix fixdr)
I="${RC_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
GROUP="${GROUPS_[$((I / 6))]}"
ARM="${ARMS[$(((I % 6) / 3))]}"
SEED=$(( (I % 3) + 1 ))

ENVS="${ENVS:-1536}"
EPOCHS="${EPOCHS:-2035}"        # 2035 x 1536 x 32 = 1.00e8 env-steps
MINIBATCH=24576
BLOCK=$((ENVS / 6))             # num_blocks is baked into the checkpoint at 6
# 65536, the cloth's own sizing, not RigidCloth.yaml's 524288. The larger buffer exists for the
# 16-17 slat chains (~119k pairs/env); at 1536 envs it is one 4.8 GB allocation, which does not fit a
# 32 GB RTX 5090 next to everything else (measured: OOM at solver init). box3-mid and box2-surface
# are 3 and 2 bodies and logged ZERO overflow warnings at 65536 in both earlier rounds.
TRI_PAIRS="${TRI_PAIRS:-65536}"
JOB_TAG="${RC_JOB_ID:-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
NAME="0_rc2_${ARM}_${GROUP}_s${SEED}_${JOB_TAG}"

DR_ARGS=()
[ "$GROUP" = "fixdr" ] && DR_ARGS=("env.rigid_cloth.randomization.enabled=true")

export PYTHONUNBUFFERED=1
echo "[train] arm=$ARM group=$GROUP seed=$SEED envs=$ENVS epochs=$EPOCHS block=$BLOCK"
echo "[train] name=$NAME host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MEMLOG="slurm_logs/${SLURM_JOB_NAME}_${JOB_TAG}_${I}.gpumem.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 \
    >"$MEMLOG" &
SMI_PID=$!

START=$(date +%s)
scripts/newton_py -m isaacsimenvs.train \
    --task Isaacsimenvs-RigidCloth-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    --checkpoint pretrained_policy/model.pth \
    --checkpoint_load_mode weights \
    --single_variant \
    "env.rigid_cloth.variant=${ARM}" \
    "${DR_ARGS[@]}" \
    env.cloth.nan_policy=reset \
    "env.scene.num_envs=${ENVS}" \
    "env.newton.per_env_triangle_pairs=${TRI_PAIRS}" \
    agent.params.config.multi_gpu=False \
    "agent.params.config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.central_value_config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.expl_coef_block_size=${BLOCK}" \
    "agent.params.config.max_epochs=${EPOCHS}" \
    "agent.params.config.name=${NAME}" \
    "agent.params.seed=${SEED}"
RC=$?

kill "$SMI_PID" 2>/dev/null
WALL=$(( $(date +%s) - START ))
echo "[train] arm=$ARM group=$GROUP seed=$SEED exit=$RC wall_s=$WALL"
if [ "$RC" -eq 0 ]; then
    echo "[train] env_steps_per_s=$(python3 -c "print(round(${EPOCHS}*${ENVS}*32/max(${WALL},1), 1))")"
else
    echo "[train] env_steps_per_s=n/a (exit $RC)"
fi
echo "[train] peak_gpu_mib=$(cut -d, -f2 "$MEMLOG" | tr -dc '0-9\n' | sort -n | tail -1)"
find outputs -type d -name "$NAME" -newermt "-25 hours" | tee "slurm_logs/${NAME}.ckptdir"
