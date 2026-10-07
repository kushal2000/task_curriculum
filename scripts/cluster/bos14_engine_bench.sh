#!/bin/bash
# PhysX vs Newton throughput on the SAME task: the upstream Play task (iiwa + SHARPA hand, one hammer
# for every env) from the pretrained policy, a few training epochs, rl_games' `fps step` read off.
#
#   sbatch --array=0-7 scripts/cluster/bos14_engine_bench.sh
#
# Array index -> (backend, envs): 0-3 PhysX, 4-7 Newton, each at 1536 / 6144 / 12288 / 24576 envs
# (multiples of 6 x 256 so the checkpoint's six SAPG blocks divide them).
#
# Each backend runs the configuration that reached parity on this task (phase3_newton_transfer.md):
# PhysX TGS 8 position iterations; Newton MJWarp elliptic, 2 substeps, njmax 8192. From 12288 envs
# Newton runs 32768 triangle pairs per env (65536 OOMs a 96 GB card at 12288 at CUDA graph capture;
# 24576 x 65536 also exceeds the int32 hashtable limit) against a measured ~26.6k/env demand.
#
# MEASURED (2026-10-05, array 157181, RTX PRO 6000; fps step = env-steps/s, mean of epochs 41-80):
#
#   envs     PhysX            Newton
#   1536      95,500   5 GB    10,600  14 GB
#   6144     239,700  10 GB    10,300  52 GB
#   12288    344,900  17 GB    10,500  75 GB   32768 pairs/env overflows once the hand grasps;
#                                              dropped contacts -> NaN std at epoch ~60
#   24576    410,600  29 GB    OOM at startup (one 15 GB alloc on top of ~82 GB)
#
# Newton is saturated from 1536 envs; PhysX still scales at 24576. Both policies manipulate
# (episode reward ~7-8.5k on both), so neither is fast because the object fell.

#SBATCH --job-name=engine_bench
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs

I="${SLURM_ARRAY_TASK_ID:-0}"
ENV_LIST=(1536 6144 12288 24576)
ENVS="${ENVS:-${ENV_LIST[$((I % 4))]}}"
if [ "$I" -lt 4 ]; then BACKEND=physx; else BACKEND=newton; fi
BACKEND="${BACKEND_OVERRIDE:-$BACKEND}"
EPOCHS="${EPOCHS:-80}"         # 80 x 16 = 1280 steps > two 600-step episodes; fps falls ~3x over the first one
HORIZON=16
MINIBATCH=$(( ENVS * HORIZON / 4 ))
BLOCK=$(( ENVS / 6 ))
NAME="0_bench_${BACKEND}_e${ENVS}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

COMMON=(
    --agent rl_games_sapg_cfg_entry_point
    --checkpoint pretrained_policy/model.pth
    --checkpoint_load_mode weights
    --single_variant
    "env.scene.num_envs=${ENVS}"
    "env.assets.handle_head_types=[hammer]"
    env.assets.num_assets_per_type=1
    agent.params.config.multi_gpu=False
    "agent.params.config.minibatch_size=${MINIBATCH}"
    "agent.params.config.central_value_config.minibatch_size=${MINIBATCH}"
    "agent.params.config.expl_coef_block_size=${BLOCK}"
    "agent.params.config.max_epochs=${EPOCHS}"
    "agent.params.config.name=${NAME}"
    agent.params.seed=1
)

export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
echo "[bench] backend=$BACKEND envs=$ENVS epochs=$EPOCHS minibatch=$MINIBATCH block=$BLOCK"
echo "[bench] host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MEMLOG="slurm_logs/engine_bench_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${I}.gpumem.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 15 >"$MEMLOG" &
SMI_PID=$!

START=$(date +%s)
if [ "$BACKEND" = physx ]; then
    .venv_isaacsim/bin/python -m isaacsimenvs.train --task Isaacsimenvs-Play-Direct-v0 --headless \
        "${COMMON[@]}"
else
    TRI=()
    if [ "$ENVS" -ge 12288 ]; then TRI=("env.newton.per_env_triangle_pairs=32768"); fi
    scripts/newton_py -m isaacsimenvs.train --task Isaacsimenvs-PlayNewton-Direct-v0 \
        "${COMMON[@]}" "${TRI[@]}"
fi
RC=$?

kill "$SMI_PID" 2>/dev/null
echo "[bench] backend=$BACKEND envs=$ENVS exit=$RC wall_s=$(( $(date +%s) - START )) peak_gpu_mib=$(cut -d, -f2 "$MEMLOG" | tr -dc '0-9\n' | sort -n | tail -1)"
