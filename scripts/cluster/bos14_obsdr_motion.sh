#!/bin/bash
# Obs-DR motion diagnosis (scripts/flexiv_wuji/probe_obs_dr_motion.py): one array task per condition,
# each runs every checkpoint in $CKPTS on the PhysX rigid pool.
#   CKPTS="tag=path ..." sbatch --array=0-11 scripts/cluster/bos14_obsdr_motion.sh
#SBATCH --job-name=obsdr_motion
#SBATCH --partition=gpu_a100,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --exclude=bos14-node-[040,042,047,051,057,062,086,092,104,105,109,110,112,121,123]
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd /home/yiboc/mit/task_curriculum_flexiv
CONDS=(off train_full obs_all pt_lag pt_lag0 pt_lag2 pt_lag5 pt_common_bias pt_common_bias_xy pt_point_noise_bias proprio_delay joint_noise)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
OUT=${OUTDIR:-docs/results/obsdr_motion}/${COND}.json
mkdir -p "$(dirname "$OUT")" slurm_logs
echo "[obsdr_motion] cond=$COND host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
export PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
setsid .venv_isaacsim/bin/python scripts/flexiv_wuji/probe_obs_dr_motion.py --headless \
    --cond "$COND" --ckpts $CKPTS --out "$OUT" ${EXTRA:-} &
PY=$!
trap 'kill -9 -- "-$PY" 2>/dev/null; exit 143' TERM INT
wait $PY
