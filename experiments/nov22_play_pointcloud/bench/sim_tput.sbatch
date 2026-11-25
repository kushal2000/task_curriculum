#!/bin/bash
#SBATCH --job-name=sim_tput
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum
#SBATCH --output=debug_outputs/train_logs/nov/bench/sim_tput_%j.out
# Physics throughput (zero actions, isaacsimenvs/scripts/evaluation/throughput.py) of the cloth fold task on
# its two backends, on the same GPU model: VBD cloth (Newton) vs the rigid-body proxy pool (PhysX).
#   sbatch experiments/nov22_play_pointcloud/bench/sim_tput.sbatch <vbd|rigid|populate> <num_envs> [TAG] [hydra overrides]
set -uo pipefail
MODE="$1"; N="$2"; TAG="${3:-}"; shift 3 2>/dev/null || shift $#
cd /share/portal/kk837/task_curriculum
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR"
OUT="debug_outputs/train_logs/nov/bench/sim_tput_${MODE}${TAG:+_$TAG}_${N}.json"
echo "[sim_tput] host=$(hostname -s) job=${SLURM_JOB_ID:-local} mode=$MODE envs=$N commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$MODE" in
  vbd)   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6   # see experiments/newton_py
         PY=.venv/isaaclab3/bin/python; TASK=Isaacsimenvs-Cloth-Direct-v0 ;;
  rigid) PY=.venv/isaacsim/bin/python;  TASK=Isaacsimenvs-RigidCloth-Direct-v0 ;;
  # The Newton backend reads robot/table USDs from .usd_cache, which only Isaac Sim can write.
  populate) exec .venv/isaacsim/bin/python -m isaacsimenvs.tasks.cloth.backends.vbd.newton_support.usd_cache --populate "$@" ;;
  *) echo "mode must be vbd, rigid or populate"; exit 2 ;;
esac
exec "$PY" -m isaacsimenvs.scripts.evaluation.throughput --task "$TASK" --num_envs "$N" \
    --warmup "${WARMUP:-60}" --steps "${STEPS:-300}" --out "$OUT" ${PROGRESS:+--progress "${OUT%.json}.progress"} "$@"
