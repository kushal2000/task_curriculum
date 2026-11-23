#!/bin/bash
#SBATCH --job-name=play_pc_throughput
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=/share/portal/kk837/task_curriculum/debug_outputs/train_logs/nov/bench/throughput_%j.out
# sbatch train_throughput.sbatch <code root: repo or a worktree of another commit> <num envs> <steps>
set -uo pipefail
ROOT="$1"; shift
REPO=/share/portal/kk837/task_curriculum
cd "$ROOT"
echo "[throughput] root=$ROOT commit=$(git rev-parse --short HEAD) host=$(hostname -s)"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$ROOT" TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
exec "$REPO/.venv/isaacsim/bin/python" "$REPO/experiments/nov22_play_pointcloud/bench/train_throughput.py" "$@"
