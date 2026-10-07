#!/bin/bash
# Live, interactive policy viewer (isaacsimenvs/eval/live_viser.py) on a GPU node.
#
#   sbatch scripts/cluster/bos14_live_viser.sh
#   TASK=Isaacsimenvs-Cloth-Direct-v0 ENVS=4 sbatch scripts/cluster/bos14_live_viser.sh
#
# Then read slurm_logs/live_viser_<job>.out for the node, and from your laptop:
#   ssh -N -L 8080:<node>:8080 <you>@<bos14 login>      then open http://localhost:8080
#
# The task can be switched from the browser (the process re-execs itself); the job holds one GPU
# until it is cancelled (scancel <job>) or the time limit is hit.

#SBATCH --job-name=live_viser
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=bos14-node-[104,105]
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs
PORT="${PORT:-8080}"
export PYTHONUNBUFFERED=1
echo "[live] node $(hostname -s) ($(hostname -I | awk '{print $1}')) port $PORT commit $(git rev-parse --short HEAD)"
echo "[live] tunnel: ssh -N -L ${PORT}:$(hostname -s):${PORT} $USER@<bos14 login>   then http://localhost:${PORT}"
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
TASK="${TASK:-Isaacsimenvs-FoldPool-Direct-v0}"
# Rigid manipulands need the evaluation-sized contact buffer; the VBD cloth keeps its default
# (the big one only slows it down). Switching task from the browser applies the same rule.
BUF=()
[ "$TASK" != Isaacsimenvs-Cloth-Direct-v0 ] && BUF=("env.newton.per_env_triangle_pairs=2097152")
exec scripts/newton_py -m isaacsimenvs.eval.live_viser \
    --task "$TASK" \
    --num_envs "${ENVS:-8}" \
    --port "$PORT" \
    "${BUF[@]}" \
    "${EXTRA[@]}"
