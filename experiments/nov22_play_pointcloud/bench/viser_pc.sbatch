#!/bin/bash
#SBATCH --job-name=viser_play_pc
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum
#SBATCH --output=debug_outputs/train_logs/nov/renders/viser_%j.out
# Live viser player of a point-cloud student on its own GPU:
#   sbatch viser_pc.sbatch <student .pth or run dir or run-name part> <port>
# then: ssh -N -L <port>:<node>:<port> <login node>, open http://localhost:<port>
set -uo pipefail
echo "[viser] host $(hostname -s) port $2"
WS=third_party/viser_websockets
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD/$WS:$PWD" TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
exec .venv/isaacsim/bin/python -m isaacsimenvs.scripts.visualization.policies.viser_play_pc --headless --student "$1" --port "$2"
