#!/bin/bash
#SBATCH --job-name=play_pc_render
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum
#SBATCH --output=debug_outputs/train_logs/nov/renders/render_%j.out
# One point-cloud student, one tool, an Isaac Sim (RTX) video:
#   sbatch render.sbatch <student .pth> <tool index> <out .mp4> [--log_contact ...]
set -uo pipefail
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
exec .venv/isaacsim/bin/python -m isaacsimenvs.scripts.visualization.policies.render_play_pc --headless --enable_cameras \
    --student "$1" --tool "$2" --out "$3" "${@:4}"
