#!/bin/bash
# Render a G1 + Wuji Hand 2 trajectory to mp4 in Isaac Sim, on a bos14 GPU node.
#
#   sbatch scripts/cluster/bos14_g1_wuji_replay_isaacsim.sh traj.npz --video out.mp4 [--variant 23 ...]
#
# Arguments go straight to scripts/g1_wuji/replay_isaacsim.py (--headless is added). Submit from
# the repo root, and keep inputs and outputs on the shared filesystem: /tmp is node-local.
#
# bos14 compute nodes lack libGLU.so.1. Without it Isaac Sim's MDL SDK fails to load, the RTX
# shaders fail to build, and the camera returns empty frames rather than an error. One-time fix,
# from the login node, which has the library:
#
#   mkdir -p .venv_isaacsim/compat-libs
#   cp /usr/lib/x86_64-linux-gnu/libGLU.so.1.3.1 .venv_isaacsim/compat-libs/libGLU.so.1
#
# (Reinstalling .venv_isaacsim deletes it; copy it again afterwards.)

#SBATCH --job-name=g1_wuji_replay
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -x .venv_isaacsim/bin/python ]] || { echo "submit from the task_curriculum repo root" >&2; exit 1; }

if [[ "$(ldconfig -p)" != *"libGLU.so.1 "* ]]; then
    compat="$PWD/.venv_isaacsim/compat-libs"
    [[ -f "$compat/libGLU.so.1" ]] || { echo "libGLU.so.1 missing here and in $compat; see the header of $0" >&2; exit 1; }
    export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "using compat libGLU from $compat"
fi

export OMNI_KIT_ACCEPT_EULA=YES
.venv_isaacsim/bin/python scripts/g1_wuji/replay_isaacsim.py --headless "$@"
