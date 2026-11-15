#!/bin/bash
# Run one VBD cloth probe in scripts/analysis on an Engaging GPU node (the bos14_*.sh launchers
# hard-code another cluster's path and partition). Submit from the repo root:
#   sbatch -J cantilever scripts/cluster/engaging_probe.sh fabric_cantilever.py [args]
# Needs .venv_isaaclab3 (scripts/cluster/engaging_install_isaaclab3.sh).
#SBATCH --job-name=probe
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
script="$1"; shift
echo "[probe] ${script} on $(hostname -s), $(nvidia-smi --query-gpu=name --format=csv,noheader)"
.venv_isaaclab3/bin/python "scripts/analysis/${script}" "$@"
