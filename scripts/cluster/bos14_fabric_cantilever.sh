#!/bin/bash
# Sim cantilever test of the VBD fabric presets: scripts/analysis/fabric_cantilever.py.
# Output: docs/results/fabric_cantilever/{cantilever.npz,summary.json}
#SBATCH --job-name=cantilever
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --exclude=bos14-node-040,bos14-node-057,bos14-node-062,bos14-node-086,bos14-node-092,bos14-node-105,bos14-node-109,bos14-node-110,bos14-node-121
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd /home/yiboc/mit/task_curriculum
export PYTHONUNBUFFERED=1
echo "[cantilever] node $(hostname -s)"
.venv_isaaclab3/bin/python scripts/analysis/fabric_cantilever.py "$@"
