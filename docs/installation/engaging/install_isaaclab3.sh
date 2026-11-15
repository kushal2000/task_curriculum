#!/bin/bash
# Build .venv/isaaclab3 (docs/installation/install_isaaclab3.sh) on an Engaging CPU node, off the login node.
#   scripts/slurm/submit.sh docs/installation/engaging/install_isaaclab3.sh   (submit from the repo root)
#SBATCH --job-name=install_il3
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=debug_outputs/train_logs/slurm/%x_%j.out
#SBATCH --error=debug_outputs/train_logs/slurm/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load cmake/3.27.9
export UV_CACHE_DIR="${UV_CACHE_DIR:-/orcd/scratch/orcd/011/hglin/.uv_cache}"
export PATH="$HOME/.local/bin:$PATH"
echo "[install_il3] node $(hostname -s)"
bash docs/installation/install_isaaclab3.sh
