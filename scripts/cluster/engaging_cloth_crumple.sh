#!/bin/bash
# Crumple / scrunch / pinch-lift check of the VBD sheet: elastic vs current single-stage vs two-stage
# plasticity (scripts/analysis/cloth_crumple_probe.py).
#   sbatch scripts/cluster/engaging_cloth_crumple.sh      (SET=80 OUT=docs/results/cloth_crumple)
#SBATCH --job-name=crumple
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
echo "[crumple] node $(hostname -s) commit $(git rev-parse --short HEAD)"
.venv_isaaclab3/bin/python scripts/analysis/cloth_crumple_probe.py --set "${SET:-80}" \
    --out "${OUT:-docs/results/cloth_crumple}" 2>&1 | grep -E "^\[(crumple|done)|Error|Traceback|line [0-9]+" | tail -80
