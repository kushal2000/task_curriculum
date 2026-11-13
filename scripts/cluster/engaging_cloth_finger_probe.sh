#!/bin/bash
# Scripted fingertip press / drag / lift on the VBD sheet (scripts/analysis/cloth_finger_probe.py):
# thickness 16 / 2 / 0.5 mm x scenario (poke into a suspended or hanging sheet, slow/fast; table press reference), full-surface contact off/on, soft_contact_ke 500/5000.
#   sbatch scripts/cluster/engaging_cloth_finger_probe.sh
#SBATCH --job-name=fingerprobe
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=node3801
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
echo "[finger] node $(hostname -s) commit $(git rev-parse --short HEAD)"
.venv_isaaclab3/bin/python scripts/analysis/cloth_finger_probe.py --out "${OUT:-docs/results/cloth_finger_probe}" ${ARGS:-} 2>&1 \
    | grep -E "^\[(finger|done)|Error|Traceback|line [0-9]+|Exception"
