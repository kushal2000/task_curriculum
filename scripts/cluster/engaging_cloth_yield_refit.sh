#!/bin/bash
# Refit plastic_yield_curvature on the CURRENT env mesh (alternate diagonals): hand-free touch fold,
# let go unpressed and after a 0.93 N press, 3 s free (scripts/analysis/cloth_plasticity_probe.py).
#   sbatch scripts/cluster/engaging_cloth_yield_refit.sh
#   two-stage plasticity (set grows only past SET /m), one set value per array task:
#   SETS="40 60 80" sbatch --array=0-2 scripts/cluster/engaging_cloth_yield_refit.sh
# Output: docs/results/cloth_yield_refit/<fabric>/fold.json, two-stage: cloth_yield_refit_set<SET>/
#SBATCH --job-name=yieldrefit
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
echo "[refit] node $(hostname -s) commit $(git rev-parse --short HEAD)"
read -r -a SET_LIST <<<"${SETS:-0}"
SET="${SET_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"
ROOT_OUT=docs/results/cloth_yield_refit
[ "$SET" != 0 ] && ROOT_OUT="docs/results/cloth_yield_refit_set${SET}"
echo "[refit] set curvature $SET -> $ROOT_OUT"
for FAB in ${FABRICS:-big_napkin chiffon cotton napkin silk}; do
    D="$ROOT_OUT/$FAB"
    [ -s "$D/fold.json" ] && { echo "[refit] exists $D"; continue; }
    .venv_isaaclab3/bin/python scripts/analysis/cloth_plasticity_probe.py --fold touch \
        --fabrics "${FAB}_measured" --kappa ${KAPPAS:-10 15 20 25 30 35 42 50} --press 0 0.93 --set-kappa "$SET" \
        --out "$D" 2>&1 | grep -E "^\[(fold|drape)|Error|Traceback" | tail -40
done
