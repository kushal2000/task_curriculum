#!/bin/bash
# Stretch-stiffness sweep for the soft-membrane fabrics: crumple/scrunch check plus the touch-fold
# refit (single-stage and two-stage set 80) at tri_ke = tri_ka = TRI, one value per array task.
#   TRIS="100 200 600" sbatch --array=0-2 scripts/cluster/engaging_cloth_tri_sweep.sh
# Output: docs/results/cloth_tri_sweep/tri<TRI>/{crumple,fold_set0/<fab>,fold_set80/<fab>}
#SBATCH --job-name=trisweep
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
read -r -a TRI_LIST <<<"${TRIS:-100 200 600}"
TRI="${TRI_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"
FABS="${FABRICS:-chiffon cotton silk}"
OUT=docs/results/cloth_tri_sweep/tri${TRI}
echo "[tri] node $(hostname -s) commit $(git rev-parse --short HEAD) tri_ke $TRI fabrics $FABS"
PY=.venv_isaaclab3/bin/python
$PY scripts/analysis/cloth_crumple_probe.py --tri-ke "$TRI" --fabrics $FABS --out "$OUT/crumple" 2>&1 \
    | grep -E "^\[(crumple|done)|Error|Traceback|line [0-9]+"
for SET in 0 80; do
    for FAB in $FABS; do
        $PY scripts/analysis/cloth_plasticity_probe.py --fold touch --tri-ke "$TRI" --fabrics "${FAB}_measured" \
            --kappa ${KAPPAS:-15 20 25 30 32 42 100} --press 0 0.93 --set-kappa "$SET" \
            --out "$OUT/fold_set${SET}/$FAB" 2>&1 | grep -E "^\[(fold|drape|done)|Error|Traceback|line [0-9]+"
    done
done
