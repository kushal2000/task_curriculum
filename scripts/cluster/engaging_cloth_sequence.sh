#!/bin/bash
# 1) chiffon edge_ke refit at tri_ke 100 (fabric_cantilever.py), 2) chiffon touch-fold yield check at
# that ke (two-stage set 80), the highest yield whose unpressed fold still holds, 3) the handling
# sequence (fold no press, lift, fold + press, lift, crumple) for every fabric, current vs new presets.
#   sbatch scripts/cluster/engaging_cloth_sequence.sh
# Output: docs/results/cloth_sequence/{chiffon_refit,chiffon_fold,sequence.*}
#SBATCH --job-name=clothseq
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
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
PY=.venv_isaaclab3/bin/python
OUT=${OUT:-docs/results/cloth_sequence}
TRI=${TRI:-100}
SET=${SET:-80}
mkdir -p "$OUT"
echo "[seq] node $(hostname -s) commit $(git rev-parse --short HEAD) tri $TRI set $SET"
if [ ! -s "$OUT/chiffon_refit/summary.json" ]; then
    $PY scripts/analysis/fabric_cantilever.py --fabrics chiffon_measured --tri-ke "$TRI" --ke-start tuned \
        --out "$OUT/chiffon_refit" 2>&1 | grep -E "^\[(target|round|final)|Error|Traceback|line [0-9]+"
fi
KE=$($PY -c "import json;print(json.load(open('$OUT/chiffon_refit/summary.json'))['fabrics']['chiffon_measured']['ke'])")
echo "[seq] chiffon edge_ke at tri $TRI: $KE"
$PY scripts/analysis/cloth_plasticity_probe.py --fold touch --tri-ke "$TRI" --edge-ke "$KE" --fabrics chiffon_measured \
    --kappa 15 20 25 30 35 42 50 60 100 --press 0 0.93 --set-kappa "$SET" --out "$OUT/chiffon_fold" 2>&1 \
    | grep -E "^\[(fold|drape|done)|Error|Traceback|line [0-9]+"
# Highest yield of the run of holding unpressed folds starting at the lowest yield.
KC=$($PY - "$OUT/chiffon_fold/fold.json" <<'PYEOF'
import json, sys
rows = sorted((r for r in json.load(open(sys.argv[1])) if r["press_N"] == 0), key=lambda r: r["kappa"])
best = None
for r in rows:
    if r["free_err_cm"] < 1.5:
        best = r["kappa"]
    else:
        break
print(best if best is not None else rows[0]["kappa"])
PYEOF
)
echo "[seq] chiffon yield: $KC"
$PY scripts/analysis/cloth_sequence_probe.py --set "$SET" --soft-tri "$TRI" --chiffon-ke "$KE" \
    --new-yield chiffon="$KC" --out "$OUT" 2>&1 | grep -E "^\[(seq|done)|Error|Traceback|line [0-9]+"
