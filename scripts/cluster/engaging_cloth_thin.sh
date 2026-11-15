#!/bin/bash
# Thin (realistic-thickness) VBD fabrics: per-fabric touch-fold yield refit at its thickness, then the
# handling sequence (fold, lift, fold + press, lift, crumple) for the 16 mm proposed presets and the thin ones.
#   THIN_KE=5000 sbatch scripts/cluster/engaging_cloth_thin.sh
# Thickness per fabric: THIN in scripts/analysis/cloth_sequence_probe.py. Particle radius = thickness / 2,
# self_contact_radius = THIN_SELF x thickness, soft_contact_ke = THIN_KE. Stretch 100 for chiffon / cotton / silk,
# chiffon edge_ke CHIFFON_KE, two-stage set 80.
# Output: docs/results/cloth_thin_ke<THIN_KE>/{fold/<fab>, sequence.*}
#SBATCH --job-name=cloththin
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=node3801
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
PY=.venv_isaaclab3/bin/python
KE=${THIN_KE:-5000}
SELF=${THIN_SELF:-1.0}
CKE=${CHIFFON_KE:-0.0012076974591784867}
OUT=${OUT:-docs/results/cloth_thin_ke${KE}}
mkdir -p "$OUT"
echo "[thin] node $(hostname -s) commit $(git rev-parse --short HEAD) soft_ke $KE self x$SELF"
declare -A TH=([chiffon]=0.0001 [silk]=0.0003 [napkin]=0.00015 [big_napkin]=0.0003 [cotton]=0.0003 [denim]=0.00035 [towel]=0.001)
YARGS=()
for FAB in big_napkin chiffon cotton napkin silk towel denim; do
    T=${TH[$FAB]}
    X=()
    case $FAB in chiffon) X=(--tri-ke 100 --edge-ke "$CKE");; cotton|silk) X=(--tri-ke 100);; esac
    D="$OUT/fold/$FAB"
    if [ ! -s "$D/fold.json" ]; then
        $PY scripts/analysis/cloth_plasticity_probe.py --fold touch --fabrics "${FAB}_measured" "${X[@]}" \
            --particle-radius $($PY -c "print($T/2)") --soft-ke "$KE" --self-contact-radius $($PY -c "print($T*$SELF)") \
            --kappa ${KAPPAS:-15 20 25 30 35 42 50 60 100} --press 0 0.93 --set-kappa 80 --skip-drape --out "$D" 2>&1 \
            | grep -E "^\[(fold|done)|Error|Traceback|line [0-9]+"
    fi
    # stays-folded fabrics: the highest yield of the run of holding unpressed folds from the bottom;
    # springs-flat fabrics (napkin, silk): the highest yield whose PRESSED fold still holds.
    K=$($PY - "$D/fold.json" "$FAB" <<'PYEOF'
import json, sys
rows = json.load(open(sys.argv[1])); fab = sys.argv[2]
press = 0.93 if fab in ("napkin", "silk") else 0.0
rs = sorted((r for r in rows if abs(r["press_N"] - press) < 1e-6), key=lambda r: r["kappa"])
best = None
for r in rs:
    if r["free_err_cm"] < 1.5 and r.get("finite", True):
        best = r["kappa"]
    else:
        break
print(best if best is not None else rs[0]["kappa"])
PYEOF
)
    echo "[thin] $FAB thickness $T yield $K"
    YARGS+=("$FAB=$K")
done
$PY scripts/analysis/cloth_sequence_probe.py --variants new thin --set 80 --soft-tri 100 --chiffon-ke "$CKE" \
    --new-yield chiffon=42 --thin-ke "$KE" --thin-self "$SELF" --thin-yield "${YARGS[@]}" --out "$OUT" 2>&1 \
    | grep -E "^\[(seq|done)|Error|Traceback|line [0-9]+"
