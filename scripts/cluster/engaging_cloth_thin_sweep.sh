#!/bin/bash
# Thin-fabric yield refit on the handling sequence's own fold (rows 0-1 held, edge over, let go;
# and the same with the 0.93 N press), one fresh sheet per (yield, press). One fabric per array task.
#   sbatch --array=0-6 scripts/cluster/engaging_cloth_thin_sweep.sh
# Output: docs/results/cloth_thin_sweep/<fab>/sweep.json
# Fabric x set grid at a self-contact margin (task i: fabric FABS[i % n], set SETS[i / n]):
#   FABS="napkin chiffon" SETS="80 120 160" MARGIN=0.003 sbatch --array=0-5 ...
#   -> docs/results/cloth_thin_sweep_m<MARGIN>/set<SET>/<fab>/sweep.json
#SBATCH --job-name=thinsweep
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=node3801
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
read -r -a FABS <<<"${FABS:-big_napkin chiffon cotton napkin silk towel denim}"
read -r -a SETS <<<"${SETS:-80}"
I=${SLURM_ARRAY_TASK_ID:-0}
FAB=${FABS[$(( I % ${#FABS[@]} ))]}; SET=${SETS[$(( I / ${#FABS[@]} ))]}
OUT="docs/results/cloth_thin_sweep/$FAB"; MARG=()
if [ -n "${MARGIN:-}" ]; then OUT="docs/results/cloth_thin_sweep_m$MARGIN/set$SET/$FAB"; MARG=(--self-margin "$MARGIN")
elif [ "$SET" != 80 ]; then OUT="docs/results/cloth_thin_sweep_set$SET/$FAB"; fi
CKE=${CHIFFON_KE:-0.0012076974591784867}
# stagger the start so the array tasks don't race on the warp kernel cache
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 45 ))
echo "[sweep] node $(hostname -s) commit $(git rev-parse --short HEAD) fabric $FAB set $SET margin ${MARGIN:-1.5x} -> $OUT"
.venv_isaaclab3/bin/python scripts/analysis/cloth_sequence_probe.py --fabrics "$FAB" --set "$SET" "${MARG[@]}" --soft-tri 100 \
    --chiffon-ke "$CKE" --thin-ke 500 --thin-self 1.0 \
    --sweep-yield ${KAPPAS:-5 10 15 20 25 30 35 42 50 60 80 100 150} --out "$OUT"
