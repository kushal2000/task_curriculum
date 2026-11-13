#!/bin/bash
# Hand-free fold -> press -> release on every MEASURED fabric at its own yield curvature
# (scripts/analysis/cloth_plasticity_probe.py --fold touch, the calibration test), with the press
# taken off over 0.2 s (the calibration) and at once (a hand leaving within a policy step), at
# 0 / 0.93 / 5 / 20 N. Question: does a pressed fold bounce open when the press vanishes?
#
#   sbatch scripts/cluster/engaging_cloth_release_ramp.sh
# Output: docs/results/cloth_release_ramp/<fabric>_ramp<s>/fold.json
#SBATCH --job-name=relramp
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
OUT=docs/results/cloth_release_ramp
echo "[relramp] node $(hostname -s) commit $(git rev-parse --short HEAD)"
for FAB in ${FABRICS:-big_napkin chiffon cotton denim napkin silk towel}; do
    K=$(.venv_isaaclab3/bin/python -I -c "import yaml; print(yaml.safe_load(open('isaacsimenvs/cfg/fabric/${FAB}_measured.yaml'))['plastic_yield_curvature'])")
    for R in ${RAMPS:-0.2 0}; do
        D="$OUT/${FAB}_ramp${R}"
        [ -s "$D/fold.json" ] && { echo "[relramp] exists $D"; continue; }
        .venv_isaaclab3/bin/python scripts/analysis/cloth_plasticity_probe.py --fold touch --skip-drape \
            --fabrics "${FAB}_measured" --kappa "$K" --press ${PRESS:-0 0.93 5 20} --release-ramp "$R" \
            --out "$D" ${EXTRA:-} 2>&1 | grep -E "^\[|Error|Traceback" | tail -12
    done
done
