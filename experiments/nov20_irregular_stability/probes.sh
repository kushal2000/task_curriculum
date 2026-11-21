#!/bin/bash
#SBATCH --job-name=irregprobe
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONUNBUFFERED=1
levels=(10 40 100 200)
tri="${levels[$((SLURM_ARRAY_TASK_ID - 1))]}"
root="${SWEEP_ROOT:-outputs/physics_sweep/irregular_stability}"
contact_ke="${CLOTH_CONTACT_KE:-500}"
table_mu="${CLOTH_TABLE_MU:-0.6}"
edge="${CLOTH_EDGE:-0.017}"
mesh_seed="${CLOTH_MESH_SEED:-0}"
out="$root/probes/tri_${tri}"
mkdir -p "$out"
py=.venv_isaaclab3/bin/python
tools=isaacsimenvs/tasks/cloth/tools
cfg="$out/fabrics.json"
"$py" - "$tri" "$cfg" "$edge" "${CLOTH_AREA_KE:-}" "${CLOTH_TRI_KD:-}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

configs = json.loads(Path("experiments/nov18_physics_sweep/probe_c17.json").read_text())
for name, fabric in configs["c17"].items():
    if float(sys.argv[3]) == 0.010:
        path = Path(f"isaacsimenvs/cfg/fabric/{name}_vbd.yaml")
        cloth = yaml.safe_load(path.read_text())["cloth"]
        fabric.update(
            density=cloth["density"], kd=float(cloth["edge_kd"]), mu=cloth["soft_contact_mu"],
            B=cloth["bending_rigidity"], stiffness_scale=cloth["bending_stiffness_scale"],
            friction_ratio=cloth["friction_ratio"], friction_slip=cloth["friction_slip_curvature"],
            yield_curvature=cloth["yield_curvature"] or float("inf"),
            hardening=cloth["yield_hardening"], creep_curvature=cloth.get("creep_curvature", float("inf")),
            creep_tau=cloth.get("creep_tau", float("inf")),
            natural_curvature=cloth.get("natural_curvature", 0.0),
        )
    fabric["tri"] = float(sys.argv[1])
    if sys.argv[4]:
        fabric["tri_ka"] = float(sys.argv[4])
    if sys.argv[5]:
        fabric["tri_kd"] = float(sys.argv[5])
Path(sys.argv[2]).write_text(json.dumps(configs, indent=2))
PY
read -r -a fabrics <<< "${PROBE_FABRICS:-denim towel chiffon cotton silk napkin big_napkin}"
scenarios=(sequence foldline cantilever)
if [[ "${PRESS_ONLY:-0}" == 1 ]]; then scenarios=(); fi
for scenario in "${scenarios[@]}"; do
    if [[ ! -s "$out/$scenario.json" ]]; then
        "$py" "$tools/cloth_probe.py" --scenario "$scenario" --out "$out" --sizes 0.10 \
            --variants new --configs "$cfg" --edge "$edge" --mesh-seed "$mesh_seed" --color-ratio 1.3 --fps 5 \
            --radius 0.004 --table-mu "$table_mu" --soft-ke "$contact_ke" --fabrics "${fabrics[@]}"
    fi
done
if [[ ! -s "$out/press.json" ]]; then
    "$py" "$tools/vbd_press_test.py" --out "$out/press.json" --variants new --configs "$cfg" --edge "$edge" \
        --mesh-seed "$mesh_seed" \
        --thickness 0.008 --fsc --table-mu "$table_mu" --soft-ke "$contact_ke" --drag 0.03 \
        --forces 1 5 20 50 --sites moving_centre crease stationary_centre --fabrics "${fabrics[@]}"
fi
