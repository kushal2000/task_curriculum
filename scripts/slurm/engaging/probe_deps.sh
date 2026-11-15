#!/bin/bash
# Engaging: finish .venv_isaaclab3 enough for the kit-less VBD cloth probes (scripts/analysis) and
# tests. scripts/install_isaaclab3.sh stops in `isaaclab.sh -i` here: the core submodule isaaclab_ov
# needs ovstage, whose only Linux wheels are manylinux_2_35 (Engaging nodes are older), and the
# installer has no way to skip a core submodule. The probes only need Newton + warp, so install
# Isaac Lab's own pins for those (pyproject [tool.uv].override-dependencies / [tool.isaaclab.versions]).
# newton[importers] is left out: its open3d dependency is also manylinux_2_35-only, and the probes build
# their sheets with ModelBuilder.add_cloth_mesh (no USD / mesh importers).
#   scripts/slurm/submit.sh scripts/slurm/engaging/probe_deps.sh   (from the repo root, after the venv exists)
#SBATCH --job-name=probe_deps
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/orcd/scratch/orcd/011/hglin/.uv_cache}"
export PATH="$HOME/.local/bin:$PATH"
PY=.venv_isaaclab3/bin/python
echo "[probe_deps] node $(hostname -s)"
uv pip install --python "$PY" \
  "newton[sim] @ git+https://github.com/newton-physics/newton.git@release-1.5" \
  "newton-usd-schemas>=0.4.1" "warp-lang==1.16.0" "mujoco~=3.11.0" "mujoco-warp~=3.11.0" \
  scipy pyyaml pytest matplotlib viser
uv pip install --python "$PY" -e . --no-deps
"$PY" - <<'PY'
import importlib
for m in ("newton", "warp", "scipy", "yaml", "torch", "viser"):
    print(f"{m:8}", getattr(importlib.import_module(m), "__version__", "ok"))
PY
