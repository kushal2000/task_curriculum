#!/bin/bash
# Isaac Sim (PhysX) on Engaging, inside an Apptainer container.
#   sbatch scripts/cluster/engaging_install_isaacsim.sh   (submit from the repo root)
#
# Isaac Sim 5.1's pip wheels need glibc >= 2.35 and every Engaging node is Rocky 8 (glibc 2.28),
# so the Kit stack cannot run natively. The official image (Ubuntu, glibc 2.35) supplies the OS;
# the Python stack is still scripts/install_isaacsim.sh (isaaclab 2.3.2 / Isaac Sim 5.1), built
# from INSIDE the container into .venv_isaacsim_ct so pip accepts the 2_35 wheels. Run it with
# scripts/isaacsim_py, the container's counterpart of scripts/newton_py.
#SBATCH --job-name=install_isim
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# GPU nodes do not have apptainer on PATH by default (login and CPU nodes do).
command -v apptainer >/dev/null || module load apptainer
SCR=/orcd/scratch/orcd/011/hglin
SIF="${ISAACSIM_SIF:-$SCR/containers/isaac-sim_5.1.0.sif}"
export APPTAINER_CACHEDIR="$SCR/.apptainer_cache"
export APPTAINER_TMPDIR="$SCR/.apptainer_tmp"
mkdir -p "$(dirname "$SIF")" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
echo "[install_isim] node $(hostname -s)"

if [[ ! -s "$SIF" ]]; then
  echo "==> Pulling nvcr.io/nvidia/isaac-sim:5.1.0 -> $SIF"
  apptainer pull "$SIF" docker://nvcr.io/nvidia/isaac-sim:5.1.0
fi
GLIBC="$(apptainer exec "$SIF" ldd --version)"   # no `| head`: SIGPIPE + pipefail kills the job
echo "    container ${GLIBC%%$'\n'*}"

echo "==> Building .venv_isaacsim_ct inside the container"
apptainer exec --bind /orcd \
  --env UV_CACHE_DIR="$SCR/.uv_cache" \
  --env UV_PYTHON_PREFERENCE=only-managed \
  --env VENV=.venv_isaacsim_ct \
  --env PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "$SIF" bash scripts/install_isaacsim.sh
rm -rf "$APPTAINER_TMPDIR"
echo "[install_isim] done"
