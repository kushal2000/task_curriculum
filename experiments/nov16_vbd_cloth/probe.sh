#!/bin/bash
# Cloth-only probes (isaacsimenvs/tasks/cloth/tools/cloth_probe.py) on one GPU, Engaging.
# Each argument is one probe command line (the arguments after cloth_probe.py), run in order:
#
#   sbatch experiments/nov16_vbd_cloth/probe.sh "--scenario cantilever --out outputs/cloth_probe/cal --iters 12"
#
#SBATCH --job-name=clothprobe
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
PY=.venv_isaaclab3/bin/python
echo "[probe] host=$(hostname -s) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
rc=0
for cmd in "$@"; do
    echo "[probe] >>> $cmd"
    # shellcheck disable=SC2086
    $PY isaacsimenvs/tasks/cloth/tools/cloth_probe.py $cmd || rc=$?
done
exit $rc
