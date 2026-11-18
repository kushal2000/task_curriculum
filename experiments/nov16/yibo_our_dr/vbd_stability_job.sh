#!/usr/bin/env bash
#SBATCH --job-name=vbd_stability
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
fi
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
config="${VBD_STABILITY_CONFIG:-experiments/nov16/yibo_our_dr/vbd_stability.yaml}"
output="${VBD_STABILITY_RUN:-experiments/nov16/yibo_our_dr/results/vbd_stability}"
candidate=$(.venv/isaaclab3/bin/python -c \
    'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(list(c["candidates"])[int(sys.argv[2])])' \
    "$config" "${SLURM_ARRAY_TASK_ID:-1}")
echo "VBD stability: candidate=$candidate commit=$(git rev-parse HEAD) host=$(hostname)"
exec .venv/isaaclab3/bin/python experiments/nov16/yibo_our_dr/benchmark_vbd_stability.py \
    --config "$config" --candidate "$candidate" --fixture "$output/fixture.npz" \
    --out-dir "$output/$candidate" "$@"
