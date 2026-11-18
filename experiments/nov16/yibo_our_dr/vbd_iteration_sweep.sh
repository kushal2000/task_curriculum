#!/usr/bin/env bash
#SBATCH --job-name=yibo_vbd_iterations
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
levels=(12 40 80 6)
iterations="${levels[${SLURM_ARRAY_TASK_ID:-0}]}"
recipe=experiments/nov16/yibo_our_dr/vbd_base_branch.yaml
if [[ "$iterations" != 12 ]]; then
    recipe="experiments/nov16/yibo_our_dr/vbd_iterations_${iterations}.yaml"
fi
echo "VBD iteration comparison: iterations=$iterations commit=$(git rev-parse HEAD) host=$(hostname)"
exec .venv/isaaclab3/bin/python experiments/nov16/yibo_our_dr/evaluate_vbd.py \
    --material base --num-envs 32 --steps 600 --seed 0 \
    --experiment "$recipe" --contact-diagnostics \
    --out-dir "experiments/nov16/yibo_our_dr/results/vbd_iterations_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}/iterations_${iterations}" "$@"
