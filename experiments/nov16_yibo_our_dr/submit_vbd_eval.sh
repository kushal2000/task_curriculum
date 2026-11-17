#!/usr/bin/env bash
#SBATCH --job-name=yibo_vbd_materials
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
# First prepare the shared cache and smoke-test on one allocated GPU:
# experiments/submit.sh experiments/nov16_yibo_our_dr/submit_vbd_eval.sh --prepare-cache --material base --num-envs 2 --steps 12
# Then run materials on up to four GPUs (one material per array index):
# experiments/submit.sh --array=0-7%4 experiments/nov16_yibo_our_dr/submit_vbd_eval.sh
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/../.."
fi
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "VBD material evaluation: commit=$(git rev-parse HEAD) job=${SLURM_JOB_ID:-local} host=$(hostname)"
args=()
prepare=false
for arg in "$@"; do
    if [[ "$arg" == --prepare-cache ]]; then prepare=true; else args+=("$arg"); fi
done
if $prepare; then
    .venv/isaacsim/bin/python -m isaacsimenvs.scripts.assets.prepare_usd_cache \
        --populate --task Isaacsimenvs-RigidBodyCloth-Direct-v0 --num_envs 1 \
        env.assets.table_urdf=assets/urdf/table_fold_pool.urdf
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    materials=(base big_napkin_measured chiffon_measured cotton_measured denim_measured napkin_measured silk_measured towel_measured)
    args+=(--material "${materials[$SLURM_ARRAY_TASK_ID]}")
fi
.venv/isaaclab3/bin/python experiments/nov16_yibo_our_dr/evaluate_vbd.py \
    --out-dir "experiments/nov16_yibo_our_dr/results/vbd_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}" "${args[@]}"
