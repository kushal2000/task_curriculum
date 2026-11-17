#!/bin/bash
# Zero-shot eval of the rigid (PhysX fold-pool) checkpoint on the VBD cloth, Engaging (one L40S).
# Protocol of docs/results/px_matrix/flat7DRABC_s1__on_vbd__*: 32 envs, 900 steps, SAPG block coef 0,
# keypoint tolerance 4 cm (cloth.keypoint_tolerance), pinned unless RAND=1.
#
#   OUT=outputs/vbd_eval/<name>.json SEED=1 sbatch experiments/nov16_vbd_cloth/eval.sh env.cloth.size=0.2 ...
#
# Extra arguments are Hydra overrides (env.cloth.*, env.termination.*).
#SBATCH --job-name=vbdeval
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs "$(dirname "${OUT:?set OUT=<json>}")"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
# Engaging has no omniverseclient wheel: the ground plane resolves from a local mirror, and the
# anaconda-based venv needs the system libstdc++ first (see scripts/newton_py at d01b7b0).
[[ -d /orcd/scratch/orcd/011/hglin/isaac_assets/6.1/Isaac ]] && export ISAACSIM_ASSET_ROOT="${ISAACSIM_ASSET_ROOT:-/orcd/scratch/orcd/011/hglin/isaac_assets/6.1}"
for lib in /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib64/libstdc++.so.6; do
    [[ -f "$lib" ]] && { export LD_PRELOAD="$lib${LD_PRELOAD:+:$LD_PRELOAD}"; break; }
done
CKPT="${CKPT:-checkpoints/last_0_pxpool_flat7DRABC_s1_175385_ep_3052_rew__1198.3557_.pth}"
CFG="${CFG:-checkpoints/config.yaml}"
EXTRA=()
[[ "${RAND:-0}" == 1 ]] && EXTRA+=(--randomize)
echo "[eval] host=$(hostname -s) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD) out=$OUT overrides: $*"
.venv_isaaclab3/bin/python scripts/evaluation/eval.py --task "${TASK:-Isaacsimenvs-Cloth-Direct-v0}" \
    --num_envs "${NUM_ENVS:-32}" --max_steps "${MAX_STEPS:-900}" \
    --checkpoint "$CKPT" --policy_config "$CFG" --sapg_expl_coef 0 --success_tolerance 0.01 \
    --single_variant --num_assets_per_type 1 --seed "${SEED:-1}" --out "$OUT" "${EXTRA[@]}" "$@"
