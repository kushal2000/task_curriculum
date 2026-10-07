#!/bin/bash
# Screen a PhysX chain physics setting with three FIXED policies, before paying for training on it.
#
#   TAG=base sbatch --array=0-5 scripts/cluster/bos14_physx_chain_screen.sh
#   TAG=k3000 EXTRA_ARGS="env.physx_chain.slat_compliant_stiffness=3000 env.physx_chain.slat_compliant_damping=100" \
#       sbatch --array=0-5 scripts/cluster/bos14_physx_chain_screen.sh
#   python3 scripts/analysis/physx_chain_screen_summary.py
#
# Index = policy*2 + start (start 0 = pinned, 1 = randomised); two eval seeds x 32 envs each, the
# protocol of bos14_rigid_cloth_eval_rc2.sh. The three policies bracket what "cloth-like" means here:
#   0 ref     the cloth-trained reference (VBD 100%; Newton fixed chain 61% pinned / 38% rand)
#   1 newton  the Newton box3-mid chain policy (rc2 fixdr s2, 155349; ~50% on the cloth)
#   2 physx   the PhysX-from-scratch box3-mid policy (157222 s1; ~3% on the cloth: flips the
#             stationary plate whole)
# A setting under which PhysX-trained policies transfer should score the first two well and the
# third badly -- the third's trick must stop working.
# Writes docs/results/pxscreen_<TAG>_<policy>_<start>_s<seed>.json.

#SBATCH --job-name=px_screen
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
: "${TAG:?set TAG to name this physics setting}"
I="${SLURM_ARRAY_TASK_ID:-0}"
P=$(( I / 2 )); S=$(( I % 2 ))
VARIANT="${VARIANT:-box3-mid}"

find_ckpt() {
    .venv_isaaclab3/bin/python -c "import sys; sys.path.insert(0, 'scripts/analysis'); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path('outputs')) or '')" "$1"
}
case "$P" in
    0) POLICY=ref;    CKPT=/home/yiboc/mit/model.pth ;;
    1) POLICY=newton; CKPT=outputs/2026-09-30/17-12-24/0_rc2_box3-mid_fixdr_s2_155349/nn/last_0_rc2_box3-mid_fixdr_s2_155349_ep_2035_rew__1054.914_.pth ;;
    2) POLICY=physx;  CKPT=$(find_ckpt 0_pxchain_rc2fix_s1_157222) ;;
    *) echo "bad index $I"; exit 2 ;;
esac
[ -f "$CKPT" ] || { echo "[px_screen] no checkpoint for $POLICY" >&2; exit 1; }
START=(env.rigid_cloth.randomization.enabled=false); SNAME=pinned
[ "$S" -eq 1 ] && { START=(--randomize env.rigid_cloth.randomization.enabled=false); SNAME=rand; }
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
echo "[px_screen] tag=$TAG policy=$POLICY start=$SNAME extra=${EXTRA_ARGS:-none} host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
for ES in 1 2; do
    .venv_isaacsim/bin/python -m isaacsimenvs.eval.episodes \
        --task Isaacsimenvs-RigidClothPhysx-Direct-v0 --checkpoint "$CKPT" \
        --policy_config pretrained_policy/config.yaml --headless \
        --sapg_expl_coef 0 --num_envs 32 --max_steps 900 --success_tolerance 0.01 \
        --num_assets_per_type 1 --single_variant --seed "$ES" \
        --out "docs/results/pxscreen_${TAG}_${POLICY}_${SNAME}_s${ES}.json" \
        "env.rigid_cloth.variant=${VARIANT}" env.cloth.nan_policy=reset env.termination.success_steps=10 \
        "${START[@]}" "${EXTRA[@]}" || echo "[px_screen] $POLICY $SNAME seed $ES FAILED"
done
echo "[px_screen] done $TAG $POLICY $SNAME"
