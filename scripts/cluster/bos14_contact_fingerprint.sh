#!/bin/bash
# Contact "fingerprint": the same hand motion, open-loop, on the VBD cloth, the Newton chain and the
# PhysX chain -- how each sheet responds (drift, lift, tilt) to an identical push.
#
#   STAGE=record sbatch --array=0-1 scripts/cluster/bos14_contact_fingerprint.sh
#   STAGE=replay sbatch --array=0-5 scripts/cluster/bos14_contact_fingerprint.sh
#
# record 0: Newton box3 policy on the Newton chain; 1: PhysX box3 policy on the PhysX chain.
# replay i: source i/3 (0 newton, 1 physx) onto target i%3 (0 vbd, 1 newton chain, 2 physx chain).
# Dumps: docs/results/fingerprint_<source>_on_<target>.pt (cloth_obs_parity.py format).

#SBATCH --job-name=px_fp
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
I="${SLURM_ARRAY_TASK_ID:-0}"
STEPS="${STEPS:-200}"
TAG="${OUT_TAG:-}"
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
NW_CKPT=outputs/2026-09-30/17-12-24/0_rc2_box3-mid_fixdr_s2_155349/nn/last_0_rc2_box3-mid_fixdr_s2_155349_ep_2035_rew__1054.914_.pth
PX_CKPT="${PX_CKPT:-outputs/2026-10-05/16-18-10/0_pxchain_rc2fix_e12288_s1_157223/nn/0_pxchain_rc2fix_e12288_s1_157223.pth}"
SRC=(newton physx)
run() {  # target-kind out extra...
    local kind=$1 out=$2; shift 2
    case $kind in
        vbd)    scripts/newton_py scripts/analysis/cloth_obs_parity.py --task Isaacsimenvs-Cloth-Direct-v0 --steps "$STEPS" --out "$out" "$@" ;;
        newton) scripts/newton_py scripts/analysis/cloth_obs_parity.py --task Isaacsimenvs-RigidCloth-Direct-v0 --steps "$STEPS" --out "$out" env.rigid_cloth.randomization.enabled=false "$@" ;;
        physx)  .venv_isaacsim/bin/python scripts/analysis/cloth_obs_parity.py --headless --task Isaacsimenvs-RigidClothPhysx-Direct-v0 --steps "$STEPS" --out "$out" env.rigid_cloth.randomization.enabled=false "$@" ;;
    esac
}
if [ "${STAGE:-record}" = record ]; then
    if [ "$I" -eq 0 ]; then run newton docs/results/fingerprint_src_newton.pt --checkpoint "$NW_CKPT"
    else run physx docs/results/fingerprint_src_physx.pt --checkpoint "$PX_CKPT" "${EXTRA[@]}"; fi
else
    S=${SRC[$((I / 3))]}; T=(vbd newton physx); T=${T[$((I % 3))]}
    X=(); [ "$T" = physx ] && X=("${EXTRA[@]}")
    run "$T" "docs/results/fingerprint_${S}_on_${T}${TAG}.pt" --replay "docs/results/fingerprint_src_${S}.pt" "${X[@]}"
fi
echo "[px_fp] stage=${STAGE:-record} task=$I exit=$?"
