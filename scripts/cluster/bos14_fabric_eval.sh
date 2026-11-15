#!/bin/bash
# The best PhysX-trained cloth policies + the VBD reference on VBD sheets set up as six real fabrics.
# The fabric parameters are the presets in isaacsimenvs/cfg/fabric (sources: isaacsimenvs/tasks/cloth/utils/fabrics.py).
# Job 238829 ran an earlier round with edge_ke = 100 x B and edge_kd 1e-3.
# Same protocol as bos14_vbd_param_sweep.sh: 1.5 cm pinned, eval seeds 3-8 (192 episodes per point).
#
#   sbatch --array=0-20 scripts/cluster/bos14_fabric_eval.sh
#   PREFIX=fabric1101 (sbatch --export=ALL,PREFIX=...) keeps an earlier round's outputs (238829 = 100 x B, kd 1e-3)
#   PLAST=1 adds bending plasticity at each preset's plastic_yield_curvature (11-04 scripted fold-press-release,
#   job 259470: the gentlest threshold that holds a pressed fold); base has none, so skip 0,7,14:
#   sbatch --array=1-6,8-13,15-20 --export=ALL,PLAST=1,PREFIX=fabricplast1104 scripts/cluster/bos14_fabric_eval.sh
#
# Index = policy * 7 + fabric.
#   policy 0 flat7 s2 ep3400 (flat7c 179734)   1 flat7DRABC s3 ep1200 (175385)   2 VBD-trained reference
#   fabric 0 base (re-run as a control for code drift since 10-19)  1 chiffon  2 silk  3 cotton  4 napkin
#          5 towel  6 denim
# Output: docs/results/px_matrix/fabric_<policy>__<fabric>_s<seed>__on_vbd__pinned__1p5cm_s<es>.json

#SBATCH --job-name=fabric
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH --exclude=bos14-node-040,bos14-node-057,bos14-node-062,bos14-node-086,bos14-node-092,bos14-node-105,bos14-node-109,bos14-node-110,bos14-node-121
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
OUTDIR=docs/results/px_matrix
mkdir -p slurm_logs "$OUTDIR"
I="${SLURM_ARRAY_TASK_ID:-0}"
NCFG=7
P=$((I / NCFG)); C=$((I % NCFG))
POLS=(flat7 flat7DRABC reference)
SEEDS=(2 3 1)
CKPTS=(outputs/2026-10-13/18-22-53/0_pxpool_flat7c_s2_179734/nn/snap_0_pxpool_flat7c_s2_179734_ep_3400.pth
       outputs/2026-10-13/01-48-46/0_pxpool_flat7DRABC_s3_175385/nn/snap_0_pxpool_flat7DRABC_s3_175385_ep_1200.pth
       outputs/2026-09-23/17-31-13/0_pretrained_s1_resume_150911/nn/0_pretrained_s1_resume_150911.pth)
TAGS=(base chiffon silk cotton napkin towel denim)
CKPT=${CKPTS[$P]}; SEED=${SEEDS[$P]}
[ -f "$CKPT" ] || { echo "[fabric] no checkpoint $CKPT" >&2; exit 1; }
# The preset's overrides (isaacsimenvs/cfg/fabric/<tag>.yaml); base = the env defaults, sent explicitly.
read -r -a OV <<<"$(.venv_isaaclab3/bin/python isaacsimenvs/tasks/cloth/utils/fabrics.py "${TAGS[$C]}" \
    $([ "${PLAST:-0}" = 1 ] && echo --plastic))"
[ "${#OV[@]}" -gt 0 ] || { echo "[fabric] no overrides for ${TAGS[$C]}" >&2; exit 1; }
TAG="${PREFIX:-fabric}_${POLS[$P]}__${TAGS[$C]}"
export PYTHONUNBUFFERED=1
echo "[fabric] $TAG s$SEED ${OV[*]} ckpt=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
for ES in ${EVAL_SEEDS:-3 4 5 6 7 8}; do
    OUT="$OUTDIR/${TAG}_s${SEED}__on_vbd__pinned__1p5cm_s${ES}.json"
    [ -s "$OUT" ] && { echo "[fabric] exists: $OUT"; continue; }
    scripts/newton_py -m isaacsimenvs.eval.episodes --task Isaacsimenvs-Cloth-Direct-v0 \
        --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml \
        --sapg_expl_coef 0 --num_envs 32 --max_steps 900 --success_tolerance 0.01 \
        --num_assets_per_type 1 --single_variant --seed "$ES" --out "$OUT" \
        env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset \
        env.termination.success_steps=10 "${OV[@]}" \
        || echo "[fabric] $OUT FAILED"
done
echo "[fabric] done"
