#!/bin/bash
# How the best PhysX-trained cloth policies hold up when the VBD sheet changes: one ClothCfg knob at
# a time, everything else at its default. VBD 1.5 cm pinned only (the record's key cell), fresh eval
# seeds 3-8 (192 episodes), the protocol of the best-snapshot re-tests.
#
#   sbatch --array=0-65 scripts/cluster/bos14_vbd_param_sweep.sh
#
# Index = policy * 22 + config.
#   policy 0 flat7 s2 ep3400 (flat7c 179734)   1 flat7DRABC s3 ep1200 (175385)   2 VBD-trained reference
#   config 0 base; density 0.25 0.5 1 4 (2); edge_ke 1e-4 4e-3 2e-2 (8e-4); tri_ke = tri_ka 1e3 1e4 (1e2);
#          soft_contact_mu 0.1 0.5 1 (0.25); size 0.13 0.16 0.20 (0.10); thickness 0.008 0.012 0.024
#          (0.016); resolution 5 9 11 (7). Defaults in brackets.
# size keeps the cloth, not the particle count: the grid grows so spacing stays ~16 mm (res 9 / 11 / 13),
# and the reset centre moves toward the robot by (size - 0.10) * sqrt(2) / 2, as the pool's reach_shift
# (bos14_cloth_size_sweep.sh does the same) -- unshifted, the far side of a 16-20 cm sheet is out of reach.
# thickness / resolution move only that one: res 9 / 11 keep the 16 mm particles on a finer grid
# (spheres overlap, ClothCfg only warns), thickness 24 mm sits above the 16.7 mm spacing the same way.
# Output: docs/results/px_matrix/vbdsw_<policy>__<config>_s<seed>__on_vbd__pinned__1p5cm_s<es>.json

#SBATCH --job-name=vbdsw
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
OUTDIR=docs/results/px_matrix
mkdir -p slurm_logs "$OUTDIR"
I="${SLURM_ARRAY_TASK_ID:-0}"
NCFG=22
P=$((I / NCFG)); C=$((I % NCFG))
POLS=(flat7 flat7DRABC reference)
SEEDS=(2 3 1)
CKPTS=(outputs/2026-10-13/18-22-53/0_pxpool_flat7c_s2_179734/nn/snap_0_pxpool_flat7c_s2_179734_ep_3400.pth
       outputs/2026-10-13/01-48-46/0_pxpool_flat7DRABC_s3_175385/nn/snap_0_pxpool_flat7DRABC_s3_175385_ep_1200.pth
       outputs/2026-09-23/17-31-13/0_pretrained_s1_resume_150911/nn/0_pretrained_s1_resume_150911.pth)
CFGS=(""
      "env.cloth.density=0.25" "env.cloth.density=0.5" "env.cloth.density=1.0" "env.cloth.density=4.0"
      "env.cloth.edge_ke=0.0001" "env.cloth.edge_ke=0.004" "env.cloth.edge_ke=0.02"
      "env.cloth.tri_ke=1000.0 env.cloth.tri_ka=1000.0" "env.cloth.tri_ke=10000.0 env.cloth.tri_ka=10000.0"
      "env.cloth.soft_contact_mu=0.1" "env.cloth.soft_contact_mu=0.5" "env.cloth.soft_contact_mu=1.0"
      "env.cloth.size=0.13 env.cloth.resolution=9 env.cloth.reset_offset_xy=[0.0,0.0212]"
      "env.cloth.size=0.16 env.cloth.resolution=11 env.cloth.reset_offset_xy=[0.0,0.0424]"
      "env.cloth.size=0.2 env.cloth.resolution=13 env.cloth.reset_offset_xy=[0.0,0.0707]"
      "env.cloth.thickness=0.008" "env.cloth.thickness=0.012" "env.cloth.thickness=0.024"
      "env.cloth.resolution=5" "env.cloth.resolution=9" "env.cloth.resolution=11")
TAGS=(base dens0.25 dens0.5 dens1 dens4 bend1e-4 bend4e-3 bend2e-2 stretch1e3 stretch1e4
      mu0.1 mu0.5 mu1 size0.13 size0.16 size0.20 thick8 thick12 thick24 res5 res9 res11)
CKPT=${CKPTS[$P]}; SEED=${SEEDS[$P]}
[ -f "$CKPT" ] || { echo "[vbdsw] no checkpoint $CKPT" >&2; exit 1; }
read -r -a OV <<<"${CFGS[$C]}"
TAG="vbdsw_${POLS[$P]}__${TAGS[$C]}"
export PYTHONUNBUFFERED=1
echo "[vbdsw] $TAG s$SEED ckpt=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
for ES in ${EVAL_SEEDS:-3 4 5 6 7 8}; do
    OUT="$OUTDIR/${TAG}_s${SEED}__on_vbd__pinned__1p5cm_s${ES}.json"
    [ -s "$OUT" ] && { echo "[vbdsw] exists: $OUT"; continue; }
    scripts/newton_py -m isaacsimenvs.eval.episodes --task Isaacsimenvs-Cloth-Direct-v0 \
        --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml \
        --sapg_expl_coef 0 --num_envs 32 --max_steps 900 --success_tolerance 0.01 \
        --num_assets_per_type 1 --single_variant --seed "$ES" --out "$OUT" \
        env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset \
        env.termination.success_steps=10 "${OV[@]}" \
        || echo "[vbdsw] $OUT FAILED"
done
echo "[vbdsw] done"
