#!/bin/bash
# The viewer's four policies on a LARGER VBD cloth (default 20 cm), protocol of bos14_px_eval_matrix.sh.
#
#   sbatch --array=0-7 scripts/cluster/bos14_large_cloth_eval.sh
#   python3 scripts/analysis/large_cloth_summary.py
#
# Index = policy*2 + start (0 pinned, 1 randomised); each task runs both cutoffs x two eval seeds.
# CLOTH_ARGS (default: 20 cm, resolution 13 = the 10 cm sheet's 16.7 mm spacing, shifted 7.07 cm toward
# the robot like the pool's reach_shift). TAG names the output: docs/results/large_cloth/<TAG>/.

#SBATCH --job-name=lc_eval
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
I="${SLURM_ARRAY_TASK_ID:-0}"
P=$(( I / 2 )); S=$(( I % 2 ))
NAMES=(run2_flat_s3 run1_initfold_s1 run3_box3_s3 prev_best_157622_s1)
CKPTS=(
  outputs/2026-10-06/14-53-54/0_pxpool_one_flat_s3_158376/nn/0_pxpool_one_flat_s3_158376.pth
  outputs/2026-10-06/14-33-35/0_pxpool_one_initfold_s1_158375/nn/0_pxpool_one_initfold_s1_158375.pth
  outputs/2026-10-06/13-32-04/0_pxchain_box3_s3_158324/nn/0_pxchain_box3_s3_158324.pth
  outputs/2026-10-05/23-51-55/0_pxpool_pxs1_arm1e3_e6144_s1_157622/nn/0_pxpool_pxs1_arm1e3_e6144_s1_157622.pth
)
read -r -a CLOTH <<<"${CLOTH_ARGS:-env.cloth.size=0.2 env.cloth.resolution=13 env.cloth.reset_offset_xy=[0.0,0.0707]}"
OUT="docs/results/large_cloth/${TAG:-s20}"
mkdir -p slurm_logs "$OUT"
[ "$S" -eq 1 ] && { START=rand; SARGS=(--randomize); } || { START=pinned; SARGS=(); }
export PYTHONUNBUFFERED=1
echo "[lc_eval] ${NAMES[$P]} $START host=$(hostname -s) cloth=${CLOTH[*]}"
for CUT in 0.04 0.015; do
    [ "$CUT" = 0.04 ] && CT=4cm || CT=1p5cm
    for ES in 1 2; do
        F="$OUT/${NAMES[$P]}__${START}__${CT}_s${ES}.json"
        [ -s "$F" ] && continue
        scripts/newton_py -m isaacsimenvs.eval.episodes --task Isaacsimenvs-Cloth-Direct-v0 \
            --checkpoint "${CKPTS[$P]}" --policy_config pretrained_policy/config.yaml \
            --sapg_expl_coef 0 --num_envs 32 --max_steps 900 --success_tolerance 0.01 \
            --num_assets_per_type 1 --single_variant "${SARGS[@]}" --seed "$ES" --out "$F" \
            "${CLOTH[@]}" "env.cloth.keypoint_tolerance=$CUT" env.cloth.nan_policy=reset \
            env.termination.success_steps=10 || echo "[lc_eval] $F FAILED"
    done
done
echo "[lc_eval] done"
