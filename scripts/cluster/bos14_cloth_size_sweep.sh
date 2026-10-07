#!/bin/bash
# Run 2 (flat pool, one_flat 158376), all three seeds, on VBD cloths of several sizes. Protocol of
# bos14_px_eval_matrix.sh. The 10 cm baseline is docs/results/px_matrix/one_flat_*__on_vbd__*.
#
#   sbatch --array=0-23 scripts/cluster/bos14_cloth_size_sweep.sh
#
# Index = size*6 + (seed-1)*2 + start. Per size: the grid whose spacing is closest to the calibrated
# 16.67 mm while >= the 16 mm particle thickness (ClothCfg guard), and the reset centre shifted toward
# the robot by (size - 0.10) * sqrt(2) / 2, as the pool's reach_shift. Output:
# docs/results/large_cloth/sweep/one_flat_s<seed>__<size>__<pinned|rand>__<4cm|1p5cm>_s<es>.json

#SBATCH --job-name=lc_sweep
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
Z=$(( I / 6 )); SEED=$(( (I % 6) / 2 + 1 )); S=$(( I % 2 ))
SIZES=(12p5cm 15cm 17p5cm 20cm)
ARGS=(
  "env.cloth.size=0.125 env.cloth.resolution=8 env.cloth.reset_offset_xy=[0.0,0.0177]"
  "env.cloth.size=0.15 env.cloth.resolution=10 env.cloth.reset_offset_xy=[0.0,0.0354]"
  "env.cloth.size=0.175 env.cloth.resolution=11 env.cloth.reset_offset_xy=[0.0,0.0530]"
  "env.cloth.size=0.2 env.cloth.resolution=13 env.cloth.reset_offset_xy=[0.0,0.0707]"
)
read -r -a CLOTH <<<"${ARGS[$Z]}"
CKPT=$(ls outputs/2026-10-06/*/0_pxpool_one_flat_s${SEED}_158376/nn/0_pxpool_one_flat_s${SEED}_158376.pth)
OUT=docs/results/large_cloth/sweep
mkdir -p slurm_logs "$OUT"
[ "$S" -eq 1 ] && { START=rand; SARGS=(--randomize); } || { START=pinned; SARGS=(); }
export PYTHONUNBUFFERED=1
echo "[lc_sweep] size=${SIZES[$Z]} seed=$SEED $START host=$(hostname -s) ckpt=$CKPT cloth=${CLOTH[*]}"
for CUT in 0.04 0.015; do
    [ "$CUT" = 0.04 ] && CT=4cm || CT=1p5cm
    for ES in 1 2; do
        F="$OUT/one_flat_s${SEED}__${SIZES[$Z]}__${START}__${CT}_s${ES}.json"
        [ -s "$F" ] && continue
        scripts/newton_py -m isaacsimenvs.eval.episodes --task Isaacsimenvs-Cloth-Direct-v0 \
            --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml \
            --sapg_expl_coef 0 --num_envs 32 --max_steps 900 --success_tolerance 0.01 \
            --num_assets_per_type 1 --single_variant "${SARGS[@]}" --seed "$ES" --out "$F" \
            "${CLOTH[@]}" "env.cloth.keypoint_tolerance=$CUT" env.cloth.nan_policy=reset \
            env.termination.success_steps=10 || echo "[lc_sweep] $F FAILED"
    done
done
echo "[lc_sweep] done"
