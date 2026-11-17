#!/usr/bin/env bash
#SBATCH --job-name=yibo_rigid_ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=00:30:00
set -eo pipefail
source experiments/_common.sh
OUTPUT="debug_outputs/yibo_our_dr_analysis/rigid_${SLURM_JOB_ID:-local}"
adopt_log "$OUTPUT"
gpu_log_start
trap gpu_log_stop EXIT
SEED="${ABLATION_SEED:-0}"
CASES="${ABLATION_CASES:-hybrid sensors_off thickness_16mm density_2 hinges_yibo sizes_small contacts_yibo yibo}"
for CASE in $CASES; do
    echo "[ablation] case=$CASE seed=$SEED"
    "${PY[@]}" experiments/nov16_yibo_our_dr/evaluate_rigid.py --headless --device cuda:0 \
        --case "$CASE" --seed "$SEED" --num_envs 64 --max_steps 600 \
        --out "$OUTPUT/${CASE}_seed${SEED}.json" 2>&1 | tee "$OUTPUT/${CASE}_seed${SEED}.log"
    "${PY[@]}" experiments/nov16_yibo_our_dr/analyze_rigid.py "$OUTPUT"
done
