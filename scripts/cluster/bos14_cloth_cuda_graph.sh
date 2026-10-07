#!/bin/bash
# Does CUDA-graph capture of the coupled MJWarp + VBD step (env.cloth.use_cuda_graph) speed the cloth
# up, and does it leave the results unchanged?
#
#   sbatch --array=0-5 scripts/cluster/bos14_cloth_cuda_graph.sh
#
#   0-3  throughput.py, zero actions, 60 warm-up + 200 timed steps:
#        0 = 4 envs graph off, 1 = 4 envs graph on, 2 = 32 envs off, 3 = 32 envs on
#   4    cloth-trained reference, graph ON, 4 cm pinned, eval seeds 1-2
#        (compare: rc_eval_reference__on_vbd_c0_s*.json, graph off, 32/32)
#   5    stage-2 pool box3-prior s3, graph ON, 4 cm pinned, eval seeds 1-2
#        (compare: rc_eval_fp_box3_s3__on_vbd_c0_s*.json, graph off, 18/64)
# Eval protocol as bos14_fold_pool_eval_cloth.sh. Results: docs/results/cloth_graph_*.json

#SBATCH --job-name=cloth_graph
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=bos14-node-[104,105]
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results
I="${SLURM_ARRAY_TASK_ID:-0}"
export PYTHONUNBUFFERED=1
echo "[graph] task $I host=$(hostname -s) commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"

if [ "$I" -le 3 ]; then
    ENVS=(4 4 32 32); GRAPH=(false true false true)
    OUT="docs/results/cloth_graph_tput_e${ENVS[$I]}_${GRAPH[$I]}.json"
    scripts/newton_py -m isaacsimenvs.eval.throughput \
        --task Isaacsimenvs-Cloth-Direct-v0 \
        --num_envs "${ENVS[$I]}" --steps 200 --warmup 60 --num_assets_per_type 1 \
        --out "$OUT" \
        env.cloth.nan_policy=reset "env.cloth.use_cuda_graph=${GRAPH[$I]}"
    exit $?
fi

if [ "$I" -eq 4 ]; then
    CKPT=outputs/2026-09-23/17-31-13/0_pretrained_s1_resume_150911/nn/0_pretrained_s1_resume_150911.pth; LABEL=reference
else
    CKPT=outputs/2026-10-04/20-17-57/0_fp_box3_s3_156746/nn/0_fp_box3_s3_156746.pth; LABEL=fp_box3_s3
fi
for ES in 1 2; do
    scripts/newton_py -m isaacsimenvs.eval.episodes \
        --task Isaacsimenvs-Cloth-Direct-v0 \
        --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml --sapg_expl_coef 0 \
        --num_envs 32 --max_steps 900 --success_tolerance 0.01 --num_assets_per_type 1 \
        --single_variant --seed "$ES" \
        --out "docs/results/cloth_graph_${LABEL}__on_vbd_c0_s${ES}.json" \
        env.cloth.nan_policy=reset env.termination.success_steps=10 env.cloth.use_cuda_graph=true \
        || echo "[graph] $LABEL seed $ES FAILED"
done
