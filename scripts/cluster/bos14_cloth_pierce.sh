#!/bin/bash
# Fingers going THROUGH the VBD cloth (scripts/analysis/cloth_pierce_probe.py): how often, what it
# costs the fold, and which physics knob changes it.
#
#   sbatch --array=0-17 scripts/cluster/bos14_cloth_pierce.sh
#
# Index = policy * 9 + config. policy 0 = stage-2 pool box3-prior s3, 1 = cloth-trained reference.
#   config 0 base (Cloth.yaml)            4 cloth.substeps 12 (6)
#          1 tri_ke = tri_ka = 1e3 (1e2)  5 cloth.proxy_mode lagged (staggered)
#          2 tri_ke = tri_ka = 1e4        6 cloth.vbd_iterations 24 (12)
#          3 soft_contact_ke 5e3 (5e2)  7 finer grid: resolution 9, thickness 12.5 mm (7, 16 mm)
#                                         8 finer grid: resolution 13, thickness 8.3 mm
# 64 envs, one episode each, randomised start, eval seed 1, CUDA graph on (same folds as off: job
# 157004). Results: docs/results/cloth_pierce_<policy>_<config>.json

#SBATCH --job-name=cloth_pierce
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
P=$((I / 9)); C=$((I % 9))
POL=(fp_box3_s3 reference)
CKPTS=(outputs/2026-10-04/20-17-57/0_fp_box3_s3_156746/nn/0_fp_box3_s3_156746.pth
       outputs/2026-09-23/17-31-13/0_pretrained_s1_resume_150911/nn/0_pretrained_s1_resume_150911.pth)
CFGS=("" "env.cloth.tri_ke=1000.0 env.cloth.tri_ka=1000.0" "env.cloth.tri_ke=10000.0 env.cloth.tri_ka=10000.0"
      "env.cloth.soft_contact_ke=5000.0" "env.cloth.substeps=12" "env.cloth.proxy_mode=lagged"
      "env.cloth.vbd_iterations=24"
      "env.cloth.resolution=9 env.cloth.thickness=0.0125" "env.cloth.resolution=13 env.cloth.thickness=0.0083")
TAGS=(base trike1e3 trike1e4 contactke5e3 substeps12 lagged iters24 res9 res13)
read -r -a OV <<<"${CFGS[$C]}"
export PYTHONUNBUFFERED=1
echo "[pierce] ${POL[$P]} ${TAGS[$C]} host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
scripts/newton_py -m scripts.analysis.cloth_pierce_probe \
    --checkpoint "${CKPTS[$P]}" --num_envs 64 --max_steps 900 --randomize --seed 1 \
    --out "docs/results/cloth_pierce_${POL[$P]}_${TAGS[$C]}.json" \
    env.cloth.nan_policy=reset env.termination.success_steps=10 env.cloth.use_cuda_graph=true \
    "${OV[@]}"
