#!/bin/bash
# Thinner VBD cloth / grippier table: does the sim break? (scripts/analysis/cloth_thin_probe.py)
#
#   sbatch --array=0-19 scripts/cluster/bos14_cloth_thin_probe.sh
#
# Index = policy * 10 + config. All runs ns6 (newton.num_substeps 6, cloth.substeps 2).
#   policy 0 = flat7RW_base s1 ep900 on Flexiv + right Wuji (HLRL), 1 = Kuka + Sharpa flat7 s1 ep3052
#   config: thickness mm / table shape mu (cloth-table mu = sqrt(0.25 * table mu)) / resolution
#     0 t16 (base)  1 t8  2 t4  3 t2  4 t16 mu2  5 t16 mu4  6 t4 mu2  7 t2 mu4  8 t4 res13  9 t4 res13 mu2
# Results: docs/results/cloth_thin/<policy>_<config>.json
# SMOKE=1: 4 envs, short phases.

#SBATCH --job-name=cloth_thin
#SBATCH --partition=gpu_5090,gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=bos14-node-[040,047,051,057,062,086,092,104,105,109,110,112,121]
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum_flexiv
mkdir -p slurm_logs docs/results/cloth_thin
I="${SLURM_ARRAY_TASK_ID:-0}"
P=$((I / 10)); C=$((I % 10))
POL=(flat7RW_base_ep900 kuka_flat7)
CKPTS=(outputs/2026-11-11/02-51-52/0_pxpool_flat7RW_base_s1_277681/nn/snap_0_pxpool_flat7RW_base_s1_277681_ep_900.pth
       /home/yiboc/mit/task_curriculum/best_ckpts/flat7_s1_ep3052.pth)
TASKS=(Isaacsimenvs-ClothFlexivWujiRightHLRL-Direct-v0 Isaacsimenvs-Cloth-Direct-v0)
ROBOT_EXTRA=("env.cloth.start_height=0.20 env.robot_self_collision=false env.newton.njmax=32768" "")
CFGS=("" "env.cloth.thickness=0.008" "env.cloth.thickness=0.004" "env.cloth.thickness=0.002"
      "" "" "env.cloth.thickness=0.004" "env.cloth.thickness=0.002"
      "env.cloth.thickness=0.004 env.cloth.resolution=13" "env.cloth.thickness=0.004 env.cloth.resolution=13")
TMU=("" "" "" "" 2.0 4.0 2.0 4.0 "" 2.0)
TAGS=(t16 t8 t4 t2 t16_mu2 t16_mu4 t4_mu2 t2_mu4 t4_res13 t4_res13_mu2)
read -r -a OV <<<"${CFGS[$C]} ${ROBOT_EXTRA[$P]}"
MUARG=(); [ -n "${TMU[$C]}" ] && MUARG=(--table_mu "${TMU[$C]}")
if [ "${SMOKE:-0}" = 1 ]; then
    SZ=(--num_envs 4 --max_steps 60 --settle_steps 40 --slide_steps 30); OUT=docs/results/cloth_thin/smoke_${POL[$P]}_${TAGS[$C]}.json
else
    SZ=(--num_envs 64 --max_steps 900); OUT=docs/results/cloth_thin/${POL[$P]}_${TAGS[$C]}${SUFFIX:-}.json
fi
export PYTHONUNBUFFERED=1
echo "[thin] ${POL[$P]} ${TAGS[$C]} host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
scripts/newton_py -m scripts.analysis.cloth_thin_probe --task "${TASKS[$P]}" --checkpoint "${CKPTS[$P]}" \
    "${SZ[@]}" "${MUARG[@]}" --seed 1 --out "$OUT" \
    env.cloth.nan_policy=reset env.termination.success_steps=10 env.cloth.keypoint_tolerance=0.04 \
    env.newton.num_substeps=6 env.cloth.substeps=2 \
    "${OV[@]}"
echo "[thin] exit=$?"
