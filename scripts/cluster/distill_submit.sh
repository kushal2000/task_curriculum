#!/bin/bash
# Submit one distillation run and its chained evaluation (PhysX rand + VBD pinned/rand x 2 seeds).
#   scripts/cluster/distill_submit.sh <base|fric> <name> <vis> <goal> [ITERS] [extra distill args...]
# Env: NOISE=1, G3_NOISE=1, ENVS, STUDENT_INIT=<pth> (warm start). Appends to slurm_logs/distill/jobs.txt.
set -euo pipefail
cd /home/yiboc/mit/task_curriculum_flexiv
tt=$1 name=$2 vis=$3 goal=$4 iters=${5:-3000}; shift $(( $# < 5 ? $# : 5 ))
FRIC=outputs/2026-11-11/06-51-20/0_pxpool_flat7RW_fric_s1_278460/nn/snap_0_pxpool_flat7RW_fric_s1_278460_ep_3000.pth
extra="--beta_iters ${BETA_ITERS:-300} --save_every 500 $*"
[ -n "${STUDENT_INIT:-}" ] && extra="$extra --student $STUDENT_INIT"
common=()
[ "$tt" = fric ] && common+=(TEACHER=$FRIC EXPERIMENT=flat7RW_fric)
j=$(env "${common[@]}" NAME="${tt}_${name}" VIS=$vis GOAL=$goal ITERS=$iters ENVS="${ENVS:-4096}" NOISE="${NOISE:-0}" \
    G3_NOISE="${G3_NOISE:-0}" EXTRA="$extra" \
    sbatch --parsable --mem=48G --time="${TIME:-06:00:00}" --job-name="dpc_${tt}_${name}" scripts/cluster/bos14_distill_pc.sh)
stu="outputs/distill/${tt}_${name}_s1_${j}/student_last.pth"
e=$(env "${common[@]}" MODE=evalall STUDENT=$stu TAG="${tt}_${name}" \
    sbatch --parsable --time=03:00:00 --dependency=afterok:$j --job-name="dpe_${tt}_${name}" scripts/cluster/bos14_distill_pc.sh)
mkdir -p slurm_logs/distill
echo "$(date +%m-%d_%H:%M) $tt $name train=$j eval=$e $stu" | tee -a slurm_logs/distill/jobs.txt
