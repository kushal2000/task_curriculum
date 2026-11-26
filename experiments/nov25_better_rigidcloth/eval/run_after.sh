#!/bin/bash
# Waits for the running old-DR rest sweep, then runs the depth probe (1 seed) and threshold sweep (2 seeds).
cd /share/portal/kk837/task_curriculum_nov25
while pgrep -u $USER -f "^xargs -P 4 -I\{\} bash -c SLURM_ARRAY_TASK_ID" >/dev/null; do sleep 15; done
O=outputs/eval_nov25_rest_probe; C=experiments/nov25_better_rigidcloth/eval/rest_probe_configs.txt; mkdir -p $O
# probe: config lines 1-7, seed 1 only -> indices 0,2,..,12; threshold sweep: lines 8-16 -> indices 14..31
{ seq 0 2 12; seq 14 31; } | PROBE=1 CONFIGS=$C OUTDIR=$O xargs -P 4 -I{} bash -c 'SLURM_ARRAY_TASK_ID={} bash experiments/nov25_better_rigidcloth/eval/ablate.sbatch > '$O'/local_{}.out 2>&1; echo "task {} rc=$?"'
