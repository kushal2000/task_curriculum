#!/bin/bash
# Evaluate a checkpoint: scripts/slurm/submit.sh [sbatch options] scripts/slurm/eval.sh \
#     --task <id> --checkpoint <run>/nn/<name>.pth [episodes.py args]
#
# Runs isaacsimenvs.eval.episodes; unless --out is given, the JSON goes to
# debug_outputs/train_logs/results/eval/<job>.json.
#
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
source scripts/slurm/_common.sh

ARGS=("$@")
if [[ " ${ARGS[*]} " != *" --out "* && " ${ARGS[*]} " != *" --out="* ]]; then
    mkdir -p debug_outputs/train_logs/results/eval
    ARGS+=(--out "debug_outputs/train_logs/results/eval/${SLURM_JOB_NAME:-eval}_${JOB_TAG}.json")
fi

gpu_log_start
"${PY[@]}" -m isaacsimenvs.eval.episodes "${HEADLESS[@]}" "${ARGS[@]}"
RC=$?
gpu_log_stop
exit $RC
