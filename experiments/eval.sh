#!/bin/bash
# Evaluate a checkpoint: experiments/submit.sh [sbatch options] experiments/eval.sh \
#     --experiment_dir experiments/<day> --task <id> --checkpoint <run>/nn/<name>.pth [episodes.py args]
#
# Runs scripts/evaluation/eval.py. The JSON and this job's Slurm log go to
# <experiment_dir>/results/<job name>_<job id>.json and slurm_<job id>.out (or pass --out yourself).
#
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
source experiments/_common.sh

EXP_DIR=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment_dir) EXP_DIR="$2"; shift 2 ;;
        --experiment_dir=*) EXP_DIR="${1#*=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
if [[ " ${ARGS[*]} " != *" --out "* && " ${ARGS[*]} " != *" --out="* ]]; then
    [[ -n "$EXP_DIR" ]] || { echo "[slurm] pass --experiment_dir experiments/<day> (or --out <json>)" >&2; exit 2; }
    ARGS+=(--out "$EXP_DIR/results/${SLURM_JOB_NAME:-eval}_${JOB_TAG}.json")
fi
[[ -n "$EXP_DIR" ]] && adopt_log "$EXP_DIR/results"

gpu_log_start
"${PY[@]}" scripts/evaluation/eval.py "${HEADLESS[@]}" "${ARGS[@]}"
RC=$?
gpu_log_stop
exit $RC
