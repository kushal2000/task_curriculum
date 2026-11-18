#!/bin/bash
# Evaluate a cloth checkpoint under its training config as one job:
#   experiments/submit.sh --job-name=<name> experiments/eval_cloth.sh \
#       --experiment_dir experiments/<day>/<folder> --config <run>/.hydra/config.yaml \
#       --checkpoint <run>/nn/<name>.pth [--num_envs N] [--out <json>] [key=value overrides]
#
# Runs isaacsimenvs/scripts/evaluation/eval_cloth.py (rigid cloth: PhysX / Kit). The job's Slurm log
# is moved into <experiment_dir>/ as slurm_<job id>.out; without --out the JSON goes next to it as
# <job name>_<job id>.json.
#
#SBATCH --job-name=eval_cloth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
source experiments/_common.sh --task Isaacsimenvs-RigidBodyCloth-Direct-v0

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
    [[ -n "$EXP_DIR" ]] || { echo "[slurm] pass --experiment_dir experiments/<day>/<folder> (or --out <json>)" >&2; exit 2; }
    ARGS+=(--out "$EXP_DIR/${SLURM_JOB_NAME:-eval_cloth}_${JOB_TAG}.json")
fi
[[ -n "$EXP_DIR" ]] && adopt_log "$EXP_DIR"

gpu_log_start
"${PY[@]}" isaacsimenvs/scripts/evaluation/eval_cloth.py "${HEADLESS[@]}" "${ARGS[@]}"
RC=$?
gpu_log_stop
exit $RC
