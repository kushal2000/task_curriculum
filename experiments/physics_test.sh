#!/bin/bash
# Run one policy-free physics test as its own job:
#   experiments/submit.sh --job-name=<name> experiments/physics_test.sh <press_test|fold_test> \
#       --experiment_dir experiments/<day>/<folder> [test args]
#
# Runs isaacsimenvs/scripts/physics_test/<test>.py (rigid cloth: PhysX / Kit). The job's Slurm log is
# moved into <experiment_dir>/ as slurm_<job id>.out; pass --out / --video there yourself, e.g.
#   experiments/submit.sh --job-name=press_d0.3 experiments/physics_test.sh press_test \
#       --experiment_dir experiments/nov18/press_test/density_sweep \
#       --config <run>/.hydra/config.yaml --design strip_4 --density 0.3 \
#       --out experiments/nov18/press_test/density_sweep/d0.3.json
#
#SBATCH --job-name=physics_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
TEST="${1:-}"
shift || true
case "$TEST" in
    press_test|fold_test) ;;
    *) echo "[slurm] first argument must be press_test or fold_test, got '${TEST}'" >&2; exit 2 ;;
esac
# _common.sh picks the interpreter from --task; these tests always run the PhysX rigid cloth.
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
[[ -n "$EXP_DIR" ]] && adopt_log "$EXP_DIR"

gpu_log_start
"${PY[@]}" "isaacsimenvs/scripts/physics_test/${TEST}.py" "${HEADLESS[@]}" "${ARGS[@]}"
RC=$?
gpu_log_stop
exit $RC
