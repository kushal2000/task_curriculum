#!/bin/bash
# Train any task: scripts/slurm/submit.sh [sbatch options] scripts/slurm/train.sh --task <id> [train.py args]
#
# The args go to isaacsimenvs.train unchanged (--experiment, --checkpoint, --checkpoint_load_mode,
# --wandb_*, hydra overrides). In an array job, task i gets agent.params.seed=i+1 unless the args
# set a seed. The run itself lands in outputs/<date>/<time>/ (Hydra) and wandb in outputs/wandb/.
#
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
source scripts/slurm/_common.sh

ARGS=("$@")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && ! printf '%s\n' "$@" | grep -q '^agent.params.seed='; then
    ARGS+=("agent.params.seed=$(( SLURM_ARRAY_TASK_ID + 1 ))")
fi
[[ " ${ARGS[*]} " == *" --agent "* || " ${ARGS[*]} " == *" --agent="* ]] || ARGS=(--agent rl_games_sapg_cfg_entry_point "${ARGS[@]}")

gpu_log_start
START=$(date +%s)
"${PY[@]}" -m isaacsimenvs.train "${HEADLESS[@]}" "${ARGS[@]}"
RC=$?
gpu_log_stop
echo "[slurm] exit=$RC wall_s=$(( $(date +%s) - START ))"
exit $RC
