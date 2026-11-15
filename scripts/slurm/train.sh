#!/bin/bash
# Train any task: scripts/slurm/submit.sh [sbatch options] scripts/slurm/train.sh --task <id> [train.py args]
#
# The args go to isaacsimenvs.train unchanged (--experiment, --checkpoint, --checkpoint_load_mode,
# --wandb_*, hydra overrides). In an array job, task i gets agent.params.seed=i+1 unless the args
# set a seed. The run itself lands in debug_outputs/train_logs/<date>/<time>/ (Hydra) and wandb in debug_outputs/train_logs/wandb/.
#
# The interactive pose viewer is on by default (--capture_viewer, every 6000 steps, logged to wandb
# and written to the run dir's interactive_viewer/); the browser fetches the robot meshes from raw
# GitHub at this branch, so push the branch before submitting. VIEWER=0 turns it off.
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
if [[ "${VIEWER:-1}" != 0 && " ${ARGS[*]} " != *" --capture_viewer "* ]]; then
    ARGS=(--capture_viewer --capture_viewer_url_check warn "${ARGS[@]}")
    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1 \
        || echo "[slurm] WARNING: branch $BRANCH is not on origin -- the viewer's meshes will 404" >&2
    [[ -z "$(git log "origin/$BRANCH..HEAD" --oneline 2>/dev/null)" ]] \
        || echo "[slurm] WARNING: HEAD has commits not pushed to origin/$BRANCH -- viewer assets may be stale" >&2
fi

# wandb's console capture (a redirect thread on stdout/stderr) is redundant here -- the slurm log
# already has every line -- and it deadlocked both 2026-11-14 7-day runs at their first viewer
# write: main thread blocked on a futex, a wandb-era thread spinning in syscalls, no traceback.
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

gpu_log_start
START=$(date +%s)
"${PY[@]}" -m isaacsimenvs.train "${HEADLESS[@]}" "${ARGS[@]}"
RC=$?
gpu_log_stop
echo "[slurm] exit=$RC wall_s=$(( $(date +%s) - START ))"
exit $RC
