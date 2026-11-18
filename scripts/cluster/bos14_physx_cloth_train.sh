#!/bin/bash
# Train the rigid fold task (the shape pool) on PhysX (Isaac Sim 5.1).
#
#   TASK=pool PRIOR=<run>.pth RUN_TAG=box3 sbatch --array=0-2 scripts/cluster/bos14_physx_cloth_train.sh
#   TASK=pool ENVS=256 EPOCHS=3 RUN_TAG=smoke sbatch --array=0 scripts/cluster/bos14_physx_cloth_train.sh
#   EXPERIMENT=experiments/nov11_right_wuji_dr_ablation/flat7RW_fric.yaml RUN_TAG=flat7RW_fric sbatch --array=0-2 ...
#       (EXPERIMENT: a name in isaacsimenvs/scripts/configs/runs/ or a recipe path; configs/recipes.py)
#   REPO=<worktree> ROBOT=FlexivWujiHumanLike PRIOR=pretrained_policy/flexiv_wuji.pth ... (another checkout / robot)
#   PRIORS="<s1>.pth <s2>.pth <s3>.pth" TASK=pool RUN_TAG=ft sbatch --array=0-2 scripts/cluster/bos14_physx_cloth_train.sh
#       (finetune: array index k starts from the k-th checkpoint)
#
# Array index = seed - 1. Same learner, prior, reward, horizon, minibatch and default env-step budget
# (1e8) as the Newton launchers (bos14_rigid_cloth_train.sh, bos14_fold_pool_train.sh); what changes is
# the simulator and therefore how many envs one GPU can carry. EPOCHS defaults to 1e8 env-steps at
# the chosen ENVS. MINIBATCH defaults to the Newton runs' 24576, so an optimizer step sees the same
# number of samples and a bigger ENVS buys more optimizer steps per epoch, not bigger ones.
#
# BEST_SAVE_S (default 60): at most one best-reward checkpoint per that many seconds. Unthrottled, a
# run whose reward keeps rising saves on most epochs, ~4 s each on the shared filesystem -- 3-4x the
# 1.2 s a 1536-env PhysX chain epoch takes. The checkpoint the evals pick is then <= 60 s stale.
#
# LOAD_MODE (default weights): weights = model + normalizers only, fresh optimizer / lr / epoch count
# (set the lr with agent.params.config.learning_rate). resume = also the optimizer, lr, epoch and frame
# counters, so EPOCHS is the TOTAL epoch count (source epochs + new ones).
#
# Policies are evaluated on Newton (the VBD cloth exists only there): the observation is identical,
# see isaacsimenvs/tasks/cloth/backends/rigid/env.py.

#SBATCH --job-name=px_train
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --exclude=bos14-node-[040,047,051,057,062,086,092,104,105,109,110,112,121,123]
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd "${REPO:-/home/yiboc/mit/task_curriculum}"
mkdir -p slurm_logs

I="${PX_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
SEED=$(( I + 1 ))
TASK_KIND="${TASK:-pool}"
ENVS="${ENVS:-1536}"
EPOCHS="${EPOCHS:-$(( (100000000 + ENVS * 32 - 1) / (ENVS * 32) ))}"
MINIBATCH="${MINIBATCH:-24576}"
BLOCK=$((ENVS / 6))             # num_blocks is baked into the checkpoint at 6
JOB_TAG="${PX_JOB_ID:-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
PRIOR="${PRIOR:-pretrained_policy/model.pth}"
# PRIORS: one checkpoint per array index (space-separated), so seed k finetunes ITS OWN source seed.
if [ -n "${PRIORS:-}" ]; then
    read -r -a PRIOR_LIST <<<"$PRIORS"
    PRIOR="${PRIOR_LIST[$I]:-}"
    [ -n "$PRIOR" ] || { echo "[px_train] PRIORS has no entry for array index $I"; exit 2; }
fi
# NO_PRIOR=1: no --checkpoint at all (random initialisation, or a point-cloud actor that takes its
# start from --pc_init_student in EXTRA_ARGS).
if [ "${NO_PRIOR:-0}" = 1 ]; then
    PRIOR=none; CKPT_ARGS=()
else
    [ -f "$PRIOR" ] || { echo "[px_train] prior not found: $PRIOR"; exit 2; }
    CKPT_ARGS=(--checkpoint "$PRIOR" --checkpoint_load_mode "${LOAD_MODE:-weights}")
fi
case "$TASK_KIND" in
    pool)  GYM_TASK=Isaacsimenvs-FoldPoolPhysx${ROBOT:-}-Direct-v0; TASK_ARGS=() ;;
    *) echo "TASK must be pool, got $TASK_KIND"; exit 2 ;;
esac
NAME="0_px${TASK_KIND}_${RUN_TAG:-run}_s${SEED}_${JOB_TAG}"
if [ $(( ENVS * 32 % MINIBATCH )) -ne 0 ]; then
    echo "ENVS x 32 = $(( ENVS * 32 )) is not a multiple of MINIBATCH $MINIBATCH"; exit 2
fi
WANDB_ARGS=()
if [ "${WANDB:-0}" = 1 ]; then
    WANDB_ARGS=(--wandb_activate --wandb_project fold_pool --wandb_group "px_${RUN_TAG:-run}_${JOB_TAG}" --wandb_name "$NAME")
fi
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
# EXPERIMENT=<name or path>: the run recipe (configs/recipes.py) supplies the env count,
# epochs, minibatch, SAPG block and DR, so the launcher's own size overrides are left out (ENVS / EPOCHS /
# MINIBATCH are ignored; override through EXTRA_ARGS instead).
if [ -n "${EXPERIMENT:-}" ]; then
    RUN_ARGS=(--experiment "$EXPERIMENT")
else
    RUN_ARGS=(env.cloth.nan_policy=reset "env.scene.num_envs=${ENVS}" agent.params.config.multi_gpu=False
              "agent.params.config.minibatch_size=${MINIBATCH}"
              "agent.params.config.central_value_config.minibatch_size=${MINIBATCH}"
              "agent.params.config.expl_coef_block_size=${BLOCK}" "agent.params.config.max_epochs=${EPOCHS}"
              "agent.params.config.best_save_min_interval_s=${BEST_SAVE_S:-60}")
fi

export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
if [ -n "${EXPERIMENT:-}" ]; then
    # envs / epochs as the run will use them (recipe, then EXTRA_ARGS), for run_record_sync.py's header parse
    read -r ENVS EPOCHS < <(PYTHONPATH=$PWD .venv_isaacsim/bin/python - "$EXPERIMENT" "${EXTRA_ARGS:-}" <<'PY'
import re, sys
from isaacsimenvs.scripts.configs.recipes import load_recipe
e = load_recipe(sys.argv[1])
envs = e["env"].get("scene", {}).get("num_envs", "?")
epochs = e["agent"].get("params", {}).get("config", {}).get("max_epochs", "?")
envs = (re.findall(r"env\.scene\.num_envs=(\d+)", sys.argv[2]) or [envs])[-1]
epochs = (re.findall(r"max_epochs=(\d+)", sys.argv[2]) or [epochs])[-1]
print(envs, epochs)
PY
)
    echo "[px_train] task=$GYM_TASK seed=$SEED envs=$ENVS epochs=$EPOCHS experiment=$EXPERIMENT name=$NAME"
else
    echo "[px_train] task=$GYM_TASK seed=$SEED envs=$ENVS epochs=$EPOCHS minibatch=$MINIBATCH name=$NAME"
fi
echo "[px_train] prior=$PRIOR load_mode=${LOAD_MODE:-weights} wandb=${WANDB:-0} extra=${EXTRA_ARGS:-none} ${TASK_ARGS[*]:-}"
echo "[px_train] host=$(hostname -s) commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MEMLOG="slurm_logs/${SLURM_JOB_NAME:-px_train}_${JOB_TAG}_${I}.gpumem.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 >"$MEMLOG" &
SMI_PID=$!
# Host RAM too: PhysX pins host memory for its GPU buffers, and a failed pinned allocation shows up
# only as a bare "PxgCudaMemoryAllocator.cpp LINE 68" warning before the CUDA context dies.
HOSTLOG="${MEMLOG%.gpumem.csv}.hostmem.csv"
( while sleep 30; do echo "$(date +%T),$(ps -u "$USER" -o rss= | awk '{s+=$1} END {printf "%d", s/1024}') MiB"; done ) >"$HOSTLOG" &
HOST_PID=$!

START=$(date +%s)
# PhysX's GPU pipeline sometimes dies at simulator start ("Fetching GPU Narrowphase failed! 700",
# preceded or followed by "PxgCudaMemoryAllocator" warnings) -- intermittently: the same config and
# node also start fine. The process then HANGS instead of exiting and holds the node until the time
# limit. A watchdog greps each attempt's log for those messages, kills the attempt and retries it from
# scratch, up to PX_TRIES (default 3) attempts.
FATAL='Fetching GPU Narrowphase failed|Cuda context manager error|PxgCudaMemoryAllocator'
RC=1
for TRY in $(seq 1 "${PX_TRIES:-3}"); do
    ALOG="slurm_logs/${NAME}.try${TRY}.log"
    setsid \
    .venv_isaacsim/bin/python -m isaacsimenvs.scripts.training.train \
        --task "$GYM_TASK" \
        --agent rl_games_sapg_cfg_entry_point \
        --headless \
        "${CKPT_ARGS[@]}" \
        --single_variant \
        "${WANDB_ARGS[@]}" \
        "${TASK_ARGS[@]}" \
        "${RUN_ARGS[@]}" \
        "agent.params.config.name=${NAME}" \
        "agent.params.seed=${SEED}" \
        "${EXTRA[@]}" \
        > >(tee "$ALOG") 2>&1 &
    PY_PID=$!
    # scancel must take python with it. Slurm here tracks a job by its process TREE
    # (proctrack/linuxproc), not a cgroup: on SIGTERM bash dies, the setsid'd python is re-parented to
    # init, and the SIGKILL that follows KillWait no longer finds it. The orphan keeps its GPU and the
    # node health check drains the node ("rogue GPU usage outside Slurm" -- 048/067/074/082 on
    # 2026-10-07). So kill python's group ourselves; `sleep & wait` lets the trap run at once.
    trap 'kill -9 -- "-$PY_PID" 2>/dev/null; kill "$SMI_PID" "$HOST_PID" 2>/dev/null; exit 143' TERM INT
    RC=""
    while kill -0 "$PY_PID" 2>/dev/null; do
        if grep -qE "$FATAL" "$ALOG" 2>/dev/null; then
            echo "[px_train] try $TRY: PhysX GPU failure ($(grep -m1 -oE "$FATAL" "$ALOG")); killing and retrying"
            kill -9 -- "-$PY_PID" 2>/dev/null
            RC=86
            break
        fi
        sleep 15 & wait $!
    done
    if [ -z "$RC" ]; then wait "$PY_PID"; RC=$?; else wait "$PY_PID" 2>/dev/null; fi
    [ "$RC" -ne 86 ] && break
done

kill "$SMI_PID" "$HOST_PID" 2>/dev/null
WALL=$(( $(date +%s) - START ))
echo "[px_train] seed=$SEED exit=$RC wall_s=$WALL peak_gpu_mib=$(cut -d, -f2 "$MEMLOG" | tr -dc '0-9\n' | sort -n | tail -1) peak_host_mib=$(cut -d, -f2 "$HOSTLOG" | tr -dc '0-9\n' | sort -n | tail -1)"
find outputs -type d -name "$NAME" -newermt "-25 hours" | tee "slurm_logs/${NAME}.ckptdir"
