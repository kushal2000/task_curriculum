#!/bin/bash
#SBATCH --job-name=cartpole_sapg
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --output=/share/portal/kk837/task_curriculum/experiments/runs/slurm_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#
# Train SAPG policies for the multi-link cartpole at 1, 2, 4 and 8 free links.
#
# Three ways to run it:
#
#   sbatch --array=0-3 experiments/train_multilink_cartpole_sapg.sh
#       One SLURM job per link count, running in PARALLEL. Each array task gets its own
#       GPU allocation, so the "one Kit process per GPU" rule
#       (docs/isaacsim_installation.md, Gotchas) is satisfied by SLURM rather than by
#       serialising. This is the intended way — it is ~4x faster than the alternatives.
#
#   sbatch experiments/train_multilink_cartpole_sapg.sh
#       A single job that walks the four link counts sequentially on one GPU.
#
#   experiments/train_multilink_cartpole_sapg.sh
#       Same sequential walk, in the foreground, on whatever GPU you already hold.
#
# The array index maps into LINKS: 0->1 link, 1->2 links, 2->4 links, 3->8 links.
#
# --------------------------------------------------------------------------------
# Why all four runs use n_max=8 rather than n_max=1/2/4/8
# --------------------------------------------------------------------------------
# Every run spawns the same 8-link articulation and *locks* a subset of its joints, so
# the "1-link" run is one rigid pendulum of the full 1.0 m chain, the "2-link" run is a
# double pendulum of that same 1.0 m, and so on. Total length, mass and inertia are
# identical across all four — the only thing that changes is how many unactuated degrees
# of freedom the single actuator has to stabilise.
#
# Two consequences, both of which are the point:
#   * the four policies share one 26-dim observation space, so any of them can be
#     restored into any other run with `--checkpoint_load_mode weights`. That is what
#     the padded observation exists for, and what the staged-curriculum arm needs.
#   * a difficulty comparison across them is not confounded by the pole also getting
#     longer or heavier.
#
# If you instead want four standalone cartpoles with equal-length links and no transfer
# between them, override per run, e.g. for the 2-link case:
#     env.geometry.n_max=2 env.geometry.link_lengths=[0.5,0.5] \
#     env.curriculum.enabled=false
# Their observation widths (5, 8, 14, 26) then differ and checkpoints stop being
# interchangeable.
#
# --------------------------------------------------------------------------------
# Difficulty -> free-joint count
# --------------------------------------------------------------------------------
# reset_utils maps  n_active = 1 + round(d * (n_max - 1)),  so with n_max = 8:
#     n = 1 -> d = 0/7      n = 2 -> d = 1/7
#     n = 4 -> d = 3/7      n = 8 -> d = 7/7
# `mode=fixed` with `init_range=[d,d]` pins every env to exactly that d for the run.
#
# Usage:
#   export OMNI_KIT_ACCEPT_EULA=YES
#   [WANDB_PROJECT=task_curriculum] [SEED=42] [MAX_EPOCHS=1500] [LINKS="1 2 4 8"] \
#     experiments/train_multilink_cartpole_sapg.sh
#
# Small-network ablation (see MultiLinkCartpoleSAPGSmall.yaml):
#   AGENT=rl_games_sapg_small_cfg_entry_point TAG=_small \
#     sbatch --array=0-3 experiments/train_multilink_cartpole_sapg.sh

set -euo pipefail

# Under sbatch, SLURM copies the batch script into /var/spool/slurmd, so BASH_SOURCE no
# longer points into the repo — resolving REPO_ROOT from it lands in the spool dir.
# Prefer SLURM_SUBMIT_DIR, fall back to BASH_SOURCE for direct invocation, and verify.
if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/isaacsimenvs/train.py" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
  else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi
if [[ ! -f "$REPO_ROOT/isaacsimenvs/train.py" ]]; then
  echo "REPO_ROOT=$REPO_ROOT does not contain isaacsimenvs/train.py." >&2
  echo "Submit from the repo root, or pass REPO_ROOT=/path/to/task_curriculum." >&2
  exit 1
fi
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv_isaacsim/bin/python}"
TRAIN="$REPO_ROOT/isaacsimenvs/train.py"
TASK_ID="Isaacsimenvs-MultiLinkCartpole-Direct-v0"

SEED="${SEED:-42}"
# Which rl_games config to train against, and a suffix keeping its runs separate from
# other arrays' (output dir, wandb group, wandb run name, checkpoint name).
#   rl_games_sapg_cfg_entry_point        play2perfect hyperparams, LSTM(1024)+4xMLP
#   rl_games_sapg_small_cfg_entry_point  same settings, MLP [256,128,64], no LSTM
AGENT="${AGENT:-rl_games_sapg_cfg_entry_point}"
TAG="${TAG:-}"

# fixed           : one run per entry in LINKS, difficulty pinned (the n=1/2/4/8 baselines)
# adaptive        : ONE run, staircase. Starts at n=1 and advances by one link each time
#                   the leader block clears ADAPT_THRESHOLD. The whole batch moves to the
#                   new rung, so earlier ns leave the distribution entirely.
# mixture         : ONE run, growing support. Envs sample UNIFORMLY over n in 1..X, and a
#                   successful check widens X by one. Earlier ns are retained and
#                   rehearsed forever.
# mixture_control : the no-curriculum control for `mixture` -- X is pinned at n_max from
#                   step 0, so envs sample uniformly over 1..n_max the whole run.
#
# mixture vs mixture_control is the cleanly-posed curriculum experiment: both arms end on
# the SAME final distribution (uniform over 1..n_max) and are scored the same way; the
# only difference is whether the support grows or is there from the start.
CURRICULUM_MODE="${CURRICULUM_MODE:-fixed}"
ADAPT_THRESHOLD="${ADAPT_THRESHOLD:-0.7}"
ADAPT_INTERVAL="${ADAPT_INTERVAL:-2000}"
ADAPT_MIN_EPISODES="${ADAPT_MIN_EPISODES:-2000}"
# SAPG's leader block; the curriculum scores only these envs (see score_last_n_envs).
LEADER_BLOCK="${LEADER_BLOCK:-4096}"
NUM_ENVS="${NUM_ENVS:-24576}"
MAX_EPOCHS="${MAX_EPOCHS:-1500}"
LINKS="${LINKS:-1 2 4 8}"
N_MAX="${N_MAX:-8}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/experiments/runs/cartpole_sapg${TAG}_s${SEED}}"

# Under `--array`, each task trains exactly one link count so the four run in parallel
# on four separate GPU allocations.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  read -r -a _ALL_LINKS <<< "$LINKS"
  if (( SLURM_ARRAY_TASK_ID >= ${#_ALL_LINKS[@]} )); then
    echo "array index $SLURM_ARRAY_TASK_ID exceeds LINKS='$LINKS'; nothing to do." >&2
    exit 0
  fi
  LINKS="${_ALL_LINKS[$SLURM_ARRAY_TASK_ID]}"
  echo "array task $SLURM_ARRAY_TASK_ID -> ${LINKS} link(s)"
fi

# 8 links x 0.125 m = 1.0 m total, identical at every free-joint count.
LINK_LENGTHS="${LINK_LENGTHS:-[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]}"

# Interactive HTML viewer, logged to wandb as `interactive_viewer`. Pose-only: it reads
# one env's joint vector per step and never touches a camera or the RTX renderer, so
# unlike --capture_video it does not need --enable_cameras and costs ~nothing.
VIEWER_LEN="${VIEWER_LEN:-300}"
VIEWER_INTERVAL="${VIEWER_INTERVAL:-2000}"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
# Keep the Omniverse shader cache off NFS; the first Kit boot builds RTX shaders and
# doing that over the network is painfully slow.
export OMNI_KIT_CACHE_PATH="${OMNI_KIT_CACHE_PATH:-/tmp/$USER/ov_cache}"
mkdir -p "$OMNI_KIT_CACHE_PATH" "$OUT_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "No interpreter at $PYTHON — run scripts/install_isaacsim.sh first, or set PYTHON=." >&2
  exit 1
fi

# Difficulty that pins exactly n free joints, as a fraction with full precision.
difficulty_for () {
  "$PYTHON" - "$1" "$N_MAX" <<'PY'
import sys
n, n_max = int(sys.argv[1]), int(sys.argv[2])
print(0.0 if n_max <= 1 else (n - 1) / (n_max - 1))
PY
}

STEP_PER_LINK="$("$PYTHON" -c "print(1.0/($N_MAX-1))")"
case "$CURRICULUM_MODE" in
  adaptive)        RUN_LIST="adaptive" ;;
  mixture)         RUN_LIST="mixture" ;;
  mixture_control) RUN_LIST="mixture_control" ;;
  *)               RUN_LIST="$LINKS" ;;
esac

for N in $RUN_LIST; do
  if [[ "$CURRICULUM_MODE" == "mixture" || "$CURRICULUM_MODE" == "mixture_control" ]]; then
    RUN="$CURRICULUM_MODE"
    # difficulty_levels=N_MAX puts the grid exactly on the integer link counts, so each
    # n is drawn equally often. advance_lo_with_hi stays false so range_lo is pinned at
    # 0 and the support GROWS instead of stepping.
    if [[ "$CURRICULUM_MODE" == "mixture" ]]; then
      MIX_MODE=adaptive; MIX_INIT="[0.0,0.0]"
    else
      MIX_MODE=fixed;    MIX_INIT="[0.0,1.0]"   # full support from step 0
    fi
    CURRICULUM_ARGS=(
      "env.curriculum.enabled=true"
      "env.curriculum.mode=$MIX_MODE"
      "env.curriculum.difficulty_levels=$N_MAX"
      "env.curriculum.advance_lo_with_hi=false"
      "env.curriculum.init_range=$MIX_INIT"
      "env.curriculum.final_range=[0.0,1.0]"
      "env.curriculum.adapt_step=$STEP_PER_LINK"
      "env.curriculum.adapt_success_threshold=$ADAPT_THRESHOLD"
      "env.curriculum.adapt_interval=$ADAPT_INTERVAL"
      "env.curriculum.adapt_min_episodes=$ADAPT_MIN_EPISODES"
      "env.curriculum.score_last_n_envs=$LEADER_BLOCK"
    )
    echo "=============================================================="
    if [[ "$CURRICULUM_MODE" == "mixture" ]]; then
      echo "  SAPG cartpole :: MIXTURE  n ~ U{1..X}, X grows 1 -> $N_MAX"
    else
      echo "  SAPG cartpole :: MIXTURE CONTROL  n ~ U{1..$N_MAX} from step 0"
    fi
    echo "  threshold=${ADAPT_THRESHOLD}  interval=${ADAPT_INTERVAL}  seed=${SEED}"
    echo "=============================================================="
  elif [[ "$CURRICULUM_MODE" == "adaptive" ]]; then
    RUN="adaptive"
    CURRICULUM_ARGS=(
      "env.curriculum.enabled=true"
      "env.curriculum.mode=adaptive"
      "env.curriculum.init_range=[0.0,0.0]"
      "env.curriculum.final_range=[0.0,1.0]"
      "env.curriculum.adapt_step=$STEP_PER_LINK"
      "env.curriculum.advance_lo_with_hi=true"
      "env.curriculum.adapt_success_threshold=$ADAPT_THRESHOLD"
      "env.curriculum.adapt_interval=$ADAPT_INTERVAL"
      "env.curriculum.adapt_min_episodes=$ADAPT_MIN_EPISODES"
      "env.curriculum.score_last_n_envs=$LEADER_BLOCK"
    )
    echo "=============================================================="
    echo "  SAPG cartpole :: ADAPTIVE  n: 1 -> $N_MAX  (+1 link per advance)"
    echo "  threshold=${ADAPT_THRESHOLD}  interval=${ADAPT_INTERVAL}  seed=${SEED}"
    echo "=============================================================="
  else
    if (( N < 1 || N > N_MAX )); then
      echo "skipping n=$N: outside [1, $N_MAX]" >&2
      continue
    fi
    D="$(difficulty_for "$N")"
    RUN="n${N}"
    CURRICULUM_ARGS=(
      "env.curriculum.enabled=true"
      "env.curriculum.mode=fixed"
      "env.curriculum.init_range=[$D,$D]"
      "env.curriculum.final_range=[$D,$D]"
    )
    echo "=============================================================="
    echo "  SAPG cartpole :: ${N} link(s)   difficulty=${D}   seed=${SEED}"
    echo "=============================================================="
  fi

  WANDB_ARGS=()
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    WANDB_ARGS=(
      --wandb_activate
      --wandb_project "$WANDB_PROJECT"
      --wandb_group "cartpole_sapg${TAG}_s${SEED}"
      --wandb_name "cartpole_sapg${TAG}_${RUN}_s${SEED}"
    )
  fi

  "$PYTHON" "$TRAIN" \
    --task "$TASK_ID" \
    --agent "$AGENT" \
    --headless \
    "${WANDB_ARGS[@]}" \
    "hydra.run.dir=$OUT_ROOT/$RUN" \
    "env.scene.num_envs=$NUM_ENVS" \
    "env.geometry.n_max=$N_MAX" \
    "env.geometry.link_lengths=$LINK_LENGTHS" \
    "${CURRICULUM_ARGS[@]}" \
    "agent.params.seed=$SEED" \
    "agent.params.config.name=0_cartpole_sapg${TAG}_${RUN}" \
    "agent.params.config.max_epochs=$MAX_EPOCHS" \
    --capture_viewer \
    --capture_viewer_len "$VIEWER_LEN" \
    --capture_viewer_interval "$VIEWER_INTERVAL" \
    2>&1 | tee "$OUT_ROOT/${RUN}.log"

  echo "finished $RUN -> $OUT_ROOT/$RUN"
done

echo
echo "all done. runs under $OUT_ROOT"
