#!/bin/bash
# A/B(/C) driver for the question this repo exists to answer:
#   does learning a task through a curriculum beat learning the hard task directly?
#
# Three arms, matched on total environment steps, all scored on the same held-out
# condition (difficulty pinned to 1.0):
#
#   A  curriculum   one run, adaptive difficulty: the range starts easy and advances
#                   only when the policy clears adapt_success_threshold
#   B  control      one run, curriculum disabled: every env at difficulty 1.0 from
#                   step 0. This is the baseline the curriculum has to beat.
#   C  staged       three runs, difficulty pinned progressively harder, each restoring
#                   the previous stage's weights. The classic "train easy, transfer,
#                   train hard" formulation, and the reason the cartpole observation is
#                   padded to n_max — the network shape is identical at every stage.
#
# Arms A and B differ in exactly one config field (curriculum.enabled) so nothing else
# can explain a gap between them. Keep it that way.
#
# Usage:
#   export OMNI_KIT_ACCEPT_EULA=YES
#   experiments/run_curriculum.sh MultiLinkCartpole 42
#   experiments/run_curriculum.sh BottleFlip 42
#
# Runs sequentially on purpose: one Isaac Sim process per GPU (see docs/ Gotchas).

set -euo pipefail

TASK_NAME="${1:-MultiLinkCartpole}"
SEED="${2:-42}"

# Under sbatch the batch script is copied to /var/spool/slurmd, so BASH_SOURCE does not
# resolve into the repo; prefer SLURM_SUBMIT_DIR when it looks right.
if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/isaacsimenvs/train.py" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
  else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi
PYTHON="${PYTHON:-$REPO_ROOT/.venv_isaacsim/bin/python}"
TRAIN="$REPO_ROOT/isaacsimenvs/train.py"

TASK_ID="Isaacsimenvs-${TASK_NAME}-Direct-v0"
GROUP="${TASK_NAME,,}_curriculum_s${SEED}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/experiments/runs/$GROUP}"

# Total training budget, split evenly across arm C's stages.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-900}"
STAGE_EPOCHS=$((TOTAL_EPOCHS / 3))
NUM_ENVS="${NUM_ENVS:-4096}"

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS=(--wandb_activate --wandb_project "$WANDB_PROJECT" --wandb_group "$GROUP")
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "No interpreter at $PYTHON. Create it per docs/isaacsim_installation.md, or set PYTHON=." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

run () {
  local name="$1"; shift
  echo "=============================================================="
  echo "  $GROUP :: $name"
  echo "=============================================================="
  "$PYTHON" "$TRAIN" \
    --task "$TASK_ID" \
    --headless \
    "${WANDB_ARGS[@]}" \
    ${WANDB_PROJECT:+--wandb_name "${GROUP}_${name}"} \
    "env.scene.num_envs=$NUM_ENVS" \
    "agent.params.seed=$SEED" \
    "hydra.run.dir=$OUT_ROOT/$name" \
    "$@"
}

# Newest rl_games checkpoint under a run dir. rl_games writes into <train_dir>/nn/.
latest_ckpt () {
  find "$OUT_ROOT/$1" -name '*.pth' -print0 2>/dev/null \
    | xargs -0 -r ls -t 2>/dev/null | head -n 1
}

# --------------------------------------------------------------------------
# Arm A — adaptive curriculum
# --------------------------------------------------------------------------
run "A_curriculum" \
  "agent.params.config.max_epochs=$TOTAL_EPOCHS" \
  "env.curriculum.enabled=true" \
  "env.curriculum.mode=adaptive" \
  "env.curriculum.init_range=[0.0,0.15]" \
  "env.curriculum.final_range=[0.0,1.0]"

# --------------------------------------------------------------------------
# Arm B — control: the hard task from scratch, same budget
# --------------------------------------------------------------------------
run "B_control" \
  "agent.params.config.max_epochs=$TOTAL_EPOCHS" \
  "env.curriculum.enabled=false"

# --------------------------------------------------------------------------
# Arm C — staged transfer. `--checkpoint_load_mode weights` restarts the optimiser and
# rollout state, keeping only the policy — which is what "transfer" should mean here.
# --------------------------------------------------------------------------
STAGE_RANGES=("[0.0,0.34]" "[0.0,0.67]" "[0.0,1.0]")
PREV_CKPT=""
for i in 0 1 2; do
  STAGE_ARGS=(
    "agent.params.config.max_epochs=$STAGE_EPOCHS"
    "env.curriculum.enabled=true"
    "env.curriculum.mode=fixed"
    "env.curriculum.init_range=${STAGE_RANGES[$i]}"
  )
  if [[ -n "$PREV_CKPT" ]]; then
    STAGE_ARGS+=(--checkpoint "$PREV_CKPT" --checkpoint_load_mode weights)
  fi
  run "C_stage$i" "${STAGE_ARGS[@]}"

  PREV_CKPT="$(latest_ckpt "C_stage$i")"
  if [[ -z "$PREV_CKPT" ]]; then
    echo "Stage $i produced no checkpoint under $OUT_ROOT/C_stage$i — cannot chain." >&2
    exit 1
  fi
  echo "stage $i checkpoint: $PREV_CKPT"
done

# --------------------------------------------------------------------------
# Evaluation — every arm on identical task instances (difficulty pinned to 1.0)
# --------------------------------------------------------------------------
for arm in A_curriculum B_control C_stage2; do
  CKPT="$(latest_ckpt "$arm")"
  if [[ -z "$CKPT" ]]; then
    echo "skipping eval for $arm: no checkpoint found" >&2
    continue
  fi
  run "eval_$arm" \
    --test --checkpoint "$CKPT" --checkpoint_load_mode weights \
    "agent.params.config.max_epochs=1" \
    "env.curriculum.enabled=true" \
    "env.curriculum.eval_difficulty=1.0"
done

echo "done. runs under $OUT_ROOT"
