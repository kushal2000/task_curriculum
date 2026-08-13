#!/bin/bash
#SBATCH --job-name=cartpole_sapg
#SBATCH --partition=portal
#SBATCH --output=/share/portal/kk837/task_curriculum/experiments/runs/slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#
# Train SAPG policies for the multi-link cartpole at 1, 2, 4 and 8 free links.
#
# Runs directly (`experiments/train_multilink_cartpole_sapg.sh`) or under SLURM
# (`sbatch experiments/train_multilink_cartpole_sapg.sh`). Either way the four runs go
# SEQUENTIALLY: booting a second Kit process on a GPU that already has one can crash the
# one that is starting (docs/isaacsim_installation.md, Gotchas).
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

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv_isaacsim/bin/python}"
TRAIN="$REPO_ROOT/isaacsimenvs/train.py"
TASK_ID="Isaacsimenvs-MultiLinkCartpole-Direct-v0"

SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_EPOCHS="${MAX_EPOCHS:-1500}"
LINKS="${LINKS:-1 2 4 8}"
N_MAX="${N_MAX:-8}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/experiments/runs/cartpole_sapg_s${SEED}}"

# 8 links x 0.125 m = 1.0 m total, identical at every free-joint count.
LINK_LENGTHS="${LINK_LENGTHS:-[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]}"

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

for N in $LINKS; do
  if (( N < 1 || N > N_MAX )); then
    echo "skipping n=$N: outside [1, $N_MAX]" >&2
    continue
  fi
  D="$(difficulty_for "$N")"
  RUN="n${N}"
  echo "=============================================================="
  echo "  SAPG cartpole :: ${N} link(s)   difficulty=${D}   seed=${SEED}"
  echo "=============================================================="

  WANDB_ARGS=()
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    WANDB_ARGS=(
      --wandb_activate
      --wandb_project "$WANDB_PROJECT"
      --wandb_group "cartpole_sapg_s${SEED}"
      --wandb_name "cartpole_sapg_n${N}_s${SEED}"
    )
  fi

  "$PYTHON" "$TRAIN" \
    --task "$TASK_ID" \
    --agent rl_games_sapg_cfg_entry_point \
    --headless \
    "${WANDB_ARGS[@]}" \
    "hydra.run.dir=$OUT_ROOT/$RUN" \
    "env.scene.num_envs=$NUM_ENVS" \
    "env.geometry.n_max=$N_MAX" \
    "env.geometry.link_lengths=$LINK_LENGTHS" \
    "env.curriculum.enabled=true" \
    "env.curriculum.mode=fixed" \
    "env.curriculum.init_range=[$D,$D]" \
    "env.curriculum.final_range=[$D,$D]" \
    "agent.params.seed=$SEED" \
    "agent.params.config.name=0_cartpole_sapg_n${N}" \
    "agent.params.config.max_epochs=$MAX_EPOCHS" \
    2>&1 | tee "$OUT_ROOT/${RUN}.log"

  echo "finished n=$N -> $OUT_ROOT/$RUN"
done

echo
echo "all done. runs under $OUT_ROOT"
