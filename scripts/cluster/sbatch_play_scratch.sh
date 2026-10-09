#!/bin/bash
# SimToolReal's Play recipe from scratch, on either robot: the Flexiv + Wuji pretraining run and its
# Kuka + Sharpa control.
#
#   sbatch scripts/cluster/sbatch_play_scratch.sh flexiv_wuji [run_name] [hydra overrides...]
#   sbatch scripts/cluster/sbatch_play_scratch.sh kuka_sharpa [run_name] [hydra overrides...]
#
# WHY A KUKA CONTROL
#
# The Flexiv cannot start from SimToolReal's checkpoint (27 joints, not 29), and this repo has never
# trained Play from scratch: every Kuka number here comes from the released policy. If the Flexiv
# run comes out weak, the control says whether that is the robot or our reproduction of the recipe.
# So the two runs are identical except for the task id -- same script, same seed, same config.
#
# WHAT IS CHANGED FROM THE RECIPE
#
# Only the random force/torque impulses on the object (domain_randomization.force_scale and
# torque_scale, 20 and 2 by default) are off, on both robots. Everything else is Play.yaml /
# PlaySAPG.yaml as shipped: 8192 envs on one GPU, SAPG with 2 blocks of 4096, seed 42.
#
# The interactive viewer loads the robot from raw GitHub at this branch, so the branch -- and for
# the Flexiv, assets/flexiv_wuji/ -- must be pushed. The launcher warns if it is not.

#SBATCH --job-name=play_scratch
#SBATCH --partition=portal
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail
cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs

ROBOT="${1:?usage: sbatch_play_scratch.sh flexiv_wuji|kuka_sharpa [run_name] [overrides...]}"
case "$ROBOT" in
    flexiv_wuji) TASK=Isaacsimenvs-PlayFlexivWuji-Direct-v0 ;;
    kuka_sharpa) TASK=Isaacsimenvs-Play-Direct-v0 ;;
    *) echo "unknown robot $ROBOT" >&2; exit 2 ;;
esac
RUN_NAME="${2:-play_scratch_${ROBOT}}_${SLURM_JOB_ID:-local}"
OVERRIDES=("${@:3}")
SEED="${SEED:-42}"
CAPTURE_INTERVAL="${CAPTURE_INTERVAL:-6000}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

if ! git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    echo "[train] WARNING: branch $BRANCH is not on origin -- viewer meshes will 404." >&2
fi

echo "[train] robot=$ROBOT task=$TASK run=$RUN_NAME seed=$SEED host=$(hostname) branch=$BRANCH"
echo "[train] commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"
echo "[train] overrides=${OVERRIDES[*]:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

.venv_isaacsim/bin/python -m isaacsimenvs.train \
    --task "$TASK" \
    --agent rl_games_sapg_cfg_entry_point \
    --headless \
    --capture_viewer \
    --capture_viewer_len 600 \
    --capture_viewer_interval "$CAPTURE_INTERVAL" \
    --capture_viewer_env_id 0 \
    --capture_viewer_github_raw_base "https://raw.githubusercontent.com/kushal2000/task_curriculum/${BRANCH}/" \
    --capture_viewer_url_check warn \
    --wandb_activate \
    --wandb_project play_pretraining \
    --wandb_group scratch_no_wrench \
    --wandb_name "$RUN_NAME" \
    env.domain_randomization.force_scale=0.0 \
    env.domain_randomization.torque_scale=0.0 \
    "agent.params.config.name=0_${RUN_NAME}" \
    "agent.params.seed=${SEED}" \
    "${OVERRIDES[@]}"

echo "[train] exit=$?"
