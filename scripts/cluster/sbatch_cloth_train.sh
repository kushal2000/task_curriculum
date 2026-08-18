#!/bin/bash
# SAPG finetune of the Play policy onto cloth folding, 4 GPUs on one node.
#
#   sbatch scripts/cluster/sbatch_cloth_train.sh [run_name]
#
# Follows the play2perfect finetune recipe (arXiv 2606.26428) with two deliberate departures,
# both recorded in the configs: domain randomisation is OFF (they keep it on for sim2real
# robustness; we want the cleanest signal first on a task with rare reward events), and the scale
# is smaller because a cloth costs 42.3 MB/env against rigid assembly's negligible footprint.
#
# `--checkpoint_load_mode weights` is what makes this a FINETUNE rather than a resume: it loads
# model + normaliser state only, and starts the optimiser, epoch counter and LR schedule fresh.
# The release README omits this flag and `train.py` defaults to `resume`, which would also restore
# the pretrained run's optimiser and env state.

#SBATCH --job-name=cloth_sapg
#SBATCH --partition=portal
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail
cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

RUN_NAME="${1:-cloth_fold_sapg}"
NGPU="${NGPU:-4}"
CHECKPOINT="${CHECKPOINT:-/share/portal/kk837/simtoolreal/pretrained_policy/model.pth}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "[train] run=$RUN_NAME gpus=$NGPU host=$(hostname) branch=$BRANCH"
echo "[train] checkpoint=$CHECKPOINT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# torchrun is not on PATH in this venv; go through the module. Learned the hard way -- a bare
# `torchrun` exits 127 here.
scripts/newton_py -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$NGPU" \
    isaacsimenvs/train.py \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    --checkpoint "$CHECKPOINT" \
    --checkpoint_load_mode weights \
    --distributed \
    --headless \
    --capture_viewer \
    --capture_viewer_len 600 \
    --capture_viewer_interval 200 \
    --capture_viewer_env_id 0 \
    --capture_viewer_github_raw_base "https://raw.githubusercontent.com/kushal2000/task_curriculum/${BRANCH}/" \
    --wandb_activate \
    --wandb_project cloth_folding \
    --wandb_group sapg_finetune \
    --wandb_name "$RUN_NAME" \
    env.cloth.nan_policy=reset \
    "agent.params.config.name=0_${RUN_NAME}"

echo "[train] run=$RUN_NAME exit=$?"
