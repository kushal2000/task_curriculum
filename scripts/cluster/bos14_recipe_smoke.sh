#!/bin/bash
# 5-epoch training smoke of a PhysX fold-pool recipe on Flexiv + right Wuji (real setup).
#   EXPERIMENT=flat7FlexivWujiRHL sbatch scripts/cluster/bos14_recipe_smoke.sh
#SBATCH --job-name=recipe_smoke
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=01:00:00
#SBATCH --exclude=bos14-node-[040,047,051,057,062,086,092,104,105,109,110,112,121]
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd /home/yiboc/mit/task_curriculum_flexiv
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
EXP="${EXPERIMENT:-flat7FlexivWujiRHL}"; ENVS="${ENVS:-1536}"
echo "[smoke] experiment=$EXP envs=$ENVS host=$(hostname -s)"
.venv_isaacsim/bin/python -m isaacsimenvs.train \
    --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 --experiment "$EXP" \
    --agent rl_games_sapg_cfg_entry_point --headless --single_variant \
    --checkpoint pretrained_policy/flexiv_right_wuji/nov8_dr_friction_0p5.pth --checkpoint_load_mode weights \
    "env.scene.num_envs=${ENVS}" agent.params.config.minibatch_size=24576 \
    agent.params.config.central_value_config.minibatch_size=24576 \
    "agent.params.config.expl_coef_block_size=$((ENVS / 6))" \
    "agent.params.config.max_epochs=${EPOCHS:-5}" \
    "agent.params.config.name=0_smoke_${EXP}_${SLURM_JOB_ID}"
echo "[smoke] exit=$?"
