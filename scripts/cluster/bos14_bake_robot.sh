#!/bin/bash
# Bake a robot URDF into the USD cache for the kit-less Newton runs (VBD cloth), via a Kit-hosted
# PhysX cloth task with the same robot overlay. The key is URDF text + fix_base / self_collision, so
# pass the self-collision value the Newton run uses.
#   TASK=Isaacsimenvs-FoldPoolPhysxFlexivWujiRightHLRL-Direct-v0 SC=false sbatch scripts/cluster/bos14_bake_robot.sh
#SBATCH --job-name=bake_robot
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --exclude=bos14-node-[040,042,047,051,057,062,086,092,104,105,109,110,112,121,123]
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd /home/yiboc/mit/task_curriculum_flexiv
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
echo "== bake $(hostname -s) task=$TASK sc=${SC:-false}"
.venv_isaacsim/bin/python -m isaacsimenvs.newton.usd_cache --populate --task "$TASK" \
    "env.robot_self_collision=${SC:-false}"
echo "[bake] exit=$?"
