#!/bin/bash
# bos14_px_eval_matrix.sh on Engaging: same index layout, protocol and output names, but run from
# the submit dir, PhysX inside the Isaac Sim container (scripts/isaacsim_py), and the policy
# config from POLICY_CFG. Submit from the repo root, choosing the GPU per target, e.g. for one
# uploaded checkpoint (seed 1 = indices 0-13; target t -> indices 2t, 2t+1):
#   export CKPT=checkpoints/<ckpt>.pth POLICY_CFG=checkpoints/config.yaml KIND=pool TRAIN_JOB=175385 RUN_TAG=flat7DRABC
#   sbatch --array=0-3,6-13              scripts/cluster/engaging_px_eval_matrix.sh   # all but the Newton pool
#   sbatch --array=4-5 --gres=gpu:h200:1 scripts/cluster/engaging_px_eval_matrix.sh   # Newton pool, 320 envs
# The Newton pool OOMs on a 48 GB L40S; h200 or rtx_pro_6000 (96 GB, what bos14 used) both fit.
#
# Partition: mit_preemptable, not mit_normal_gpu. mit_normal_gpu caps a user at 2 GPUs and has few
# free big-memory cards (H200 estimates ran to days); mit_preemptable allows 4 and starts in minutes
# to hours. A task is 1-19 min and the matrix skips outputs that already exist, so a preempted task
# requeues and only redoes the runs it had not finished. -p mit_normal_gpu on the command line
# restores the old behaviour.
#SBATCH --job-name=px_mx
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
export REPO="${SLURM_SUBMIT_DIR}" PHYSX_PY="${SLURM_SUBMIT_DIR}/scripts/isaacsim_py"
export POLICY_CFG="${POLICY_CFG:-checkpoints/config.yaml}"
exec bash "${SLURM_SUBMIT_DIR}/scripts/cluster/bos14_px_eval_matrix.sh"
