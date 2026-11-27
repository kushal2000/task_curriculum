#!/bin/bash
#SBATCH --job-name=eval_pinned_default_high_table_friction
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/controlled_evals/slurm_%x_%A_%a.out
# VBD cloth, eval/default_high_table_friction.yaml (default.yaml, cloth-table friction 1.0), pinned
# start, sampled actions, both policies x 2 seeds -> outputs/controlled_evals/eval_pinned_default_high_table_friction/.
#   sbatch experiments/nov27_controlled_evals/eval_pinned_default_high_table_friction.sbatch
EVAL=eval_pinned_default_high_table_friction PROTOCOL=isaacsimenvs/tasks/cloth/config/eval/default_high_table_friction.yaml \
  source experiments/nov27_controlled_evals/_vbd_eval.sh "$@"
