#!/bin/bash
# Generic GPU array: one shell command per line of $1, run from the flexiv worktree.
#   sbatch --array=0-$(( $(wc -l < cmds.txt) - 1 )) scripts/cluster/bos14_cmd_array.sh cmds.txt
#SBATCH --job-name=cmd_array
#SBATCH --partition=gpu_a100,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --exclude=bos14-node-[040,042,047,051,057,062,086,092,104,105,109,110,112,121,123]
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd /home/yiboc/mit/task_curriculum_flexiv
CMD="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")"
echo "[cmd_array] host=$(hostname -s) commit=$(git rev-parse --short HEAD) cmd=$CMD"
export PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
setsid bash -c "$CMD" &
PY=$!
trap 'kill -9 -- "-$PY" 2>/dev/null; exit 143' TERM INT
wait $PY
