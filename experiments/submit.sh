#!/usr/bin/env bash
# Submit one of the job scripts here, from the repo root. sbatch writes the log to
# debug_outputs/train_logs/.slurm/ until the job knows its own folder: a training job then moves it
# into its run folder, an eval job into <experiment>/results/ (a symlink stays behind).
#
#   experiments/submit.sh [sbatch options] experiments/<train|eval>.sh [job args]
#
#   experiments/submit.sh \
#       experiments/train.sh --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
#       --experiment rigid_body_cloth
#   experiments/submit.sh --array=0-2 experiments/train.sh ...   # seeds 1-3 (agent.params.seed)
#
# sbatch options must be written --opt=value (the first argument not starting with '-' is the job
# script). They override the job script's #SBATCH defaults (portal: --partition=portal, one RTX 6000
# Ada), which is how one script serves every cluster -- e.g. on bos14
#   experiments/submit.sh --partition=gpu_a100 --gres=gpu:rtxpro6000:1 experiments/train.sh ...
#
# Works on every cluster checkout: the repo root is this file's location, not a hardcoded path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/debug_outputs/train_logs/.slurm"

opts=()
while [[ $# -gt 0 && "$1" == -* ]]; do opts+=("$1"); shift; done
[[ $# -gt 0 ]] || { sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 2; }
job="$1"; shift

log="%x_%j"
for o in "${opts[@]}"; do [[ "$o" == --array=* ]] && log="%x_%A_%a"; done

exec sbatch --chdir="${ROOT}" --output="${ROOT}/debug_outputs/train_logs/.slurm/${log}.out" \
    --error="${ROOT}/debug_outputs/train_logs/.slurm/${log}.out" "${opts[@]}" "${job}" "$@"
