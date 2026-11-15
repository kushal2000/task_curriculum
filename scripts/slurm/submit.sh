#!/usr/bin/env bash
# Submit one of the job scripts here, from the repo root, with its log in debug_outputs/train_logs/slurm/.
#
#   scripts/slurm/submit.sh [sbatch options] scripts/slurm/<train|eval|viser>.sh [job args]
#
#   scripts/slurm/submit.sh \
#       scripts/slurm/train.sh --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightHLRL-Direct-v0 \
#       --experiment flat7RW_base --checkpoint pretrained_policy/kuka_sharpa/model.pth
#   scripts/slurm/submit.sh --array=0-2 scripts/slurm/train.sh ...   # seeds 1-3 (agent.params.seed)
#
# sbatch options must be written --opt=value (the first argument not starting with '-' is the job
# script). They override the job script's #SBATCH defaults (portal: --partition=portal, one RTX 6000
# Ada), which is how one script serves every cluster -- e.g. on bos14
#   scripts/slurm/submit.sh --partition=gpu_a100 --gres=gpu:rtxpro6000:1 scripts/slurm/train.sh ...
#
# Works on every cluster checkout: the repo root is this file's location, not a hardcoded path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${ROOT}/debug_outputs/train_logs/slurm"

opts=()
while [[ $# -gt 0 && "$1" == -* ]]; do opts+=("$1"); shift; done
[[ $# -gt 0 ]] || { sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 2; }
job="$1"; shift

log="%x_%j"
for o in "${opts[@]}"; do [[ "$o" == --array=* ]] && log="%x_%A_%a"; done

exec sbatch --chdir="${ROOT}" --output="${ROOT}/debug_outputs/train_logs/slurm/${log}.out" \
    --error="${ROOT}/debug_outputs/train_logs/slurm/${log}.out" "${opts[@]}" "${job}" "$@"
