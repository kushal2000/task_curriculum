#!/bin/bash
# Submit the prior-vs-scratch comparison: {pretrained, scratch} x SEEDS, each a 4-node array job.
#
#   scripts/cluster/submit_prior_vs_scratch.sh            # seeds 1 2 3 -> 6 runs, 24 GPUs
#   SEEDS="4 5" scripts/cluster/submit_prior_vs_scratch.sh
#
# Refuses to submit from a dirty tree: the run log records the commit, and results are only
# attributable to a commit if that commit is what ran.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -n "$(git status --porcelain)" ]; then
    echo "working tree is dirty; commit first" >&2
    exit 1
fi

for seed in ${SEEDS:-1 2 3}; do
    for arm in pretrained scratch; do
        jid=$(sbatch --parsable --array=0-3 --job-name="cloth_pvs_${arm}_s${seed}" \
            scripts/cluster/bos14_cloth_prior_vs_scratch.sh "$arm" "$seed")
        echo "$arm seed=$seed job=$jid"
    done
done
