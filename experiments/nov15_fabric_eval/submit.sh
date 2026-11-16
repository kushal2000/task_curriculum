#!/bin/bash
# Fabric eval of the two 2-GPU rigid-cloth runs (nov15_rigid_cloth_init: 659538 prior-init,
# 659539 scratch) on the deformable cloth, 7 measured fabrics x {fold, unfold}.
# Run from the repo root: experiments/nov15_fabric_eval/submit.sh
#
# The runs are still training, so each one's newest snapshot is resolved NOW and the path is passed
# to the job (later snapshots do not change what it evaluates). JSONs: results/<tag>__<task>__<fabric>_s<seed>.json.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Every call is appended to submissions.log: when, the commit, the command, each checkpoint, the job ids.
LOG=experiments/nov15_fabric_eval/submissions.log
{ echo "## $(date '+%Y-%m-%d %H:%M:%S') host=$(hostname -s) commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"
  echo "\$ experiments/nov15_fabric_eval/submit.sh"; } >>"$LOG"
RUNS=debug_outputs/train_logs/nov
declare -A JOB=([prior]=659538 [scratch]=659539)
for TAG in prior scratch; do
    D=$(ls -d "$RUNS"/*_"${JOB[$TAG]}")
    CKPT=$(ls "$D"/nn/snap_*_ep_*.pth | sed -E 's/.*_ep_([0-9]+)\.pth$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    CMD=(experiments/submit.sh --array=0-13 --job-name="fabeval_$TAG"
         experiments/nov15_fabric_eval/job.sh "$TAG" "$CKPT")
    OUT=$("${CMD[@]}")
    echo "[submit] $TAG: $CKPT -> $OUT"
    { echo "$TAG: ${CMD[*]}"; echo "  -> $OUT"; } >>"$LOG"
done
