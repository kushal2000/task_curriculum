#!/usr/bin/env bash
# One allocated GPU; cases run sequentially in separate simulator processes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec experiments/submit.sh --job-name=yibo_rigid_ablation \
    experiments/nov16_yibo_our_dr/rigid_eval_job.sh --task Isaacsimenvs-RigidBodyCloth-Direct-v0 "$@"
