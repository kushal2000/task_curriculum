#!/bin/bash
# Pretrain Play on the Flexiv + right human-like Wuji with all domain randomization on: the same
# task as submit_play_flexiv_right_nodr.sh, with PlayFlexivWujiRightHumanLike.yaml's DR as
# shipped -- obs and action delay (up to 3 steps), object-state delay (up to 10) and noise (1 cm,
# 5 deg), joint-velocity obs noise (0.1) -- and the random force/torque impulses on the object
# on, at force_scale 5 and torque_scale 1.0 (down from Play.yaml's 20 and 2).
#
#   scripts/cluster/submit_play_flexiv_right_dr.sh [extra hydra overrides...]
#   SEED=1 scripts/cluster/submit_play_flexiv_right_dr.sh
#
# Refuses a dirty tree, as submit_play_flexiv_right_nodr.sh does.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -n "$(git status --porcelain)" ]; then
    echo "working tree is dirty; commit (and push) first" >&2
    exit 1
fi

WRENCH_DR=on WANDB_GROUP=flexiv_right_dr sbatch --job-name=play_flexiv_right_dr \
    scripts/cluster/sbatch_play_scratch.sh flexiv_wuji_right_human_like play_scratch_flexiv_right_dr \
    env.domain_randomization.force_scale=5.0 \
    env.domain_randomization.torque_scale=1.0 \
    "$@"
