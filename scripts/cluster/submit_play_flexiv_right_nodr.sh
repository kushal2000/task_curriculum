#!/bin/bash
# Pretrain Play on the Flexiv + right human-like Wuji with no domain randomization:
# Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0 -- the real setup's table, the real home pose
# with joint 7 at 120, goals kept 1 cm clear of the table, Wuji's Isaac Lab hand drives.
#
# Off: obs and action delay, object-state delay and noise, joint-velocity obs noise, and the random
# force/torque impulses on the object. Kept: the reset distribution (object pose noise, arm and
# finger joint noise, table height +-1 cm), which is the task, not DR.
#
#   scripts/cluster/submit_play_flexiv_right_nodr.sh [extra hydra overrides...]
#   SEED=1 scripts/cluster/submit_play_flexiv_right_nodr.sh
#
# Refuses a dirty tree, as submit_prior_vs_scratch.sh does: the run log records the commit, and
# the viewer loads the robot from GitHub at this branch, so commit and push first.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -n "$(git status --porcelain)" ]; then
    echo "working tree is dirty; commit (and push) first" >&2
    exit 1
fi

WRENCH_DR=off WANDB_GROUP=flexiv_right_nodr sbatch --job-name=play_flexiv_right_nodr \
    scripts/cluster/sbatch_play_scratch.sh flexiv_wuji_right_human_like play_scratch_flexiv_right_nodr \
    env.domain_randomization.use_obs_delay=false \
    env.domain_randomization.use_action_delay=false \
    env.domain_randomization.use_object_state_delay_noise=false \
    env.domain_randomization.joint_velocity_obs_noise_std=0.0 \
    "$@"
