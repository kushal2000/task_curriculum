#!/bin/bash
# The comparison run: five rigid approximations and the VBD cloth, matched on everything but physics.
#
#   sbatch --array=0-17 scripts/cluster/bos14_rigid_cloth_train.sh
#
# Array index = arm * 3 + (seed - 1), i.e. tasks 0-2 are box3-mid seeds 1/2/3, 3-5 box-mid-odd, ...
# 15-17 the VBD cloth control. Run a subset with e.g. `--array=0-2,15-17`.
#
# WHAT IS HELD FIXED
#
# The learner, the prior, the reward, the observation layout, the env count, the horizon, the
# minibatch and the env-step budget. `RigidClothEnv` subclasses `ClothEnv` and overrides six members,
# five of them setup or physics, so the reward and the success criterion are literally the same code
# on the same quantities. The sixth, `_init_fold_targets`, lifts the fold target to the CHAIN's own
# ply gap: identical to the cloth's 2 mm for box3-mid and the surface-hinge variants, 5.88 mm for the
# two mid-hinge uniform ones, whose fold error therefore is not the cloth's. Otherwise the only
# difference between an arm and the control is the manipuland and the solver topology it implies.
#
# WHY THE BUDGET IS IN ENV-STEPS, NOT WALL-CLOCK
#
# Two different questions, and both are wanted. Equal env-steps answers "is this approximation as
# learnable per sample" -- the one that would be confounded by a speed difference. Wall-clock is then
# MEASURED per arm, so the equal-wall-clock comparison is recoverable from these runs while the
# equal-sample one is not recoverable from wall-clock-matched runs. 1e8 steps also matches
# `docs/results/cloth_prior_vs_scratch.md` exactly, so the control has a published reference value.
#
#   1536 envs x 32 horizon = 49152 steps/epoch; 2035 epochs = 1.00e8 steps.
#
# WHY THE PRETRAINED PRIOR ON EVERY ARM
#
# `cloth_prior_vs_scratch.md`: at this budget the prior arm learned to fold in 3/3 seeds and
# from-scratch produced 0 held folds in 480 evaluation episodes. A from-scratch comparison at 1e8
# steps would measure nothing but that, for all six arms.
#
# WHY THREE SEEDS
#
# The same document measured 93% / 18% / 13% held-fold rates across three seeds of an IDENTICAL
# config. One seed per arm could not distinguish an approximation from a draw.

#SBATCH --job-name=rc_train
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
# 80G, not 96G: of the 44 gpu_5090 nodes, 24 report RealMemory=92000 MB and 18 report 93000 MB
# (re-checked 2026-10-03, `sinfo -p gpu_5090 -o '%n %m'`), so a 96G (98304 MB) request fits only the
# two outliers that have 126 GB -- the other 42 nodes are silently ineligible and the array
# serialises behind those two, pending with Reason=None while the cluster sits empty.
#SBATCH --mem=80G
# 16 h, not 24 h: a 24 h array of this script sat at `Reason=InvalidAccount` while the same array at
# 16 h ran immediately. The 24 h was a SYMPTOM, not the cause -- both partitions report
# MaxTime=UNLIMITED, and `bos14_rigid_cloth_retrain_rc3.sh` submits a 24 h array to gpu_a100 without
# complaint. On this cluster `InvalidAccount` really means "no node currently satisfies the whole
# request", so what the 24 h ask ran into was gpu_5090's state at the time, not a time limit. 16 h
# is kept only because it is ample: the budget needs ~8 h for the VBD control at ~3.5 k env-steps/s
# and far less for the rigid arms.
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs

# Index 5 is the control: the VBD sheet, i.e. the unmodified Cloth task.
ARMS=(box3-mid box-mid-odd cyl-mid-odd box2-surface box-surface vbd)

# RC_TASK_ID / RC_JOB_ID let a single failed array task be requeued on its own while keeping the run
# NAME it would have had, so the checkpoint still lands where the evaluation manifest looks for it.
# They exist because exporting `SLURM_ARRAY_JOB_ID` / `SLURM_ARRAY_TASK_ID` to do the same job makes
# the scheduler reject the submission with `Reason=InvalidAccount` -- those names are reserved, and
# setting them in the job environment is a footgun rather than an override.
I="${RC_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
ARM="${ARMS[$((I / 3))]}"
SEED=$(( (I % 3) + 1 ))

# 768 envs, set by the CONTROL's memory ceiling, not by choice. The VBD cloth OOMs on a 32 GB
# RTX 5090 at 1536 and at 1152 envs (it needs another 3.7 GB); the published 1536/rank runs used
# 96 GB RTX PRO 6000s. The rigid arms fit far more, but a speed comparison across different env
# counts or different GPUs measures the hardware, so every arm gets the same 768 on the same card.
ENVS="${ENVS:-768}"
EPOCHS="${EPOCHS:-4069}"        # 4069 x 768 x 32 = 1.00e8 env-steps, the same budget as at 1536
MINIBATCH=24576
BLOCK=$((ENVS / 6))             # num_blocks is baked into the checkpoint at 6 and cannot change
JOB_TAG="${RC_JOB_ID:-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
NAME="0_rc_${ARM}_s${SEED}_${JOB_TAG}"

if [ "$ARM" = "vbd" ]; then
    TASK=Isaacsimenvs-Cloth-Direct-v0
    VARIANT_ARGS=()
else
    TASK=Isaacsimenvs-RigidCloth-Direct-v0
    VARIANT_ARGS=("env.rigid_cloth.variant=${ARM}")
fi

export PYTHONUNBUFFERED=1
echo "[train] arm=$ARM seed=$SEED task=$TASK envs=$ENVS epochs=$EPOCHS block=$BLOCK"
echo "[train] name=$NAME host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
echo "[train] dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MEMLOG="slurm_logs/${SLURM_JOB_NAME}_${JOB_TAG}_${I}.gpumem.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 \
    >"$MEMLOG" &
SMI_PID=$!

START=$(date +%s)
scripts/newton_py -m isaacsimenvs.train \
    --task "$TASK" \
    --agent rl_games_sapg_cfg_entry_point \
    --checkpoint pretrained_policy/model.pth \
    --checkpoint_load_mode weights \
    --single_variant \
    "${VARIANT_ARGS[@]}" \
    env.cloth.nan_policy=reset \
    "env.scene.num_envs=${ENVS}" \
    agent.params.config.multi_gpu=False \
    "agent.params.config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.central_value_config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.expl_coef_block_size=${BLOCK}" \
    "agent.params.config.max_epochs=${EPOCHS}" \
    "agent.params.config.name=${NAME}" \
    "agent.params.seed=${SEED}"
RC=$?

kill "$SMI_PID" 2>/dev/null
WALL=$(( $(date +%s) - START ))
echo "[train] arm=$ARM seed=$SEED exit=$RC wall_s=$WALL"
# Only for a run that actually finished. A crashed job divides the FULL budget by its own short
# wall-clock and prints a throughput of 10 million env-steps/s, which is exactly the kind of number
# that ends up in a results table.
if [ "$RC" -eq 0 ]; then
    echo "[train] env_steps_per_s=$(python3 -c "print(round(${EPOCHS}*${ENVS}*32/max(${WALL},1), 1))")"
else
    echo "[train] env_steps_per_s=n/a (exit $RC)"
fi
echo "[train] peak_gpu_mib=$(cut -d, -f2 "$MEMLOG" | tr -dc '0-9\n' | sort -n | tail -1)"
# The checkpoint directory, so the eval matrix can find it without guessing the Hydra timestamp.
find outputs -type d -name "$NAME" -newermt "-25 hours" | tee "slurm_logs/${NAME}.ckptdir"
