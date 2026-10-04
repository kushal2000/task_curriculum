#!/bin/bash
# Round 3: the three REMAINING approximations under the same three fixes that took box3-mid and
# box2-surface from ~0% cloth transfer to ~50%, with and without physics DR.
#
#   sbatch --array=0-17 scripts/cluster/bos14_rigid_cloth_retrain_rc3.sh
#
# Index = group * 9 + arm * 3 + (seed - 1):
#
#   group 0  "fix"    the three fixes only -- and all three are RigidClothCfg DEFAULTS, so this arm
#                     passes no physics override at all: reset heading applied (the adapter's
#                     `task_yaw`), contact surface at the cloth's height (`contact_margin: -1.0`
#                     => 7 mm), cloth friction per contact pair (`friction_priority: true`).
#   group 1  "fixdr"  the same, plus per-env physics randomisation re-drawn at every reset:
#                     slat and fingertip friction, hinge damping and stiffness, contact height, mass.
#
#   arm 0 cyl-mid-odd (17 bodies), arm 1 box-mid-odd (17), arm 2 box-surface (16); seeds 1-3.
#
# Held identical to round 2 (array 155349) so these rows land in the SAME table as box3-mid and
# box2-surface: pretrained prior, SAPG, 1536 envs, horizon 32, minibatch 24576, 2035 epochs =
# 1.0e8 env-steps. Only two things differ, and both are forced by the arms themselves:
#
# 1. THE CONTACT BUFFER. Round 2 hardcoded `per_env_triangle_pairs=65536`, which is correct for a
#    2-3 body chain and was measured to log zero overflow there. These arms are 16-17 bodies, and
#    round 1 measured ~119k pairs/env for them -- while they were still NOT FOLDING. A folded
#    16-slat chain is two plies of eight slats face to face, the configuration that maximises
#    triangle pairs, and the eval guard caught ~285k/env there. Overflow silently DROPS contacts,
#    which makes the sim both wrong and faster -- i.e. it flatters exactly the transfer number this
#    experiment exists to measure, and it appears precisely when the policy starts succeeding.
#
#    Cutting the other way: `contact_margin != 0` removes non-adjacent slat-slat pairs
#    (`RigidClothEnv._drop_slat_slat_pairs`), which for a 17-slat chain is C(17,2) - 16 = 120 pairs
#    per env -- the very pairs that drove that 285k. So post-fix demand is far lower than round 1's,
#    by an amount that had to be MEASURED rather than assumed: see the driven-fold probe at 1536
#    envs (`OUT_TAG=_e1536_p65k` / `_p524k`) that chose the default below.
#
#    The default errs large, because over-sizing costs only GPU memory while under-sizing fails
#    silently. Override once the probe says a smaller budget survives a driven fold:
#    `TRI_PAIRS=65536 sbatch --partition=gpu_5090 --gres=gpu:rtx5090:1 ...`
#
# 2. THE CARD. 524288 pairs x 1536 envs does not fit a 32 GB RTX 5090 (round 2 measured an OOM at
#    solver init with only 262144), so the default partition is the 96 GB RTX PRO 6000. Wall-clock
#    is therefore NOT comparable with round 2 -- only with itself.
#
# box-surface is the one variant the approximation study registered "to be measured, not trained
# on": it is the only one with any instability (it diverged in both drape configs) and its `solimp`
# reach through Isaac Lab -> Newton -> MJWarp has never been verified. It is trained here anyway, by
# request. `nan_policy=reset` absorbs a non-finite env instead of killing the run, so READ THE
# DIVERGENCE COUNT before reading its score -- a run that resets constantly can post a plausible
# number while modelling nothing.

#SBATCH --job-name=rc3_train
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=24:00:00
# Node-health exclusion from 2026-09-30; drop it once the node is confirmed good. 104/105 are
# gpu_5090 nodes and cannot be picked for this gpu_a100 job, so naming them here excluded nothing.
#SBATCH --exclude=bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs

ARMS=(cyl-mid-odd box-mid-odd box-surface)
GROUPS_=(fix fixdr)
I="${RC_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
GROUP="${GROUPS_[$((I / 9))]}"
ARM="${ARMS[$(((I % 9) / 3))]}"
SEED=$(( (I % 3) + 1 ))

ENVS="${ENVS:-1536}"
EPOCHS="${EPOCHS:-2035}"        # 2035 x 1536 x 32 = 1.00e8 env-steps, round 2's budget exactly
MINIBATCH=24576
BLOCK=$((ENVS / 6))             # num_blocks is baked into the checkpoint at 6
TRI_PAIRS="${TRI_PAIRS:-524288}"   # see note 1 in the header
JOB_TAG="${RC_JOB_ID:-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
NAME="0_rc3_${ARM}_${GROUP}_s${SEED}_${JOB_TAG}"

DR_ARGS=()
[ "$GROUP" = "fixdr" ] && DR_ARGS=("env.rigid_cloth.randomization.enabled=true")

export PYTHONUNBUFFERED=1
echo "[train] arm=$ARM group=$GROUP seed=$SEED envs=$ENVS epochs=$EPOCHS block=$BLOCK tri_pairs=$TRI_PAIRS"
echo "[train] name=$NAME host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MEMLOG="slurm_logs/${SLURM_JOB_NAME}_${JOB_TAG}_${I}.gpumem.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 \
    >"$MEMLOG" &
SMI_PID=$!

START=$(date +%s)
scripts/newton_py -m isaacsimenvs.train \
    --task Isaacsimenvs-RigidCloth-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    --checkpoint pretrained_policy/model.pth \
    --checkpoint_load_mode weights \
    --single_variant \
    "env.rigid_cloth.variant=${ARM}" \
    "${DR_ARGS[@]}" \
    env.cloth.nan_policy=reset \
    "env.scene.num_envs=${ENVS}" \
    "env.newton.per_env_triangle_pairs=${TRI_PAIRS}" \
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
echo "[train] arm=$ARM group=$GROUP seed=$SEED exit=$RC wall_s=$WALL"
if [ "$RC" -eq 0 ]; then
    echo "[train] env_steps_per_s=$(python3 -c "print(round(${EPOCHS}*${ENVS}*32/max(${WALL},1), 1))")"
else
    echo "[train] env_steps_per_s=n/a (exit $RC)"
fi
echo "[train] peak_gpu_mib=$(cut -d, -f2 "$MEMLOG" | tr -dc '0-9\n' | sort -n | tail -1)"
# The two numbers that decide whether this run's score means anything. A buffer overflow drops
# contacts silently, and `nan_policy=reset` hides divergence behind a healthy-looking reset.
# Slurm names the log by `%A_%a`, so the path has to be built from SLURM_ARRAY_JOB_ID /
# SLURM_ARRAY_TASK_ID -- NOT from JOB_TAG, which is RC_JOB_ID on a requeued task and names a file
# that does not exist. Grepping a missing file printed 0 for both, i.e. a clean bill of health on
# exactly the run that needed checking, so the absent case now says so instead.
LOG="slurm_logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.out"
if [ -f "$LOG" ]; then
    echo "[train] overflow_warnings=$(grep -ciE 'buffer overflow|overflowed' "$LOG" || true)"
    echo "[train] nonfinite_mentions=$(grep -ci 'non-finite' "$LOG" || true)"
else
    echo "[train] WARNING: log $LOG not found -- overflow_warnings/nonfinite_mentions NOT CHECKED" >&2
fi
find outputs -type d -name "$NAME" -newermt "-25 hours" | tee "slurm_logs/${NAME}.ckptdir"
