#!/bin/bash
# Contact-layer curriculum: finetune one fabric's 16 mm release policy on its thin preset while the
# hand/table contact layer steps down (default 8 -> 4 -> 2 -> 1 mm). Everything else is the THIN
# preset of engaging_fabric_finetune.sh (real-thickness self-contact, two-stage plasticity,
# full-surface contact). A stage trains in CHUNK-epoch pieces (resuming its own run) and advances
# once the mean `episode_final/done_persist` over the last 20 summary points reaches THRESH; a stage
# that has not got there by MAX epochs stops the curriculum.
#
#   sbatch scripts/cluster/engaging_fabric_curriculum.sh                       # cotton
#   FABRIC=towel sbatch scripts/cluster/engaging_fabric_curriculum.sh
#   CHUNK=2 MAX=4 THRESH=0 STAGES="0.008 0.004" sbatch --time=01:00:00 ...   # smoke
#
# Runs: 0_ftfabcur<um>_<fabric>_<job> (one per stage). Progress: [cur] lines in this log, and
# outputs/curriculum/<fabric>_<job>/<um>.done (the stage's final checkpoint) -- a requeued job
# skips finished stages and resumes the current one from its newest periodic checkpoint.
# mit_normal_gpu caps a job at 6 h, so each job queues its own successor (afterany) under the same
# CUR_ID; the successor picks up from the .done files and the newest periodic checkpoint (every 25
# epochs), and exits at once if the curriculum has already ended. Stop the chain: scancel the queued job.
#SBATCH --job-name=fabcur
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs
PY=.venv_isaaclab3/bin/python
FABRIC="${FABRIC:-cotton}"
STAGES="${STAGES:-0.008 0.004 0.002 0.001}"
CHUNK="${CHUNK:-100}"
MAX="${MAX:-1500}"
THRESH="${THRESH:-0.30}"
# FINAL_THRESH: pass bar of the last stage (a polish stage: the policy settles at the final layer).
FINAL_THRESH="${FINAL_THRESH:-$THRESH}"
JOB="${CUR_ID:-${SLURM_JOB_ID:-local}}"
STATE="outputs/curriculum/${FABRIC}_${JOB}"
mkdir -p "$STATE"
# A stall under a smaller MAX than this job's is reopened (the stage resumes from its newest
# checkpoint); `ended` files without "max <n>" were written with the old default, 600.
if [ -e "$STATE/ended" ] && grep -q "^stalled" "$STATE/ended"; then
    OLDMAX=$(grep -o "max [0-9]*" "$STATE/ended" | awk '{print $2}'); OLDMAX="${OLDMAX:-600}"
    if [ "$OLDMAX" -lt "$MAX" ]; then
        mv "$STATE/ended" "$STATE/ended.max$OLDMAX"
        echo "[cur] reopening curriculum $JOB: stalled under max $OLDMAX, this job has max $MAX"
    fi
fi
[ -e "$STATE/ended" ] && { echo "[cur] curriculum $JOB already ended: $(cat "$STATE/ended")"; exit 0; }
# AFTER_CUR=<fabric>_<curriculum id>: a follow-on curriculum (e.g. one more stage after a finished
# chain, with RESUME_RUN = that chain's last stage run): run only once that chain has finished.
if [ -n "${AFTER_CUR:-}" ] && ! grep -q "^finished" "outputs/curriculum/$AFTER_CUR/ended" 2>/dev/null; then
    echo "[cur] waiting on curriculum $AFTER_CUR, not finished: $(cat "outputs/curriculum/$AFTER_CUR/ended" 2>/dev/null || echo running)"; exit 0
fi
if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_RESTART_COUNT:-0}" = 0 ]; then
    NEXT=$(CUR_ID="$JOB" sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" --job-name="${SLURM_JOB_NAME:-fabcur}" \
           scripts/cluster/engaging_fabric_curriculum.sh)
    echo "[cur] curriculum id $JOB, this job $SLURM_JOB_ID, successor queued: $NEXT"
fi
REL_ARGS="env.cloth.persist_fail_penalty=0.0 env.cloth.persist_rew_per_step=5.0 env.cloth.release_dist_rew_scale=20000.0 env.cloth.release_bonus=1000.0"
# Stage 0: the fabric's 16 mm release finetune (the THIN table's start run in engaging_fabric_finetune.sh).
declare -A START=([big_napkin]=0_ftfabrel3sRfsC_big_napkin_28602237 [chiffon]=0_ftfabrel3sRfsC_chiffon_28607526
                  [cotton]=0_ftfabrel3sRfsC2_cotton_28647593 [denim]=0_ftfabrel3sRfsC_denim_28607527
                  [napkin]=0_ftfabrel3sRfsC_napkin_28607529 [silk]=0_ftfabrel3sRfsC_silk_28607530
                  [towel]=0_ftfabrel3sRfsC_towel_28607531)
last_ckpt() { $PY -I -c "import sys; sys.path.insert(0, 'scripts/analysis'); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path('outputs'), 'last') or '')" "$1"; }
# START_CKPT=<path>: start stage 1 from this checkpoint instead (e.g. an earlier direct thin run).
PREV="${START_CKPT:-$(last_ckpt "${START[$FABRIC]}")}"
[ -f "$PREV" ] || { echo "[cur] no start checkpoint for $FABRIC" >&2; exit 1; }
echo "[cur] fabric=$FABRIC stages=($STAGES) chunk=$CHUNK max=$MAX thresh=$THRESH final=$FINAL_THRESH margin=${THIN_MARGIN:-0.003} start=$PREV restart=${SLURM_RESTART_COUNT:-0}"

# Stage "base": more training on the fabric's 16 mm release setup (the measured preset, 2 mm
# self-contact, full-surface contact: the THIN=0 config of its start run) until BASE_THRESH, before
# the thin stages. E.g. STAGES="base 0.012 0.008 0.004 0.002 0.001".
BASE_THRESH="${BASE_THRESH:-0.60}"
# True resume (optimizer, adaptive LR, epoch counter kept) instead of a weights-only restart: the
# source run's newest periodic checkpoint is copied under the stage's run name, which the finetune
# script then resumes; the stage's MAX counts from that epoch.
#   RESUME_RUN=<run>: the first stage resumes that run.  RESUME_CHAIN=1: each later stage resumes
#   the stage before it.
# A stage may set the self-contact margin as well: "<contact>@<margin>", e.g. "0.00035@0.001"
# (default margin: engaging_fabric_finetune.sh THIN_MARGIN).
seed_run() {  # seed_run <run> <source run>; prints the run's first epoch
    if ! ls outputs/*/*/"$1"/nn/periodic_"$1"_ep_*.pth >/dev/null 2>&1; then
        local src se
        src=$(ls outputs/*/*/"$2"/nn/periodic_"$2"_ep_*.pth | awk -F_ep_ '{print $NF"\t"$0}' | sort -n | tail -1 | cut -f2)  # newest by epoch
        se=${src##*_ep_}; se=${se%.pth}
        mkdir -p "outputs/seed/$JOB/$1/nn" && cp "$src" "outputs/seed/$JOB/$1/nn/periodic_${1}_ep_${se}.pth"
        echo "[cur] resume: seeded $1 from $src" >&2
    fi
    ls outputs/*/*/"$1"/nn/periodic_"$1"_ep_*.pth | sed 's/.*_ep_\([0-9]*\)\.pth/\1/' | sort -n | head -1
}
SEED_FROM="${RESUME_RUN:-}"
for T in $STAGES; do
    E0=0; EMAX="$MAX"
    if [ "$T" = base ]; then
        UM=base; TAG=base; GATE="$BASE_THRESH"
        STAGE_ENV=(THIN=0 "TRAIN_ARGS=$REL_ARGS env.cloth.full_surface_contact=true")
        # BASE_RESUME=1: a true resume of the start run (optimizer, adaptive LR, epoch counter): its
        # newest periodic checkpoint is copied under this stage's run name, which the finetune script
        # then resumes (--checkpoint_load_mode resume); MAX counts from epoch 0 of the start run.
        if [ "${BASE_RESUME:-0}" = 1 ]; then
            TAG=baser; E0=$(seed_run "0_ftfab${TAG}_${FABRIC}_${JOB}" "${START[$FABRIC]}")
        fi
    else
        C="${T%@*}"; GATE="$THRESH"
        UM=$($PY -I -c "print(round($C * 1e6))"); TAG="cur${UM}"
        STAGE_ENV=(THIN=1 "CONTACT=$C" "TRAIN_ARGS=$REL_ARGS")
        if [ "$C" != "$T" ]; then
            MUM=$($PY -I -c "print(round(${T#*@} * 1e6))"); UM="${UM}m${MUM}"; TAG="cur${UM}"
            STAGE_ENV+=("THIN_MARGIN=${T#*@}")
        fi
        [ "$T" = "${STAGES##* }" ] && GATE="$FINAL_THRESH"
        if [ -n "$SEED_FROM" ]; then
            E0=$(seed_run "0_ftfab${TAG}_${FABRIC}_${JOB}" "$SEED_FROM"); EMAX=$(( E0 + MAX ))
        fi
    fi
    RUN="0_ftfab${TAG}_${FABRIC}_${JOB}"
    E0=$(( E0 / CHUNK * CHUNK ))
    if [ -s "$STATE/$UM.done" ]; then
        PREV=$(cat "$STATE/$UM.done"); echo "[cur] contact $T done earlier -> $PREV"
        [ "${RESUME_CHAIN:-0}" = 1 ] && SEED_FROM="$RUN"; continue
    fi
    passed=0
    for ((E = E0 + CHUNK; E <= EMAX; E += CHUNK)); do
        # Skip chunks a previous (preempted) attempt already finished.
        have=$(ls outputs/*/*/"$RUN"/nn/last_"$RUN"_ep_"$E"_*.pth 2>/dev/null | head -1)
        if [ -z "$have" ]; then
            env "${STAGE_ENV[@]}" FABRICS="$FABRIC" SLURM_ARRAY_TASK_ID=0 RUN_JOB="$JOB" RELEASE=1 RUN_TAG="$TAG" \
                EPOCHS="$E" PERIODIC="${PERIODIC:-25}" PRIOR="$PREV" \
                bash scripts/cluster/engaging_fabric_finetune.sh 2>&1 | grep -v -E "^\s*$"
        fi
        P=$($PY -I scripts/analysis/train_persist.py "$RUN" --last 20)
        R=$($PY -I scripts/analysis/train_persist.py "$RUN" --last 20 --tag episode_cumulative/total_reward)
        echo "[cur] contact $T epoch $E persist $P reward $R"
        if $PY -I -c "import sys; p = float(sys.argv[1]); t = float(sys.argv[2]); sys.exit(0 if t <= 0 or (p == p and p >= t) else 1)" "$P" "$GATE"; then
            passed=1; break
        fi
    done
    CK=$(last_ckpt "$RUN")
    if [ "$passed" = 1 ] && [ -f "$CK" ]; then
        echo "$CK" >"$STATE/$UM.done"; PREV="$CK"
        echo "[cur] contact $T PASSED at epoch $E -> $CK"
        [ "${RESUME_CHAIN:-0}" = 1 ] && SEED_FROM="$RUN"
    else
        echo "[cur] contact $T STALLED (persist $P < $GATE after $MAX epochs); last checkpoint $CK"
        echo "stalled at $T max $MAX: $CK" >"$STATE/ended"
        exit 0
    fi
done
echo "[cur] curriculum finished for $FABRIC: final $PREV"
echo "finished: $PREV" >"$STATE/ended"
