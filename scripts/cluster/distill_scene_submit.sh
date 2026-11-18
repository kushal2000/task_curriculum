#!/bin/bash
# Submit one scene-cloud DAgger run (xyz of everything in the box over the table) and its evaluations.
#   scripts/cluster/distill_scene_submit.sh <config> [seed]
# config: A-clean A-noise A-noise-FK B-clean B-clean-aux B-noise-nodelay B-noise B-noise-FK B-noise-aux-FK
#   A = no goal input, B = first-frame cloth cloud with moving / stationary labels;
#   noise = version C cloud noise + delays (cloud 0-6, proprio 0-3, action 0-3 steps); nodelay = noise only;
#   aux = auxiliary keypoint head; FK = 128 forward-kinematics points on the forearm and hand.
# Each student is evaluated without noise (@clean) and with the full noise + delays (@noise): PhysX pinned +
# rand, VBD ns6 pinned + rand x seeds 1-6 (384 episodes each). Env: ITERS (3000), ENVS (4096), TIME.
# PREFIX=<p> prepends <p> to the run name (e.g. fric_ with TEACHER=<pth> EXPERIMENT=flat7RW_fric); NO_EVAL=1 trains only.
# evals only, for an existing student:  STUDENT=<pth> scripts/cluster/distill_scene_submit.sh <config> [seed]
set -euo pipefail
cd /home/yiboc/mit/task_curriculum_flexiv
cfg=$1 seed=${2:-1}
DL="--pc_delay 6 --proprio_delay 3 --action_delay 3"
extra="--cloud scene --beta_iters ${BETA_ITERS:-300} --save_every 500"
# B: the reference cloud is taken at step 12 (0.2 s), when the sheet has landed (it spawns ~4 cm up).
case "$cfg" in A-*) goal=none ;; B-*) goal=first; extra="$extra --ref_step 12" ;; *) echo "config must start with A- or B-"; exit 2 ;; esac
noise=0
case "$cfg" in *-noise*) noise=1 ;; esac
case "$cfg" in *-noise*) case "$cfg" in *-nodelay*) ;; *) extra="$extra $DL" ;; esac ;; esac
case "$cfg" in *-aux*) ;; *) extra="$extra --aux_coef 0" ;; esac
case "$cfg" in *-FK*) extra="$extra --fk_points 128" ;; esac
name="${PREFIX:-}sc_${cfg}"
mkdir -p slurm_logs/distill
dep=()
if [ -z "${STUDENT:-}" ]; then
    j=$(NAME="$name" GOAL=$goal NOISE=$noise SEED=$seed ITERS="${ITERS:-3000}" ENVS="${ENVS:-4096}" EXTRA="$extra" \
        sbatch --parsable --mem=60G --time="${TIME:-10:00:00}" --job-name="d${PREFIX:-}sc_${cfg}_s${seed}" scripts/cluster/bos14_distill_pc.sh)
    STUDENT="outputs/distill/${name}_s${seed}_${j}/student_last.pth"
    dep=(--dependency=afterok:$j)
else
    j=existing
fi
evals=""
conds="clean noise"; [ "${NO_EVAL:-0}" = 1 ] && conds=""
for cond in $conds; do
    [ "$cond" = clean ] && eargs="--eval_clean" || eargs="--eval_noise $DL"
    for grp in "1 2 3" "4 5 6"; do
        sims="vbd"; [ "$grp" = "1 2 3" ] && sims="physx vbd"
        e=$(MODE=evalx STUDENT=$STUDENT TAG="${name}_s${seed}@${cond}" SIMS="$sims" SEEDS="$grp" STARTS="pinned rand" \
            PX_STARTS="pinned rand" EVAL_ARGS="$eargs" \
            sbatch --parsable --time=06:00:00 ${EVAL_SBATCH:-} "${dep[@]}" --job-name="dse_${cfg}_s${seed}_${cond}" scripts/cluster/bos14_distill_pc.sh)
        evals="$evals $e"
    done
done
echo "$(date +%m-%d_%H:%M) scene ${PREFIX:-}$cfg s$seed train=$j evals=$evals $STUDENT" | tee -a slurm_logs/distill/jobs.txt
