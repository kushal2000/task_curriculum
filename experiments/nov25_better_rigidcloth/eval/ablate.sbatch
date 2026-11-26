#!/bin/bash
#SBATCH --job-name=ablate_64live
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-33
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/eval_nov25_ablate/slurm_%A_%a.out
# One-at-a-time DR ablation of 64live (ablate_configs.txt: name|recipe|shapes old/new|envs|overrides).
# index = config * 2 + seed - 1; settings otherwise as eval_old_new.sbatch.
set -uo pipefail
I=$SLURM_ARRAY_TASK_ID; SEED=$((I%2+1))
IFS='|' read -r NAME RECIPE SET NENV OVR < <(sed -n "$((I/2+1))p" "${CONFIGS:-experiments/nov25_better_rigidcloth/eval/ablate_configs.txt}")
read -r -a OVR <<<"$OVR"
names() { (cd "isaacsimenvs/tasks/cloth/config/designs/$1" && ls *.yaml | sed 's/\.yaml$//' | paste -sd,); }
if [ "$SET" = old ]; then SHAPES=box2-mid,box2diag4-mid,box2x8-mid,cross4-mid,corners8-mid,box4-mid,star8-mid
else SHAPES=plate2,$(names 2vert+hinge),$(names 2flaps+hinge); fi
PYBIN=/share/portal/kk837/task_curriculum/.venv/isaacsim/bin/python
source experiments/_common.sh --task Isaacsimenvs-RigidCloth-Direct-v0
export LOG_DIR=${OUTDIR:-outputs/eval_nov25_ablate} PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO:$REPO/rl_games"
OUT=${OUTDIR:-outputs/eval_nov25_ablate}/64live__${NAME}__s${SEED}.json
[ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
echo "[ablate] $NAME seed $SEED recipe $RECIPE shapes $SET envs $NENV overrides ${OVR[*]}"
provenance eval
if [ -n "${PROBE:-}" ] && [[ $NAME == probe_* ]]; then export DEPTH_OUT=${OUT%.json}.depth.json; MOD=(experiments/nov25_better_rigidcloth/eval/depth_probe.py); else MOD=(-m isaacsimenvs.scripts.evaluation.eval); fi
run_watched "${NAME}_s${SEED}_$SLURM_JOB_ID" "$PYBIN" "${MOD[@]}" --headless \
  --task Isaacsimenvs-RigidCloth-Direct-v0 --checkpoint pretrained_policy/play_pc_rigidclothRL_64live.pth --experiment "$RECIPE" \
  --num_envs $NENV --sapg_expl_coef 0 --max_steps 900 --success_tolerance 0.01 --num_assets_per_type 1 \
  --single_variant --randomize --seed $SEED --out "$OUT" \
  env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset env.termination.success_steps=10 \
  "env.fold_pool.shapes=[$SHAPES]" env.fold_pool.one_shape_per_env=true env.fold_pool.init_fold_prob=0.0 \
  env.fold_pool.goal_task.mode=half_fold "${OVR[@]}"
[ -s "$OUT" ] || { echo "[ablate] FAILED $OUT"; exit 1; }
