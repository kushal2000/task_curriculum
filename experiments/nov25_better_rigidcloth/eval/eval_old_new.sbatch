#!/bin/bash
#SBATCH --job-name=eval_64live
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-15
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/eval_nov25/slurm_%A_%a.out
# Success rates of the two 64live policies: old designs + old DR vs new designs + new DR.
# index = policy(2) * 8 + cell(4: old plate2 2vert 2flaps) * 2 + seed(2). Settings as experiments/eval.sh's
# rigidbody_flat:rand, with this checkout's code on the original checkout's Isaac Sim venv.
set -uo pipefail
I=$SLURM_ARRAY_TASK_ID
POLS=(play_pc_rigidclothRL_64live play_pc_rigidclothRL_64live_inplace); CELLS=(old plate2 2vert 2flaps)
POL=${POLS[$((I/8))]}; CELL=${CELLS[$((I/2%4))]}; SEED=$((I%2+1))
PYBIN=/share/portal/kk837/task_curriculum/.venv/isaacsim/bin/python
NEW=experiments/nov25_better_rigidcloth/viewer/new_shapes_pc.yaml; NENV=448; SHAPES=
names() { (cd "isaacsimenvs/tasks/cloth/config/designs/$1" && ls *.yaml | sed 's/\.yaml$//' | paste -sd,); }
case $CELL in
  old)    RECIPE=experiments/nov24_play_pc_rl/play_pc_rl_A.yaml
          SHAPES=box2-mid,box2diag4-mid,box2x8-mid,cross4-mid,corners8-mid,box4-mid,star8-mid ;;
  plate2) RECIPE=$NEW; SHAPES=plate2 ;;
  2vert)  RECIPE=$NEW; SHAPES=$(names 2vert+hinge); NENV=450 ;;
  2flaps) RECIPE=$NEW; SHAPES=$(names 2flaps+hinge) ;;
esac
source experiments/_common.sh --task Isaacsimenvs-RigidCloth-Direct-v0
export LOG_DIR=outputs/eval_nov25 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO:$REPO/rl_games"
OUT=outputs/eval_nov25/${POL#play_pc_rigidclothRL_}__${CELL}__s${SEED}.json
[ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
provenance eval
run_watched "${POL}_${CELL}_s${SEED}_$SLURM_JOB_ID" "$PYBIN" -m isaacsimenvs.scripts.evaluation.eval --headless \
  --task Isaacsimenvs-RigidCloth-Direct-v0 --checkpoint pretrained_policy/$POL.pth --experiment "$RECIPE" \
  --num_envs $NENV --sapg_expl_coef 0 --max_steps 900 --success_tolerance 0.01 --num_assets_per_type 1 \
  --single_variant --randomize --seed $SEED --out "$OUT" \
  env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset env.termination.success_steps=10 \
  "env.fold_pool.shapes=[$SHAPES]" env.fold_pool.one_shape_per_env=true env.fold_pool.init_fold_prob=0.0 \
  env.fold_pool.goal_task.mode=half_fold
[ -s "$OUT" ] || { echo "[eval] FAILED $OUT"; exit 1; }
