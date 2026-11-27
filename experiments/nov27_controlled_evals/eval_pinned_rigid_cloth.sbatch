#!/bin/bash
#SBATCH --job-name=eval_pinned_rigid_cloth
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/controlled_evals/slurm_%x_%A_%a.out
# The training world: hinged rigid plates on PhysX (Isaac Lab 2.3), eval/rigid_cloth.yaml (the policies'
# recipe, plate physics written out, Hausdorff success), the 10 cm half fold (box2-mid), from the VBD evals' pinned start
# (identity pose -> 180 deg, base.reset_yaw), sampled actions, both policies x 2 seeds
# -> outputs/controlled_evals/eval_pinned_rigid_cloth/. The reference the VBD evals transfer to.
#   sbatch experiments/nov27_controlled_evals/eval_pinned_rigid_cloth.sbatch
set -uo pipefail
EVAL=eval_pinned_rigid_cloth; SEEDS=${SEEDS:-2}
[ -n "${PINNED_YAW_DEG:-}" ] && [ "${PINNED_YAW_DEG}" != 180 ] && EVAL=${EVAL}_yaw${PINNED_YAW_DEG}  # another pinned heading
I=$SLURM_ARRAY_TASK_ID; S=$((I % SEEDS + 1)); P=(64live 64live_inplace); N=${P[$((I / SEEDS))]}
O=outputs/controlled_evals/$EVAL; mkdir -p "$O"; OUT=$O/${N}__s$S.json
[ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/rl_games"
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}_$I; mkdir -p "$TMPDIR"
/share/portal/kk837/task_curriculum/.venv/isaacsim/bin/python -m isaacsimenvs.scripts.evaluation.eval --task Isaacsimenvs-RigidCloth-Direct-v0 \
  --checkpoint pretrained_policy/play_pc_rigidclothRL_$N.pth --policy_config pretrained_policy/config.yaml --sapg_expl_coef 0 \
  --max_steps ${MAX_STEPS:-900} --success_tolerance 0.01 --num_assets_per_type 1 --single_variant --stochastic --headless \
  --experiment isaacsimenvs/tasks/cloth/config/eval/rigid_cloth.yaml --num_envs ${NUM_ENVS:-32} --seed $S --out "$OUT" "$@"
[ -s "$OUT" ] || { echo "[$EVAL] FAILED $OUT"; exit 1; }
