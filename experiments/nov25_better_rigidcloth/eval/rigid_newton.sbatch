#!/bin/bash
#SBATCH --job-name=rigid_newton
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/rigid_newton/slurm_%A_%a.out
# A policy on the PhysX plate pool's recipe with robot AND plates on Newton/MJWarp
# (Isaacsimenvs-RigidClothNewton-Direct-v0). Array task i = line (i / SEEDS + 1) of $CONFIGS
# ("name|checkpoint|recipe|hydra overrides"), seed i % SEEDS + 1; 32 envs, rand start, box2-mid.
set -uo pipefail
SEEDS=${SEEDS:-2}; I=$SLURM_ARRAY_TASK_ID; S=$((I % SEEDS + 1))
IFS='|' read -r NAME CKPT RECIPE OVR < <(sed -n "$((I / SEEDS + 1))p" "${CONFIGS:-outputs/rigid_newton/configs.txt}")
read -r -a TOK <<<"$OVR"; OVR=()
for t in "${TOK[@]}"; do if [[ "$t" =~ ^[A-Z_][A-Z0-9_]*= ]]; then export "$t"; else OVR+=("$t"); fi; done  # KEY=VAL -> env var
O=${OUTDIR:-outputs/rigid_newton}; OUT=$O/${NAME}__s$S.json
[ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/rl_games" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
FREE_KB=$(df -Pk /tmp | awk 'NR==2{print $4}'); if [ "${FREE_KB:-0}" -gt 8000000 ]; then TROOT=/tmp; else TROOT=$PWD/.cache/jobtmp; fi
export TMPDIR=$TROOT/${USER}_${SLURM_JOB_ID}_${I}; export WARP_CACHE_PATH=$TMPDIR/warp; mkdir -p "$WARP_CACHE_PATH"
cp -a .cache/warp_vbd/. "$WARP_CACHE_PATH/" 2>/dev/null
echo "[rigid_newton] $NAME seed $S: ${OVR[*]} env: $(env | grep -E '^RN_' | tr '\n' ' ')"
.venv/isaaclab3/bin/python -m isaacsimenvs.scripts.evaluation.eval --task Isaacsimenvs-RigidClothNewton-Direct-v0 \
  --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml --sapg_expl_coef 0 --max_steps 900 \
  --success_tolerance 0.01 --num_assets_per_type 1 --single_variant --randomize --experiment "$RECIPE" \
  --num_envs 32 --seed $S --out "$OUT" \
  env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset env.termination.success_steps=10 \
  "env.fold_pool.shapes=[box2-mid]" env.fold_pool.one_shape_per_env=false env.fold_pool.init_fold_prob=0.0 \
  env.fold_pool.goal_task.mode=half_fold "${OVR[@]}"
[ -s "$OUT" ] || { echo "[rigid_newton] FAILED $OUT"; exit 1; }
