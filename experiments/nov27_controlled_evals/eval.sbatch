#!/bin/bash
#SBATCH --job-name=ctrl_eval
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/controlled_evals/slurm_%A_%a.out
# The controlled eval (README.md): default.yaml, pinned start, sampled actions. Array task i = line
# (i / SEEDS + 1) of $CONFIGS ("name|checkpoint|overrides"), seed (i % SEEDS + 1). KEY=VAL tokens in the
# overrides become env vars (VBD_STIFF_FACTOR=..., NUM_ENVS=..., RANDOMIZE=1 for the robustness check); the
# rest are hydra overrides on default.yaml.
#   CONFIGS=my.txt sbatch --array=0-$((2 * $(wc -l < my.txt) - 1)) experiments/nov27_controlled_evals/eval.sbatch
set -uo pipefail
SEEDS=${SEEDS:-2}; I=$SLURM_ARRAY_TASK_ID; S=$((I % SEEDS + 1))
IFS='|' read -r NAME CKPT OVR < <(sed -n "$((I / SEEDS + 1))p" "$CONFIGS")
read -r -a TOK <<<"$OVR"; OVR=()
for t in "${TOK[@]}"; do if [[ "$t" =~ ^[A-Z_][A-Z0-9_]*= ]]; then export "$t"; else OVR+=("$t"); fi; done
O=outputs/controlled_evals; mkdir -p $O; OUT=$O/${NAME}__s$S.json
[ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/rl_games" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
FREE_KB=$(df -Pk /tmp | awk 'NR==2{print $4}'); if [ "${FREE_KB:-0}" -gt 8000000 ]; then TROOT=/tmp; else TROOT=$PWD/.cache/jobtmp; fi
export TMPDIR=$TROOT/${USER}_${SLURM_JOB_ID}_${I}; export WARP_CACHE_PATH=$TMPDIR/warp; mkdir -p "$WARP_CACHE_PATH"
cp -a .cache/warp_vbd/. "$WARP_CACHE_PATH/" 2>/dev/null
echo "[ctrl] $NAME seed $S ckpt $CKPT: ${OVR[*]} env: $(env | grep -E '^(NEWTON_|VBD_|NUM_ENVS|RANDOMIZE)' | tr '\n' ' ')"
( while sleep 20; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES%%,*} 2>/dev/null; done ) > $O/${NAME}__s$S.gpumem & GPUMON=$!
.venv/isaaclab3/bin/python -m isaacsimenvs.scripts.evaluation.eval --task Isaacsimenvs-Cloth-Direct-v0 \
  --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml --sapg_expl_coef 0 --max_steps ${MAX_STEPS:-900} \
  --success_tolerance 0.01 --num_assets_per_type 1 --single_variant $([ "${RANDOMIZE:-0}" = 1 ] && echo --randomize) \
  --protocol isaacsimenvs/tasks/cloth/config/eval/default.yaml --num_envs ${NUM_ENVS:-32} --seed $S --stochastic \
  --out "$OUT" "${OVR[@]}"
kill $GPUMON 2>/dev/null; echo "[ctrl] peak GPU MiB: $(sort -n $O/${NAME}__s$S.gpumem | tail -1)"
[ -s "$OUT" ] || { echo "[ctrl] FAILED $OUT"; exit 1; }
