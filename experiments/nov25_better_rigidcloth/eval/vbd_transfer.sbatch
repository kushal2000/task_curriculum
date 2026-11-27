#!/bin/bash
#SBATCH --job-name=vbd_xfer
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/vbd_transfer/slurm_%A_%a.out
# Zero-shot VBD transfer of a policy under a physics fix. Array task i = line (i / SEEDS + 1) of $CONFIGS
# ("name|checkpoint|hydra overrides"), seed (i % SEEDS + 1). eval.py --protocol vbd.yaml, 32 envs, rand start.
set -uo pipefail
SEEDS=${SEEDS:-2}; I=$SLURM_ARRAY_TASK_ID; S=$((I % SEEDS + 1))
IFS='|' read -r NAME CKPT OVR < <(sed -n "$((I / SEEDS + 1))p" "${CONFIGS:-outputs/vbd_transfer/configs.txt}")
read -r -a TOK <<<"$OVR"; OVR=()
for t in "${TOK[@]}"; do if [[ "$t" =~ ^[A-Z_][A-Z0-9_]*= ]]; then export "$t"; else OVR+=("$t"); fi; done  # KEY=VAL -> env var
O=outputs/vbd_transfer; OUT=$O/${NAME}__s$S.json
[ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/rl_games" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
FREE_KB=$(df -Pk /tmp | awk 'NR==2{print $4}'); if [ "${FREE_KB:-0}" -gt 8000000 ]; then TROOT=/tmp; else TROOT=$PWD/.cache/jobtmp; fi
export TMPDIR=$TROOT/${USER}_${SLURM_JOB_ID}_${I}; export WARP_CACHE_PATH=$TMPDIR/warp; mkdir -p "$WARP_CACHE_PATH"
cp -a .cache/warp_vbd/. "$WARP_CACHE_PATH/" 2>/dev/null
echo "[xfer] $NAME seed $S ckpt $CKPT: ${OVR[*]} env: $(env | grep -E '^(NEWTON_|RN_|VBD_)' | tr '\n' ' ')"
( while sleep 20; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES%%,*} 2>/dev/null; done ) > $O/${NAME}__s$S.gpumem & GPUMON=$!
.venv/isaaclab3/bin/python -m isaacsimenvs.scripts.evaluation.eval --task Isaacsimenvs-Cloth-Direct-v0 \
  --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml --sapg_expl_coef 0 --max_steps ${MAX_STEPS:-900} \
  --success_tolerance 0.01 --num_assets_per_type 1 --single_variant $([ -n "${PINNED:-}" ] || echo --randomize) \
  --protocol isaacsimenvs/tasks/cloth/config/eval/vbd.yaml --num_envs ${NUM_ENVS:-32} --seed $S $([ "${DETERMINISTIC:-0}" = 1 ] || echo --stochastic) --out "$OUT" "${OVR[@]}"
kill $GPUMON 2>/dev/null; echo "[xfer] peak GPU MiB: $(sort -n $O/${NAME}__s$S.gpumem | tail -1)"
[ -s "$OUT" ] || { echo "[xfer] FAILED $OUT"; exit 1; }
