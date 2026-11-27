#!/bin/bash
# Shared body of the VBD controlled evals (sourced by eval_pinned_*.sbatch). Array task i: policy (i / SEEDS),
# seed (i % SEEDS + 1), both policies; pinned start, sampled actions. Needs PROTOCOL (an eval/*.yaml) and
# EVAL (the output folder name); optional PINNED_YAW_DEG (180: the cloth's pinned heading, scripts/evaluation/protocol.py), NUM_ENVS (32), SEEDS (2), MAX_STEPS (900), extra hydra overrides
# in "$@", KEY=VAL env vars set by the caller (e.g. VBD_STIFF_FACTOR).
set -uo pipefail
[ -n "${VBD_STIFF_FACTOR:-}" ] && EVAL=${EVAL}_stiffhalves${VBD_STIFF_FACTOR}${VBD_CREASE_FACTOR:+_crease$VBD_CREASE_FACTOR}  # plate-like halves
[ -n "${PINNED_YAW_DEG:-}" ] && [ "${PINNED_YAW_DEG}" != 180 ] && EVAL=${EVAL}_yaw${PINNED_YAW_DEG}  # another pinned heading
SEEDS=${SEEDS:-2}; I=$SLURM_ARRAY_TASK_ID; S=$((I % SEEDS + 1)); P=(64live 64live_inplace); N=${P[$((I / SEEDS))]}
O=outputs/controlled_evals/$EVAL; mkdir -p "$O"; OUT=$O/${N}__s$S.json
[ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/rl_games" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
FREE_KB=$(df -Pk /tmp | awk 'NR==2{print $4}'); if [ "${FREE_KB:-0}" -gt 8000000 ]; then TROOT=/tmp; else TROOT=$PWD/.cache/jobtmp; fi
export TMPDIR=$TROOT/${USER}_${SLURM_JOB_ID}_${I}; export WARP_CACHE_PATH=$TMPDIR/warp; mkdir -p "$WARP_CACHE_PATH"
cp -a .cache/warp_vbd/. "$WARP_CACHE_PATH/" 2>/dev/null
echo "[$EVAL] $N seed $S: $PROTOCOL $* env: $(env | grep -E '^(NEWTON_|VBD_)' | tr '\n' ' ')"
( while sleep 20; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES%%,*} 2>/dev/null; done ) > "$O/${N}__s$S.gpumem" & GPUMON=$!
.venv/isaaclab3/bin/python -m isaacsimenvs.scripts.evaluation.eval --task Isaacsimenvs-Cloth-Direct-v0 \
  --checkpoint pretrained_policy/play_pc_rigidclothRL_$N.pth --policy_config pretrained_policy/config.yaml --sapg_expl_coef 0 \
  --max_steps ${MAX_STEPS:-900} --success_tolerance 0.01 --num_assets_per_type 1 --single_variant --stochastic \
  --protocol "$PROTOCOL" --num_envs ${NUM_ENVS:-32} --seed $S --out "$OUT" "$@"
kill $GPUMON 2>/dev/null; echo "[$EVAL] peak GPU MiB: $(sort -n "$O/${N}__s$S.gpumem" | tail -1)"
[ -s "$OUT" ] || { echo "[$EVAL] FAILED $OUT"; exit 1; }
