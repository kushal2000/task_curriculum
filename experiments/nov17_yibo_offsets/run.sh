#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
results_dir="${RESULTS_DIR:-experiments/nov17_yibo_offsets/results/offset_aware}"
mkdir -p "$results_dir"
for seed in ${SEEDS:-0 1 2}; do
  for variant in yibo yibo_with_our_offsets; do
    output="${results_dir}/${variant}_seed${seed}"
    .venv/isaacsim/bin/python experiments/nov16_yibo_our_dr/evaluate_rigid.py \
      --headless --device cuda:0 --case "$variant" --seed "$seed" \
      --task_yaml isaacsimenvs/scripts/configs/task/RigidBodyCloth.yaml \
      --num_envs 256 --max_steps 600 --out "${output}.json" > "${output}.log" 2>&1
    test -f "${output}.json"
  done
done
