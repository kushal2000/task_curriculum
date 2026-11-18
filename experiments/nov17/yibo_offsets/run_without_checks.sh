#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
mkdir -p experiments/nov17/yibo_offsets/results/strict
for seed in 0 1 2; do
  for variant in yibo yibo_with_our_offsets; do
    output="experiments/nov17/yibo_offsets/results/strict/${variant}_no_checks_seed${seed}"
    .venv/isaacsim/bin/python experiments/nov16/yibo_our_dr/evaluate_rigid.py \
      --headless --device cuda:0 --case "$variant" --seed "$seed" --episode_checks off \
      --task_yaml isaacsimenvs/scripts/configs/task/RigidBodyCloth.yaml \
      --num_envs 256 --max_steps 600 --out "${output}.json" > "${output}.log" 2>&1
    test -f "${output}.json"
  done
done
