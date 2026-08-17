#!/bin/bash
# Aggregate throughput across N GPUs on one node.
#
#   NGPU=4 ENVS=1024 sbatch --gres=gpu:nvidia_rtx_6000_ada_generation:4 \
#       scripts/cluster/multi_gpu_throughput.sh
#
# A Newton env instance runs on ONE GPU -- it cannot span devices. So "multi-GPU" here means N
# independent processes, one per device, which is also how distributed RL training is structured
# (each worker owns a GPU; only gradients cross). Simulation itself is embarrassingly parallel.
#
# The question this answers is therefore NOT whether it scales in principle, but whether N
# processes on a shared node actually achieve N x single-GPU throughput, or lose some of it to
# contention on host CPU, PCIe, system memory bandwidth and the filesystem during startup.
#
# Each worker gets its own CUDA_VISIBLE_DEVICES, so it sees exactly one device as cuda:0.

#SBATCH --job-name=mgpu
#SBATCH --partition=portal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail

cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs docs/results

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

NGPU="${NGPU:-2}"
ENVS="${ENVS:-1024}"
STEPS="${STEPS:-100}"

echo "[mgpu] ngpu=$NGPU envs_per_gpu=$ENVS host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -"$NGPU"

pids=()
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i scripts/newton_py -m isaacsimenvs.eval.throughput \
        --num_envs "$ENVS" \
        --steps "$STEPS" \
        --warmup 40 \
        --num_assets_per_type 1 \
        --out "docs/results/mgpu_${NGPU}x${ENVS}_rank${i}.json" \
        'env.assets.handle_head_types=[hammer]' \
        env.cable.thickness=0.03 \
        > "slurm_logs/mgpu_rank${i}_${SLURM_JOB_ID}.log" 2>&1 &
    pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "[mgpu] workers finished (fail=$fail); per-rank results:"
grep -h THROUGHPUT slurm_logs/mgpu_rank*_"${SLURM_JOB_ID}".log 2>/dev/null

python3 - "$NGPU" "$ENVS" <<'PY'
import glob, json, sys
ngpu, envs = int(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(open(f).read()) for f in glob.glob(f"docs/results/mgpu_{ngpu}x{envs}_rank*.json")]
if not rows:
    print("[mgpu] no rank results"); raise SystemExit(1)
tot = sum(r["env_steps_per_s"] for r in rows)
per = [r["env_steps_per_s"] for r in rows]
print(f"[mgpu] ranks={len(rows)} envs/gpu={envs} "
      f"total_env_steps_per_s={tot:.1f} per_rank={[round(p,1) for p in per]} "
      f"slowest_rank_ms={max(r['ms_per_step'] for r in rows):.1f} "
      f"fastest_rank_ms={min(r['ms_per_step'] for r in rows):.1f}")
PY
echo "[mgpu] done"
