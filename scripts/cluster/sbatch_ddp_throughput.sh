#!/bin/bash
# Distributed-RL throughput: N ranks with a real gradient all-reduce between them.
#
#   NGPU=4 ENVS=2048 HORIZON=32 sbatch --gres=gpu:nvidia_rtx_6000_ada_generation:4 \
#       scripts/cluster/sbatch_ddp_throughput.sh
#
# The plain multi-GPU benchmark (multi_gpu_throughput.sh) lets ranks free-run, which is an UPPER
# BOUND: real RL blocks on the slowest rank at every gradient sync and pays interconnect time for
# the payload. This launches under torchrun and all-reduces a policy-sized buffer every HORIZON
# steps, which is the PPO rollout length, so the measured rate includes both costs.
#
# Compare against the free-running number to get the true cost of synchronisation.

#SBATCH --job-name=ddp
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

NGPU="${NGPU:-4}"
ENVS="${ENVS:-2048}"
STEPS="${STEPS:-60}"
HORIZON="${HORIZON:-32}"
GRAD_MB="${GRAD_MB:-8}"

# Isaac Lab divides host threads by world size so ranks do not fight over cores; match that.
export OMP_NUM_THREADS=$(( $(nproc) / NGPU ))

echo "[ddp] ngpu=$NGPU envs/gpu=$ENVS horizon=$HORIZON grad=${GRAD_MB}MB host=$(hostname)"
echo "[ddp] OMP_NUM_THREADS=$OMP_NUM_THREADS"

# torchrun must run *through* scripts/newton_py: it is not on PATH (it lives in the venv), and
# the wrapper's LD_PRELOAD of the system libstdc++ has to be set before the workers start, since
# they inherit it as an environment variable. Invoking `torchrun` bare gives exit 127.
scripts/newton_py -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$NGPU" \
    -m isaacsimenvs.eval.throughput \
    --num_envs "$ENVS" \
    --steps "$STEPS" \
    --warmup 40 \
    --num_assets_per_type 1 \
    --sync_every "$HORIZON" \
    --grad_mb "$GRAD_MB" \
    --out "docs/results/ddp_${NGPU}x${ENVS}.json" \
    'env.assets.handle_head_types=[hammer]' \
    env.cable.thickness=0.03

echo "[ddp] exit=$?"
