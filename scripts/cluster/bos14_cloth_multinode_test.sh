#!/bin/bash
# Cross-node DDP check for the cloth SAPG run on bos14, where every node has exactly one GPU.
#
#   N=2; sbatch --array=0-$((N-1)) scripts/cluster/bos14_cloth_multinode_test.sh
#
# WHY AN ARRAY AND NOT --nodes=N. On bos14 neither in-allocation launcher works: `srun` fails to
# create a step ("Invalid generic resource (gres) specification" with the job's gres, "Communication
# connection failure" with --gres=none), and ssh between allocated nodes is refused (publickey). So
# each node is its own single-node array task, and the tasks rendezvous over TCP: task 0 writes its
# IP to a shared file, the others wait for it, and every task runs torchrun's c10d rendezvous
# against that address. Nothing but the shared filesystem and the network is assumed.
#
# Two steps per task, each its own rendezvous:
#   1. bos14_multinode_comm_test.py -- raw gloo timings at rl_games' real payload sizes;
#   2. train.py --distributed for EPOCHS epochs at 1536 envs/rank, to compare with the single-GPU
#      probe (job 146459: 3,516 env-steps/s at 1536 envs).
#
# The 8-GPU launcher pins GLOO_SOCKET_IFNAME=lo, which is only correct on one node; here it is the
# interface carrying the node's cluster IP (eth0, 5 Gb/s, measured in job 146469).

#SBATCH --job-name=cloth_mnode
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum

NNODES=$SLURM_ARRAY_TASK_COUNT
RANK_HINT=$SLURM_ARRAY_TASK_ID
JOB=$SLURM_ARRAY_JOB_ID
ENVS="${ENVS:-1536}"
BLOCK="${BLOCK:-256}"
EPOCHS="${EPOCHS:-25}"
RDZV_FILE="slurm_logs/.rdzv_${JOB}"
export PYTHONUNBUFFERED=1

MY_IP="$(hostname -I | awk '{print $1}')"
IFACE="$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ "^"ip"/" {print $2; exit}')"
export GLOO_SOCKET_IFNAME="$IFACE" TP_SOCKET_IFNAME="$IFACE"

if [ "$RANK_HINT" = "0" ]; then
    echo "$MY_IP" >"$RDZV_FILE"
    HEAD_IP="$MY_IP"
else
    for _ in $(seq 600); do [ -s "$RDZV_FILE" ] && break; sleep 1; done
    HEAD_IP="$(cat "$RDZV_FILE" 2>/dev/null)"
    [ -n "$HEAD_IP" ] || { echo "[mnode] no rendezvous file after 10 min"; exit 1; }
fi

echo "[mnode] task=$RANK_HINT/$NNODES host=$(hostname -s) ip=$MY_IP iface=$IFACE" \
     "speed=$(cat /sys/class/net/$IFACE/speed 2>/dev/null)Mb/s head=$HEAD_IP envs/rank=$ENVS"
echo "[mnode] commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"

# STATIC rendezvous, not c10d. c10d decides whether this node hosts the store by matching the
# endpoint against its own hostname, and bos14's /etc/hosts resolves every hostname to 127.0.1.1 --
# so no node recognised 10.14.1.x as itself, nobody hosted, and every node (task 0 included) timed
# out connecting (jobs 146470/146471). Raw TCP between nodes works (job 146476). With static,
# node_rank 0 hosts the store unconditionally.
torchrun() {  # $1 = port offset, rest = program
    local port=$((29500 + JOB % 1000 + $1)); shift
    scripts/newton_py -m torch.distributed.run \
        --nnodes="$NNODES" --nproc_per_node=1 --node_rank="$RANK_HINT" \
        --master_addr="$HEAD_IP" --master_port="$port" \
        "$@"
}

echo "[mnode] ===== step 1: collective timings"
torchrun 0 scripts/cluster/bos14_multinode_comm_test.py
echo "[mnode] step1 exit=$?"

echo "[mnode] ===== step 2: distributed train.py"
START=$(date +%s)
torchrun 1 -m isaacsimenvs.train \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    --checkpoint pretrained_policy/model.pth --checkpoint_load_mode weights \
    --distributed --single_variant \
    env.cloth.nan_policy=reset \
    "env.scene.num_envs=$ENVS" \
    agent.params.config.minibatch_size=24576 \
    agent.params.config.central_value_config.minibatch_size=24576 \
    "agent.params.config.expl_coef_block_size=$BLOCK" \
    "agent.params.config.max_epochs=$EPOCHS" \
    "agent.params.config.name=0_mnode_${NNODES}n_${JOB}"
echo "[mnode] step2 exit=$? wall_s=$(( $(date +%s) - START ))"
