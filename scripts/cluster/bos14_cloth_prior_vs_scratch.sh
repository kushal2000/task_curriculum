#!/bin/bash
# Prior-vs-scratch on cloth folding: one 4-node SAPG run, as one bos14 array task per node.
#
#   scripts/cluster/submit_prior_vs_scratch.sh          # all 6 runs: {pretrained, scratch} x seeds 1-3
#   sbatch --array=0-3 scripts/cluster/bos14_cloth_prior_vs_scratch.sh <pretrained|scratch> <seed>
#
# The two arms differ in exactly one thing: whether the SimToolReal checkpoint is loaded
# (`--checkpoint ... --checkpoint_load_mode weights`). Every other setting -- task YAML, reward
# (keypoint_rew_scale 1500), SAPG config, env count, epochs -- is identical.
#
# Launch mechanics are those validated by scripts/cluster/bos14_cloth_multinode_test.sh (jobs
# 146478/146479); read that header for why: no srun, no ssh, a shared rendezvous file, STATIC
# torchrun rendezvous, and gloo on eth0.
#
# Budget. 1536 envs/rank x 4 ranks x horizon 32 = 196,608 env-steps per epoch, so 509 epochs is
# 1.0e8 env-steps. At the measured 11,450 env-steps/s on 4 nodes that is ~2.4 h; the time limit
# leaves room for slower 2.5 Gb/s nodes and the scratch arm's ~6% slower stepping.
#
# Per optimizer step: 4 ranks x minibatch 24576 = 98304 samples, play2perfect's batch.
#
# `--wandb_tags` is nargs="*" and must be followed by another --flag: placed last it would swallow
# every hydra override after it as a tag.

#SBATCH --job-name=cloth_pvs
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum

ARM="$1"
SEED="$2"
case "$ARM" in pretrained|scratch) ;; *) echo "arm must be pretrained|scratch, got '$ARM'"; exit 2;; esac

NNODES=$SLURM_ARRAY_TASK_COUNT
NODE_RANK=$SLURM_ARRAY_TASK_ID
JOB=$SLURM_ARRAY_JOB_ID
ENVS="${ENVS:-1536}"
BLOCK="${BLOCK:-256}"
MINIBATCH="${MINIBATCH:-24576}"
EPOCHS="${EPOCHS:-509}"
RUN_NAME="${ARM}_s${SEED}_${JOB}"
RDZV_FILE="slurm_logs/.rdzv_${JOB}"
export PYTHONUNBUFFERED=1

MY_IP="$(hostname -I | awk '{print $1}')"
IFACE="$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ "^"ip"/" {print $2; exit}')"
export GLOO_SOCKET_IFNAME="$IFACE" TP_SOCKET_IFNAME="$IFACE"

if [ "$NODE_RANK" = "0" ]; then
    echo "$MY_IP" >"$RDZV_FILE"
    HEAD_IP="$MY_IP"
else
    for _ in $(seq 3600); do [ -s "$RDZV_FILE" ] && break; sleep 1; done
    HEAD_IP="$(cat "$RDZV_FILE" 2>/dev/null)"
    [ -n "$HEAD_IP" ] || { echo "[pvs] no rendezvous file after 60 min"; exit 1; }
fi

CKPT_ARGS=()
if [ "$ARM" = "pretrained" ]; then
    CKPT_ARGS=(--checkpoint pretrained_policy/model.pth --checkpoint_load_mode weights)
fi

echo "[pvs] run=$RUN_NAME arm=$ARM seed=$SEED node=$NODE_RANK/$NNODES host=$(hostname -s)" \
     "ip=$MY_IP iface=$IFACE speed=$(cat /sys/class/net/$IFACE/speed 2>/dev/null)Mb/s head=$HEAD_IP"
echo "[pvs] envs/rank=$ENVS block=$BLOCK minibatch=$MINIBATCH epochs=$EPOCHS" \
     "env-steps=$((NNODES * ENVS * 32 * EPOCHS))"
echo "[pvs] commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"

PORT=$((29500 + JOB % 1000))
scripts/newton_py -m torch.distributed.run \
    --nnodes="$NNODES" --nproc_per_node=1 --node_rank="$NODE_RANK" \
    --master_addr="$HEAD_IP" --master_port="$PORT" \
    -m isaacsimenvs.train \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    "${CKPT_ARGS[@]}" \
    --distributed \
    --single_variant \
    --wandb_activate \
    --wandb_tags "$ARM" "seed$SEED" bos14 \
    --wandb_project cloth_folding \
    --wandb_group prior_vs_scratch_bos14 \
    --wandb_name "$RUN_NAME" \
    env.cloth.nan_policy=reset \
    "env.scene.num_envs=$ENVS" \
    "agent.params.seed=$SEED" \
    "agent.params.config.minibatch_size=$MINIBATCH" \
    "agent.params.config.central_value_config.minibatch_size=$MINIBATCH" \
    "agent.params.config.expl_coef_block_size=$BLOCK" \
    "agent.params.config.max_epochs=$EPOCHS" \
    agent.params.config.save_frequency=50 \
    "agent.params.config.name=0_$RUN_NAME"
echo "[pvs] exit=$?"
