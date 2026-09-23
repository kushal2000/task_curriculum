#!/bin/bash
# Resume a finished cloth SAPG run from its checkpoint, as one bos14 array task per node.
#
#   CKPT=<...>.pth RUN=pretrained_s1_resume EPOCHS=3800 sbatch --array=0-3 \
#       scripts/cluster/bos14_cloth_resume.sh
#
# Launch mechanics are bos14_cloth_prior_vs_scratch.sh's -- read that header for why there is no
# srun, no ssh, a shared rendezvous file and a STATIC torchrun rendezvous. What differs is the
# checkpoint mode, and three things follow from it:
#
# 1. EPOCHS IS A TOTAL, NOT AN INCREMENT. `--checkpoint_load_mode resume` (the default) calls
#    set_full_state_weights with set_epoch=True, which restores `epoch_num` and `frame` from the
#    checkpoint. A run resumed at epoch 509 with max_epochs=509 therefore stops immediately.
# 2. NNODES MUST MATCH THE CHECKPOINT'S RANK COUNT. A multi-GPU checkpoint is a dict keyed by
#    global rank (`{0: state, 1: state, ...}`) and each rank restores its own entry
#    (a2c_continuous.py:91-99). Asserted below rather than discovered at epoch 1.
# 3. ENVS MUST MATCH. set_full_state_weights silently skips the rollout buffers when
#    `current_rewards.shape[0] != num_actors`, printing "Skipping loading of many things".
#
# Budget. 4 ranks x 1536 envs x horizon 32 = 196,608 env-steps per epoch, the SAME per-epoch
# figure as the 8x768 reference run, so epoch numbers compare directly: the reference reached
# 100% held folds at epoch 3800 (7.47e8 env-steps). At the measured 11,450 env-steps/s that is
# ~4.7 h per 1000 epochs.

#SBATCH --job-name=cloth_resume
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum

CKPT="${CKPT:?set CKPT=<checkpoint.pth>}"
[ -f "$CKPT" ] || { echo "[resume] checkpoint not found: $CKPT" >&2; exit 1; }

NNODES=$SLURM_ARRAY_TASK_COUNT
NODE_RANK=$SLURM_ARRAY_TASK_ID
JOB=$SLURM_ARRAY_JOB_ID
ENVS="${ENVS:-1536}"
BLOCK="${BLOCK:-256}"
MINIBATCH="${MINIBATCH:-24576}"
EPOCHS="${EPOCHS:-3800}"
SEED="${SEED:-1}"
RUN_NAME="${RUN:-resume}_${JOB}"
RDZV_FILE="slurm_logs/.rdzv_${JOB}"
export PYTHONUNBUFFERED=1

# Fail fast on the two mismatches that do not raise: a rank count the checkpoint has no entry for,
# and an env count that silently drops the restored rollout state.
.venv_isaaclab3/bin/python - "$CKPT" "$NNODES" "$ENVS" <<'PY' || exit 1
import sys, torch
ckpt, nnodes, envs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = torch.load(ckpt, map_location="cpu", weights_only=False)
groups = sorted(k for k in c if isinstance(k, int))
g0 = c[groups[0]]
actors = g0["current_rewards"].shape[0]
print(f"[resume] checkpoint: ranks={groups} epoch={g0['epoch']} frame={g0['frame']:,} "
      f"envs/rank={actors} last_mean_rew={float(g0['last_mean_rewards']):.1f}")
if len(groups) != nnodes:
    sys.exit(f"[resume] checkpoint has {len(groups)} ranks, this job has {nnodes}")
if actors != envs:
    sys.exit(f"[resume] checkpoint has {actors} envs/rank, this job has {envs}")
PY

MY_IP="$(hostname -I | awk '{print $1}')"
IFACE="$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ "^"ip"/" {print $2; exit}')"
export GLOO_SOCKET_IFNAME="$IFACE" TP_SOCKET_IFNAME="$IFACE"

if [ "$NODE_RANK" = "0" ]; then
    echo "$MY_IP" >"$RDZV_FILE"
    HEAD_IP="$MY_IP"
else
    for _ in $(seq 3600); do [ -s "$RDZV_FILE" ] && break; sleep 1; done
    HEAD_IP="$(cat "$RDZV_FILE" 2>/dev/null)"
    [ -n "$HEAD_IP" ] || { echo "[resume] no rendezvous file after 60 min"; exit 1; }
fi

echo "[resume] run=$RUN_NAME node=$NODE_RANK/$NNODES host=$(hostname -s) ip=$MY_IP iface=$IFACE head=$HEAD_IP"
echo "[resume] envs/rank=$ENVS target_epoch=$EPOCHS (TOTAL, epoch is restored from the checkpoint)"
echo "[resume] commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"

PORT=$((29500 + JOB % 1000))
scripts/newton_py -m torch.distributed.run \
    --nnodes="$NNODES" --nproc_per_node=1 --node_rank="$NODE_RANK" \
    --master_addr="$HEAD_IP" --master_port="$PORT" \
    -m isaacsimenvs.train \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    --checkpoint "$CKPT" \
    --checkpoint_load_mode resume \
    --distributed \
    --single_variant \
    --wandb_activate \
    --wandb_tags resume "seed$SEED" bos14 \
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
echo "[resume] exit=$?"
