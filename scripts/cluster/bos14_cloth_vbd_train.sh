#!/bin/bash
# SAPG directly on the VBD cloth (Newton), several seeds x NPS nodes in ONE array job, so the trend
# driver sees one job id: array index i -> seed i/NPS + 1, node rank i%NPS.
#
#   ROBOT=FlexivWujiHumanLike PRIOR=pretrained_policy/flexiv_wuji.pth RUN_TAG=vbdFWHL EXTRA_ARGS=env.robot_self_collision=false \
#       sbatch --array=0-11 scripts/cluster/bos14_cloth_vbd_train.sh        # 3 seeds x 4 nodes
#   ... EPOCHS=30 RUN_TAG=vbdsmoke sbatch --array=0-3 ...                    # smoke, one seed
#   resume all seeds: CKPTS="<s1.pth> <s2.pth> <s3.pth>" LOAD_MODE=resume sbatch --array=0-11 ...
#   resume a seed: SEEDS_OFFSET=<seed-1> CKPT=<periodic .pth> LOAD_MODE=resume sbatch --array=0-3 ...
#
# Launch mechanics as bos14_cloth_prior_vs_scratch.sh (Kuka, 09-13): no srun / ssh, rank 0 of each
# seed writes its IP to a rendezvous file, static torchrun rendezvous, gloo on the node's interface.
# Budget: 1488 envs/rank x NPS ranks x horizon 32 per epoch; 4 nodes x 3750 epochs = 7.14e8 env-steps
# (the Kuka cloth reference reached 100% held folds at 7.47e8). 1488, not the Kuka's 1536: VBD allocates
# rigid_body_particle_contact_buffer_size (65536) x body count, and the Flexiv's 22 bodies per env
# overflow Warp's int32 array size above 1489 envs. Minibatch 23808 = half the rollout (1488 x 32 / 2).
# Robot self-collision OFF for the Flexiv (EXTRA_ARGS above): on, MJWarp steps 26x slower (159 vs 4207
# env-steps/s at 1488 envs, 10-26 probe); the Kuka cloth reference ran with it off too (Cloth task default).
# Run names are 0_pxvbd_<RUN_TAG>_s<seed>_<job>, the trend driver's pattern with --kind vbd.

#SBATCH --job-name=vbd_train
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=bos14-node-[040,047,051,057,062,086,092,104,105,109,110,112,121,123]
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd "${REPO:-/home/yiboc/mit/task_curriculum_flexiv}"

NPS="${NPS:-4}"
I=$SLURM_ARRAY_TASK_ID
SEED=$(( I / NPS + 1 + ${SEEDS_OFFSET:-0} ))
NODE_RANK=$(( I % NPS ))
JOB=$SLURM_ARRAY_JOB_ID
ENVS="${ENVS:-1488}"
BLOCK=$(( ENVS / 6 ))
MINIBATCH="${MINIBATCH:-23808}"
EPOCHS="${EPOCHS:-3750}"
PRIOR="${PRIOR:-pretrained_policy/flexiv_wuji.pth}"
CKPT="${CKPT:-$PRIOR}"
# CKPTS: one checkpoint per seed (space-separated, seed order), to resume all seeds in one job.
if [ -n "${CKPTS:-}" ]; then read -r -a _CK <<<"$CKPTS"; CKPT="${_CK[$(( I / NPS ))]}"; fi
NAME="0_pxvbd_${RUN_TAG:-run}_s${SEED}_${JOB}"
RDZV_FILE="slurm_logs/.rdzv_${JOB}_s${SEED}"
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
    [ -n "$HEAD_IP" ] || { echo "[vbd_train] no rendezvous file after 60 min"; exit 1; }
fi

echo "[vbd_train] name=$NAME task=Isaacsimenvs-Cloth${ROBOT:-}-Direct-v0 seed=$SEED node=$NODE_RANK/$NPS" \
     "host=$(hostname -s) ip=$MY_IP iface=$IFACE head=$HEAD_IP"
echo "[vbd_train] envs/rank=$ENVS block=$BLOCK minibatch=$MINIBATCH epochs=$EPOCHS" \
     "env-steps=$(( NPS * ENVS * 32 * EPOCHS )) ckpt=$CKPT load_mode=${LOAD_MODE:-weights} experiment=${EXPERIMENT:-none} extra=${EXTRA_ARGS:-none}"
echo "[vbd_train] commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"

WANDB_ARGS=()
if [ "${WANDB:-1}" = 1 ]; then
    WANDB_ARGS=(--wandb_activate --wandb_project fold_pool --wandb_group "vbd_${RUN_TAG:-run}_${JOB}" --wandb_name "$NAME")
fi
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
# NO_PRIOR=1: no --checkpoint (a point-cloud actor that starts from --pc_init_student in EXTRA_ARGS).
CKPT_ARGS=(--checkpoint "$CKPT" --checkpoint_load_mode "${LOAD_MODE:-weights}")
[ "${NO_PRIOR:-0}" = 1 ] && CKPT_ARGS=()

PORT=$(( 20000 + (JOB % 10000) * 3 + SEED ))
scripts/newton_py -m torch.distributed.run \
    --nnodes="$NPS" --nproc_per_node=1 --node_rank="$NODE_RANK" \
    --master_addr="$HEAD_IP" --master_port="$PORT" ${RDZV_TIMEOUT:+--rdzv_conf=timeout=$RDZV_TIMEOUT} \
    -m isaacsimenvs.scripts.training.train \
    --task "Isaacsimenvs-Cloth${ROBOT:-}-Direct-v0" \
    --agent rl_games_sapg_cfg_entry_point \
    "${CKPT_ARGS[@]}" \
    --distributed \
    --single_variant \
    ${EXPERIMENT:+--experiment "$EXPERIMENT"} \
    "${WANDB_ARGS[@]}" \
    env.cloth.nan_policy=reset \
    env.newton.njmax=32768 \
    "env.scene.num_envs=$ENVS" \
    "agent.params.seed=$SEED" \
    "agent.params.config.minibatch_size=$MINIBATCH" \
    "agent.params.config.central_value_config.minibatch_size=$MINIBATCH" \
    "agent.params.config.expl_coef_block_size=$BLOCK" \
    "agent.params.config.max_epochs=$EPOCHS" \
    agent.params.config.save_frequency=50 \
    agent.params.config.periodic_save_every=50 \
    agent.params.config.snapshot_every=150 \
    agent.params.config.best_save_min_interval_s=300 \
    "agent.params.config.name=$NAME" \
    "${EXTRA[@]}"
echo "[vbd_train] exit=$?"
