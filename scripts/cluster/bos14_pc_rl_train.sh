#!/bin/bash
# SAPG with the point-cloud student as the actor (isaacsimenvs/distill/pc_rl.py), NPS nodes per seed in
# ONE array job: array index i -> seed i/NPS + 1, node rank i%NPS. Launch mechanics as
# bos14_cloth_vbd_train.sh (no srun / ssh, static torchrun rendezvous over a file).
#
#   SIM=physx (default): the flat7RW_base PhysX fold pool, EXPERIMENT pcRW_scratch / pcRW_ft.
#   SIM=vbd:             the VBD cloth (Newton, ns6), SAPG settings through EXTRA_ARGS.
#   PC_ARGS='{"goal":"first","noise":true,...}'   the actor's input (distill_pc pc_args keys)
#   STUDENT=<pth> SIGMA=<teacher snap> CRITIC=<teacher full ckpt>   start from a distilled student (M3)
#   NPS ENVS per rank (7680 physx / 1488 vbd, mix) MINIBATCH per rank EPOCHS RUN_TAG EXTRA_ARGS
#   SIM=mix: half the ranks PhysX, half VBD (same ENVS / MINIBATCH on every rank).
#
#   PC_ARGS=... RUN_TAG=M2_A-noise sbatch --array=0 scripts/cluster/bos14_pc_rl_train.sh          (1 GPU)
#   SIM=vbd NPS=4 MINIBATCH=2976 ... sbatch --array=0-3 ...                                        (4 nodes)
# PhysX runs on ONE RTX PRO 6000: 7680 envs x 3052 epochs = 7.5e8 env-steps, MINIBATCH=3840 (a PointNet
# over 1024+ points per sample: ~45 GB). Not smaller: SAPG keeps every minibatch's flat gradient on the
# host for a diagnostic, and 320 minibatches per epoch (MINIBATCH 1536) overran 100 GB of RAM. More PhysX ranks do not pay: gloo syncs the gradients
# every minibatch, ~40 s per epoch (11-16 probe), and a 32 GB RTX 5090 cannot hold 7680 envs' rollout.
# Run names 0_pcrl_<RUN_TAG>_s<seed>_<job>; pc_args.json is written into the Hydra run directory.

#SBATCH --job-name=pc_rl
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=115G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=bos14-node-[040,047,051,057,059,062,086,092,104,105,109,110,112,121,123]
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd "${REPO:-/home/yiboc/mit/task_curriculum_flexiv}"

SIM="${SIM:-physx}"
NPS="${NPS:-1}"
I=$SLURM_ARRAY_TASK_ID
SEED=$(( I / NPS + 1 + ${SEEDS_OFFSET:-0} ))
NODE_RANK=$(( I % NPS ))
JOB=$SLURM_ARRAY_JOB_ID
if [ "$SIM" = physx ]; then ENVS="${ENVS:-7680}"; else ENVS="${ENVS:-1488}"; fi
BLOCK=$(( ENVS / 6 ))
MINIBATCH="${MINIBATCH:-$ENVS}"
EPOCHS="${EPOCHS:-3052}"
CKPT="${CKPT:-}"
# CKPTS: one checkpoint per seed (space-separated, seed order), to resume all seeds in one job.
if [ -n "${CKPTS:-}" ]; then read -r -a _CK <<<"$CKPTS"; CKPT="${_CK[$(( I / NPS ))]}"; fi
NAME="0_pcrl_${RUN_TAG:-run}_s${SEED}_${JOB}"
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
    [ -n "$HEAD_IP" ] || { echo "[pc_rl] no rendezvous file after 60 min"; exit 1; }
fi

echo "[pc_rl] name=$NAME task=Isaacsimenvs-Cloth${ROBOT:-}-Direct-v0 seed=$SEED node=$NODE_RANK/$NPS" \
     "host=$(hostname -s) ip=$MY_IP iface=$IFACE head=$HEAD_IP"
echo "[pc_rl] envs/rank=$ENVS block=$BLOCK minibatch=$MINIBATCH epochs=$EPOCHS" \
     "env-steps=$(( NPS * ENVS * 32 * EPOCHS )) ckpt=$CKPT load_mode=${LOAD_MODE:-weights} experiment=${EXPERIMENT:-none} extra=${EXTRA_ARGS:-none}"
echo "[pc_rl] commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"

WANDB_ARGS=()
if [ "${WANDB:-0}" = 1 ]; then
    WANDB_ARGS=(--wandb_activate --wandb_project fold_pool --wandb_group "vbd_${RUN_TAG:-run}_${JOB}" --wandb_name "$NAME")
fi
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
# NO_PRIOR=1: no --checkpoint (a point-cloud actor that starts from --pc_init_student in EXTRA_ARGS).
# No --checkpoint unless CKPT is given (resume: CKPT=<periodic .pth> LOAD_MODE=resume).
CKPT_ARGS=()
[ -n "${CKPT:-}" ] && CKPT_ARGS=(--checkpoint "$CKPT" --checkpoint_load_mode "${LOAD_MODE:-resume}")

PORT=$(( 20000 + (JOB % 10000) * 3 + SEED ))
PC=(--pc_args "${PC_ARGS:?set PC_ARGS}")
[ -n "${STUDENT:-}" ] && PC+=(--pc_init_student "$STUDENT" --pc_freeze_norm)
[ -n "${SIGMA:-}" ] && PC+=(--pc_init_sigma "$SIGMA")
[ -n "${CRITIC:-}" ] && PC+=(--critic_checkpoint "$CRITIC")
# SIM=mix: the first half of the ranks run PhysX, the second half the VBD cloth; one actor, gradients
# averaged over the ranks. Every rank needs the same ENVS and MINIBATCH (same optimizer steps per epoch).
RSIM="$SIM"
if [ "$SIM" = mix ]; then [ "$NODE_RANK" -lt $(( NPS / 2 )) ] && RSIM=physx || RSIM=vbd; fi
echo "[pc_rl] rank $NODE_RANK simulator: $RSIM"
if [ "$RSIM" = vbd ]; then
    PY=(scripts/newton_py)
    TASK_ARGS=(--task "Isaacsimenvs-Cloth${ROBOT:-FlexivWujiRightHLRL}-Direct-v0" env.newton.njmax=32768
               env.cloth.start_height=0.20 env.robot_self_collision=false env.newton.num_substeps=6 env.cloth.substeps=2
               env.cloth.keypoint_tolerance=0.015
               agent.params.config.save_frequency=50 agent.params.config.periodic_save_every=50
               agent.params.config.snapshot_every=100 agent.params.config.best_save_min_interval_s=300)
else
    export OMNI_KIT_ACCEPT_EULA=YES TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
    PY=(.venv_isaacsim/bin/python)
    TASK_ARGS=(--task "Isaacsimenvs-FoldPoolPhysx${ROBOT:-FlexivWujiRightHLRL}-Direct-v0" --headless
               --experiment "${EXPERIMENT:-pcRW_scratch}")
fi
"${PY[@]}" -m torch.distributed.run \
    --nnodes="$NPS" --nproc_per_node=1 --node_rank="$NODE_RANK" \
    --master_addr="$HEAD_IP" --master_port="$PORT" ${RDZV_TIMEOUT:+--rdzv_conf=timeout=$RDZV_TIMEOUT} \
    -m isaacsimenvs.train \
    --agent rl_games_sapg_cfg_entry_point \
    "${CKPT_ARGS[@]}" \
    --distributed \
    --single_variant \
    "${PC[@]}" \
    "${TASK_ARGS[@]}" \
    env.cloth.nan_policy=reset \
    "env.scene.num_envs=$ENVS" \
    "agent.params.seed=$SEED" \
    "agent.params.config.minibatch_size=$MINIBATCH" \
    "agent.params.config.central_value_config.minibatch_size=$MINIBATCH" \
    "agent.params.config.expl_coef_block_size=$BLOCK" \
    "agent.params.config.max_epochs=$EPOCHS" \
    "agent.params.config.name=$NAME" \
    "${EXTRA[@]}"
echo "[pc_rl] exit=$?"
