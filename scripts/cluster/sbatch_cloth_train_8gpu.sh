#!/bin/bash
# SAPG finetune of the Play policy onto cloth folding, 8 GPUs on one node.
#
#   sbatch scripts/cluster/sbatch_cloth_train_8gpu.sh [run_name]
#
# Identical to sbatch_cloth_train.sh except for the GPU count and the three values that GPU count
# forces. Read that script for the shared reasoning (gloo loopback pin, module-form invocations,
# --checkpoint_load_mode weights, capture interval); only the differences are documented here.
#
# WHY THIS IS NOT JUST `--gres=...:8`.
#
# `batch_size = horizon_length * num_actors` is PER RANK (`a2c_common.py:243`) -- world_size never
# enters it. Only the GRADIENT is aggregated, via all_reduce(SUM)/world_size. So one optimizer step
# sees `world_size * minibatch_size`, and at 4 x 24576 that is 98304, exactly play2perfect's batch.
# Doubling the ranks while keeping 768 envs/GPU would silently double the effective batch to 196608
# -- a different optimisation problem, not a faster version of the same one.
#
# Halving envs/GPU instead keeps every quantity that matters identical:
#
#                        4 GPU (sbatch_cloth_train.sh)      8 GPU (this, DEFAULT)
#   envs / GPU           768                                768
#   total envs           3072                               6144      <- 2x the data
#   batch / rank         32 x 768 = 24576                   32 x 768 = 24576
#   minibatches / rank   1                                  2
#   effective batch      4 x 24576 = 98304                  8 x 12288 = 98304   <- unchanged
#   blocks x size        6 x 128                            6 x 128
#
# The default was 384 envs/GPU, which held TOTAL envs at 3072 and gave one minibatch per rank. At
# 768 envs/GPU the rank collects twice as much per epoch, so the minibatch is halved to 12288 and
# each rank takes TWO optimizer steps per epoch instead of one. `world_size * minibatch_size` is
# still 98304 -- the per-step batch play2perfect used -- and the extra data buys more gradient
# steps rather than a bigger, differently-behaved one. Two minibatches still satisfies the live
# assert at `a2c_common.py:537`: (32 * 768 // 2) %% 16 == 0.
#
# `expl_coef_block_size` returns to 128 because 768 / 128 = 6, the block count baked into the
# checkpoint's (6, 29) sigma. Memory is the measured 28.9 GB of 48 GB at 768 envs/GPU (run 629081),
# so the card has headroom.
#
#   ENVS_PER_GPU=384 MINIBATCH=12288 BLOCK=64 sbatch ...   # the previous 3072-env configuration
#
# `num_blocks` MUST stay 6: `a2c_network.sigma` in the pretrained checkpoint is (6, 29), a per-block
# parameter, so any other block count fails to load. 384 / 6 = 64, so `expl_coef_block_size` drops
# to 64 to hold num_blocks at 6 -- and 384 % 64 == 0 satisfies the assert at `a2c_common.py:329`.
#
# One minibatch per rank is preserved (12288 == batch/rank), which keeps the live assert at
# `a2c_common.py:537` satisfied: (32 * 384 // 1) %% 16 == 0. That matters because the
# `batch_size %% minibatch_size == 0` assert at `:252` is COMMENTED OUT upstream -- a non-dividing
# value would silently floor and drop the remainder rather than error.
#
# `num_actors` is NOT overridden: it interpolates from `${....env.scene.num_envs}`, and hydra
# resolves interpolations after the CLI merge, so setting env.scene.num_envs carries it.
#
# Memory: 36.46 GB of 48 measured at 768 envs/GPU on the OLD physics. Half the envs is well inside
# the card, but the new physics (self-contact on, 6 substeps) has not been memory-profiled -- if a
# rank OOMs, that is the first thing to measure rather than assume.

#SBATCH --job-name=cloth_sapg8
#SBATCH --partition=portal
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail
cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs

export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

if ! git ls-remote --exit-code --heads origin "$(git rev-parse --abbrev-ref HEAD)" >/dev/null 2>&1; then
    echo "[train] WARNING: branch $(git rev-parse --abbrev-ref HEAD) is not on origin --" \
         "viewer meshes will 404. Push it, or drop --capture_viewer." >&2
fi

RUN_NAME="${1:-cloth_fold_sapg8}_${SLURM_JOB_ID}"
# Anything after <run_name> is forwarded to `train.py` as a hydra override, and echoed below so the
# log records what actually ran. Used for `env.cloth.keypoint_tolerance`, which is only meaningful
# to tighten AFTER the fold-criterion fixes in e01d7f8: a geometrically perfect fold scored 0.0160
# before them, so any tolerance below ~2 cm was unreachable no matter what the policy did. It now
# scores 0.0022, which leaves real headroom at 0.01.
OVERRIDES=("${@:2}")
NGPU="${NGPU:-8}"
ENVS_PER_GPU="${ENVS_PER_GPU:-768}"
MINIBATCH="${MINIBATCH:-12288}"
BLOCK="${BLOCK:-128}"
CHECKPOINT="${CHECKPOINT:-/share/portal/kk837/simtoolreal/pretrained_policy/model.pth}"
# Capture cadence is in POLICY STEPS and therefore depends on throughput, which changed. The
# inherited 12000 was tuned when the sim ran ~3x faster and the launcher comment still claims it
# gives "~36 min and ~40 captures"; at the current ~19 s/epoch (32 steps) it is one capture every
# ~2 h, so ~12 over a 24 h run. 4000 restores roughly the original intent: ~40 min apart, ~36 over
# 24 h. Capture DUTY is `capture_viewer_len / interval` = 600/4000 = 15%, and the viewer runs on
# rank 0 only while the other ranks wait at the gradient all-reduce -- so this is a throughput tax
# on the whole job, not just on rank 0. Do not push it far below this.
CAPTURE_INTERVAL="${CAPTURE_INTERVAL:-4000}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "[train] run=$RUN_NAME gpus=$NGPU host=$(hostname) branch=$BRANCH overrides=${OVERRIDES[*]:-none}"
echo "[train] envs/gpu=$ENVS_PER_GPU total=$((NGPU * ENVS_PER_GPU)) minibatch=$MINIBATCH block=$BLOCK capture_interval=$CAPTURE_INTERVAL"
echo "[train] effective batch = $NGPU x $MINIBATCH = $((NGPU * MINIBATCH)) (target 98304)"
echo "[train] checkpoint=$CHECKPOINT"
echo "[train] commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"
git --no-pager log -1 --format="[train] subject=%s"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/newton_py -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$NGPU" \
    -m isaacsimenvs.train \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    --checkpoint "$CHECKPOINT" \
    --checkpoint_load_mode weights \
    --distributed \
    --single_variant \
    --capture_viewer \
    --capture_viewer_len 600 \
    --capture_viewer_interval "$CAPTURE_INTERVAL" \
    --capture_viewer_env_id 0 \
    --capture_viewer_github_raw_base "https://raw.githubusercontent.com/kushal2000/task_curriculum/${BRANCH}/" \
    --capture_viewer_url_check warn \
    --wandb_activate \
    --wandb_project cloth_folding \
    --wandb_group sapg_finetune \
    --wandb_name "$RUN_NAME" \
    env.cloth.nan_policy=reset \
    "env.scene.num_envs=${ENVS_PER_GPU}" \
    "agent.params.config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.central_value_config.minibatch_size=${MINIBATCH}" \
    "agent.params.config.expl_coef_block_size=${BLOCK}" \
    "agent.params.config.name=0_${RUN_NAME}" \
    "${OVERRIDES[@]}"

echo "[train] exit=$?"
