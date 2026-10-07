#!/bin/bash
# Validate the PhysX port of the rigid fold tasks against Newton, before training on it.
#
#   sbatch --array=0-11 scripts/cluster/bos14_physx_cloth_validate.sh
#   python3 scripts/analysis/cloth_obs_parity_compare.py docs/results/physx_parity_{newton,physx}_chain_zero.pt
#
# Tasks 0-5: observation parity dumps (scripts/analysis/cloth_obs_parity.py), pinned start, 8 envs:
#   0/1 chain, zero actions, Newton/PhysX    2/3 chain, the box3-mid policy, Newton/PhysX
#   4/5 pool, zero actions, Newton/PhysX
# Tasks 6-11: Newton-trained policies replayed on PhysX (episodes.py, same protocol as the Newton
# evals: 900 steps, held fold = 10 steps, two eval seeds), to compare with their Newton scores:
#   6/7  box3-mid chain policy (rc2 fixdr s2, 155349) on the PhysX chain, pinned / randomised
#   8/9  stage-2 pool policy (156746 s3) on the PhysX pool, pinned / randomised
#   10/11 the same pool policy on the NEWTON pool, pinned / randomised (same session's reference)

#SBATCH --job-name=px_valid
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES

I="${SLURM_ARRAY_TASK_ID:-0}"
CHAIN_CKPT=outputs/2026-09-30/17-12-24/0_rc2_box3-mid_fixdr_s2_155349/nn/last_0_rc2_box3-mid_fixdr_s2_155349_ep_2035_rew__1054.914_.pth
POOL_CKPT=outputs/2026-10-04/20-17-57/0_fp_box3_s3_156746/nn/last_0_fp_box3_s3_156746_ep_2035_rew__1063.0068_.pth
NEWTON=(scripts/newton_py)
PHYSX=(.venv_isaacsim/bin/python)
echo "[px_valid] task=$I host=$(hostname -s) commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"

parity() {  # backend kind ckpt-or-empty label
    local py task extra=()
    if [ "$1" = newton ]; then py=("${NEWTON[@]}"); else py=("${PHYSX[@]}"); extra=(--headless); fi
    case "$1:$2" in
        newton:chain) task=Isaacsimenvs-RigidCloth-Direct-v0 ;;
        physx:chain)  task=Isaacsimenvs-RigidClothPhysx-Direct-v0 ;;
        newton:pool)  task=Isaacsimenvs-FoldPool-Direct-v0 ;;
        physx:pool)   task=Isaacsimenvs-FoldPoolPhysx-Direct-v0 ;;
    esac
    [ -n "$3" ] && extra+=(--checkpoint "$3")
    "${py[@]}" scripts/analysis/cloth_obs_parity.py --task "$task" "${extra[@]}" \
        --out "docs/results/physx_parity_$1_$2_$4.pt" env.rigid_cloth.randomization.enabled=false
}

replay() {  # backend kind ckpt mode label
    local py task extra=() mode=()
    if [ "$1" = newton ]; then
        py=("${NEWTON[@]}"); extra=("env.newton.per_env_triangle_pairs=2097152")
    else
        py=("${PHYSX[@]}"); extra=(--headless)
    fi
    case "$1:$2" in
        physx:chain)  task=Isaacsimenvs-RigidClothPhysx-Direct-v0; extra+=(env.rigid_cloth.variant=box3-mid) ;;
        physx:pool)   task=Isaacsimenvs-FoldPoolPhysx-Direct-v0 ;;
        newton:pool)  task=Isaacsimenvs-FoldPool-Direct-v0 ;;
    esac
    if [ "$4" = rand ]; then mode=(--randomize); else mode=(env.rigid_cloth.randomization.enabled=false); fi
    local envs=32; [ "$2" = pool ] && envs=320
    for ES in 1 2; do
        "${py[@]}" -m isaacsimenvs.eval.episodes \
            --task "$task" --checkpoint "$3" --policy_config pretrained_policy/config.yaml \
            --sapg_expl_coef 0 --num_envs "$envs" --max_steps 900 --success_tolerance 0.01 \
            --num_assets_per_type 1 --single_variant --seed "$ES" \
            --out "docs/results/$5_s${ES}.json" \
            env.cloth.nan_policy=reset env.termination.success_steps=10 \
            "${mode[@]}" "${extra[@]}" || echo "[px_valid] $5 seed $ES FAILED"
    done
}

case "$I" in
    0) parity newton chain "" zero ;;
    1) parity physx chain "" zero ;;
    2) parity newton chain "$CHAIN_CKPT" policy ;;
    3) parity physx chain "$CHAIN_CKPT" policy ;;
    4) parity newton pool "" zero ;;
    5) parity physx pool "" zero ;;
    6) replay physx chain "$CHAIN_CKPT" pinned physx_replay_box3_chain_pinned ;;
    7) replay physx chain "$CHAIN_CKPT" rand physx_replay_box3_chain_rand ;;
    8) replay physx pool "$POOL_CKPT" pinned physx_replay_pool156746_pinned ;;
    9) replay physx pool "$POOL_CKPT" rand physx_replay_pool156746_rand ;;
    10) replay newton pool "$POOL_CKPT" pinned newton_replay_pool156746_pinned ;;
    11) replay newton pool "$POOL_CKPT" rand newton_replay_pool156746_rand ;;
esac
echo "[px_valid] task=$I done"
