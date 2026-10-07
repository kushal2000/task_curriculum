#!/bin/bash
# Does one shape per env (fold_pool.one_shape_per_env) score a trained pool policy like the 5-shape env?
#
#   sbatch --array=0-3 scripts/cluster/bos14_one_shape_parity.sh
#
# Index: bit 0 = one_shape_per_env (0 = every env carries all five), bit 1 = randomised start.
# PhysX stage-2 policy 157622 s1, 320 envs (64 per shape in one-shape mode), two eval seeds, the
# usual protocol (900 steps, fold held 10 steps, 1.5 cm cutoff of FoldPoolPhysx.yaml).

#SBATCH --job-name=px_1shape
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
I="${SLURM_ARRAY_TASK_ID:-0}"
ONE=$(( I % 2 )); RAND=$(( I / 2 ))
CKPT=outputs/2026-10-05/23-51-55/0_pxpool_pxs1_arm1e3_e6144_s1_157622/nn/0_pxpool_pxs1_arm1e3_e6144_s1_157622.pth
[ "$ONE" -eq 1 ] && M=one && OARGS=(env.fold_pool.one_shape_per_env=true) || { M=all; OARGS=(); }
[ "$RAND" -eq 1 ] && S=rand && SARGS=(--randomize) || { S=pinned; SARGS=(env.rigid_cloth.randomization.enabled=false); }
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
echo "[1shape] mode=$M start=$S host=$(hostname -s)"
for ES in 1 2; do
    .venv_isaacsim/bin/python -m isaacsimenvs.eval.episodes --headless \
        --task Isaacsimenvs-FoldPoolPhysx-Direct-v0 --checkpoint "$CKPT" \
        --policy_config pretrained_policy/config.yaml --sapg_expl_coef 0 --num_envs 320 \
        --max_steps 900 --success_tolerance 0.01 --num_assets_per_type 1 --single_variant \
        "${SARGS[@]}" --seed "$ES" --out "docs/results/one_shape_parity/${M}_${S}_s${ES}.json" \
        env.cloth.nan_policy=reset env.termination.success_steps=10 "${OARGS[@]}" \
        || echo "[1shape] $M $S seed $ES FAILED"
done
