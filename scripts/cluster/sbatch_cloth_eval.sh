#!/bin/bash
# Five-seed cloth-fold evaluation, matched to the baseline protocol.
#
#   sbatch scripts/cluster/sbatch_cloth_eval.sh <checkpoint.pth> <tag>
#
# The ONLY thing that may differ from docs/results/cloth_nohold_s*.json is the checkpoint. Every
# other flag below is copied from what those files record about themselves, because a comparison
# against a baseline run under a different protocol is worse than no comparison -- it looks
# quantitative and is not. `scripts/analysis/cloth_fold_summary.py` re-reads the emitted JSON and
# will complain if the protocol drifted.
#
# Baseline (cloth_nohold, d774ba2, pretrained policy): 3/160 entered, 0/160 held,
# best_fold_err 0.0824 +- 0.0021 across seeds. See docs/results/cloth_fold_baseline.md.
#
# --success_tolerance 0.01 pins the INHERITED rigid-tool criterion low enough that it cannot fire.
# It is not the fold threshold (that is cloth.fold tolerance 0.04, set in the env). Leaving it
# unpinned lets the tolerance curriculum drift it and gives `_successes` a second writer, at which
# point "goals/episode" stops meaning folds.

#SBATCH --job-name=cloth_eval
#SBATCH --partition=portal
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail
cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs docs/results

CKPT="${1:?usage: sbatch_cloth_eval.sh <checkpoint.pth> <tag>}"
TAG="${2:?usage: sbatch_cloth_eval.sh <checkpoint.pth> <tag>}"

if [ ! -f "$CKPT" ]; then
    echo "[eval] checkpoint not found: $CKPT" >&2
    exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

echo "[eval] checkpoint=$CKPT tag=$TAG host=$(hostname)"

for SEED in 1 2 3 4 5; do
    OUT="docs/results/cloth_${TAG}_s${SEED}.json"
    echo "[eval] === seed $SEED -> $OUT ==="
    # Module form, not a path: running episodes.py as a path puts isaacsimenvs/ on sys.path[0], so
    # `import newton` resolves to our patch package instead of the real one.
    scripts/newton_py -m isaacsimenvs.eval.episodes \
        --task Isaacsimenvs-Cloth-Direct-v0 \
        --checkpoint "$CKPT" \
        --num_envs 32 \
        --max_steps 900 \
        --success_tolerance 0.01 \
        --num_assets_per_type 1 \
        --single_variant \
        --seed "$SEED" \
        --out "$OUT" \
        || echo "[eval] seed $SEED FAILED"
done

echo "[eval] === summary ==="
.venv_isaaclab3/bin/python scripts/analysis/cloth_fold_summary.py --label "$TAG" docs/results/cloth_${TAG}_s*.json
echo "[eval] === baseline, for comparison ==="
.venv_isaaclab3/bin/python scripts/analysis/cloth_fold_summary.py --label baseline docs/results/cloth_nohold_s*.json
