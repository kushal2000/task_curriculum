#!/bin/bash
# VBD-cloth-only evaluation of a LIST of checkpoints -- a score-vs-epoch curve from the snapshots a run
# writes with `snapshot_every` / `periodic_save_every`. Same protocol as bos14_px_eval_matrix.sh's
# target 4 (VBD cloth): pinned and randomised starts, 4 cm and 1.5 cm cutoffs, two eval seeds.
#
#   LIST=<file> sbatch --array=0-$(( 2 * $(wc -l < <file>) - 1 )) scripts/cluster/bos14_px_eval_trend.sh
#   python3 scripts/analysis/px_eval_matrix_summary.py <tag> [<tag> ...]
#
# LIST: one checkpoint per line, "<tag> <seed> <checkpoint path>", e.g.
#   ft7_flat0_ep1500 2 outputs/.../nn/snap_0_pxpool_ft7_flat0_s2_170001_ep_1500.pth
# Index = line*2 + start (0 = pinned, 1 = randomised).
# Output: docs/results/px_matrix/<tag>_s<seed>__on_vbd__<pinned|rand>__<4cm|1p5cm>_s<es>.json, the
# matrix's naming, so px_eval_matrix_summary.py and px_eval_fold_steps.py read it as is.

#SBATCH --job-name=px_trend
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
: "${LIST:?set LIST to the checkpoint list file}"
OUTDIR="${OUTDIR:-docs/results/px_matrix}"
mkdir -p slurm_logs "$OUTDIR"
I="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(( I / 2 + 1 )); S=$(( I % 2 ))
read -r TAG SEED CKPT < <(sed -n "${LINE}p" "$LIST")
[ -n "${CKPT:-}" ] && [ -f "$CKPT" ] || { echo "[px_trend] line $LINE of $LIST: no checkpoint '${CKPT:-}'" >&2; exit 1; }
if [ "$S" -eq 1 ]; then START=rand; SARGS=(--randomize); else START=pinned; SARGS=(); fi
export PYTHONUNBUFFERED=1
echo "[px_trend] $TAG s$SEED vbd/$START ckpt=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
for CUT in 0.04 0.015; do
    [ "$CUT" = 0.04 ] && CT=4cm || CT=1p5cm
    for ES in 1 2; do
        OUT="$OUTDIR/${TAG}_s${SEED}__on_vbd__${START}__${CT}_s${ES}.json"
        [ -s "$OUT" ] && { echo "[px_trend] exists: $OUT"; continue; }
        scripts/newton_py -m isaacsimenvs.eval.episodes --task Isaacsimenvs-Cloth-Direct-v0 \
            --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml \
            --sapg_expl_coef 0 --num_envs 32 --max_steps 900 --success_tolerance 0.01 \
            --num_assets_per_type 1 --single_variant "${SARGS[@]}" --seed "$ES" --out "$OUT" \
            "env.cloth.keypoint_tolerance=$CUT" env.cloth.nan_policy=reset \
            env.termination.success_steps=10 \
            || echo "[px_trend] $OUT FAILED"
    done
done
echo "[px_trend] done"
