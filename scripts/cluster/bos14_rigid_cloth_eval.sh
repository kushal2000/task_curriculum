#!/bin/bash
# Held-fold evaluation of one (policy, manipuland) pair from the cross-evaluation manifest.
#
#   .venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_eval_manifest.py --job <train_job> \
#       --reference outputs/.../nn/0_pretrained_s1_resume_150911.pth
#   sbatch --array=0-N scripts/cluster/bos14_rigid_cloth_eval.sh
#
# The protocol is copied from `bos14_cloth_eval_ckpt.sh` value for value -- 32 envs x 900 steps,
# `--success_tolerance 0.01` so the inherited rigid-tool criterion cannot fire and only a real fold
# counts, `env.termination.success_steps=10` so a fold must persist 10 steps -- because that is what
# makes these numbers comparable to `docs/results/cloth_prior_vs_scratch.md` rather than only to each
# other.
#
# `--sapg_expl_coef 0` evaluates block 5, the one trained on task reward alone. The eval player's
# default is 50.0 (block 0, the released checkpoint's value), and using it once reported 133/0/4 held
# folds for three policies whose real rates were 149/28/21 -- right about the first, badly wrong about
# the other two. See the "SAPG block" section of that document.

#SBATCH --job-name=rc_eval
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results

MANIFEST="${MANIFEST:-docs/results/rc_eval_manifest.tsv}"
[ -f "$MANIFEST" ] || { echo "[eval] no manifest at $MANIFEST" >&2; exit 1; }

LINE=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$MANIFEST")
[ -n "$LINE" ] || { echo "[eval] no manifest line for index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
IFS=$'\t' read -r LABEL CKPT TASK VARIANT <<<"$LINE"
[ -f "$CKPT" ] || { echo "[eval] checkpoint missing: $CKPT" >&2; exit 1; }

# Env count is part of the protocol, not a free knob: 32 is what every number in the comparison
# was measured at. Exposed only so the render path (which films 8) can be checked against the
# evaluation harness at the SAME count, after a clip and the table disagreed.
NUM_ENVS="${NUM_ENVS:-32}"
EVAL_SEEDS="${EVAL_SEEDS:-1 2}"
COEF="${COEF:-0}"

VARIANT_ARGS=()
[ "$VARIANT" != "-" ] && VARIANT_ARGS=("env.rigid_cloth.variant=${VARIANT}")

# The default (0) is the protocol the whole comparison is quoted under: `disable_randomization`,
# i.e. the sheet starts dead-centre, axis-aligned, with no reset noise and no DR. RANDOMIZE=1 keeps
# the training distribution instead, which answers a different and worth-asking question -- whether
# a policy that scores 100% from the pinned pose still works when the object does not start where it
# was trained to expect it. LABEL_SUFFIX keeps the two sets of JSON apart.
RANDOMIZE="${RANDOMIZE:-0}"
LABEL_SUFFIX="${LABEL_SUFFIX:-}"
RANDOM_ARGS=()
[ "$RANDOMIZE" != "0" ] && RANDOM_ARGS=(--randomize)

export PYTHONUNBUFFERED=1
echo "[eval] label=$LABEL task=$TASK variant=$VARIANT coef=$COEF seeds='$EVAL_SEEDS'"
echo "[eval] ckpt=$CKPT host=$(hostname -s)"

for SEED in $EVAL_SEEDS; do
    OUT="docs/results/rc_eval_${LABEL}${LABEL_SUFFIX}_c${COEF}_s${SEED}.json"
    echo "[eval] === seed $SEED -> $OUT ==="
    scripts/newton_py -m isaacsimenvs.eval.episodes \
        --task "$TASK" \
        --checkpoint "$CKPT" \
        --policy_config pretrained_policy/config.yaml \
        --sapg_expl_coef "$COEF" \
        --num_envs "$NUM_ENVS" \
        --max_steps 900 \
        --success_tolerance 0.01 \
        --num_assets_per_type 1 \
        --single_variant \
        "${RANDOM_ARGS[@]}" \
        --seed "$SEED" \
        --out "$OUT" \
        "${VARIANT_ARGS[@]}" \
        env.cloth.nan_policy=reset \
        env.termination.success_steps=10 \
        || echo "[eval] $LABEL seed $SEED FAILED"
done
echo "[eval] done $LABEL"
