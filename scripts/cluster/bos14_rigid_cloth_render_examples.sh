#!/bin/bash
# Film single scored episodes of the round-2 policies, to pick one success on the own chain, one
# success on the cloth and one failure on the cloth per policy.
#
#   sbatch --array=0-127 scripts/cluster/bos14_rigid_cloth_render_examples.sh
#   .venv_isaaclab3/bin/python scripts/analysis/rigid_cloth_pick_examples.py
#
# Index = policy * 32 + r: r 0-5 film worlds 0-5 on the policy's own chain, r 6-31 film worlds 0-25
# on the cloth. Outcomes cannot be chosen in advance (a render does not replay an eval episode, and
# VBD is not run-to-run reproducible), so each task films ONE episode (`--stop_on_episode_end`) and
# writes its outcome next to the clip; the picker selects afterwards. 26 cloth attempts at the ~17%
# random-start success rate these policies have at 1.5 cm leave a ~1% chance of no success.
#
# Settings: the 1.5 cm fold tolerance (TOL), the training start distribution (`--randomize`),
# no goal marker (`--no_goal`), the evaluation's 32 envs and success_steps=10.

#SBATCH --job-name=rc2_examples
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# Node-health exclusion from 2026-09-30; drop it once the two nodes are confirmed good.
#SBATCH --exclude=bos14-node-104,bos14-node-105
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs

TRAIN_JOB="${TRAIN_JOB:-155349}"
# name <TAB> arm <TAB> group <TAB> seed -- the best 1.5 cm cloth random-start seed of each policy
POLICIES=(
    "box2-surface	box2-surface	fix	1"
    "box2-surface-DR	box2-surface	fixdr	1"
    "box3-mid	box3-mid	fix	1"
    "box3-mid-DR	box3-mid	fixdr	3"
)
I="${SLURM_ARRAY_TASK_ID:-0}"
IFS=$'\t' read -r PNAME ARM GROUP SEED <<<"${POLICIES[$((I / 32))]}"
R=$((I % 32))
RUN="0_rc2_${ARM}_${GROUP}_s${SEED}_${TRAIN_JOB}"
CKPT=$(.venv_isaaclab3/bin/python -c "import sys; sys.path.insert(0, \"scripts/analysis\"); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path(\"outputs\")) or \"\")" "$RUN")
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "[examples] no checkpoint for $RUN" >&2; exit 1; }

if [ "$R" -lt 6 ]; then
    WORLD=$R; TASK=Isaacsimenvs-RigidCloth-Direct-v0; TARGET="$ARM"
    VARIANT_ARGS=("env.rigid_cloth.variant=${ARM}")
else
    WORLD=$((R - 6)); TASK=Isaacsimenvs-Cloth-Direct-v0; TARGET=cloth; VARIANT_ARGS=()
fi

TOL="${TOL:-0.015}"
OUTDIR="${OUTDIR:-videos/rigid_cloth_rc2_examples/candidates}"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/${PNAME}__on_${TARGET}_w${WORLD}.mp4"

export PYTHONUNBUFFERED=1 PYGLET_HEADLESS=true
echo "[examples] policy=$PNAME ($RUN) target=$TARGET world=$WORLD tol=$TOL host=$(hostname -s)"
scripts/newton_py -m isaacsimenvs.eval.render_newton \
    --task "$TASK" \
    --checkpoint "$CKPT" \
    --policy_config pretrained_policy/config.yaml \
    --sapg_expl_coef 0 \
    --num_envs 32 \
    --world "$WORLD" \
    --steps 610 \
    --stride 3 \
    --cam_offset 0.26 -0.26 0.80 \
    --look_at 0.0 0.0 0.645 \
    --seed 1 \
    --success_tolerance 0.01 \
    --num_assets_per_type 1 \
    --randomize \
    --no_goal \
    --stop_on_episode_end \
    --out "$OUT" \
    "${VARIANT_ARGS[@]}" \
    env.cloth.nan_policy=reset \
    env.termination.success_steps=10 \
    "env.cloth.keypoint_tolerance=${TOL}" \
    || echo "[examples] $OUT FAILED"
echo "[examples] done $OUT"
