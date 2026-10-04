#!/bin/bash
# Film one (policy, manipuland) pair of the rigid-cloth comparison, one array task per pair.
#
#   sbatch --array=0-3 scripts/cluster/bos14_rigid_cloth_render.sh
#   sbatch --array=0-3 --export=ALL,JOBSET=rc2  scripts/cluster/bos14_rigid_cloth_render.sh
#   sbatch --array=0-5 --export=ALL,JOBSET=many scripts/cluster/bos14_rigid_cloth_render.sh
#
# THE ARRAY RANGE IS PER JOBSET: the default set and `rc2` hold 4 entries (0-3), `many` holds 6 (0-5).
#
# The array index picks a line of RENDER_JOBS below: a label, a checkpoint, the task to run it in,
# and the chain variant (`-` for the VBD cloth). So the same policy is filmed on its OWN manipuland
# and on the cloth, which is the whole comparison -- the numbers say 100% vs 0-11%, and the point of
# the video is to show WHAT the 0% looks like.
#
# Everything about the rollout matches `bos14_rigid_cloth_eval.sh` value for value -- coef 0,
# `--success_tolerance 0.01`, `success_steps=10` -- so a clip is a sample from the distribution the
# table reports, not a differently-configured demo that happens to look good.
#
# PYGLET_HEADLESS=true, NOT xvfb-run: bos14's compute nodes have no Xvfb (only the login node does).
# See `bos14_render_cloth.sh`, which this is the rigid-aware sibling of.

#SBATCH --job-name=rc_render
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs videos/rigid_cloth

B3M=outputs/2026-09-30/02-54-22/0_rc_box3-mid_s1_155046/nn/0_rc_box3-mid_s1_155046.pth
B2S=outputs/2026-09-30/06-54-23/0_rc_box2-surface_s1_155046/nn/0_rc_box2-surface_s1_155046.pth

# label <TAB> checkpoint <TAB> task <TAB> variant
RENDER_JOBS=(
    "box3-mid_s1__on_box3-mid	$B3M	Isaacsimenvs-RigidCloth-Direct-v0	box3-mid"
    "box3-mid_s1__on_vbd	$B3M	Isaacsimenvs-Cloth-Direct-v0	-"
    "box2-surface_s1__on_box2-surface	$B2S	Isaacsimenvs-RigidCloth-Direct-v0	box2-surface"
    "box2-surface_s1__on_vbd	$B2S	Isaacsimenvs-Cloth-Direct-v0	-"
)

# JOBSET=rc2: round 2's `fix` s1 policies (array 155349), the best cloth random-start seed of
# either arm (36/64 each). The 5th field is the default worlds to film (override with WORLDS).
# A clip does NOT replay the eval's episode for the same world: under RANDOMIZE=full the render's
# start states differ from `episodes.py`'s for the same seed (step-0 obs checksums 547 vs 519), and
# VBD runs are not bit-reproducible run to run either. What matches is the distribution: the
# render's batch-wide line ("batch first episode: folded k/32") is the number to set against the
# table (19/32 vs the eval's 16/32 for box3-mid fix s1 on the cloth). Caption each clip by its own
# `folds` count, never by the eval's per_env_goals.
B3M2=outputs/2026-09-30/17-12-23/0_rc2_box3-mid_fix_s1_155349/nn/0_rc2_box3-mid_fix_s1_155349.pth
B2S2=outputs/2026-09-30/17-12-24/0_rc2_box2-surface_fix_s1_155349/nn/0_rc2_box2-surface_fix_s1_155349.pth
RC2_JOBS=(
    "rc2_box3-mid_fix_s1__on_box3-mid	$B3M2	Isaacsimenvs-RigidCloth-Direct-v0	box3-mid	0 1 20"
    "rc2_box3-mid_fix_s1__on_vbd	$B3M2	Isaacsimenvs-Cloth-Direct-v0	-	0 1 5"
    "rc2_box2-surface_fix_s1__on_box2-surface	$B2S2	Isaacsimenvs-RigidCloth-Direct-v0	box2-surface	0 1 4"
    "rc2_box2-surface_fix_s1__on_vbd	$B2S2	Isaacsimenvs-Cloth-Direct-v0	-	1 3 0"
)
# JOBSET=many: the three MANY-body variants, which the full matrix showed are the cloth-like ones
# (`box-mid-odd` and `cyl-mid-odd` take 53-62% of a cloth policy against 3-19% for the 2- and 3-body
# chains). One seed each -- the best of the three, since the many-body arms scatter badly across
# seeds and filming a 12% seed would show the variance, not the variant. Worlds are chosen from the
# matching `rc_eval_*_c0_s1.json` `per_env_goals`: two that held the fold then one that did not, so
# each cloth clip shows both halves of a ~30% success rate rather than implying it always fails.
BMO=outputs/2026-09-30/09-19-04/0_rc_box-mid-odd_s2_155046/nn/0_rc_box-mid-odd_s2_155046.pth
CMO=outputs/2026-09-30/13-18-54/0_rc_cyl-mid-odd_s1_155046/nn/0_rc_cyl-mid-odd_s1_155046.pth
BSF=outputs/2026-09-30/09-19-04/0_rc_box-surface_s2_155046/nn/0_rc_box-surface_s2_155046.pth
MANY_JOBS=(
    "box-mid-odd_s2__on_box-mid-odd	$BMO	Isaacsimenvs-RigidCloth-Direct-v0	box-mid-odd	0 1"
    "box-mid-odd_s2__on_vbd	$BMO	Isaacsimenvs-Cloth-Direct-v0	-	0 2 1"
    "cyl-mid-odd_s1__on_cyl-mid-odd	$CMO	Isaacsimenvs-RigidCloth-Direct-v0	cyl-mid-odd	0 1"
    "cyl-mid-odd_s1__on_vbd	$CMO	Isaacsimenvs-Cloth-Direct-v0	-	5 6 0"
    "box-surface_s2__on_box-surface	$BSF	Isaacsimenvs-RigidCloth-Direct-v0	box-surface	0 2 1"
    "box-surface_s2__on_vbd	$BSF	Isaacsimenvs-Cloth-Direct-v0	-	8 13 0"
)

[ "${JOBSET:-}" = "rc2" ] && RENDER_JOBS=("${RC2_JOBS[@]}")
[ "${JOBSET:-}" = "many" ] && RENDER_JOBS=("${MANY_JOBS[@]}")

IDX="${SLURM_ARRAY_TASK_ID:-0}"
# Checked explicitly: under `set -u` an out-of-range read aborts with a bare "unbound variable" and
# no hint that the --array range did not match the JOBSET's length.
if [ "$IDX" -ge "${#RENDER_JOBS[@]}" ]; then
    echo "[render] index $IDX out of range: JOBSET=${JOBSET:-<default>} has ${#RENDER_JOBS[@]} entries, use --array=0-$(( ${#RENDER_JOBS[@]} - 1 ))" >&2
    exit 1
fi
IFS=$'\t' read -r LABEL CKPT TASK VARIANT JOB_WORLDS <<<"${RENDER_JOBS[$IDX]}"
[ -f "$CKPT" ] || { echo "[render] checkpoint missing: $CKPT" >&2; exit 1; }

OUTDIR="${OUTDIR:-videos/rigid_cloth}"
COEF="${COEF:-0}"
STEPS="${STEPS:-900}"
# 32, not 8: the env count is part of the evaluated protocol (`bos14_rigid_cloth_eval.sh` runs 32),
# and a clip filmed at a different count is not a sample from the distribution the table reports.
NUM_ENVS="${NUM_ENVS:-32}"
SEED="${SEED:-1}"
STRIDE="${STRIDE:-3}"
WORLDS="${WORLDS:-${JOB_WORLDS:-0 1}}"
# OFF by default: `episodes.py` evaluates with `disable_randomization` -- sheet dead-centre and
# axis-aligned, no reset noise, no DR -- unless `--randomize` is passed, and
# `bos14_rigid_cloth_eval.sh` does not pass it. A clip filmed with `--randomize_reset` is therefore
# not a sample from the distribution the results table reports. RANDOMIZE=1 films the harder
# distribution on purpose, which is a real and much worse one: measured at 32 episodes, box3-mid s1
# drops 100% -> 81% on its own chain and box2-surface s1 drops 100% -> 34%, while the cloth policy
# holds 100% -> 97% on cloth.
RANDOMIZE="${RANDOMIZE:-0}"
# RANDOMIZE=full keeps the DR block too (`render_newton.py --randomize`), which is exactly what
# `episodes.py --randomize` -- the tables' "randomised" columns -- evaluates.
RANDOM_ARGS=(); SUFFIX=""
case "$RANDOMIZE" in
    0) ;;
    full) RANDOM_ARGS=(--randomize); SUFFIX="__rand" ;;
    *) RANDOM_ARGS=(--randomize_reset); SUFFIX="__resetonly" ;;
esac
# Pulled in from the cloth launcher's 1.6/-1.6/1.15, which sits ~2.3 m from a 10 cm manipuland: the
# first clips drew the sheet as a few dozen pixels, so the fold was not visible at all. ~0.4 m keeps
# the fingertips and the whole sheet in frame. Keep it fixed -- these clips are meant to be watched
# side by side.
CAM_OFFSET="${CAM_OFFSET:-0.26 -0.26 0.80}"
LOOK_AT="${LOOK_AT:-0.0 0.0 0.645}"

VARIANT_ARGS=()
[ "$VARIANT" != "-" ] && VARIANT_ARGS=("env.rigid_cloth.variant=${VARIANT}")

export PYTHONUNBUFFERED=1 PYGLET_HEADLESS=true
echo "[render] label=$LABEL task=$TASK variant=$VARIANT coef=$COEF worlds='$WORLDS'"
echo "[render] ckpt=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"

for WORLD in $WORLDS; do
    OUT="$OUTDIR/${LABEL}${SUFFIX}_w${WORLD}.mp4"
    echo "[render] === world $WORLD -> $OUT ==="
    scripts/newton_py -m isaacsimenvs.eval.render_newton \
        --task "$TASK" \
        --checkpoint "$CKPT" \
        --policy_config pretrained_policy/config.yaml \
        --sapg_expl_coef "$COEF" \
        --num_envs "$NUM_ENVS" \
        --world "$WORLD" \
        --steps "$STEPS" \
        --stride "$STRIDE" \
        --cam_offset $CAM_OFFSET \
        --look_at $LOOK_AT \
        --seed "$SEED" \
        --success_tolerance 0.01 \
        --num_assets_per_type 1 \
        "${RANDOM_ARGS[@]}" \
        --out "$OUT" \
        "${VARIANT_ARGS[@]}" \
        env.cloth.nan_policy=reset \
        env.termination.success_steps=10 \
        || echo "[render] $LABEL world $WORLD FAILED"
done
echo "[render] done $LABEL"
