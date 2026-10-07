#!/bin/bash
# Sim-to-sim cross evaluation: a run's policies on a rigid simulator other than (or the same as) the one
# they trained in. Same protocol as bos14_physx_eval_on_newton.sh (32 envs x 2 eval seeds, 900 steps,
# fold held 10 steps), so the numbers sit next to the VBD-cloth ones.
#
#   RUN=0_pxchain_arm1e3_s{SEED}_157449 LABEL=px_arm1e3 TARGET=physx_chain \
#       sbatch --array=0-5 scripts/cluster/bos14_cross_eval.sh
#
# RUN has {SEED} where the seed goes. TARGET: newton_chain | physx_chain | newton_pool | physx_pool.
# Index = (seed-1)*2 + start; start 0 = pinned, 1 = randomised.
# Writes docs/results/xeval_<LABEL>__on_<TARGET>[__rand]_s<seed>_c0_s<es>.json.

#SBATCH --job-name=px_xeval
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
: "${RUN:?set RUN (with {SEED})}"; : "${LABEL:?set LABEL}"; : "${TARGET:?set TARGET}"
I="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( I / 2 + 1 )); S=$(( I % 2 ))
NAME="${RUN//\{SEED\}/$SEED}"
CKPT=$(.venv_isaaclab3/bin/python -c "import sys; sys.path.insert(0, 'scripts/analysis'); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path('outputs')) or '')" "$NAME")
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "[xeval] no checkpoint for $NAME" >&2; exit 1; }
case "$TARGET" in
    newton_chain) PY=(scripts/newton_py); T=Isaacsimenvs-RigidCloth-Direct-v0; X=(env.rigid_cloth.variant=box3-mid) ;;
    physx_chain)  PY=(.venv_isaacsim/bin/python --); T=Isaacsimenvs-RigidClothPhysx-Direct-v0; X=(--headless env.rigid_cloth.variant=box3-mid) ;;
    newton_pool)  PY=(scripts/newton_py); T=Isaacsimenvs-FoldPool-Direct-v0; X=(env.newton.per_env_triangle_pairs=2097152) ;;
    physx_pool)   PY=(.venv_isaacsim/bin/python --); T=Isaacsimenvs-FoldPoolPhysx-Direct-v0; X=(--headless) ;;
    *) echo "bad TARGET $TARGET"; exit 2 ;;
esac
[ "${PY[1]:-}" = "--" ] && PY=("${PY[0]}")
START=(); SUFFIX=""
[ "$S" -eq 1 ] && { START=(--randomize); SUFFIX="__rand"; }
echo "[xeval] $NAME on $TARGET$SUFFIX ckpt=$CKPT host=$(hostname -s)"
for ES in 1 2; do
    "${PY[@]}" -m isaacsimenvs.eval.episodes \
        --task "$T" --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml \
        --sapg_expl_coef 0 --num_envs 32 --max_steps 900 --success_tolerance 0.01 \
        --num_assets_per_type 1 --single_variant "${START[@]}" --seed "$ES" \
        --out "docs/results/xeval_${LABEL}__on_${TARGET}${SUFFIX}_s${SEED}_c0_s${ES}.json" \
        env.cloth.nan_policy=reset env.termination.success_steps=10 "${X[@]}" \
        || echo "[xeval] $NAME seed $ES FAILED"
done
echo "[xeval] done"
