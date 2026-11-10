#!/bin/bash
# Cloth contact A/B and sweeps on one fabric: pierce stats, fold outcomes and step time per config
# (scripts/analysis/cloth_pierce_probe.py). See env.cloth.full_surface_contact / self_contact_mu /
# table_contact_mu / friction_epsilon and docs/reports/4_physx/cloth_finger_pierce.md.
#
#   sbatch --array=0-1 scripts/cluster/engaging_cloth_full_surface.sh               # SET=fs: particle vs full-surface
#   SET=mu sbatch --array=0-7 scripts/cluster/engaging_cloth_full_surface.sh        # friction sweep (below)
#
# SET=mu (all full-surface except 7): 0 preset mu, 1 cloth-on-cloth 1.0, 2 2.0, 3 5.67 (tan 80),
#   4 5.67 + friction_epsilon 1e-3, 5 5.67 + cloth-on-table 0.7, 6 5.67 + table 0.7 + eps 1e-3,
#   7 5.67 with PARTICLE contact.
# Same policy, preset, envs and seed throughout, engaging_fabric_eval.sh's protocol (measured fabric +
# plasticity, pinned start, 1.5 cm keypoint tolerance, fold held 10 steps). Override with CKPT=,
# FABRIC=, NENVS=, SEED=, EXTRA= (more hydra overrides, applied to every config), OUT_SUFFIX=.
# Output: docs/results/cloth_full_surface/<fabric>_<tag>_s<seed>.json (pierce stats + timing).
#SBATCH --job-name=clothfs
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUTDIR=docs/results/cloth_full_surface
mkdir -p slurm_logs "$OUTDIR"
export PYTHONUNBUFFERED=1
I="${SLURM_ARRAY_TASK_ID:-0}"
FS="env.cloth.full_surface_contact=true"
M="env.cloth.self_contact_mu=5.67"
if [ "${SET:-fs}" = mu ]; then
    TAGS=(fs_mupreset fs_mucc1 fs_mucc2 fs_mucc5.7 fs_mucc5.7_eps1e-3 fs_mucc5.7_tab0.7 fs_mucc5.7_tab0.7_eps1e-3 particle_mucc5.7)
    ARMS=("$FS" "$FS env.cloth.self_contact_mu=1.0" "$FS env.cloth.self_contact_mu=2.0" "$FS $M"
          "$FS $M env.cloth.friction_epsilon=0.001" "$FS $M env.cloth.table_contact_mu=0.7"
          "$FS $M env.cloth.table_contact_mu=0.7 env.cloth.friction_epsilon=0.001"
          "env.cloth.full_surface_contact=false $M")
else
    TAGS=(particle fullsurf)
    ARMS=("env.cloth.full_surface_contact=false" "$FS")
fi
FABRIC="${FABRIC:-big_napkin}"
CKPT="${CKPT:-outputs/2026-11-09/01-39-08/0_ftfabrel3sR_big_napkin_28411871/nn/last_0_ftfabrel3sR_big_napkin_28411871_ep_500_rew__892.33405_.pth}"
SEED="${SEED:-3}"
[ -f "$CKPT" ] || { echo "[clothfs] no checkpoint $CKPT" >&2; exit 1; }
read -r -a FAB_OV <<<"$(.venv_isaaclab3/bin/python -I isaacsimenvs/tasks/cloth/utils/fabrics.py "${FABRIC}_measured" --plastic)"
OUT="$OUTDIR/${FABRIC}_${TAGS[$I]}${OUT_SUFFIX:-}_s${SEED}.json"
echo "[clothfs] ${TAGS[$I]} fabric=${FABRIC}_measured ckpt=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader
scripts/newton_py -m scripts.analysis.cloth_pierce_probe \
    --checkpoint "$CKPT" --policy_config "${POLICY_CFG:-checkpoints/config.yaml}" \
    --num_envs "${NENVS:-64}" --max_steps 900 --seed "$SEED" --out "$OUT" \
    env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset env.termination.success_steps=10 \
    "${FAB_OV[@]}" ${ARMS[$I]} ${EXTRA:-}
