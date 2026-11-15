#!/bin/bash
# Live policy viewer (isaacsimenvs/eval/live_viser.py) on the VBD cloth set up as a MEASURED fabric
# (plasticity ON, as engaging_fabric_eval.sh), with that fabric's finetune and the prior in the menu.
#
#   sbatch scripts/cluster/engaging_live_viser.sh                         # denim, finetune job 28276252
#   FABRIC=cotton FT_JOB=28276252 sbatch scripts/cluster/engaging_live_viser.sh
#
# Then read slurm_logs/live_viser_<job>.out for the node, and from your laptop:
#   ssh -N -L 8080:<node>:8080 <you>@orcd-login.mit.edu      then open http://localhost:8080
# The job holds one GPU until scancel <job> or the time limit.
#SBATCH --job-name=live_viser
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
PORT="${PORT:-8080}"
FABRIC="${FABRIC:-denim}"
FT_JOB="${FT_JOB:-28276252}"
PRIOR="${PRIOR:-checkpoints/last_0_pxpool_flat7DRABC_s1_175385_ep_3052_rew__1198.3557_.pth}"
PRESET="${FABRIC}_measured"

FT="${FT_CKPT:-}"; [ -n "$FT" ] || FT=$(.venv_isaaclab3/bin/python -I -c "import sys; sys.path.insert(0, 'scripts/analysis'); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path('outputs'), 'last') or '')" "0_ftfab_${FABRIC}_${FT_JOB}")
[ -n "$FT" ] && [ -f "$FT" ] || { echo "[live] no finetune checkpoint for $FABRIC job $FT_JOB" >&2; exit 1; }
read -r -a FAB_OV <<<"$(.venv_isaaclab3/bin/python -I isaacsimenvs/tasks/cloth/utils/fabrics.py "$PRESET" --plastic)"
[ "${#FAB_OV[@]}" -gt 0 ] || { echo "[live] no overrides for $PRESET" >&2; exit 1; }

# Menu: CKPT (if set) first as the default, then the fabric finetune, then the prior it started from.
#   CKPT=<path.pth> LABEL="release ep120" VISER_ARGS="env.cloth.release_persist=true ..." sbatch ...
export LIVE_VISER_POLICIES="slurm_logs/live_viser_policies_${SLURM_JOB_ID:-local}.json"
export LIVE_VISER_POLICIES_ONLY=1
DEFAULT="${FABRIC} finetune (${FT_JOB})"
EXTRA_ENTRY=""
if [ -n "${CKPT:-}" ]; then
    [ -f "$CKPT" ] || { echo "[live] CKPT not found: $CKPT" >&2; exit 1; }
    DEFAULT="${LABEL:-$(basename "$CKPT" .pth)}"
    EXTRA_ENTRY="\"${DEFAULT}\": [\"$CKPT\", 0.0],"
fi
cat >"$LIVE_VISER_POLICIES" <<EOF
{${EXTRA_ENTRY}
 "${FABRIC} finetune (${FT_JOB})": ["$FT", 0.0],
 "prior (no finetune)": ["$PRIOR", 0.0]}
EOF
read -r -a VISER_OV <<<"${VISER_ARGS:-}"
# A VISER_ARGS key replaces the preset's value for that key (hydra takes each key once).
KEEP=()
for o in "${FAB_OV[@]}"; do
    dup=0; for v in ${VISER_OV[@]+"${VISER_OV[@]}"}; do [ "${o%%=*}" = "${v%%=*}" ] && dup=1; done
    [ "$dup" = 0 ] && KEEP+=("$o")
done
FAB_OV=("${KEEP[@]}")

echo "[live] node $(hostname -s) port $PORT fabric=$PRESET (plastic) commit $(git rev-parse --short HEAD)"
echo "[live] finetune=$FT default=$DEFAULT ckpt=${CKPT:-} extra=${VISER_ARGS:-}"
echo "[live] tunnel: ssh -N -L ${PORT}:$(hostname -s):${PORT} $USER@orcd-login.mit.edu   then http://localhost:${PORT}"
exec scripts/newton_py -m isaacsimenvs.eval.live_viser \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --policy "$DEFAULT" \
    --policy_config "${POLICY_CFG:-checkpoints/config.yaml}" \
    --num_envs "${ENVS:-4}" \
    --port "$PORT" \
    env.cloth.keypoint_tolerance=0.015 \
    "${FAB_OV[@]}" ${VISER_OV[@]+"${VISER_OV[@]}"}
