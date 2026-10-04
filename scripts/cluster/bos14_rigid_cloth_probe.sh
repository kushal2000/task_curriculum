#!/bin/bash
# Build / step / fold-test the rigid-cloth env in the real stack, one array task per variant.
#
#   sbatch --array=0-4 scripts/cluster/bos14_rigid_cloth_probe.sh
#   TASK=Isaacsimenvs-Cloth-Direct-v0 sbatch --array=0-0 scripts/cluster/bos14_rigid_cloth_probe.sh
#
# The second form probes the VBD sheet under the same protocol, for the speed baseline. Its fold
# drive is skipped (no hinges), so only the flat snapshot and the timing section run.
#
# Everything this measures is something `rigid_cloth_task_check.py` could NOT: it ran in MuJoCo off
# hand-built body poses. Here the joint limits, the slat self-collision, the contact buffers and the
# `body_state_w` fallback all have to survive URDF -> USD -> Isaac Lab -> Newton -> mjModel.

#SBATCH --job-name=rc_probe
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results

VARIANTS=(box3-mid box-mid-odd cyl-mid-odd box2-surface box-surface)
VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
TASK="${TASK:-Isaacsimenvs-RigidCloth-Direct-v0}"
NUM_ENVS="${NUM_ENVS:-64}"

export PYTHONUNBUFFERED=1
# Forwarded verbatim to the probe, which hands unknown args to hydra -- so a contact-buffer
# budget can be swept without editing RigidCloth.yaml:
#   EXTRA_ARGS="env.newton.per_env_triangle_pairs=65536"
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"

if [ "$TASK" = "Isaacsimenvs-RigidCloth-Direct-v0" ]; then
    OUT="docs/results/rigid_cloth_env_probe_${VARIANT}${OUT_TAG:-}.json"
    ARGS=(--variant "$VARIANT")
    echo "[probe] variant=$VARIANT envs=$NUM_ENVS host=$(hostname -s)"
else
    OUT="docs/results/rigid_cloth_env_probe_vbd${OUT_TAG:-}.json"
    ARGS=()
    echo "[probe] VBD cloth baseline envs=$NUM_ENVS host=$(hostname -s)"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MEMLOG="slurm_logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.gpumem.csv"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 5 >"$MEMLOG" &
SMI_PID=$!

scripts/newton_py -m scripts.analysis.rigid_cloth_env_probe \
    --task "$TASK" \
    "${ARGS[@]}" \
    --num_envs "$NUM_ENVS" \
    --out "$OUT" \
    --trace_steps "${TRACE_STEPS:-0}" \
    --nan_policy "${NAN_POLICY:-raise}" \
    "${EXTRA[@]}"
RC=$?
kill "$SMI_PID" 2>/dev/null
echo "[probe] exit=$RC"
echo "[probe] peak_gpu_mib=$(cut -d, -f2 "$MEMLOG" | tr -dc '0-9\n' | sort -n | tail -1)"
