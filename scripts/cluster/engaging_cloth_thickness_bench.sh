#!/bin/bash
# Finger pierce and step latency vs cloth thickness / grid (scripts/analysis/cloth_pierce_probe.py),
# cotton_measured preset (plastic), the extended cotton release policy, release-and-persist settings.
#   sbatch --array=0-6 scripts/cluster/engaging_cloth_thickness_bench.sh
# Per variant: (a) pierce: 64 envs, one 12 s episode each; (b) latency: 768 envs (the finetune's
# count), 80 steps, env.step wall time (warm-up excluded).
# Output: docs/results/cloth_thickness_bench/<tag>_{pierce,timing}.json
#SBATCH --job-name=thickbench
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=node3801
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
I="${SLURM_ARRAY_TASK_ID:-0}"
TAGS=(t16_r7_nofs t16_r7_fs t2_r7_fs t05_r7_fs t8_r13_fs_i24 t4_r25_fs_i12 t4_r25_fs_i96)
CFGS=("env.cloth.full_surface_contact=false"
      "env.cloth.full_surface_contact=true"
      "env.cloth.full_surface_contact=true env.cloth.thickness=0.002"
      "env.cloth.full_surface_contact=true env.cloth.thickness=0.0005"
      "env.cloth.full_surface_contact=true env.cloth.resolution=13 env.cloth.thickness=0.008 env.cloth.vbd_iterations=24"
      "env.cloth.full_surface_contact=true env.cloth.resolution=25 env.cloth.thickness=0.004"
      "env.cloth.full_surface_contact=true env.cloth.resolution=25 env.cloth.thickness=0.004 env.cloth.vbd_iterations=96")
TAG=${TAGS[$I]}
OUT=docs/results/cloth_thickness_bench
mkdir -p "$OUT"
CKPT=outputs/2026-11-10/19-49-11/0_ftfabrel3sRfsC2_cotton_28647593/nn/last_0_ftfabrel3sRfsC2_cotton_28647593_ep_500_rew__2788.9155_.pth
read -r -a FAB <<<"$(.venv_isaaclab3/bin/python -I isaacsimenvs/tasks/cloth/utils/fabrics.py cotton_measured --plastic)"
read -r -a CFG <<<"${CFGS[$I]}"
BASE=(env.cloth.keypoint_tolerance=0.015 env.cloth.release_persist=true env.episode_length_s=12.0)
echo "[bench] node $(hostname -s) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) tag $TAG: ${CFG[*]}"
# MODES="timing pierce" (default); TIMING_N overrides the 768-env latency run (output *_timing<N>.json).
for MODE in ${MODES:-timing pierce}; do
    SUF=$MODE
    if [ "$MODE" = timing ]; then N=${TIMING_N:-768}; S=80; [ -n "${TIMING_N:-}" ] && SUF="timing${N}"; else N=64; S=730; fi
    [ -s "$OUT/${TAG}_${SUF}.json" ] && { echo "[bench] exists ${TAG}_${SUF}"; continue; }
    scripts/newton_py -m scripts.analysis.cloth_pierce_probe --checkpoint "$CKPT" \
        --policy_config checkpoints/config.yaml --num_envs $N --max_steps $S --out "$OUT/${TAG}_${SUF}.json" \
        "${BASE[@]}" "${FAB[@]}" "${CFG[@]}" 2>&1 | grep -E "^\[(pierce|bench)|Error|Traceback|line [0-9]+|ValueError" | tail -15
    echo "[bench] $SUF done: $(ls -la $OUT/${TAG}_${SUF}.json 2>&1 | tail -1)"
done
