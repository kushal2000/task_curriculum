#!/bin/bash
# Can the cotton release policy still grip the cloth as the contact layer gets thinner?
# scripts/analysis/cloth_pierce_probe.py (outcomes, finger pierce, cloth lift/move) on the thin cotton
# preset (yield 20, set 80, stretch 100, full-surface contact) at contact thickness T, 64 envs, one
# 12 s release episode each, the extended cotton release policy (16 mm-trained).
#   sbatch --array=0-9 scripts/cluster/engaging_cloth_grip_bench.sh
# Output: docs/results/cloth_grip_bench/<tag>.json
#SBATCH --job-name=gripbench
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=node3801
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
I="${SLURM_ARRAY_TASK_ID:-0}"
#      tag            thickness  sdf_res
VAR=("t16            0.016      0"
     "t8             0.008      0"
     "t4             0.004      0"
     "t2             0.002      0"
     "t1             0.001      0"
     "t03            0.0003     0"
     "t2_sdf128      0.002      128"
     "t03_sdf128     0.0003     128"
     "t2_sdf256      0.002      256"
     "t03_sdf256     0.0003     256"
     "t12            0.012      0"
     "t10            0.010      0")
read -r TAG T SDF <<<"${VAR[$I]}"
R=$(.venv_isaaclab3/bin/python -I -c "print(min($T, 0.002))")
M=$(.venv_isaaclab3/bin/python -I -c "print(1.5 * min($T, 0.002))")
OUT=docs/results/cloth_grip_bench
mkdir -p "$OUT"
CKPT=outputs/2026-11-10/19-49-11/0_ftfabrel3sRfsC2_cotton_28647593/nn/last_0_ftfabrel3sRfsC2_cotton_28647593_ep_500_rew__2788.9155_.pth
read -r -a FAB <<<"$(.venv_isaaclab3/bin/python -I isaacsimenvs/tasks/cloth/utils/fabrics.py cotton_measured --plastic)"
KEEP=(); for o in "${FAB[@]}"; do case "${o%%=*}" in env.cloth.tri_ke|env.cloth.tri_ka|env.cloth.self_contact_radius|env.cloth.plastic_yield_curvature) ;; *) KEEP+=("$o");; esac; done
CFG=("${KEEP[@]}" env.cloth.tri_ke=100.0 env.cloth.tri_ka=100.0 env.cloth.plastic_yield_curvature=20.0 env.cloth.plastic_set_curvature=80.0
     env.cloth.thickness=$T env.cloth.self_contact_radius=$R env.cloth.self_contact_margin=$M
     env.cloth.full_surface_contact=true env.cloth.full_surface_sdf_resolution=$SDF
     env.cloth.nan_policy=reset env.cloth.keypoint_tolerance=0.015 env.cloth.release_persist=true env.episode_length_s=12.0)
echo "[grip] node $(hostname -s) tag $TAG: ${CFG[*]}"
[ -s "$OUT/$TAG.json" ] && { echo "[grip] exists $TAG"; exit 0; }
scripts/newton_py -m scripts.analysis.cloth_pierce_probe --checkpoint "$CKPT" \
    --policy_config checkpoints/config.yaml --num_envs 64 --max_steps 730 --out "$OUT/$TAG.json" \
    "${CFG[@]}" 2>&1 | grep -E "^\[(pierce|grip)|Error|Traceback|line [0-9]+|ValueError" | tail -15
