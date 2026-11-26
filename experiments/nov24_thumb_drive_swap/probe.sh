#!/bin/bash
# Thumb-jump diagnosis (2026-11-24): is the Wuji's thumb thrown by the cloth because of its drive
# (kp / kd / torque limit), its inertia (armature) or saturation (torque limit)? Each array task runs
# hand_tracking_probe with one env.hand_drive_swap (robots.HAND_DRIVE_SWAPS) on the current VBD
# defaults (8 mm, 12/1, table mu 0.6, contact ke 500), the protocol of
# outputs/physics_sweep/hand_probe5/HLRL_cand_g8_12x1_mu06_ke500.json (flat7RW_base prior, the same
# fabric, 32 envs, 600 steps).
#
#   sbatch --array=0-9 experiments/nov24_thumb_drive_swap/probe.sh
#
# index = 2 * swap + (seed - 1); swaps: none sharpa_drive sharpa_armature sharpa_effort sharpa_all
#SBATCH --job-name=thumbswap
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --requeue
#SBATCH --exclude=node3801,node5900-c1
#SBATCH --output=slurm_logs/%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source experiments/_common.sh --task Isaacsimenvs-Cloth-Direct-v0
set -e

swaps=(none sharpa_drive sharpa_armature sharpa_effort sharpa_all)
swap="${swaps[$((INDEX / 2))]}"
seed=$((INDEX % 2 + 1))
out_dir="${OUT_DIR:-outputs/thumb_drive_swap}"
mkdir -p "$out_dir"
provenance thumbswap
echo "[thumbswap] swap=$swap seed=$seed"
out="$out_dir/${swap}_s${seed}.json"
if [[ -s "$out" ]]; then echo "[thumbswap] $out exists (requeued after finishing?), skipping"; exit 0; fi

# hand_probe5's overrides, each key once (its last value).
overrides=(
    env.cloth.density=0.1822 env.cloth.tri_ke=200.0 env.cloth.tri_ka=200.0
    env.cloth.edge_ke=0.0006594 env.cloth.edge_kd=9.11e-05 env.cloth.soft_contact_mu=0.265
    env.cloth.plastic_yield_curvature=100.0 env.cloth.self_contact_radius=0.002
    env.cloth.self_contact_margin=0.003 env.cloth.start_height=0.20
    env.assets.finger_tip_friction=0.5 env.assets.table_friction=1.0
    env.newton.njmax=16384 env.newton.nconmax=4096 env.termination.success_steps=1
    env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset
    env.cloth.full_surface_contact=true env.cloth.thickness=0.008 env.newton.num_substeps=12
    env.cloth.substeps=1 env.cloth.table_contact_mu=0.6 env.cloth.soft_contact_ke=500.0
    env.robot_self_collision=false
    "env.hand_drive_swap=$swap"
)

"${PY[@]}" isaacsimenvs/tasks/cloth/tools/hand_tracking_probe.py \
    --task Isaacsimenvs-Cloth-Direct-v0 --policy_config checkpoints/config.yaml \
    --checkpoint checkpoints/flat7RW_base.pth --num_envs 32 --max_steps 600 --seed "$seed" \
    --joints thumb --out "$out" "${overrides[@]}"
