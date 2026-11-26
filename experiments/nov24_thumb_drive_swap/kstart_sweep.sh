#!/bin/bash
# Round 4 (2026-11-25), on the REAL hand (hand_sysid), robot_contact_kd 0.5 (default) and friction_epsilon 0.1:
# start the AVBD contact-stiffness ramp higher (rigid_contact_k_start 100 -> 500 / 1000 N/m, ramp kept, ceiling
# ~1500). Fixed stiffness (no ramp) cut near-cloth thumb jumps ~58% but folds fell to 6/64; this looks for a
# start that keeps the pinch. Baseline: friction_sweep.sh's feps0p1 (k_start 100, same protocol). Runs
#   sbatch --array=0-3 experiments/nov24_thumb_drive_swap/kstart_sweep.sh
# index = 2 * variant + (seed - 1). The track probe records cloth forces (--forces, tools/thumb_force.py).
#SBATCH --job-name=thumbkstart
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --exclude=node3801,node5900-c1
#SBATCH --output=slurm_logs/%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source experiments/_common.sh --task Isaacsimenvs-Cloth-Direct-v0
set -e

names=(kstart500 kstart1000)
extras=(
    "env.cloth.friction_epsilon=0.1 env.cloth.rigid_contact_k_start=500.0"
    "env.cloth.friction_epsilon=0.1 env.cloth.rigid_contact_k_start=1000.0"
)
v=$((INDEX / 2))
name="${names[$v]}"
read -r -a extra <<< "${extras[$v]}"
seed=$((INDEX % 2 + 1))
out_dir="${OUT_DIR:-outputs/thumb_kstart_sweep}"
mkdir -p "$out_dir"
provenance thumbkstart
echo "[thumbkstart] variant=$name seed=$seed extra=${extra[*]:-none}"

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
    env.robot_self_collision=false env.hand_sysid=true
    "${extra[@]}"
)
common=(--task Isaacsimenvs-Cloth-Direct-v0 --policy_config checkpoints/config.yaml
        --checkpoint checkpoints/flat7RW_base.pth --num_envs 32 --seed "$seed")

track="$out_dir/${name}_s${seed}_track.json"
if [[ ! -s "$track" ]]; then
    "${PY[@]}" isaacsimenvs/tasks/cloth/tools/hand_tracking_probe.py "${common[@]}" --max_steps 600 \
        --joints thumb --forces --out "$track" "${overrides[@]}"
fi
pierce="$out_dir/${name}_s${seed}_pierce.json"
if [[ ! -s "$pierce" ]]; then
    "${PY[@]}" isaacsimenvs/tasks/cloth/tools/cloth_pierce_probe.py "${common[@]}" \
        --out "$pierce" "${overrides[@]}"
fi
