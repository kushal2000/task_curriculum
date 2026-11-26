#!/bin/bash
# Round 5 (2026-11-25): Alloy-ported contact guards (newton_support/contact_guards.py) on the round-4 protocol
# (REAL hand, robot_contact_kd 0.5, friction_epsilon 0.1, k_start 100). Every variant records cloth health
# (--health: particle speed, edge strain, exit kick, self-contact overflow) with the kick measured.
#   base      guards off (kick measured only)
#   cap1      exit_speed_cap 1.0 m/s (Alloy's default)
#   cap0p25   exit_speed_cap 0.25 m/s (our dt is ~20x Alloy's, so kicks are ~20x smaller for the same depth)
#   mass      robot_contact_gains mass (per-record ke/kd from proxy link mass, safety 0.5)
#   mass_cap1 both
# Runs  sbatch --array=0-9 experiments/nov24_thumb_drive_swap/alloy_guards.sh     index = 2 * variant + (seed - 1)
# Smoke SMOKE=1 sbatch --array=8 --time=0:30:00 ...   (4 envs, 60 steps, track probe only, to outputs/alloy_guards_smoke)
#SBATCH --job-name=thumballoy
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --requeue
#SBATCH --exclude=node3801,node5900-c1
#SBATCH --output=slurm_logs/%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source experiments/_common.sh --task Isaacsimenvs-Cloth-Direct-v0
set -e

names=(base cap1 cap0p25 mass mass_cap1)
extras=(
    ""
    "env.cloth.exit_speed_cap=1.0"
    "env.cloth.exit_speed_cap=0.25"
    "env.cloth.robot_contact_gains=mass"
    "env.cloth.robot_contact_gains=mass env.cloth.exit_speed_cap=1.0"
)
v=$((INDEX / 2))
name="${names[$v]}"
read -r -a extra <<< "${extras[$v]}"
seed="${SEED:-$((INDEX % 2 + 1))}"   # SEED=3 sbatch --array=<2*variant> ... for an extra seed
out_dir="${OUT_DIR:-outputs/alloy_guards}"
envs=32; steps=600
if [[ -n "${SMOKE:-}" ]]; then out_dir="${OUT_DIR:-outputs/alloy_guards_smoke}"; envs=4; steps=60; fi
mkdir -p "$out_dir"
provenance thumballoy
echo "[thumballoy] variant=$name seed=$seed extra=${extra[*]:-none} envs=$envs steps=$steps"

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
    env.cloth.friction_epsilon=0.1 env.cloth.robot_contact_kd=0.5 env.cloth.exit_speed_measure=true
    "${extra[@]}"
)
# EXTRA="k=v ..." appends overrides (e.g. EXTRA=env.cloth.use_cuda_graph=true for a graph-mode smoke)
read -r -a more <<< "${EXTRA:-}"
overrides+=("${more[@]}")
common=(--task Isaacsimenvs-Cloth-Direct-v0 --policy_config checkpoints/config.yaml
        --checkpoint checkpoints/flat7RW_base.pth --num_envs "$envs" --seed "$seed")

track="$out_dir/${name}_s${seed}_track.json"
if [[ ! -s "$track" ]]; then
    "${PY[@]}" isaacsimenvs/tasks/cloth/tools/hand_tracking_probe.py "${common[@]}" --max_steps "$steps" \
        --joints thumb --forces --health --out "$track" "${overrides[@]}"
fi
[[ -n "${SMOKE:-}" || -n "${SKIP_PIERCE:-}" ]] && exit 0   # SKIP_PIERCE=1: track probe only
pierce="$out_dir/${name}_s${seed}_pierce.json"
if [[ ! -s "$pierce" ]]; then
    "${PY[@]}" isaacsimenvs/tasks/cloth/tools/cloth_pierce_probe.py "${common[@]}" \
        --out "$pierce" "${overrides[@]}"
fi
