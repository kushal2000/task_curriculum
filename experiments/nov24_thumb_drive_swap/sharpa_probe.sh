#!/bin/bash
# Is the VBD cloth physics stable on the Sharpa hand? The restored legacy Kuka iiwa14 + left Sharpa
# (env.robot=kuka_sharpa, its URDF and base position from before 37391a7) on the same cloth and
# physics as probe.sh / contact_sweep.sh (8 mm, 12/1, table mu 0.6, soft_contact_ke 500, the same
# fabric), driven by its own rigid-plate fold prior (flat7DRABC, 29 actions). Runs
# hand_tracking_probe (uncommanded thumb jumps) and cloth_pierce_probe.
#
#   sbatch --array=0-1 experiments/nov24_thumb_drive_swap/sharpa_probe.sh     # index = seed - 1
#SBATCH --job-name=sharpaprobe
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

seed=$((INDEX + 1))
out_dir="${OUT_DIR:-outputs/thumb_sharpa}"
mkdir -p "$out_dir"
provenance sharpaprobe
echo "[sharpaprobe] seed=$seed"

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
    env.robot=kuka_sharpa
    env.assets.robot_urdf=assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf
    "env.robot_root_pos=[0.0,0.8,0.0]"
)
common=(--task Isaacsimenvs-Cloth-Direct-v0 --policy_config checkpoints/config.yaml
        --checkpoint checkpoints/last_0_pxpool_flat7DRABC_s1_175385_ep_3052_rew__1198.3557_.pth
        --num_envs 32 --seed "$seed")

track="$out_dir/sharpa_s${seed}_track.json"
if [[ ! -s "$track" ]]; then
    "${PY[@]}" isaacsimenvs/tasks/cloth/tools/hand_tracking_probe.py "${common[@]}" --max_steps 600 \
        --joints thumb --out "$track" "${overrides[@]}"
fi
pierce="$out_dir/sharpa_s${seed}_pierce.json"
if [[ ! -s "$pierce" ]]; then
    "${PY[@]}" isaacsimenvs/tasks/cloth/tools/cloth_pierce_probe.py "${common[@]}" \
        --out "$pierce" "${overrides[@]}"
fi
