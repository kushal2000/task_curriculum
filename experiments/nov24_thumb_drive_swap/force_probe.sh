#!/bin/bash
# How hard does the cloth push on the thumb? hand_tracking_probe --forces (tools/thumb_force.py): the cloth's
# torque about each thumb joint vs its torque limit, and the force on each thumb link vs the cloth's weight.
# Same cloth/physics/protocol as contact_sweep.sh, seed 1.
#   0 wuji_base   real Wuji hand (hand_sysid), default contact damping (~50 N s/m)
#   1 wuji_kd0p5  real Wuji hand, contact damping 0.5 on every robot link
#   2 sharpa      restored Kuka + left Sharpa with its own prior (sharpa_probe.sh's setup)
#
#   sbatch --array=0-2 experiments/nov24_thumb_drive_swap/force_probe.sh
#SBATCH --job-name=thumbforce
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --requeue
#SBATCH --exclude=node3801,node5900-c1
#SBATCH --output=slurm_logs/%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source experiments/_common.sh --task Isaacsimenvs-Cloth-Direct-v0
set -e

names=(wuji_base wuji_kd0p5 sharpa)
name="${names[$INDEX]}"
out_dir="${OUT_DIR:-outputs/thumb_force}"
mkdir -p "$out_dir"
provenance thumbforce
echo "[thumbforce] variant=$name"

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
)
ckpt=checkpoints/flat7RW_base.pth
case "$name" in
    wuji_base)  overrides+=(env.hand_sysid=true) ;;
    wuji_kd0p5) overrides+=(env.hand_sysid=true env.cloth.robot_contact_kd=0.5) ;;
    sharpa)     overrides+=(env.robot=kuka_sharpa
                            env.assets.robot_urdf=assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf
                            "env.robot_root_pos=[0.0,0.8,0.0]")
                ckpt=checkpoints/last_0_pxpool_flat7DRABC_s1_175385_ep_3052_rew__1198.3557_.pth ;;
esac

"${PY[@]}" isaacsimenvs/tasks/cloth/tools/hand_tracking_probe.py --task Isaacsimenvs-Cloth-Direct-v0 \
    --policy_config checkpoints/config.yaml --checkpoint "$ckpt" --num_envs 32 --seed 1 --max_steps 600 \
    --joints thumb --forces --out "$out_dir/${name}_s1.json" "${overrides[@]}"
