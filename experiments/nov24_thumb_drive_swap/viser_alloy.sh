#!/bin/bash
# Live viser viewer for the thumb-jump physics variants (viser_alloy_tasks.json; switch from the browser's task
# menu, ~2-3 min restart each). Real Wuji hand on the probes' denim sheet with flat7RW_base, plus the Kuka +
# Sharpa with its own prior. Holds a GPU until cancelled.
#   sbatch experiments/nov24_thumb_drive_swap/viser.sh      # then see the log for the host and tunnel
#SBATCH --job-name=alloyviser
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --exclude=node3801,node5900-c1
#SBATCH --output=slurm_logs/%x_%j.out
cd "${SLURM_SUBMIT_DIR:-$PWD}"
D=experiments/nov24_thumb_drive_swap
export LIVE_VISER_TASKS="$PWD/$D/viser_alloy_tasks.json" LIVE_VISER_POLICIES="$PWD/$D/viser_alloy_policies.json" LIVE_VISER_POLICIES_ONLY=1
PORT="${PORT:-8080}"
echo "[alloyviser] open: ssh -N -L ${PORT}:$(hostname -s):${PORT} <your login node>  then http://localhost:${PORT}"
# Denim, as the probes (not task choices, so they survive a restart), then the NEW-defaults entry.
exec experiments/viewer.sh --task Isaacsimenvs-Cloth-Direct-v0 --num_envs 8 --port "$PORT" \
    --policy "flat7RW_base (right Wuji, zero-shot)" --policy_config checkpoints/config.yaml \
    env.cloth.density=0.1822 env.cloth.tri_ke=200.0 env.cloth.tri_ka=200.0 env.cloth.edge_ke=0.0006594 \
    env.cloth.edge_kd=9.11e-05 env.cloth.soft_contact_mu=0.265 env.cloth.plastic_yield_curvature=100.0 \
    env.cloth.self_contact_radius=0.002 env.cloth.self_contact_margin=0.003 env.cloth.start_height=0.20 \
    env.assets.finger_tip_friction=0.5 env.assets.table_friction=1.0 env.newton.njmax=16384 env.newton.nconmax=4096 \
    env.termination.success_steps=1 env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset \
    env.cloth.full_surface_contact=true env.cloth.thickness=0.008 env.newton.num_substeps=12 env.cloth.substeps=1 \
    env.cloth.table_contact_mu=0.6 env.cloth.soft_contact_ke=500.0 env.robot_self_collision=false \
    env.hand_sysid=true env.robot=flexiv_wuji_right env.assets.robot_urdf=assets/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf \
    "env.robot_root_pos=[0.0,0.51,0.20]" env.cloth.robot_contact_kd=0.5 env.cloth.friction_epsilon=0.1 \
    env.cloth.rigid_contact_k_start=100.0 env.cloth.rigid_avbd_beta=500000.0 \
    env.cloth.exit_speed_cap=0.0 env.cloth.robot_contact_gains=material
