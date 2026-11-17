#!/bin/bash
# Presentation renders (textured, studio look) for the demo page, Engaging (one GPU).
#
#   sbatch experiments/nov16_vbd_cloth/render_demo.sh denim  [extra render_newton.py args]
#   sbatch experiments/nov16_vbd_cloth/render_demo.sh rigid  [extra render_physx.py args]
#
#   denim: the denim contact-curriculum finetune (denim_28974159, final 0.35 mm stage) on the VBD
#          denim cloth with the overrides of its curriculum eval (denim_final_c035_m3): recorded on
#          Newton (record_newton.py), replayed in the same RTX scene as `rigid` (render_physx.py --replay).
#   rigid: the PhysX fold-pool checkpoint (flat7DRABC) on the hinged rigid-plate pool, Isaac Sim RTX
#          inside the Isaac Sim 5.1 container (scripts/visualization/render_physx.py).
# Output: videos/demo/<mode>.mp4 (OUT=... to override).
#SBATCH --job-name=render_demo
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs videos/demo
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
MODE="${1:?denim|rigid}"; shift
OUT="${OUT:-videos/demo/${MODE}.mp4}"
echo "[render_demo] mode=$MODE host=$(hostname -s) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD) out=$OUT"

case "$MODE" in
denim)
    # 1) record the episode on Newton (the physics), 2) replay it in the shared RTX scene (the look).
    [[ -d /orcd/scratch/orcd/011/hglin/isaac_assets/6.1/Isaac ]] && export ISAACSIM_ASSET_ROOT="${ISAACSIM_ASSET_ROOT:-/orcd/scratch/orcd/011/hglin/isaac_assets/6.1}"
    CKPT="${CKPT:-outputs/2026-11-14/00-24-53/0_ftfabcur350_denim_28974159/nn/last_0_ftfabcur350_denim_28974159_ep_1600_rew__3179.03_.pth}"
    TRAJ="${OUT%.mp4}.npz"
    # REUSE=1 replays an existing recording instead of simulating again.
    [[ "${REUSE:-0}" == 1 && -s "$TRAJ" ]] || ( for lib in /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib64/libstdc++.so.6; do
          [[ -f "$lib" ]] && { export LD_PRELOAD="$lib${LD_PRELOAD:+:$LD_PRELOAD}"; break; }
      done
      .venv_isaaclab3/bin/python scripts/visualization/record_newton.py --task Isaacsimenvs-Cloth-Direct-v0 \
        --checkpoint "$CKPT" --policy_config checkpoints/config.yaml --sapg_expl_coef 0 \
        --steps "${STEPS:-730}" --seed "${SEED:-1}" --out "$TRAJ" --num_envs "${NUM_ENVS:-16}" --randomize_reset \
        env.cloth.density=0.1822 env.cloth.tri_ke=200.0 env.cloth.tri_ka=200.0 env.cloth.edge_ke=0.0006594 \
        env.cloth.edge_kd=9.11e-05 env.cloth.soft_contact_mu=0.265 env.cloth.thickness=0.00035 \
        env.cloth.self_contact_radius=0.00035 env.cloth.self_contact_margin=0.003 \
        env.cloth.plastic_yield_curvature=100.0 env.cloth.plastic_set_curvature=80.0 \
        env.cloth.full_surface_contact=true env.cloth.nan_policy=reset env.cloth.keypoint_tolerance=0.015 \
        env.termination.success_steps=10 env.cloth.release_persist=true env.episode_length_s=12.0 ) || exit 1
    command -v apptainer >/dev/null || module load apptainer
    SIF="${ISAACSIM_SIF:-/orcd/scratch/orcd/011/hglin/containers/isaac-sim_5.1.0.sif}"
    apptainer exec --nv --bind /orcd --pwd "$PWD" --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y "$SIF" \
        .venv_isaacsim_ct/bin/python scripts/visualization/render_physx.py --headless --enable_cameras \
        --task Isaacsimenvs-FoldPoolPhysx-Direct-v0 --replay "$TRAJ" --out "$OUT" --hold_s 1.5 \
        --caption "Denim (deformable, VBD)" --subtitle "policy finetuned on denim" "$@"
    ;;
rigid)
    command -v apptainer >/dev/null || module load apptainer
    SIF="${ISAACSIM_SIF:-/orcd/scratch/orcd/011/hglin/containers/isaac-sim_5.1.0.sif}"
    CKPT="${CKPT:-checkpoints/last_0_pxpool_flat7DRABC_s1_175385_ep_3052_rew__1198.3557_.pth}"
    apptainer exec --nv --bind /orcd --pwd "$PWD" --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y "$SIF" \
        .venv_isaacsim_ct/bin/python scripts/visualization/render_physx.py --headless --enable_cameras \
        --task Isaacsimenvs-FoldPoolPhysx-Direct-v0 --checkpoint "$CKPT" --policy_config checkpoints/config.yaml \
        --out "$OUT" --caption "Rigid hinged plates (PhysX)" --subtitle "policy trained on the rigid fold pool" \
        "env.fold_pool.shapes=[box2-mid,box2diag4-mid,box2x8-mid,cross4-mid,corners8-mid,box4-mid,star8-mid]" \
        env.rigid_cloth.randomization.enabled=false env.cloth.keypoint_tolerance=0.04 \
        env.cloth.nan_policy=reset env.termination.success_steps=10 "$@"
    ;;
*) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
