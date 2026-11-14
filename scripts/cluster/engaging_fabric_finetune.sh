#!/bin/bash
# Finetune one PhysX-trained policy on the VBD cloth, one array task per MEASURED fabric, with no
# randomisation at all: the question is whether it can learn the exact case the fabric eval scores.
#
#   sbatch --array=0-6 scripts/cluster/engaging_fabric_finetune.sh          # submit from the repo root
#   FABRICS="cotton" EPOCHS=6 PERIODIC=2 sbatch --array=0 scripts/cluster/engaging_fabric_finetune.sh   # smoke
#
# Per task: Isaacsimenvs-Cloth-Direct-v0 with the preset isaacsimenvs/cfg/fabric/<fabric>_measured.yaml
# (plasticity ON, via fabrics.py --plastic, as bos14_fabric_eval.sh PLAST=1), `--pinned` (DR and reset
# noise off, fixed start: eval/protocol.py disable_randomization), keypoint cutoff 1.5 cm (the fabric
# eval's, and the prior's stage-2 cutoff). 768 envs on one GPU: 768 x 32 = 24,576 env-steps per epoch,
# so the default 2035 epochs is 5e7 env-steps (~5 h at ~3,000 env-steps/s).
#
# Starts from PRIOR with --checkpoint_load_mode weights (fresh optimiser and epoch counter: a finetune,
# not a resume). Preemption: the run name carries the ARRAY job id, which a requeue keeps, so a
# restarted task finds its own newest nn/periodic_*.pth and resumes from it (--checkpoint_load_mode resume).
#
# Outputs: outputs/<date>/<time>/0_ftfab_<fabric>_<arrayjob>/ (nn/ checkpoints, TensorBoard summaries).
# Then score with scripts/cluster/engaging_fabric_eval.sh.
#SBATCH --job-name=ftfab
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES

read -r -a FAB_LIST <<<"${FABRICS:-big_napkin chiffon cotton denim napkin silk towel}"
I="${SLURM_ARRAY_TASK_ID:-0}"
FABRIC="${FAB_LIST[$I]:?no fabric for index $I}"
PRESET="${FABRIC}_measured"
PRIOR_SET="${PRIOR:-}"
PRIOR="${PRIOR:-checkpoints/last_0_pxpool_flat7DRABC_s1_175385_ep_3052_rew__1198.3557_.pth}"
ENVS="${ENVS:-768}"
EPOCHS="${EPOCHS:-2035}"
PERIODIC="${PERIODIC:-100}"
SNAPSHOT="${SNAPSHOT:-250}"
# Steps the fold must hold before the +1000 bonus pays and the episode ends (cloth_task.py _get_dones:
# entered = _fold_hold >= success_steps). 10 = the eval's held-fold rule; 1 = bonus on first entry,
# which on light fabrics (chiffon) trains a fold that springs back before the eval counts it.
SUCCESS_STEPS="${SUCCESS_STEPS:-10}"
# RELEASE=1: cloth.release_persist (cloth_cfg.py) -- the fold only scores once it survives the hand
# leaving; 12 s episodes leave room for the release and the 1 s persist window. Start it from the
# fabric's own ftfab checkpoint (PRIOR=...), which already folds, with RUN_TAG=rel EPOCHS=500.
REL_OV=()
[ "${RELEASE:-0}" = 1 ] && REL_OV=(env.cloth.release_persist=true env.episode_length_s=12.0)
# TRAIN_ARGS: extra Hydra overrides, space-separated (e.g. TRAIN_ARGS="env.cloth.persist_fail_penalty=0.0").
read -r -a EXTRA_OV <<<"${TRAIN_ARGS:-}"
JOB="${RUN_JOB:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"
NAME="0_ftfab${RUN_TAG:-}_${FABRIC}_${JOB}"

read -r -a FAB_OV <<<"$(.venv_isaaclab3/bin/python -I isaacsimenvs/tasks/cloth/utils/fabrics.py "$PRESET" --plastic)"
[ "${#FAB_OV[@]}" -gt 0 ] || { echo "[ftfab] no overrides for $PRESET" >&2; exit 1; }

# THIN=1: the real-thickness presets of docs/results/cloth_thin_final (cloth_sequence_probe.py THIN):
# thickness (particle radius = half), self-contact radius = thickness, self-contact margin THIN_MARGIN (default 3 mm), two-stage
# plasticity (yield, set), full-surface contact, and the stiffer stretch / chiffon bending of that study.
# Starts from the fabric's 16 mm release finetune (THIN_PRIOR) unless PRIOR is given.
if [ "${THIN:-0}" = 1 ]; then
    #                  thickness  yield set  start run (16 mm release finetune)
    declare -A TH=([big_napkin]="0.0003  20 80  0_ftfabrel3sRfsC_big_napkin_28602237"
                   [chiffon]="   0.0001  100 80 0_ftfabrel3sRfsC_chiffon_28607526"
                   [cotton]="    0.0003  20 80  0_ftfabrel3sRfsC2_cotton_28647593"
                   [denim]="     0.00035 100 80 0_ftfabrel3sRfsC_denim_28607527"
                   [napkin]="    0.00015 20 120 0_ftfabrel3sRfsC_napkin_28607529"
                   [silk]="      0.0003  42 120 0_ftfabrel3sRfsC_silk_28607530"
                   [towel]="     0.001   100 80 0_ftfabrel3sRfsC_towel_28607531")
    read -r T Y SET START <<<"${TH[$FABRIC]:?no thin preset for $FABRIC}"
    Y="${THIN_YIELD:-$Y}"; Y="${Y%.0}"
    # The margin caps cloth speed (VBD truncation: 0.5 x 0.42 x margin per 1/1440 s substep): 1.5 x T
    # held thin sheets to 0.05-0.16 m/s (slow spawn drop); 3 mm gives 0.91 m/s as in the 16 mm setup.
    # THIN_MARGIN=1.5x restores the old value.
    MARGIN="${THIN_MARGIN:-0.003}"
    [ "$MARGIN" = 1.5x ] && MARGIN=$(.venv_isaaclab3/bin/python -I -c "print(1.5 * $T)")
    # CONTACT=<m>: the hand/table contact layer (env.cloth.thickness = 2 x particle radius) on its own,
    # while cloth-cloth contact (self_contact_radius) stays at the real thickness T. The contact
    # curriculum (engaging_fabric_curriculum.sh) steps this down; unset = T as well.
    THIN_OV=(env.cloth.thickness=${CONTACT:-$T} env.cloth.self_contact_radius=$T env.cloth.self_contact_margin=$MARGIN
             env.cloth.plastic_yield_curvature=$Y.0 env.cloth.plastic_set_curvature=$SET.0 env.cloth.full_surface_contact=true)
    case $FABRIC in
        chiffon) THIN_OV+=(env.cloth.tri_ke=100.0 env.cloth.tri_ka=100.0 env.cloth.edge_ke=0.0012076974591784867);;
        cotton|silk) THIN_OV+=(env.cloth.tri_ke=100.0 env.cloth.tri_ka=100.0);;
    esac
    # A thin key replaces the preset's (Hydra takes each key once).
    KEEP=()
    for o in "${FAB_OV[@]}"; do
        dup=0; for v in "${THIN_OV[@]}"; do [ "${o%%=*}" = "${v%%=*}" ] && dup=1; done
        [ "$dup" = 0 ] && KEEP+=("$o")
    done
    FAB_OV=("${KEEP[@]}" "${THIN_OV[@]}")
    if [ -z "${PRIOR_SET:-}" ]; then
        PRIOR=$(.venv_isaaclab3/bin/python -I -c "import sys; sys.path.insert(0, 'scripts/analysis'); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path('outputs'), 'last') or '')" "$START")
    fi
fi

# Resume from this run's own newest periodic checkpoint if a previous attempt (preempted) left one.
# Newest by EPOCH, not path: a seeded run (engaging_fabric_curriculum.sh) also has outputs/seed/..., which
# sorts after the dated folders.
LATEST="$(ls outputs/*/*/"$NAME"/nn/periodic_"$NAME"_ep_*.pth 2>/dev/null | awk -F_ep_ '{print $NF"\t"$0}' | sort -n | tail -1 | cut -f2)"
if [ -n "$LATEST" ]; then
    CKPT_ARGS=(--checkpoint "$LATEST" --checkpoint_load_mode resume)
else
    [ -f "$PRIOR" ] || { echo "[ftfab] prior not found: $PRIOR" >&2; exit 1; }
    CKPT_ARGS=(--checkpoint "$PRIOR" --checkpoint_load_mode weights)
fi

echo "[ftfab] run=$NAME fabric=$PRESET (plastic) envs=$ENVS epochs=$EPOCHS env-steps=$((ENVS * 32 * EPOCHS)) success_steps=$SUCCESS_STEPS release=${RELEASE:-0}"
echo "[ftfab] start=${CKPT_ARGS[*]} restart=${SLURM_RESTART_COUNT:-0}"
echo "[ftfab] overrides: ${FAB_OV[*]} ${REL_OV[*]:-} ${EXTRA_OV[*]:-}"
echo "[ftfab] host=$(hostname -s) commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

scripts/newton_py -m isaacsimenvs.train \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    "${CKPT_ARGS[@]}" \
    --single_variant \
    --pinned \
    "env.scene.num_envs=$ENVS" \
    env.cloth.keypoint_tolerance=0.015 \
    env.cloth.nan_policy=reset \
    "env.termination.success_steps=$SUCCESS_STEPS" \
    "${FAB_OV[@]}" ${REL_OV[@]+"${REL_OV[@]}"} ${EXTRA_OV[@]+"${EXTRA_OV[@]}"} \
    agent.params.config.multi_gpu=false \
    "agent.params.config.max_epochs=$EPOCHS" \
    "agent.params.config.periodic_save_every=$PERIODIC" \
    "agent.params.config.snapshot_every=$SNAPSHOT" \
    agent.params.config.best_save_min_interval_s=60 \
    "agent.params.config.name=$NAME" \
    2>&1 | grep --line-buffered -v "Contact reduction hashtable fill ratio exceeded"
# ^ That Newton warning is spurious at 768 envs: narrow_phase.py tests `active*100 >= capacity*80` in
#   int32, and capacity*80 overflows at capacity 33,554,432, so it fires every substep (~190 lines/epoch)
#   at a real fill of ~0.07%. Real losses report as "insert failures" / "overflowed" and still print.
echo "[ftfab] run=$NAME exit=${PIPESTATUS[0]}"
