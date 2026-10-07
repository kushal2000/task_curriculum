#!/bin/bash
# Stiffer MJWarp contacts on the Newton fold pool: what they do to the fold rate and to the speed.
#
#   MODE=eval  sbatch --array=0-35 scripts/cluster/bos14_contact_stiffness_bench.sh
#   MODE=speed sbatch --array=0-2  scripts/cluster/bos14_contact_stiffness_bench.sh
#   .venv_isaaclab3/bin/python scripts/analysis/contact_stiffness_summary.py
#
# Why: at Newton's default contact (ke 2500 / kd 100 = solref 0.02) the 7-27 g plates are soft
# enough that a pressing finger reaches a plate's mid-plane in ~1/3 of env-steps and plates sink up
# to ~17 mm into the table (scripts/analysis/rigid_overlap_probe.py). Configs (rigid_cloth.contact_ke):
#   0 base     ke 2500 (default) kd 100, 2 substeps   solref 0.02
#   1 stiff2   ke 1e4  kd 200,   2 substeps           solref 0.01
#   2 stiff4   ke 4e4  kd 400,   4 substeps           solref 0.005
#
# eval:  index = (policy * 3 + config) * 2 + start. Policies 0-2 = PhysX run 2 flat pool s1-s3
#        (158376), 3-5 = Newton stage-2 box3-prior pool s1-s3 (156746). Same protocol as
#        bos14_fold_pool_eval.sh: 320 envs, 900 steps, fold held 10 steps at 1.5 cm, eval seeds 1, 2;
#        pinned = reset noise and contact-shell DR off, rand = the training regime.
# speed: index = config. 80 training epochs at 1536 envs from the Newton box3-prior s1 weights (a
#        policy that folds, so contacts are at their busiest), rl_games `fps step` read off.

#SBATCH --job-name=cstiff
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
OUTDIR=docs/results/contact_stiffness
mkdir -p slurm_logs "$OUTDIR"
MODE="${MODE:?set MODE=eval or MODE=speed}"
I="${SLURM_ARRAY_TASK_ID:-0}"

CFG_TAGS=(base stiff2 stiff4)
CFG_ARGS=("" "env.rigid_cloth.contact_ke=10000.0 env.rigid_cloth.contact_kd=200.0"
          "env.rigid_cloth.contact_ke=40000.0 env.rigid_cloth.contact_kd=400.0 env.newton.num_substeps=4")
POL_TAGS=(px_flat_s1 px_flat_s2 px_flat_s3 nt_box3_s1 nt_box3_s2 nt_box3_s3)
CKPTS=(outputs/2026-10-06/14-33-34/0_pxpool_one_flat_s1_158376/nn/0_pxpool_one_flat_s1_158376.pth
       outputs/2026-10-06/14-48-53/0_pxpool_one_flat_s2_158376/nn/0_pxpool_one_flat_s2_158376.pth
       outputs/2026-10-06/14-53-54/0_pxpool_one_flat_s3_158376/nn/0_pxpool_one_flat_s3_158376.pth
       outputs/2026-10-04/20-17-57/0_fp_box3_s1_156746/nn/0_fp_box3_s1_156746.pth
       outputs/2026-10-04/20-17-57/0_fp_box3_s2_156746/nn/0_fp_box3_s2_156746.pth
       outputs/2026-10-04/20-17-57/0_fp_box3_s3_156746/nn/0_fp_box3_s3_156746.pth)
export PYTHONUNBUFFERED=1

if [ "$MODE" = eval ]; then
    S=$(( I % 2 )); C=$(( (I / 2) % 3 )); P=$(( I / 6 ))
    read -r -a CA <<<"${CFG_ARGS[$C]}"
    if [ "$S" -eq 1 ]; then START=rand; SARGS=(--randomize)
    else START=pinned; SARGS=(env.rigid_cloth.randomization.enabled=false); fi
    echo "[cstiff] eval ${POL_TAGS[$P]} ${CFG_TAGS[$C]} $START host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
    for ES in 1 2; do
        OUT="$OUTDIR/${POL_TAGS[$P]}__${CFG_TAGS[$C]}__${START}_s${ES}.json"
        [ -s "$OUT" ] && { echo "[cstiff] exists: $OUT"; continue; }
        T0=$(date +%s)
        scripts/newton_py -m isaacsimenvs.eval.episodes \
            --task Isaacsimenvs-FoldPool-Direct-v0 --checkpoint "${CKPTS[$P]}" \
            --policy_config pretrained_policy/config.yaml --sapg_expl_coef 0 \
            --num_envs 320 --max_steps 900 --success_tolerance 0.01 \
            --num_assets_per_type 1 --single_variant "${SARGS[@]}" --seed "$ES" --out "$OUT" \
            env.newton.per_env_triangle_pairs=2097152 env.cloth.nan_policy=reset \
            env.termination.success_steps=10 "${CA[@]}" \
            || echo "[cstiff] $OUT FAILED"
        echo "[cstiff] $OUT wall_s=$(( $(date +%s) - T0 ))"
    done
elif [ "$MODE" = speed ]; then
    C=$I
    read -r -a CA <<<"${CFG_ARGS[$C]}"
    ENVS=1536; EPOCHS="${EPOCHS:-80}"
    NAME="0_cstiff_speed_${CFG_TAGS[$C]}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
    echo "[cstiff] speed ${CFG_TAGS[$C]} envs=$ENVS epochs=$EPOCHS host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
    scripts/newton_py -m isaacsimenvs.train \
        --task Isaacsimenvs-FoldPool-Direct-v0 \
        --agent rl_games_sapg_cfg_entry_point \
        --checkpoint "${CKPTS[3]}" --checkpoint_load_mode weights --single_variant \
        env.cloth.nan_policy=reset "env.scene.num_envs=${ENVS}" \
        agent.params.config.multi_gpu=False \
        agent.params.config.minibatch_size=24576 \
        agent.params.config.central_value_config.minibatch_size=24576 \
        "agent.params.config.expl_coef_block_size=$(( ENVS / 6 ))" \
        "agent.params.config.max_epochs=${EPOCHS}" \
        "agent.params.config.name=${NAME}" \
        agent.params.seed=1 \
        "${CA[@]}"
    echo "[cstiff] speed ${CFG_TAGS[$C]} exit=$?"
else
    echo "[cstiff] unknown MODE=$MODE" >&2; exit 1
fi
