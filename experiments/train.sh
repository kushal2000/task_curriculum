#!/bin/bash
# Train a task: experiments/train.sh --task <gym id> [--experiment <recipe>] [train.py args] [hydra overrides]
#
#   experiments/train.sh --task Isaacsimenvs-RigidCloth-Direct-v0 --experiment rigid_cloth_right_wuji \
#       --checkpoint <prior.pth>
#   RUN_TAG=trkCr experiments/train.sh --task Isaacsimenvs-RigidCloth-Direct-v0 \
#       --experiment experiments/nov15_dr_delay/flat7RW_DR_delay.yaml --checkpoint <prior.pth>
#
# The arguments go to isaacsimenvs/scripts/training/train.py unchanged; this script adds:
#   --agent rl_games_sapg_cfg_entry_point   unless --agent is given
#   --checkpoint_load_mode weights          with --checkpoint, unless given (LOAD_MODE=resume continues a run)
#   agent.params.seed=<INDEX + 1>           unless given: an array's index k trains seed k + 1
#   agent.params.config.name=<name>         unless given: 0_px<kind>_<RUN_TAG>_s<seed>_<job>, kind pool (PhysX
#                                           fold pool), vbd (VBD cloth), play, or 0_pcrl_... for a point-cloud
#                                           actor (recipe or override env.obs.object=pointcloud)
# Environment:
#   RUN_TAG          the run's tag in its name (default run)
#   PRIORS="a b c"   one --checkpoint per array index (index k starts from the k-th), e.g. seed-wise finetunes
#   SMOKE=1          a 5-epoch smoke at 192 envs (overrides the recipe's size)
#   WANDB=0          no wandb (default on, off with SMOKE=1). Project $WANDB_PROJECT (default task_curriculum);
#                    group = the study, i.e. the recipe's folder experiments/<study>/ ($WANDB_GROUP overrides;
#                    "misc" for a recipe outside experiments/ or none); tags kind, recipe, RUN_TAG, seed, job.
#                    The best and last checkpoints are uploaded at the end as artifact ckpt-<name>.
#   VIEWER=0         no interactive HTML rollouts (default: one episode every 500 epochs and at epoch 0, on wandb)
# The header lines ([px_train] / [vbd_train]) are what the run record parses; keep their format.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"
[ -n "$TASK" ] || { echo "[train] pass --task <gym id>" >&2; exit 2; }

ARGS=("$@")
has() { printf '%s\n' "${ARGS[@]}" | grep -qE "$1"; }
SEED="$(( INDEX + 1 ))"
if has '^agent\.params\.seed='; then
    SEED="$(printf '%s\n' "${ARGS[@]}" | sed -n 's/^agent\.params\.seed=//p' | tail -1)"
else
    ARGS+=("agent.params.seed=$SEED")
fi
has '^--agent(=|$)' || ARGS=(--agent rl_games_sapg_cfg_entry_point "${ARGS[@]}")

if [ -n "${PRIORS:-}" ]; then
    read -r -a _PRIORS <<<"$PRIORS"
    PRIOR="${_PRIORS[$INDEX]:-}"
    [ -n "$PRIOR" ] || { echo "[train] PRIORS has no entry for index $INDEX" >&2; exit 2; }
    ARGS+=(--checkpoint "$PRIOR")
fi
PRIOR="$(printf '%s\n' "${ARGS[@]}" | grep -A1 -x -- '--checkpoint' | tail -1)"
[ "$PRIOR" = "--checkpoint" ] && PRIOR=""
if [ -n "$PRIOR" ]; then
    [ -f "$PRIOR" ] || { echo "[train] checkpoint not found: $PRIOR" >&2; exit 2; }
    has '^--checkpoint_load_mode' || ARGS+=(--checkpoint_load_mode "${LOAD_MODE:-weights}")
fi

case "$TASK" in
    Isaacsimenvs-Cloth*)                              KIND=pxvbd ;;
    Isaacsimenvs-RigidCloth*|Isaacsimenvs-FoldPool*)  KIND=pxpool ;;
    Isaacsimenvs-Play*)                               KIND=play ;;
    *)                                                KIND=run ;;
esac
EXPERIMENT="$(printf '%s\n' "${ARGS[@]}" | grep -A1 -x -- '--experiment' | tail -1)"
[ "$EXPERIMENT" = "--experiment" ] && EXPERIMENT=""
if has '^env\.obs\.object=pointcloud$' || { [ -n "$EXPERIMENT" ] && "${PY[@]}" -c \
    "import sys; from isaacsimenvs.tasks.recipes import load_recipe; sys.exit(load_recipe(sys.argv[1])['env'].get('obs', {}).get('object') != 'pointcloud')" \
    "$EXPERIMENT" 2>/dev/null; }; then
    KIND=pcrl
fi
NAME="0_${KIND}_${RUN_TAG:-run}_s${SEED}_${JOB}"
if has '^agent\.params\.config\.name='; then
    NAME="$(printf '%s\n' "${ARGS[@]}" | sed -n 's/^agent\.params\.config\.name=//p' | tail -1)"
else
    ARGS+=("agent.params.config.name=$NAME")
fi

if [ "${SMOKE:-0}" = 1 ]; then
    ARGS+=(env.scene.num_envs=192 agent.params.config.max_epochs=5 agent.params.config.minibatch_size=6144
           agent.params.config.central_value_config.minibatch_size=6144 agent.params.config.expl_coef_block_size=32)
fi
[ "${SMOKE:-0}" = 1 ] && WANDB="${WANDB:-0}"
WANDB="${WANDB:-1}"
if [ "$WANDB" = 1 ]; then
    case "$EXPERIMENT" in
        *experiments/*/*) STUDY="$(basename "$(dirname "$EXPERIMENT")")" ;;
        *)                STUDY=misc ;;
    esac
    RECIPE="$(basename "${EXPERIMENT:-none}" .yaml)"
    ARGS+=(--wandb_activate --wandb_project "${WANDB_PROJECT:-task_curriculum}" --wandb_group "${WANDB_GROUP:-$STUDY}"
           --wandb_name "$NAME" --wandb_tags "$KIND" "recipe:$RECIPE" "tag:${RUN_TAG:-run}" "s$SEED" "job:$JOB")
fi
[ "${VIEWER:-1}" = 1 ] && ! has '^--capture_viewer$' && ARGS+=(--capture_viewer)

# The run's size as it will be used (recipe, then the command line), for the header.
read -r ENVS EPOCHS MINIBATCH DR < <("${PY[@]}" - "$EXPERIMENT" "${ARGS[@]}" <<'PY' 2>/dev/null
import re, sys
from isaacsimenvs.tasks.recipes import load_recipe
exp, args = sys.argv[1], sys.argv[2:]
r = load_recipe(exp) if exp else {"env": {}, "agent": {}, "dr": []}
cfg = r["agent"].get("params", {}).get("config", {})
val = {"envs": r["env"].get("scene", {}).get("num_envs", "?"), "epochs": cfg.get("max_epochs", "?"),
       "minibatch": cfg.get("minibatch_size", "?")}
for key, pat in (("envs", r"env\.scene\.num_envs=(\d+)"), ("epochs", r"agent\.params\.config\.max_epochs=(-?\d+)"),
                 ("minibatch", r"agent\.params\.config\.minibatch_size=(\d+)")):
    for a in args:
        m = re.fullmatch(pat, a)
        if m:
            val[key] = m.group(1)
print(val["envs"], val["epochs"], val["minibatch"], ",".join(r["dr"]) or "none")
PY
)
EXTRA="$(printf '%s ' "${ARGS[@]}" | grep -oE '(env|agent)\.[^ ]+' | tr '\n' ' ')"
if [ "$BACKEND" = newton ]; then
    echo "[vbd_train] name=$NAME task=$TASK seed=$SEED"
    echo "[vbd_train] envs/rank=${ENVS:-?} minibatch=${MINIBATCH:-?} epochs=${EPOCHS:-?} ckpt=${PRIOR:-none}" \
         "load_mode=${LOAD_MODE:-weights} experiment=${EXPERIMENT:-none} extra=${EXTRA:-none}"
else
    echo "[px_train] task=$TASK seed=$SEED envs=${ENVS:-?} epochs=${EPOCHS:-?} experiment=${EXPERIMENT:-none} name=$NAME"
    echo "[px_train] prior=${PRIOR:-none} load_mode=${LOAD_MODE:-weights} wandb=${WANDB:-0} extra=${EXTRA:-none}"
    echo "[px_train] dr=${DR:-none}"
fi
provenance train

mem_log_start "$NAME"
START=$(date +%s)
run_watched "$NAME" "${PY[@]}" -m isaacsimenvs.scripts.training.train "${HEADLESS[@]}" "${ARGS[@]}"
RC=$?
echo "[train] name=$NAME seed=$SEED exit=$RC wall_s=$(( $(date +%s) - START )) $(mem_log_stop)"
exit $RC
