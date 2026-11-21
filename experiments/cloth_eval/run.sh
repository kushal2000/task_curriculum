#!/usr/bin/env bash
# Run from the repository root with the cloth-physics changes present.
set -euo pipefail
: "${CKPT:?Set CKPT to a policy checkpoint}"
: "${OUTDIR:?Set OUTDIR to a directory unique to this checkpoint}"
manifest=experiments/cloth_eval/starts-v1.json
fabrics=(denim towel chiffon cotton silk napkin big_napkin)
if [[ -n ${FABRIC:-} ]]; then fabrics=("$FABRIC"); fi
total=${ENVS:-256}
batch=${BATCH_ENVS:-32}
start=${OFFSET:-0}
(( total > 0 && batch > 0 && start >= 0 && start + total <= 256 ))
for fabric in "${fabrics[@]}"; do
    preset=$(experiments/newton_py isaacsimenvs/tasks/cloth/utils/fabrics.py "${fabric}_measured" --plastic)
    read -ra overrides <<< "$preset"
    for (( offset=start; offset<start+total; offset+=batch )); do
        count=$(( start + total - offset ))
        if (( count > batch )); then count=$batch; fi
        experiments/newton_py -m isaacsimenvs.scripts.evaluation.eval \
        --task Isaacsimenvs-Cloth-Direct-v0 --checkpoint "$CKPT" \
        --policy_config "${POLICY_CONFIG:-checkpoints/config.yaml}" --sapg_expl_coef 0 \
        --num_envs "$count" --seed 1 --cloth-standard "$manifest" \
        --cloth-start-offset "$offset" --cloth-fabric "$fabric" \
        --out "$OUTDIR/$fabric/$offset.json" env.cloth.start_height=0.20 \
        env.robot_self_collision=false "${overrides[@]}" "$@"
    done
done
