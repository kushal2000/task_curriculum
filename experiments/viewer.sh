#!/bin/bash
# Live policy viewer (isaacsimenvs/scripts/visualization/live_viser/) on a GPU machine: one target per viewer.
#
#   experiments/viewer.sh vbd [--policy <name or .pth>] [--policies <menu.yaml>] [--start pinned|rand] [--port 8080]
#   experiments/viewer.sh rigidbody_flat --policy outputs/distill/<run>/student_last.pth     # port 8081
#   experiments/viewer.sh rigidbody_folded --policy <pth> --experiment <its training recipe>
#
# Targets as experiments/eval.sh's: vbd (Newton, .venv_isaaclab3), rigidbody_flat / rigidbody_folded (PhysX,
# .venv_isaacsim, in the starting policy's training world: run.json / args.json beside it, or --experiment).
# Open http://<host>:<port> (through an ssh tunnel from a laptop: ssh -N -L <port>:<host>:<port> <login>).
# The process holds the GPU until it is stopped.
# PhysX: viser needs websockets >= 13 and Isaac Sim ships 12, so the copy of websockets 16 in
# third_party/viser_websockets (install_isaacsim.sh; older setups: .venv_isaacsim/viser_overlay) goes first on
# PYTHONPATH for this process only.
TARGET="${1:?usage: experiments/viewer.sh <vbd|rigidbody_flat|rigidbody_folded> [live_viser args]}"
shift
case "$TARGET" in
    vbd)         TASK=Isaacsimenvs-Cloth-Direct-v0 ;;
    rigidbody_*) TASK=Isaacsimenvs-RigidCloth-Direct-v0 ;;
    *) echo "[viewer] unknown target $TARGET (vbd, rigidbody_flat, rigidbody_folded)" >&2; exit 2 ;;
esac
ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" --task "$TASK"
printf '%s\n' "${ARGS[@]}" | grep -qE '^--port' || ARGS+=(--port "$([ "$BACKEND" = physx ] && echo 8081 || echo 8080)")
if [ "$BACKEND" = physx ]; then
    WS=""
    for d in third_party/viser_websockets .venv_isaacsim/viser_overlay; do [ -d "$d/websockets" ] && { WS="$REPO/$d"; break; }; done
    [ -n "$WS" ] || { echo "[viewer] no websockets>=13 in third_party/viser_websockets (docs/installation/install_isaacsim.sh)" >&2; exit 2; }
    export PYTHONPATH="$WS:$PYTHONPATH"
fi
echo "[viewer] host $(hostname -s) ($(hostname -I | awk '{print $1}')) target $TARGET args: ${ARGS[*]}"
provenance viewer
exec "${PY[@]}" -m isaacsimenvs.scripts.visualization.live_viser "${HEADLESS[@]}" "$TARGET" "${ARGS[@]}"
