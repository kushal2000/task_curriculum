#!/bin/bash
# Live, interactive policy viewer (isaacsimenvs/scripts/visualization/policies/live_viser.py) on a GPU machine.
#
#   experiments/viewer.sh --task Isaacsimenvs-ClothFlexivWujiRightHumanLike-Direct-v0 [--num_envs 8] [--port 8080] [live_viser args]
#   experiments/viewer.sh --task Isaacsimenvs-RigidCloth-Direct-v0 --port 8081 [--policy "<name>"]
#
# Open http://<host>:<port> (through an ssh tunnel from a laptop: ssh -N -L <port>:<host>:<port> <login>).
# The process holds the GPU until it is stopped. On Newton the task can be switched from the browser.
# PhysX (.venv_isaacsim): viser needs websockets >= 13 and Isaac Sim ships 12, so the copy of websockets 16
# in third_party/viser_websockets (install_isaacsim.sh; older setups: .venv_isaacsim/viser_overlay) goes
# first on PYTHONPATH for this process only.
ARGS=("$@")
printf '%s\n' "${ARGS[@]}" | grep -qE '^--task' || ARGS=(--task Isaacsimenvs-ClothFlexivWujiRightHumanLike-Direct-v0 "${ARGS[@]}")
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "${ARGS[@]}"
printf '%s\n' "${ARGS[@]}" | grep -qE '^--num_envs' || ARGS+=(--num_envs 8)
printf '%s\n' "${ARGS[@]}" | grep -qE '^--port' || ARGS+=(--port "$([ "$BACKEND" = physx ] && echo 8081 || echo 8080)")
if [ "$BACKEND" = physx ]; then
    WS=""
    for d in third_party/viser_websockets .venv_isaacsim/viser_overlay; do [ -d "$d/websockets" ] && { WS="$REPO/$d"; break; }; done
    [ -n "$WS" ] || { echo "[viewer] no websockets>=13 in third_party/viser_websockets (docs/installation/install_isaacsim.sh)" >&2; exit 2; }
    export PYTHONPATH="$WS:$PYTHONPATH"
elif [[ "$TASK" != Isaacsimenvs-Cloth* ]]; then
    # Rigid manipulands on Newton need the evaluation-sized contact buffer (the VBD cloth keeps its default).
    ARGS+=("env.newton.per_env_triangle_pairs=2097152")
fi
echo "[viewer] host $(hostname -s) ($(hostname -I | awk '{print $1}')) args: ${ARGS[*]}"
provenance viewer
exec "${PY[@]}" -m isaacsimenvs.scripts.visualization.policies.live_viser "${HEADLESS[@]}" "${ARGS[@]}"
