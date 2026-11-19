#!/bin/bash
# Live, interactive policy viewer (isaacsimenvs/scripts/visualization/policies/live_viser.py) on a GPU machine.
#
#   experiments/viewer.sh --task Isaacsimenvs-Cloth-Direct-v0 [--num_envs 8] [--port 8080] [live_viser args]
#   experiments/viewer.sh --task Isaacsimenvs-RigidCloth-Direct-v0 --port 8081 [--policy "<name>"]
#
# Open http://<host>:<port> (through an ssh tunnel from a laptop: ssh -N -L <port>:<host>:<port> <login>).
# The process holds the GPU until it is stopped. On Newton the task can be switched from the browser.
# PhysX (.venv_isaacsim): viser needs websockets >= 13 and Isaac Sim ships 12, so a copy of websockets 16
# in .venv_isaacsim/viser_overlay goes first on PYTHONPATH for this process only.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"
ARGS=("$@")
[ -n "$TASK" ] || ARGS=(--task Isaacsimenvs-Cloth-Direct-v0 "${ARGS[@]}")
printf '%s\n' "${ARGS[@]}" | grep -qE '^--num_envs' || ARGS+=(--num_envs 8)
printf '%s\n' "${ARGS[@]}" | grep -qE '^--port' || ARGS+=(--port "$([ "$BACKEND" = physx ] && echo 8081 || echo 8080)")
if [ "$BACKEND" = physx ]; then
    [ -d .venv_isaacsim/viser_overlay/websockets ] || { echo "[viewer] missing .venv_isaacsim/viser_overlay/websockets (copy websockets>=13 there)" >&2; exit 2; }
    export PYTHONPATH="$REPO/.venv_isaacsim/viser_overlay:$PYTHONPATH"
elif [[ "${TASK:-Isaacsimenvs-Cloth}" != Isaacsimenvs-Cloth* ]]; then
    # Rigid manipulands on Newton need the evaluation-sized contact buffer (the VBD cloth keeps its default).
    ARGS+=("env.newton.per_env_triangle_pairs=2097152")
fi
echo "[viewer] host $(hostname -s) ($(hostname -I | awk '{print $1}')) args: ${ARGS[*]}"
provenance viewer
exec "${PY[@]}" -m isaacsimenvs.scripts.visualization.policies.live_viser "${HEADLESS[@]}" "${ARGS[@]}"
