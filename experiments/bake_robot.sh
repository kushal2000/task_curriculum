#!/bin/bash
# Bake a robot URDF into the USD cache (.usd_cache/) that the kit-less Newton runs (VBD cloth) read, through
# a Kit-hosted PhysX cloth task with the same robot setup. The cache key is the URDF text + fix_base /
# self_collision, so pass the self-collision value the Newton run uses.
#
#   ROBOT_SETUP=flexiv_wuji_right_hl_reallimits SC=false experiments/bake_robot.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" --task Isaacsimenvs-RigidCloth-Direct-v0
echo "[bake] robot_setup=${ROBOT_SETUP:?set ROBOT_SETUP} self_collision=${SC:-false}"
provenance bake
"${PY[@]}" -m isaacsimenvs.tasks.cloth.backends.vbd.newton_support.usd_cache --populate \
    --task Isaacsimenvs-RigidCloth-Direct-v0 "env.robot_setup=$ROBOT_SETUP" "env.robot_self_collision=${SC:-false}"
echo "[bake] exit=$?"
