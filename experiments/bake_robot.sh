#!/bin/bash
# Bake a robot URDF into the USD cache (.usd_cache/) that the kit-less Newton runs (VBD cloth) read, through
# a Kit-hosted PhysX cloth task. The cache key is the URDF text + fix_base / self_collision, so pass the
# self-collision value the Newton run uses.
#
#   SC=false experiments/bake_robot.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" --task Isaacsimenvs-RigidCloth-Direct-v0
echo "[bake] self_collision=${SC:-false}"
provenance bake
"${PY[@]}" -m isaacsimenvs.tasks.cloth.backends.vbd.newton_support.usd_cache --populate \
    --task Isaacsimenvs-RigidCloth-Direct-v0 "env.robot_self_collision=${SC:-false}"
echo "[bake] exit=$?"
