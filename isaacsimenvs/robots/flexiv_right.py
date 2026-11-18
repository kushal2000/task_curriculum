"""Flexiv Rizon 4s + RIGHT Wuji Hand 2, as the real robot is set up: human-like limits, the
identified hand (sysid), and the drive variants trained on it."""

from __future__ import annotations

import dataclasses

from .flexiv_left import FLEXIV_WUJI, FLEXIV_WUJI_HUMAN_LIKE_SHARPAGAIN, _RIZON_JOINTS, _SHARPA_URDF_EFFORT, _WUJI_JOINTS, _WUJI_S2R_DRIVE, _WUJI_SELF_COLLISION_FILTER, _WUJI_TO_SHARPA
from .kuka_sharpa import KUKA_SHARPA
from .spec import RobotSpec

# ----------------------------------------------------------------------------
# Flexiv Rizon 4s + RIGHT Wuji Hand 2, human-like limits, as the real robot is set up: the
# robot of kushal/2026-10-30_pretraining_real (cdf26eb, Isaacsimenvs-PlayFlexivWujiRightHumanLike):
# the human-like hand mirrored (r_ joints), mounted 180 deg about the flange axis so the palm faces
# the table at the same arm pose; Wuji's HAND2_DRIVE on the hand (_WUJI_S2R_DRIVE, no joint
# friction, 100 rad/s); the real robot's home pose; per-finger pad offsets (thumb 22.9 mm, the same
# 77% of its longer distal link). Where the table sits relative to it is the env's
# ``robot_root_pos`` (configs/robot/flexiv_wuji_right_human_like.yaml).
# ----------------------------------------------------------------------------

def _right(name: str) -> str:
    return "r_" + name[2:] if name.startswith("l_") else name


_WUJI_RIGHT_JOINTS = tuple(_right(n) for n in _WUJI_JOINTS)

FLEXIV_WUJI_RIGHT_HUMAN_LIKE = RobotSpec(
    name="flexiv_wuji_right_human_like",
    arm_joint_regex=FLEXIV_WUJI.arm_joint_regex,
    hand_joint_regex="r_.*",
    joint_names_canonical=_RIZON_JOINTS + _WUJI_RIGHT_JOINTS,
    # The importer merges flange, r_mount and r_wrist into link7.
    palm_body_name="link7",
    # FLEXIV_WUJI's palm centre mirrored across the hand (the branch's value).
    palm_center_offset=(0.0018, -0.0149, 0.1948),
    fingertip_body_regex="r_(thumb|index_finger|middle_finger|ring_finger|pinky)_distal",
    # PhysX find_bodies order, as on the left hand (smoke-checked on this URDF).
    fingertip_link_names=tuple(_right(n) for n in FLEXIV_WUJI.fingertip_link_names),
    # index, middle, pinky, ring: 19 mm; thumb (last): 22.9 mm.
    fingertip_offset=((0.0, 0.0, -0.019),) * 4 + ((0.0, 0.0, -0.0229),),
    arm_stiffness=FLEXIV_WUJI.arm_stiffness,
    arm_damping=FLEXIV_WUJI.arm_damping,
    hand_stiffness={j: d[0] for j, d in zip(_WUJI_RIGHT_JOINTS, _WUJI_S2R_DRIVE)},
    hand_damping={j: d[1] for j, d in zip(_WUJI_RIGHT_JOINTS, _WUJI_S2R_DRIVE)},
    hand_effort_limit={j: d[2] for j, d in zip(_WUJI_RIGHT_JOINTS, _WUJI_S2R_DRIVE)},
    hand_armature={_right(k): v for k, v in FLEXIV_WUJI.hand_armature.items()},
    hand_velocity_limit=100.0,
    hand_friction=0.0,
    # The real robot's home pose, (-120, 0, 30, 100, 150, 80, 120) deg: palm to the table.
    arm_default_joint_pos={
        "joint1": -2.0944, "joint2": 0.0, "joint3": 0.5236, "joint4": 1.7453,
        "joint5": 2.618, "joint6": 1.3963, "joint7": 2.0944,
    },
    # The left's pairs, renamed (the branch's self_collision_pairs.py derives exactly these).
    self_collision_filter={
        _right(a): tuple(_right(b) for b in bs) for a, bs in _WUJI_SELF_COLLISION_FILTER.items()
    },
    finger_segment_regex={k: "r_" + v[2:] for k, v in FLEXIV_WUJI.finger_segment_regex.items()},
    palm_side_link_names=(),
)

# ----------------------------------------------------------------------------
# FLEXIV_WUJI_RIGHT_HUMAN_LIKE with the real hand's identified joint dynamics and the real robot's
# limits (the "real" variant; configs/robot/flexiv_wuji_right_real.yaml):
#   * hand armature, joint friction and passive damping from the sysid of the deployed right hand
#     (firmware 2.10.0; aiowuji sysid_fit.py MuJoCo fit of the 2026-09-30 fingers-up and 2026-10-09
#     palm-down runs, assets/flexiv_wuji/wuji_hand2_right_sysid_fit_20261009.json, kept out of git).
#     MuJoCo `frictionloss` is a Coulomb torque [N m]; Isaac Sim >= 5.0 joint friction is an effort
#     too, so it is used as both the static and the dynamic friction. MuJoCo joint `damping` is
#     passive, so it goes in as viscous friction and the PD damping stays HAND2_DRIVE's (the deploy
#     converts that kd to A s/rad). The fit's passive springs are applied as a joint effort
#     outside the PD (_WUJI_RIGHT_SPRING, RobotSpec.hand_spring_stiffness).
#   * hand torque limit = the deploy's 1.0 A current limit x each joint's identified torque constant
#     (the fit's `gear`, N m/A; deploy config.yaml hand_kt). The firmware multiplies the PD output by
#     ~1.97 BEFORE clipping the current, so that factor changes the gain conversion, not the limit.
#   * arm limits 5 deg inside the spec on both sides and finger PIP <= 100 deg: in the URDF
#     (rizon4s_right_wuji_real_limits.urdf), not here.
# The PD gains, geometry, home pose and collision filter are FLEXIV_WUJI_RIGHT_HUMAN_LIKE's.
# ----------------------------------------------------------------------------

#: Per hand joint: (torque constant N m/A, armature kg m^2, passive damping N m s/rad, Coulomb
#: friction N m), the sysid fit rounded to 4 significant figures (tests/test_robot_specs.py checks
#: it against the JSON).
_WUJI_RIGHT_SYSID = {
    "r_thumb_cmc_flex": (0.439, 0.0001347, 4.39e-06, 0.05382),  # thumb_S1
    "r_thumb_cmc_abd": (0.2042, 6.801e-05, 8.075e-06, 0.04521),  # thumb_S2
    "r_thumb_mcp": (0.2546, 7.132e-05, 2.546e-06, 0.02253),  # thumb_S3
    "r_thumb_ip": (0.2219, 0.0002385, 2.359e-06, 0.05255),  # thumb_S4
    "r_index_finger_mcp_flex": (0.5986, 0.0003551, 1.064e-05, 0.0783),  # index_S1
    "r_index_finger_mcp_abd": (0.2324, 0.0001063, 0.003128, 0.02839),  # index_S2
    "r_index_finger_pip": (0.1795, 2.622e-05, 2.901e-06, 0.01979),  # index_S3
    "r_index_finger_dip": (0.135, 4.875e-05, 4.08e-06, 0.01011),  # index_S4
    "r_middle_finger_mcp_flex": (0.6611, 0.0004131, 6.611e-06, 0.07499),  # middle_S1
    "r_middle_finger_mcp_abd": (0.2009, 6.992e-05, 0.002858, 0.02198),  # middle_S2
    "r_middle_finger_pip": (0.1636, 1.913e-05, 2.555e-06, 0.01594),  # middle_S3
    "r_middle_finger_dip": (0.1138, 2.834e-05, 0.001117, 0.00786),  # middle_S4
    "r_ring_finger_mcp_flex": (0.5567, 0.0002835, 5.886e-06, 0.07003),  # ring_S1
    "r_ring_finger_mcp_abd": (0.2053, 4.993e-05, 0.001846, 0.01984),  # ring_S2
    "r_ring_finger_pip": (0.1686, 3.891e-05, 0.001516, 0.01526),  # ring_S3
    "r_ring_finger_dip": (0.1635, 4.443e-05, 0.0004419, 0.0138),  # ring_S4
    "r_pinky_mcp_flex": (0.4777, 0.0001032, 0.0008498, 0.03306),  # pinky_S1
    "r_pinky_mcp_abd": (0.109, 7.36e-05, 0.004938, 0.01344),  # pinky_S2
    "r_pinky_pip": (0.1002, 2.487e-05, 0.000616, 0.0146),  # pinky_S3
    "r_pinky_dip": (0.157, 5.652e-05, 0.004776, 0.01838),  # pinky_S4
}
#: Per hand joint: the sysid fit's passive spring (stiffness N m/rad, rest angle rad), 4 significant
#: figures. Replayed in PhysX (docs/results/flexiv_right_sysid_replay, 11-10) it halves the tracking
#: error against the real trials: 23.6 -> 14.3 mrad at dt 1/120 (12.0 at 1/1000, = the MuJoCo fit).
_WUJI_RIGHT_SPRING = {
    "r_thumb_cmc_flex": (0.0843, 0.0619),  # thumb_S1
    "r_thumb_cmc_abd": (0.06436, 0.2691),  # thumb_S2
    "r_thumb_mcp": (0.02401, 0.2373),  # thumb_S3
    "r_thumb_ip": (0.0197, 0.2892),  # thumb_S4
    "r_index_finger_mcp_flex": (0.1053, 0.2061),  # index_S1
    "r_index_finger_mcp_abd": (0.05887, 0.0683),  # index_S2
    "r_index_finger_pip": (0.01922, 0.002333),  # index_S3
    "r_index_finger_dip": (0.0119, 0.1165),  # index_S4
    "r_middle_finger_mcp_flex": (0.1447, 0.2237),  # middle_S1
    "r_middle_finger_mcp_abd": (0.05022, -0.0176),  # middle_S2
    "r_middle_finger_pip": (0.01633, 0.01584),  # middle_S3
    "r_middle_finger_dip": (0.01095, 0.1609),  # middle_S4
    "r_ring_finger_mcp_flex": (0.1296, 0.05509),  # ring_S1
    "r_ring_finger_mcp_abd": (0.03698, 0.06116),  # ring_S2
    "r_ring_finger_pip": (0.01341, 0.07124),  # ring_S3
    "r_ring_finger_dip": (0.01487, 0.09207),  # ring_S4
    "r_pinky_mcp_flex": (0.1199, 0.1068),  # pinky_S1
    "r_pinky_mcp_abd": (0.04985, -0.01235),  # pinky_S2
    "r_pinky_pip": (0.01597, -0.0003289),  # pinky_S3
    "r_pinky_dip": (0.0224, 0.062),  # pinky_S4
}
#: Coulomb-friction scale per joint type, on the fit's frictionloss (static and dynamic alike), for
#: PhysX at dt 1/120 WITH the spring: the light PIP/DIP stick more at 1/120 than at 1/1000, which this
#: undoes. The replay's "Rs" variant: 13.3 mrad over all trials vs 14.3 at x1 (S3/S4 11.6/12.4 ->
#: 6.6/7.4 mrad). Finger S1 (MCP flex) x1, S2 (abduction) / S3 (PIP) x0.8, S4 (DIP) x0.6; thumb
#: S1-S3 x0.8, S4 (IP) x1.
_WUJI_FRICTION_SCALE_FINGER = (1.0, 0.8, 0.8, 0.6)
_WUJI_FRICTION_SCALE_THUMB = (0.8, 0.8, 0.8, 1.0)
_WUJI_RIGHT_FRICTION_SCALE = {
    j: (_WUJI_FRICTION_SCALE_THUMB if k < 4 else _WUJI_FRICTION_SCALE_FINGER)[k % 4]
    for k, j in enumerate(_WUJI_RIGHT_SYSID)  # thumb S1..S4, then index, middle, ring, pinky
}
#: The deploy's hand current limit [A] (deploy/simtoolreal/config.yaml hand_effort_limit).
WUJI_HAND_CURRENT_LIMIT_A = 1.0

FLEXIV_WUJI_RIGHT_REAL = dataclasses.replace(
    FLEXIV_WUJI_RIGHT_HUMAN_LIKE,
    name="flexiv_wuji_right_real",
    hand_armature={j: v[1] for j, v in _WUJI_RIGHT_SYSID.items()},
    hand_effort_limit={j: WUJI_HAND_CURRENT_LIMIT_A * v[0] for j, v in _WUJI_RIGHT_SYSID.items()},
    hand_friction={j: v[3] * _WUJI_RIGHT_FRICTION_SCALE[j] for j, v in _WUJI_RIGHT_SYSID.items()},
    hand_dynamic_friction={j: v[3] * _WUJI_RIGHT_FRICTION_SCALE[j] for j, v in _WUJI_RIGHT_SYSID.items()},
    hand_viscous_friction={j: v[2] for j, v in _WUJI_RIGHT_SYSID.items()},
    hand_spring_stiffness={j: v[0] for j, v in _WUJI_RIGHT_SPRING.items()},
    hand_spring_ref={j: v[1] for j, v in _WUJI_RIGHT_SPRING.items()},
)

# Right-hand drive variants for the 11-11 training set (stage 12). Both keep FLEXIV_WUJI_RIGHT_HUMAN_LIKE's
# armature, no joint friction and no spring; their overlays use the real-limits URDF.
#   * _S2R_EFF: HAND2_DRIVE kp/kd (as RIGHT_HUMAN_LIKE) with the real hand's torque limit, 1.0 A x Kt
#     (FLEXIV_WUJI_RIGHT_REAL's), instead of HAND2_DRIVE's limits.
#   * _SHARPAGAIN: the Sharpa's kp / kd / URDF torque limit by joint role, as
#     FLEXIV_WUJI_HUMAN_LIKE_SHARPAGAIN does for the left hand.
FLEXIV_WUJI_RIGHT_S2R_EFF = dataclasses.replace(
    FLEXIV_WUJI_RIGHT_HUMAN_LIKE,
    name="flexiv_wuji_right_s2r_eff",
    hand_effort_limit=dict(FLEXIV_WUJI_RIGHT_REAL.hand_effort_limit),
)
FLEXIV_WUJI_RIGHT_SHARPAGAIN = dataclasses.replace(
    FLEXIV_WUJI_RIGHT_HUMAN_LIKE,
    name="flexiv_wuji_right_sharpagain",
    hand_stiffness={_right(w): KUKA_SHARPA.hand_stiffness[_WUJI_TO_SHARPA[w]] for w in _WUJI_JOINTS},
    hand_damping={_right(w): KUKA_SHARPA.hand_damping[_WUJI_TO_SHARPA[w]] for w in _WUJI_JOINTS},
    hand_effort_limit={_right(w): _SHARPA_URDF_EFFORT[_WUJI_TO_SHARPA[w]] for w in _WUJI_JOINTS},
)
