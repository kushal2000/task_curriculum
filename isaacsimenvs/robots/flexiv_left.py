"""Flexiv Rizon 4s with the LEFT hands: the left Wuji Hand 2 (vendor and human-like limits, S2R
drive), the Rizon carrying the Kuka's Sharpa, and the drive-swap controls. The original setup, kept
for its checkpoints; the right-hand robot (``flexiv_right.py``) is built from these tables."""

from __future__ import annotations

import dataclasses
import math

from .kuka_sharpa import KUKA_SHARPA, _SHARPA_SELF_COLLISION_FILTER
from .spec import RobotSpec

# ----------------------------------------------------------------------------
# Flexiv Rizon 4s + left Wuji Hand 2 (scripts/flexiv_wuji/). Geometry is derived from the
# models; the DYNAMICS ARE PROVISIONAL placeholders, to be refined against the hardware.
# ----------------------------------------------------------------------------

_RIZON_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
_RIZON_LINKS = ("base_link", *(f"link{i}" for i in range(1, 8)))
#: Rizon 4s joint torque limits (flexiv_description joint_limits.yaml), Nm.
_RIZON_EFFORT = (123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0)
_KUKA_ARM_K = tuple(KUKA_SHARPA.arm_stiffness.values())
_KUKA_ARM_D = tuple(KUKA_SHARPA.arm_damping.values())
#: Provisional: stiffness 2 Nm/rad per Nm of torque limit, so every joint saturates at the same
#: 0.5 rad of tracking error. Damping keeps the Kuka's per-joint damping ratio at equal link
#: inertia (d scales with sqrt(k)), which over-damps a little since the Rizon's links are lighter.
_RIZON_K = tuple(2.0 * e for e in _RIZON_EFFORT)
_RIZON_D = tuple(d * math.sqrt(k / kk) for d, k, kk in zip(_KUKA_ARM_D, _RIZON_K, _KUKA_ARM_K))

#: Wuji Hand 2 joints, thumb to pinky, 4 per finger (vendor URDF order).
_WUJI_JOINTS = tuple(
    f"l_{finger}_{joint}"
    for finger, joints in (
        ("thumb", ("cmc_flex", "cmc_abd", "mcp", "ip")),
        ("index_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("middle_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("ring_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("pinky", ("mcp_flex", "mcp_abd", "pip", "dip")),
    )
    for joint in joints
)
#: Provisional: the vendor MJCF's position-servo gains (wuji-description hand2_beta2
#: mjcf/left.xml, <position kp kv>), in _WUJI_JOINTS order. These are the vendor's own model of
#: the real hand and much softer than the Sharpa's (0.18-0.69 against 0.9-13.2 Nm/rad).
_WUJI_KP = (
    0.40844710645048043, 0.6858601063643346, 0.2391112196099482, 0.20736128319711747,
    0.37352218155860073, 0.45592448909027794, 0.24368366649522863, 0.18026971340925335,
    0.3687093483646485, 0.4164253443634641, 0.22218607502059182, 0.19427606072023446,
    0.35718151495111794, 0.42977315313086895, 0.24930151196247122, 0.2285032688178066,
    0.3655325975433942, 0.41393113081120425, 0.22729367621965954, 0.1964723550341816,
)
_WUJI_KV = (
    0.020882010257063675, 0.030610939996373314, 0.010181475050560962, 0.00909698844675045,
    0.01882274330029718, 0.019798167597016643, 0.010477031953162727, 0.008240212147903584,
    0.01848622487024593, 0.018032947229953678, 0.009592200014076666, 0.009152994605972402,
    0.018376606800780005, 0.01867700966212433, 0.01059512121009555, 0.009917602877441107,
    0.018616960278988272, 0.018732177029667153, 0.00951005441616486, 0.009017295241756363,
)
#: Vendor MJCF armature: 0.0005 on the two thumb CMC joints and each finger's MCP flexion,
#: the 0.0002 default elsewhere.
_WUJI_ARMATURE = tuple(
    0.0005 if name.endswith(("cmc_flex", "cmc_abd", "mcp_flex")) else 0.0002
    for name in _WUJI_JOINTS
)

# Derived by scripts/flexiv_wuji/self_collision_pairs.py (tests/test_robot_specs.py reruns it):
# each jointed pair, plus the palm (link7, into which l_wrist merges) with every finger's
# proximal_abd -- touching at reset on the ring and pinky, and through the knuckle's range on
# the rest, as SimToolReal filters the Sharpa's palm-to-proximal pairs.
_WUJI_SELF_COLLISION_FILTER = {
        'base_link': ('link1',),
        'l_index_finger_distal': ('l_index_finger_middle',),
        'l_index_finger_middle': ('l_index_finger_distal', 'l_index_finger_proximal_abd'),
        'l_index_finger_proximal': ('l_index_finger_proximal_abd', 'link7'),
        'l_index_finger_proximal_abd': ('l_index_finger_middle', 'l_index_finger_proximal', 'link7'),
        'l_middle_finger_distal': ('l_middle_finger_middle',),
        'l_middle_finger_middle': ('l_middle_finger_distal', 'l_middle_finger_proximal_abd'),
        'l_middle_finger_proximal': ('l_middle_finger_proximal_abd', 'link7'),
        'l_middle_finger_proximal_abd': ('l_middle_finger_middle', 'l_middle_finger_proximal', 'link7'),
        'l_pinky_distal': ('l_pinky_middle',),
        'l_pinky_middle': ('l_pinky_distal', 'l_pinky_proximal_abd'),
        'l_pinky_proximal': ('l_pinky_proximal_abd', 'link7'),
        'l_pinky_proximal_abd': ('l_pinky_middle', 'l_pinky_proximal', 'link7'),
        'l_ring_finger_distal': ('l_ring_finger_middle',),
        'l_ring_finger_middle': ('l_ring_finger_distal', 'l_ring_finger_proximal_abd'),
        'l_ring_finger_proximal': ('l_ring_finger_proximal_abd', 'link7'),
        'l_ring_finger_proximal_abd': ('l_ring_finger_middle', 'l_ring_finger_proximal', 'link7'),
        'l_thumb_distal': ('l_thumb_middle',),
        'l_thumb_middle': ('l_thumb_distal', 'l_thumb_proximal_abd'),
        'l_thumb_proximal': ('l_thumb_proximal_abd', 'link7'),
        'l_thumb_proximal_abd': ('l_thumb_middle', 'l_thumb_proximal', 'link7'),
        'link1': ('base_link', 'link2'),
        'link2': ('link1', 'link3'),
        'link3': ('link2', 'link4'),
        'link4': ('link3', 'link5'),
        'link5': ('link4', 'link6'),
        'link6': ('link5', 'link7'),
        'link7': ('l_index_finger_proximal', 'l_index_finger_proximal_abd', 'l_middle_finger_proximal', 'l_middle_finger_proximal_abd', 'l_pinky_proximal', 'l_pinky_proximal_abd', 'l_ring_finger_proximal', 'l_ring_finger_proximal_abd', 'l_thumb_proximal', 'l_thumb_proximal_abd', 'link6'),
}

FLEXIV_WUJI = RobotSpec(
    name="flexiv_wuji",
    arm_joint_regex="joint[1-7]",
    hand_joint_regex="l_.*",
    joint_names_canonical=_RIZON_JOINTS + _WUJI_JOINTS,
    # The importer merges flange, l_mount and l_wrist into link7.
    palm_body_name="link7",
    # The Kuka's palm centre carried over through the canonical hand frame (middle MCP origin,
    # knuckle-defined axes; isaacsimenvs/scripts/visualization/robots/viser_compare.py): same spot relative to the
    # knuckles. Depends on flexiv_wuji.MOUNT_*; tests/test_robot_specs.py recomputes it.
    palm_center_offset=(-0.0018, -0.015, 0.1948),
    fingertip_body_regex="l_(thumb|index_finger|middle_finger|ring_finger|pinky)_distal",
    # The order the policy observes them in. PhysX's find_bodies(fingertip_body_regex) returns body
    # index order, which for this hand is index, middle, pinky, ring, thumb (not the Kuka's index,
    # middle, ring, thumb, pinky); the Play prior was trained on it. The cloth env resolves its
    # fingertips by this tuple on both backends (cloth/mdp/robot.py), so it must list the PhysX order.
    fingertip_link_names=(
        "l_index_finger_distal", "l_middle_finger_distal", "l_pinky_distal",
        "l_ring_finger_distal", "l_thumb_distal",
    ),
    # The tip is 24.8 mm down each distal link's -z (29.8 on the thumb); the Sharpa's offset sits
    # ~77% of the way to its tip, and this is the same fraction.
    fingertip_offset=(0.0, 0.0, -0.019),
    arm_stiffness=dict(zip(_RIZON_JOINTS, _RIZON_K)),
    arm_damping=dict(zip(_RIZON_JOINTS, _RIZON_D)),
    hand_stiffness=dict(zip(_WUJI_JOINTS, _WUJI_KP)),
    hand_damping=dict(zip(_WUJI_JOINTS, _WUJI_KV)),
    hand_armature=dict(zip(_WUJI_JOINTS, _WUJI_ARMATURE)),
    # viser_compare.py's IK match of the Kuka's default pose (flexiv_wuji.FLEXIV_DEFAULT_ARM_POS).
    arm_default_joint_pos={
        "joint1": -1.7604, "joint2": -0.2933, "joint3": 0.0197, "joint4": 1.4941,
        "joint5": -1.6591, "joint6": 1.6462, "joint7": 0.0369,
    },
    self_collision_filter=_WUJI_SELF_COLLISION_FILTER,
    # `l_*_tip` and `l_*_tip_sensor_frame` merge into the distal link; both proximal links carry
    # collision (the flexion one, then the abduction one).
    finger_segment_regex={
        "distal": "l_(thumb|index_finger|middle_finger|ring_finger|pinky)_distal",
        "middle": "l_(thumb|index_finger|middle_finger|ring_finger|pinky)_middle",
        "proximal": "l_(thumb|index_finger|middle_finger|ring_finger|pinky)_proximal(_abd)?",
    },
    # `l_mount` and `l_wrist` merge into link7, the palm body.
    palm_side_link_names=(),
)

# The same robot with scripts/flexiv_wuji/flexiv_wuji.py's HUMAN_LIKE_LIMITS_DEG: finger and thumb
# joints' backward bend cut to rough human values, finger abduction held to +-2 deg. Joints, gains,
# geometry and default pose are FLEXIV_WUJI's; the limits live in its URDF. The self-collision
# filter is re-derived on that URDF (self_collision_pairs.py), since the knuckle sweep depends on
# the joint ranges.
FLEXIV_WUJI_HUMAN_LIKE = dataclasses.replace(
    FLEXIV_WUJI,
    name="flexiv_wuji_human_like",
    self_collision_filter=_WUJI_SELF_COLLISION_FILTER,
)

#: Wuji's own sim2real drive for Hand 2 (wuji-isaaclab tasks/reorient/config/wuji_hand2/
#: hand2_constants.py HAND2_DRIVE, identical to wuji-mjlab's HAND2_REORIENT_ACTUATOR_GAINS /
#: FORCE_LIMITS): (stiffness N m/rad, damping N m s/rad, effort limit N m) per joint, thumb to
#: pinky in _WUJI_JOINTS order. Given for the right hand; the left mirrors it joint for joint.
#: Kp 1.95-3.79x (mean 3.2x) the MJCF servo's, with the MCP flexion torque at ~1 N m instead of 2.
_WUJI_S2R_DRIVE = (
    (1.354985, 0.04769, 1.02), (2.198821, 0.062838, 0.83), (0.886467, 0.027982, 0.35), (0.654026, 0.020457, 0.21),
    (1.345207, 0.047402, 1.05), (0.88946, 0.027239, 0.38), (0.779361, 0.024489, 0.31), (0.656424, 0.020508, 0.30),
    (1.299313, 0.045699, 1.06), (0.942641, 0.027846, 0.39), (0.73725, 0.023408, 0.32), (0.710314, 0.022366, 0.29),
    (1.276892, 0.046209, 1.02), (1.013334, 0.028704, 0.41), (0.859159, 0.027436, 0.28), (0.648064, 0.020448, 0.30),
    (1.102915, 0.043816, 0.87), (0.910201, 0.026593, 0.36), (0.861968, 0.027476, 0.28), (0.656704, 0.020611, 0.30),
)

# FLEXIV_WUJI_HUMAN_LIKE with that drive on the hand (armature is the same 0.0005 / 0.0002 split).
FLEXIV_WUJI_HUMAN_LIKE_S2R = dataclasses.replace(
    FLEXIV_WUJI_HUMAN_LIKE,
    name="flexiv_wuji_human_like_s2r",
    hand_stiffness={j: d[0] for j, d in zip(_WUJI_JOINTS, _WUJI_S2R_DRIVE)},
    hand_damping={j: d[1] for j, d in zip(_WUJI_JOINTS, _WUJI_S2R_DRIVE)},
    hand_effort_limit={j: d[2] for j, d in zip(_WUJI_JOINTS, _WUJI_S2R_DRIVE)},
)

# ----------------------------------------------------------------------------
# Flexiv Rizon 4s + the Kuka's left Sharpa (scripts/flexiv_wuji/flexiv_wuji.py
# compose_flexiv_sharpa_urdf): the Rizon of FLEXIV_WUJI, at its default pose, carrying the hand of
# KUKA_SHARPA -- arm entries from the one, hand entries from the other.
# ----------------------------------------------------------------------------


def _rename(pairs: dict[str, tuple[str, ...]], names: dict[str, str]) -> dict[str, tuple[str, ...]]:
    return {names.get(a, a): tuple(names.get(b, b) for b in bs) for a, bs in pairs.items()}


_IIWA_TO_RIZON = {"iiwa14_link_7": "link7"}
_FLEXIV_SHARPA_SELF_COLLISION_FILTER = {
    # The Rizon's chain, as FLEXIV_WUJI filters it ...
    **{a: tuple(b for b in bs if not b.startswith("l_"))
       for a, bs in _WUJI_SELF_COLLISION_FILTER.items() if a in _RIZON_LINKS},
}
# ... and SimToolReal's Sharpa pairs, with the palm in the Rizon's link7 instead of the iiwa's.
for _a, _bs in _rename(_SHARPA_SELF_COLLISION_FILTER, _IIWA_TO_RIZON).items():
    if _a.startswith("iiwa14_"):
        continue
    _kept = tuple(b for b in _bs if not b.startswith("iiwa14_"))
    _FLEXIV_SHARPA_SELF_COLLISION_FILTER[_a] = tuple(
        dict.fromkeys(_FLEXIV_SHARPA_SELF_COLLISION_FILTER.get(_a, ()) + _kept))
del _a, _bs, _kept

FLEXIV_SHARPA = RobotSpec(
    name="flexiv_sharpa",
    arm_joint_regex=FLEXIV_WUJI.arm_joint_regex,
    hand_joint_regex=KUKA_SHARPA.hand_joint_regex,
    joint_names_canonical=_RIZON_JOINTS + KUKA_SHARPA.hand_joint_names,
    # The importer merges flange, sharpa_mount and left_hand_C_MC into link7.
    palm_body_name="link7",
    # The Kuka's palm centre, carried rigidly with the hand (the same hand, so exact). Depends on
    # flexiv_wuji.FLEXIV_SHARPA_MOUNT_*; tests/test_robot_specs.py recomputes it.
    palm_center_offset=(-0.0049, -0.0194, 0.239),
    fingertip_body_regex=KUKA_SHARPA.fingertip_body_regex,
    fingertip_link_names=KUKA_SHARPA.fingertip_link_names,
    fingertip_offset=KUKA_SHARPA.fingertip_offset,
    arm_stiffness=FLEXIV_WUJI.arm_stiffness,
    arm_damping=FLEXIV_WUJI.arm_damping,
    hand_stiffness=KUKA_SHARPA.hand_stiffness,
    hand_damping=KUKA_SHARPA.hand_damping,
    hand_armature=KUKA_SHARPA.hand_armature,
    # The Flexiv keeps one default pose whatever hand it carries; the mount yaw puts the palm down.
    arm_default_joint_pos=FLEXIV_WUJI.arm_default_joint_pos,
    self_collision_filter=_FLEXIV_SHARPA_SELF_COLLISION_FILTER,
    finger_segment_regex=KUKA_SHARPA.finger_segment_regex,
    palm_side_link_names=KUKA_SHARPA.palm_side_link_names,
)

# ----------------------------------------------------------------------------
# Drive-swap controls: each hand with the other hand's drive (stiffness, damping and torque limit;
# armature stays the hand's own, a motor property). Joints are matched by role; the Sharpa's two
# joints without a Wuji counterpart take the weakest analogous Wuji joint: thumb MCP_AA <- thumb
# mcp, pinky CMC <- pinky mcp_abd. Torque limits are the URDF ``effort`` of the source hand.
# ----------------------------------------------------------------------------

#: Wuji URDF effort [N m], _WUJI_JOINTS order (rizon4s_left_wuji_human_like_joint_limits.urdf).
_WUJI_URDF_EFFORT = dict(zip(_WUJI_JOINTS, (
    0.6, 0.6, 0.3, 0.3,
    2.0, 0.2, 0.3, 0.3,
    2.0, 0.2, 0.3, 0.3,
    2.0, 0.2, 0.3, 0.3,
    2.0, 0.2, 0.3, 0.3,
)))
#: Sharpa URDF effort [N m] (rizon4s_left_sharpa.urdf).
_SHARPA_URDF_EFFORT = {
    "left_1_thumb_CMC_FE": 3.3, "left_thumb_CMC_AA": 3.3, "left_thumb_MCP_FE": 1.864,
    "left_thumb_MCP_AA": 1.864, "left_thumb_IP": 0.638,
    **{f"left_{n}{f}_MCP_FE": 1.864 for n, f in (("2_", "index"), ("3_", "middle"), ("4_", "ring"))},
    **{f"left_{f}_{j}": e for f in ("index", "middle", "ring", "pinky")
       for j, e in (("MCP_AA", 1.864), ("PIP", 0.638), ("DIP", 0.189369))},
    "left_5_pinky_CMC": 0.5285, "left_pinky_MCP_FE": 1.864,
}
#: Sharpa joint -> the Wuji joint of the same role.
_SHARPA_TO_WUJI = {
    "left_1_thumb_CMC_FE": "l_thumb_cmc_flex", "left_thumb_CMC_AA": "l_thumb_cmc_abd",
    "left_thumb_MCP_FE": "l_thumb_mcp", "left_thumb_MCP_AA": "l_thumb_mcp", "left_thumb_IP": "l_thumb_ip",
    "left_5_pinky_CMC": "l_pinky_mcp_abd",
    **{s: f"l_{w}_{j}" for f, s_mcp, w in (("index", "left_2_index_MCP_FE", "index_finger"),
                                            ("middle", "left_3_middle_MCP_FE", "middle_finger"),
                                            ("ring", "left_4_ring_MCP_FE", "ring_finger"),
                                            ("pinky", "left_pinky_MCP_FE", "pinky"))
       for s, j in ((s_mcp, "mcp_flex"), (f"left_{f}_MCP_AA", "mcp_abd"),
                    (f"left_{f}_PIP", "pip"), (f"left_{f}_DIP", "dip"))},
}
#: Wuji joint -> the Sharpa joint of the same role (the inverse, minus the two extra Sharpa joints).
_WUJI_TO_SHARPA = {w: s for s, w in _SHARPA_TO_WUJI.items()
                   if s not in ("left_thumb_MCP_AA", "left_5_pinky_CMC")}

# Flexiv + Wuji (human-like limits) with the Sharpa's drive.
FLEXIV_WUJI_HUMAN_LIKE_SHARPAGAIN = dataclasses.replace(
    FLEXIV_WUJI_HUMAN_LIKE,
    name="flexiv_wuji_human_like_sharpagain",
    hand_stiffness={w: KUKA_SHARPA.hand_stiffness[_WUJI_TO_SHARPA[w]] for w in _WUJI_JOINTS},
    hand_damping={w: KUKA_SHARPA.hand_damping[_WUJI_TO_SHARPA[w]] for w in _WUJI_JOINTS},
    hand_effort_limit={w: _SHARPA_URDF_EFFORT[_WUJI_TO_SHARPA[w]] for w in _WUJI_JOINTS},
)
# Flexiv + Sharpa with the Wuji's drive (vendor MJCF gains, as FLEXIV_WUJI_HUMAN_LIKE runs).
FLEXIV_SHARPA_WUJIGAIN = dataclasses.replace(
    FLEXIV_SHARPA,
    name="flexiv_sharpa_wujigain",
    hand_stiffness={s: FLEXIV_WUJI_HUMAN_LIKE.hand_stiffness[_SHARPA_TO_WUJI[s]]
                    for s in KUKA_SHARPA.hand_joint_names},
    hand_damping={s: FLEXIV_WUJI_HUMAN_LIKE.hand_damping[_SHARPA_TO_WUJI[s]]
                  for s in KUKA_SHARPA.hand_joint_names},
    hand_effort_limit={s: _WUJI_URDF_EFFORT[_SHARPA_TO_WUJI[s]] for s in KUKA_SHARPA.hand_joint_names},
)
