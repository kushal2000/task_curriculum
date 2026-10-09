"""Robot embodiments the Play task can run: everything about the robot that the task code reads.

``PlayEnvCfg.robot`` names one of :data:`ROBOT_SPECS`; each registered task fixes it, so a robot
is chosen by task id (``Isaacsimenvs-Play[SelfCollision]-Direct-v0`` / ``Isaacsimenvs-PlayFlexivWuji-Direct-v0``)
rather than by a runtime flag. ``cfg.assets.robot_urdf`` must be the matching URDF; the env checks
that its joints are exactly the spec's.

Kit-free on purpose (stdlib only), so tests and scripts can read a spec without booting Isaac.

The action pipeline (``action_utils.py``, byte-identical to upstream SimToolReal) treats the first
7 joints in Isaac Lab's order as the arm and the rest as the hand. ``allocate_state_buffers``
asserts that ordering for whichever robot is loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobotSpec:
    name: str
    arm_joint_regex: str
    hand_joint_regex: str
    #: Policy-facing joint order (arm first). Isaac Lab tensors are permuted to and from it.
    joint_names_canonical: tuple[str, ...]
    #: Body the palm observation is read from, and the offset (in that body's frame) to the
    #: palm centre the policy sees.
    palm_body_name: str
    palm_center_offset: tuple[float, float, float]
    #: Bodies that carry the fingertip collision after the URDF importer merges fixed joints,
    #: and the offset (in each one's frame) to the pad centre.
    fingertip_body_regex: str
    fingertip_link_names: tuple[str, ...]
    fingertip_offset: tuple[float, float, float]
    arm_stiffness: dict[str, float]
    arm_damping: dict[str, float]
    hand_stiffness: dict[str, float]
    hand_damping: dict[str, float]
    hand_armature: dict[str, float]
    #: Reset pose; hand joints not listed default to 0.
    arm_default_joint_pos: dict[str, float]
    #: Body pairs excluded from robot self-collision (the task turns it on), body -> its partners,
    #: by post-merge_fixed_joints body name. scene_utils authors them as FilteredPairsAPI and
    #: fails if any name is not a body of the imported robot.
    self_collision_filter: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names_canonical)

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return tuple(self.arm_stiffness)

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return tuple(self.hand_stiffness)

    def __post_init__(self) -> None:
        arm, hand = self.arm_joint_names, self.hand_joint_names
        if len(arm) != 7:
            raise ValueError(f"{self.name}: action_utils assumes a 7-DoF arm, got {len(arm)}")
        if self.joint_names_canonical != arm + hand:
            raise ValueError(f"{self.name}: canonical order must be the arm gains' keys, then the hand's")
        for table in ("arm_damping", "arm_default_joint_pos"):
            if set(getattr(self, table)) != set(arm):
                raise ValueError(f"{self.name}: {table} must cover exactly the arm joints")
        for table in ("hand_damping", "hand_armature"):
            if set(getattr(self, table)) != set(hand):
                raise ValueError(f"{self.name}: {table} must cover exactly the hand joints")
        if len(self.fingertip_link_names) != 5:
            raise ValueError(f"{self.name}: the task observes 5 fingertips")


# ----------------------------------------------------------------------------
# Kuka iiwa14 + left Sharpa: SimToolReal's robot. Values verified with the pretrained checkpoint;
# scene_utils / obs_utils re-export them under their original names.
# ----------------------------------------------------------------------------

_SHARPA_JOINTS = (
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)

# SimToolReal's own list (isaacgymenvs/tasks/simtoolreal/adjacent_links.py,
# LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS), verbatim: the recipe, so the Kuka control runs it.
# Its hand pairs are each jointed pair, the palm with the thumb MC (touching at rest), and the
# palm (or MC) with each proximal phalanx across the two-joint MCP knuckle.
_SHARPA_SELF_COLLISION_FILTER = {
    "iiwa14_link_0": ("iiwa14_link_1",),
    "iiwa14_link_1": ("iiwa14_link_0", "iiwa14_link_2"),
    "iiwa14_link_2": ("iiwa14_link_1", "iiwa14_link_3"),
    "iiwa14_link_3": ("iiwa14_link_2", "iiwa14_link_4"),
    "iiwa14_link_4": ("iiwa14_link_3", "iiwa14_link_5"),
    "iiwa14_link_5": ("iiwa14_link_4", "iiwa14_link_6"),
    "iiwa14_link_6": ("iiwa14_link_5", "iiwa14_link_7"),
    "iiwa14_link_7": (
        "iiwa14_link_6", "left_thumb_CMC_VL", "left_thumb_MC", "left_index_MCP_VL", "left_index_PP",
        "left_middle_MCP_VL", "left_middle_PP", "left_ring_MCP_VL", "left_ring_PP", "left_pinky_MC",
    ),
    "left_index_MCP_VL": ("iiwa14_link_7", "left_index_PP"),
    "left_index_PP": ("iiwa14_link_7", "left_index_MCP_VL", "left_index_MP"),
    "left_index_MP": ("left_index_PP", "left_index_DP"),
    "left_index_DP": ("left_index_MP",),
    "left_middle_MCP_VL": ("iiwa14_link_7", "left_middle_PP"),
    "left_middle_PP": ("iiwa14_link_7", "left_middle_MCP_VL", "left_middle_MP"),
    "left_middle_MP": ("left_middle_PP", "left_middle_DP"),
    "left_middle_DP": ("left_middle_MP",),
    "left_pinky_MC": ("iiwa14_link_7", "left_pinky_MCP_VL", "left_pinky_PP"),
    "left_pinky_MCP_VL": ("left_pinky_MC", "left_pinky_PP"),
    "left_pinky_PP": ("left_pinky_MC", "left_pinky_MCP_VL", "left_pinky_MP"),
    "left_pinky_MP": ("left_pinky_PP", "left_pinky_DP"),
    "left_pinky_DP": ("left_pinky_MP",),
    "left_ring_MCP_VL": ("iiwa14_link_7", "left_ring_PP"),
    "left_ring_PP": ("iiwa14_link_7", "left_ring_MCP_VL", "left_ring_MP"),
    "left_ring_MP": ("left_ring_PP", "left_ring_DP"),
    "left_ring_DP": ("left_ring_MP",),
    "left_thumb_CMC_VL": ("iiwa14_link_7", "left_thumb_MC"),
    "left_thumb_MC": ("iiwa14_link_7", "left_thumb_CMC_VL", "left_thumb_MCP_VL", "left_thumb_PP"),
    "left_thumb_MCP_VL": ("left_thumb_MC", "left_thumb_PP"),
    "left_thumb_PP": ("left_thumb_MC", "left_thumb_MCP_VL", "left_thumb_DP"),
    "left_thumb_DP": ("left_thumb_PP",),
}

KUKA_SHARPA = RobotSpec(
    name="kuka_sharpa",
    arm_joint_regex="iiwa14_joint_.*",
    hand_joint_regex="left_.*",
    joint_names_canonical=tuple(f"iiwa14_joint_{i}" for i in range(1, 8)) + _SHARPA_JOINTS,
    palm_body_name="iiwa14_link_7",
    # Policy was trained against the palm center, not the raw wrist body.
    palm_center_offset=(-0.0, -0.02, 0.16),
    # Merged fingertip bodies land on the DP links in both sims.
    fingertip_body_regex="left_(index|middle|ring|thumb|pinky)_DP",
    fingertip_link_names=(
        "left_index_DP", "left_middle_DP", "left_ring_DP",
        "left_thumb_DP", "left_pinky_DP",
    ),
    # Shift fingertip body origins to the approximate pad centers.
    fingertip_offset=(0.02, 0.002, 0.0),
    arm_stiffness={
        "iiwa14_joint_1": 600.0, "iiwa14_joint_2": 600.0, "iiwa14_joint_3": 500.0,
        "iiwa14_joint_4": 400.0, "iiwa14_joint_5": 200.0, "iiwa14_joint_6": 200.0,
        "iiwa14_joint_7": 200.0,
    },
    arm_damping={
        "iiwa14_joint_1": 27.027026473513512, "iiwa14_joint_2": 27.027026473513512,
        "iiwa14_joint_3": 24.672186769721083, "iiwa14_joint_4": 22.067474708266914,
        "iiwa14_joint_5": 9.752538131173853, "iiwa14_joint_6": 9.147747263670984,
        "iiwa14_joint_7": 9.147747263670984,
    },
    hand_stiffness={
        "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2, "left_thumb_MCP_FE": 4.76,
        "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
        "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
        "left_index_PIP": 0.9, "left_index_DIP": 0.9,
        "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
        "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
        "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
        "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
        "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76, "left_pinky_MCP_AA": 6.62,
        "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
    },
    hand_damping={
        "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
        "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
        "left_thumb_IP": 0.04190723,
        "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
        "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
        "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
        "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
        "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
        "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
        "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
        "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
        "left_pinky_DIP": 0.0350446,
    },
    hand_armature={
        "left_1_thumb_CMC_FE": 0.0032, "left_thumb_CMC_AA": 0.0032,
        "left_thumb_MCP_FE": 0.00265, "left_thumb_MCP_AA": 0.00265, "left_thumb_IP": 0.0006,
        "left_2_index_MCP_FE": 0.00265, "left_index_MCP_AA": 0.00265,
        "left_index_PIP": 0.0006, "left_index_DIP": 0.00042,
        "left_3_middle_MCP_FE": 0.00265, "left_middle_MCP_AA": 0.00265,
        "left_middle_PIP": 0.0006, "left_middle_DIP": 0.00042,
        "left_4_ring_MCP_FE": 0.00265, "left_ring_MCP_AA": 0.00265,
        "left_ring_PIP": 0.0006, "left_ring_DIP": 0.00042,
        "left_5_pinky_CMC": 0.00012, "left_pinky_MCP_FE": 0.00265,
        "left_pinky_MCP_AA": 0.00265, "left_pinky_PIP": 0.0006, "left_pinky_DIP": 0.00042,
    },
    # Proven-working default arm pose.
    arm_default_joint_pos={
        "iiwa14_joint_1": -1.571, "iiwa14_joint_2": 1.571, "iiwa14_joint_3": 0.0,
        "iiwa14_joint_4": 1.376, "iiwa14_joint_5": 0.0, "iiwa14_joint_6": 1.485,
        "iiwa14_joint_7": 1.308,
    },
    self_collision_filter=_SHARPA_SELF_COLLISION_FILTER,
)


# ----------------------------------------------------------------------------
# Flexiv Rizon 4s + left Wuji Hand 2 (scripts/flexiv_wuji/). Geometry is derived from the
# models; the DYNAMICS ARE PROVISIONAL placeholders, to be refined against the hardware.
# ----------------------------------------------------------------------------

_RIZON_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
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
    # knuckle-defined axes; scripts/flexiv_wuji/viser_compare.py): same spot relative to the
    # knuckles. Depends on flexiv_wuji.MOUNT_*; tests/test_robot_specs.py recomputes it.
    palm_center_offset=(-0.0018, -0.015, 0.1948),
    fingertip_body_regex="l_(thumb|index_finger|middle_finger|ring_finger|pinky)_distal",
    fingertip_link_names=(
        "l_index_finger_distal", "l_middle_finger_distal", "l_ring_finger_distal",
        "l_thumb_distal", "l_pinky_distal",
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
)

ROBOT_SPECS: dict[str, RobotSpec] = {spec.name: spec for spec in (KUKA_SHARPA, FLEXIV_WUJI)}


def get_robot_spec(name: str) -> RobotSpec:
    try:
        return ROBOT_SPECS[name]
    except KeyError:
        raise ValueError(f"unknown robot {name!r}; one of {sorted(ROBOT_SPECS)}") from None
