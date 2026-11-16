"""The one robot the cloth envs run: Flexiv Rizon 4s + right Wuji Hand 2 with the identified hand.

Copied from ``tasks/play/robots.py`` ``FLEXIV_WUJI_RIGHT_REAL`` (values resolved, the sources noted
there) so the cloth package stands alone. Stdlib only. Field names match Play's ``RobotSpec``,
which the Newton helpers (``vbd/newton_support/patches.author_drive_gains`` etc.) read by attribute.

The policy-facing joint order (``joint_names_canonical``) and the fingertip order are what the Play
prior was trained on; ``tests/test_cloth_utils.py`` pins both.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

#: Play's robot base position; ``palm_pos`` is observed as if the base were here.
PLAY_ROBOT_ROOT_POS = (0.0, 0.8, 0.0)

URDF = "assets/urdf/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf"
#: Where the base sits relative to the env origin (isaacsimenvs/scripts/configs/robot/flexiv_wuji_right_real.yaml): table top
#: 0.33 m above the robot frame, the near side 0.31 m in front.
ROOT_POS = (0.0, 0.51, 0.20)

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
HAND_JOINTS = tuple(
    f"r_{finger}_{joint}"
    for finger, joints in (
        ("thumb", ("cmc_flex", "cmc_abd", "mcp", "ip")),
        ("index_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("middle_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("ring_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("pinky", ("mcp_flex", "mcp_abd", "pip", "dip")),
    )
    for joint in joints
)

#: Rizon 4s PD: stiffness 2 N m/rad per N m of torque limit; damping the Kuka's per-joint ratio.
_ARM_K = (246.0, 246.0, 128.0, 128.0, 78.0, 78.0, 78.0)
_ARM_D = (17.305740827829364, 17.305740827829364, 12.483248807902532, 12.483248807902532,
          6.090458110848477, 5.712766335147973, 5.712766335147973)
#: The real robot's home pose, (-120, 0, 30, 100, 150, 80, 120) deg: palm to the table.
_ARM_HOME = (-2.0944, 0.0, 0.5236, 1.7453, 2.618, 1.3963, 2.0944)

#: Wuji HAND2_DRIVE (stiffness N m/rad, damping N m s/rad), thumb to pinky.
_HAND_DRIVE = (
    (1.354985, 0.04769), (2.198821, 0.062838), (0.886467, 0.027982), (0.654026, 0.020457),
    (1.345207, 0.047402), (0.88946, 0.027239), (0.779361, 0.024489), (0.656424, 0.020508),
    (1.299313, 0.045699), (0.942641, 0.027846), (0.73725, 0.023408), (0.710314, 0.022366),
    (1.276892, 0.046209), (1.013334, 0.028704), (0.859159, 0.027436), (0.648064, 0.020448),
    (1.102915, 0.043816), (0.910201, 0.026593), (0.861968, 0.027476), (0.656704, 0.020611),
)
#: Sysid of the deployed right hand, per joint: (torque constant N m/A, armature kg m^2, passive
#: damping N m s/rad, Coulomb friction N m).
_SYSID = (
    (0.439, 0.0001347, 4.39e-06, 0.05382), (0.2042, 6.801e-05, 8.075e-06, 0.04521),
    (0.2546, 7.132e-05, 2.546e-06, 0.02253), (0.2219, 0.0002385, 2.359e-06, 0.05255),
    (0.5986, 0.0003551, 1.064e-05, 0.0783), (0.2324, 0.0001063, 0.003128, 0.02839),
    (0.1795, 2.622e-05, 2.901e-06, 0.01979), (0.135, 4.875e-05, 4.08e-06, 0.01011),
    (0.6611, 0.0004131, 6.611e-06, 0.07499), (0.2009, 6.992e-05, 0.002858, 0.02198),
    (0.1636, 1.913e-05, 2.555e-06, 0.01594), (0.1138, 2.834e-05, 0.001117, 0.00786),
    (0.5567, 0.0002835, 5.886e-06, 0.07003), (0.2053, 4.993e-05, 0.001846, 0.01984),
    (0.1686, 3.891e-05, 0.001516, 0.01526), (0.1635, 4.443e-05, 0.0004419, 0.0138),
    (0.4777, 0.0001032, 0.0008498, 0.03306), (0.109, 7.36e-05, 0.004938, 0.01344),
    (0.1002, 2.487e-05, 0.000616, 0.0146), (0.157, 5.652e-05, 0.004776, 0.01838),
)
#: The sysid's passive spring per joint: (stiffness N m/rad, rest angle rad).
_SPRING = (
    (0.0843, 0.0619), (0.06436, 0.2691), (0.02401, 0.2373), (0.0197, 0.2892),
    (0.1053, 0.2061), (0.05887, 0.0683), (0.01922, 0.002333), (0.0119, 0.1165),
    (0.1447, 0.2237), (0.05022, -0.0176), (0.01633, 0.01584), (0.01095, 0.1609),
    (0.1296, 0.05509), (0.03698, 0.06116), (0.01341, 0.07124), (0.01487, 0.09207),
    (0.1199, 0.1068), (0.04985, -0.01235), (0.01597, -0.0003289), (0.0224, 0.062),
)
#: Coulomb-friction scale per joint type (PhysX at 1/120 with the spring): thumb S1-S4, finger S1-S4.
_FRICTION_SCALE_THUMB = (0.8, 0.8, 0.8, 1.0)
_FRICTION_SCALE_FINGER = (1.0, 0.8, 0.8, 0.6)
#: The deploy's hand current limit [A].
HAND_CURRENT_LIMIT_A = 1.0

_FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def _self_collision_filter() -> dict[str, tuple[str, ...]]:
    """Jointed pairs, plus the palm (link7) with every finger's proximal and proximal_abd."""
    pairs: dict[str, set[str]] = {}

    def add(a: str, b: str) -> None:
        pairs.setdefault(a, set()).add(b)
        pairs.setdefault(b, set()).add(a)

    links = ("base_link",) + tuple(f"link{i}" for i in range(1, 8))
    for a, b in zip(links, links[1:]):
        add(a, b)
    for f in _FINGERS:
        add(f"r_{f}_distal", f"r_{f}_middle")
        add(f"r_{f}_middle", f"r_{f}_proximal_abd")
        add(f"r_{f}_proximal", f"r_{f}_proximal_abd")
        add(f"r_{f}_proximal", "link7")
        add(f"r_{f}_proximal_abd", "link7")
    return {k: tuple(sorted(v)) for k, v in sorted(pairs.items())}


@dataclass(frozen=True)
class RobotSpec:
    name: str
    arm_joint_regex: str
    hand_joint_regex: str
    joint_names_canonical: tuple[str, ...]
    palm_body_name: str
    palm_center_offset: tuple[float, float, float]
    fingertip_body_regex: str
    #: The order the policy observes the fingertips in (PhysX's find_bodies order for this hand).
    fingertip_link_names: tuple[str, ...]
    fingertip_offsets: tuple[tuple[float, float, float], ...]
    arm_stiffness: dict[str, float]
    arm_damping: dict[str, float]
    hand_stiffness: dict[str, float]
    hand_damping: dict[str, float]
    hand_armature: dict[str, float]
    hand_effort_limit: dict[str, float]
    hand_velocity_limit: float
    hand_friction: dict[str, float]
    hand_dynamic_friction: dict[str, float]
    hand_viscous_friction: dict[str, float]
    hand_spring_stiffness: dict[str, float]
    hand_spring_ref: dict[str, float]
    arm_default_joint_pos: dict[str, float]
    self_collision_filter: dict[str, tuple[str, ...]] = field(default_factory=dict)
    finger_segment_regex: dict[str, str] = field(default_factory=dict)
    palm_side_link_names: tuple[str, ...] = ()

    @property
    def num_joints(self) -> int:
        return len(self.joint_names_canonical)

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return tuple(self.arm_stiffness)

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return tuple(self.hand_stiffness)

    @property
    def fingertip_offset(self):  # Play's name (Newton helpers read either)
        return self.fingertip_offsets

    def finger_links_regex(self, *segments: str) -> str:
        """One regex group over the finger bodies of ``segments`` (keys of ``finger_segment_regex``)."""
        return "(" + "|".join(self.finger_segment_regex[s] for s in segments) + ")"


def _friction() -> dict[str, float]:
    out = {}
    for k, (j, s) in enumerate(zip(HAND_JOINTS, _SYSID)):
        scale = (_FRICTION_SCALE_THUMB if k < 4 else _FRICTION_SCALE_FINGER)[k % 4]
        out[j] = s[3] * scale
    return out


_TIPS = "r_(thumb|index_finger|middle_finger|ring_finger|pinky)"

FLEXIV_WUJI_RIGHT_REAL = RobotSpec(
    name="flexiv_wuji_right_real",
    arm_joint_regex="joint[1-7]",
    hand_joint_regex="r_.*",
    joint_names_canonical=ARM_JOINTS + HAND_JOINTS,
    palm_body_name="link7",
    palm_center_offset=(0.0018, -0.0149, 0.1948),
    fingertip_body_regex=f"{_TIPS}_distal",
    fingertip_link_names=(
        "r_index_finger_distal", "r_middle_finger_distal", "r_pinky_distal",
        "r_ring_finger_distal", "r_thumb_distal",
    ),
    fingertip_offsets=((0.0, 0.0, -0.019),) * 4 + ((0.0, 0.0, -0.0229),),
    arm_stiffness=dict(zip(ARM_JOINTS, _ARM_K)),
    arm_damping=dict(zip(ARM_JOINTS, _ARM_D)),
    hand_stiffness={j: d[0] for j, d in zip(HAND_JOINTS, _HAND_DRIVE)},
    hand_damping={j: d[1] for j, d in zip(HAND_JOINTS, _HAND_DRIVE)},
    hand_armature={j: s[1] for j, s in zip(HAND_JOINTS, _SYSID)},
    hand_effort_limit={j: HAND_CURRENT_LIMIT_A * s[0] for j, s in zip(HAND_JOINTS, _SYSID)},
    hand_velocity_limit=100.0,
    hand_friction=_friction(),
    hand_dynamic_friction=_friction(),
    hand_viscous_friction={j: s[2] for j, s in zip(HAND_JOINTS, _SYSID)},
    hand_spring_stiffness={j: s[0] for j, s in zip(HAND_JOINTS, _SPRING)},
    hand_spring_ref={j: s[1] for j, s in zip(HAND_JOINTS, _SPRING)},
    arm_default_joint_pos=dict(zip(ARM_JOINTS, _ARM_HOME)),
    self_collision_filter=_self_collision_filter(),
    finger_segment_regex={
        "distal": f"{_TIPS}_distal",
        "middle": f"{_TIPS}_middle",
        "proximal": f"{_TIPS}_proximal(_abd)?",
    },
)

ROBOT = FLEXIV_WUJI_RIGHT_REAL

# The flat7RW fine-tunes used HAND2_DRIVE, with real limits supplied by the URDF.
# Keep this checkpoint profile independent of the current identified hand.
YIBO_ROBOT = replace(
    ROBOT,
    name="flexiv_wuji_right_human_like",
    hand_armature={j: (0.0005 if i in (0, 1, 4, 8, 12, 16) else 0.0002)
                   for i, j in enumerate(HAND_JOINTS)},
    hand_effort_limit=dict(zip(HAND_JOINTS, (
        1.02, .83, .35, .21, 1.05, .38, .31, .30, 1.06, .39,
        .32, .29, 1.02, .41, .28, .30, .87, .36, .28, .30,
    ))),
    hand_friction=0.0, hand_dynamic_friction=None, hand_viscous_friction=None,
    hand_spring_stiffness={}, hand_spring_ref={},
)


def get_robot_spec(preset: str) -> RobotSpec:
    """Select the identified hand or the historical checkpoint's hand dynamics."""
    if preset == "real":
        return ROBOT
    if preset == "yibo":
        return YIBO_ROBOT
    raise ValueError(f"unknown robot_preset {preset!r}; expected real or yibo")
