"""``RobotSpec``: everything about a robot the Play / cloth task code reads. Stdlib only."""

from __future__ import annotations

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
    #: One offset for all five, or one per ``fingertip_link_names`` entry.
    fingertip_offset: tuple[float, float, float] | tuple[tuple[float, float, float], ...]
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
    #: The cloth tasks' view of the hand: a regex over body names per finger segment ("distal",
    #: "middle", "proximal"; the thumb's segments included), and the palm-side bodies other than
    #: ``palm_body_name``. The VBD cloth couples to these bodies and the pierce detector watches
    #: them. Empty on a robot the cloth tasks have not been set up for.
    finger_segment_regex: dict[str, str] = field(default_factory=dict)
    palm_side_link_names: tuple[str, ...] = ()
    #: Hand joint torque limits [N m]; empty keeps the URDF's ``effort``. Both backends apply it
    #: (PhysX via ``effort_limit_sim``, Newton via the drives' ``maxForce``).
    hand_effort_limit: dict[str, float] = field(default_factory=dict)
    #: Hand joint velocity limit [rad/s] and joint friction; None keeps the asset's. Friction is an
    #: EFFORT [N m] from Isaac Sim 5.0 on (a unitless coefficient before): static breakaway, Coulomb
    #: once moving, and a viscous term [N m s/rad]. One value for every hand joint, or one per joint.
    hand_velocity_limit: float | None = None
    hand_friction: float | dict[str, float] | None = None
    hand_dynamic_friction: float | dict[str, float] | None = None
    hand_viscous_friction: float | dict[str, float] | None = None
    #: Passive joint spring per hand joint: stiffness [N m / rad] and rest angle [rad]; empty for
    #: none. Applied as a joint effort ``ks (ref - q)`` every physics step (PlayEnv._apply_action,
    #: ``action_utils.hand_spring_effort``), next to the PD drive, NOT folded into it: the drive's
    #: maxForce (``hand_effort_limit``) clips only the PD, as the firmware's current limit does on the
    #: real hand, where the spring is mechanical. Folded in, the spring was clipped with the PD and
    #: a saturated flexion got the spring's 20-30% of the limit back for free.
    hand_spring_stiffness: dict[str, float] = field(default_factory=dict)
    hand_spring_ref: dict[str, float] = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names_canonical)

    def finger_links_regex(self, *segments: str) -> str:
        """One regex group matching the finger bodies of ``segments`` (keys of ``finger_segment_regex``)."""
        if not self.finger_segment_regex:
            raise ValueError(f"{self.name}: no finger_segment_regex; the cloth tasks need it")
        return "(" + "|".join(self.finger_segment_regex[s] for s in segments) + ")"

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
        if self.hand_effort_limit and set(self.hand_effort_limit) != set(hand):
            raise ValueError(f"{self.name}: hand_effort_limit must be empty or cover exactly the hand joints")
        for table in ("hand_friction", "hand_dynamic_friction", "hand_viscous_friction"):
            value = getattr(self, table)
            if isinstance(value, dict) and set(value) != set(hand):
                raise ValueError(f"{self.name}: a per-joint {table} must cover exactly the hand joints")
        if set(self.hand_spring_stiffness) != set(self.hand_spring_ref):
            raise ValueError(f"{self.name}: hand_spring_stiffness and hand_spring_ref need the same joints")
        if self.hand_spring_stiffness and set(self.hand_spring_stiffness) != set(hand):
            raise ValueError(f"{self.name}: hand_spring_stiffness must be empty or cover exactly the hand joints")
        if any(v < 0.0 for v in self.hand_spring_stiffness.values()):
            raise ValueError(f"{self.name}: hand_spring_stiffness must be >= 0")
        if len(self.fingertip_link_names) != 5:
            raise ValueError(f"{self.name}: the task observes 5 fingertips")
        off = self.fingertip_offset
        if not (len(off) == 3 and all(isinstance(v, (int, float)) for v in off)) and not (
            len(off) == 5 and all(len(o) == 3 for o in off)
        ):
            raise ValueError(f"{self.name}: fingertip_offset must be one 3-vector or five")
