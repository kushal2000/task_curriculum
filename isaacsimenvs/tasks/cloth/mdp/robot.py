"""The robot as the cloth task sees it: body and joint indices, the policy's joint order, joint
limits, the reset pose, and the hand's passive spring.

Two joint orders meet here. The simulator's ("Lab") order is whatever the URDF importer produced;
the policy's ("canonical") order is ``RobotSpec.joint_names_canonical``, the order every checkpoint
was trained in. ``perm_canon_to_lab`` maps a policy action into the simulator's columns and
``perm_lab_to_canon`` maps joint state back for the observation.
"""

from __future__ import annotations

import torch

__all__ = ["NUM_FINGERTIPS", "RobotLayout", "to_torch"]

NUM_FINGERTIPS = 5


def to_torch(value) -> torch.Tensor:
    """Isaac Lab 3.0 asset data are warp-backed ProxyArrays; Isaac Lab 2.3 already gives torch."""
    return value.torch if hasattr(value, "torch") else value


class RobotLayout:
    """Indices, joint orders, limits, reset pose and passive spring of the robot articulation.

    Read once from the live articulation; nothing here changes during a run.
    """

    def __init__(self, robot, spec, *, num_joints: int, robot_urdf: str, device) -> None:
        self.spec = spec
        self.arm_joint_ids = robot.find_joints(spec.arm_joint_regex)[0]
        self.hand_joint_ids = robot.find_joints(spec.hand_joint_regex)[0]
        self.palm_body_id = robot.find_bodies(spec.palm_body_name)[0][0]
        # By name, in the spec's order: the order every policy observes its fingertips in. (A regex
        # lookup returns body-index order, which differs between PhysX and Newton.)
        self.body_names = list(robot.body_names)
        self.fingertip_body_ids = [self.body_names.index(n) for n in spec.fingertip_link_names]
        assert len(self.fingertip_body_ids) == NUM_FINGERTIPS

        lab_names = list(robot.data.joint_names)
        canonical = spec.joint_names_canonical
        assert set(lab_names) == set(canonical), (
            f"robot {spec.name!r} expects joints {sorted(canonical)}, but {robot_urdf} has {sorted(lab_names)}"
        )
        self.joint_names = lab_names
        # The action code slices the arm as Lab columns [:n_arm] and the hand as the rest.
        n_arm = len(spec.arm_joint_names)
        assert list(self.arm_joint_ids) == list(range(n_arm)), self.arm_joint_ids
        assert list(self.hand_joint_ids) == list(range(n_arm, len(lab_names))), self.hand_joint_ids
        self.perm_canon_to_lab = torch.tensor([canonical.index(n) for n in lab_names], device=device, dtype=torch.long)
        self.perm_lab_to_canon = torch.tensor([lab_names.index(n) for n in canonical], device=device, dtype=torch.long)

        limits = robot.data.joint_pos_limits  # (N, J, 2), Lab order
        # Canonical order, to normalise the joint_pos observation; Lab order, to clamp targets.
        self.joint_lower_canon = limits[0, :, 0][self.perm_lab_to_canon]
        self.joint_upper_canon = limits[0, :, 1][self.perm_lab_to_canon]
        self.arm_lower = limits[:, self.arm_joint_ids, 0]
        self.arm_upper = limits[:, self.arm_joint_ids, 1]
        self.hand_lower = limits[:, self.hand_joint_ids, 0]
        self.hand_upper = limits[:, self.hand_joint_ids, 1]
        # The reset pose is drawn around the default pose, within the limits.
        self.default_joint_pos = to_torch(robot.data.default_joint_pos).clone()
        self.joint_pos_limits = to_torch(limits).clone()
        self.num_joints = int(num_joints)
        self._init_spring(robot)

    def _init_spring(self, robot) -> None:
        """The hand's passive spring (``RobotSpec.hand_spring_stiffness``) per sim joint:
        ``spring_ks`` / ``spring_ref`` (J,), ``None`` without a spring. The drives carry the PD gains
        only; :meth:`spring_effort` adds ``ks (ref - q)`` as a joint effort."""
        spec = self.spec
        self.spring_ks = self.spring_ref = None
        if not spec.hand_spring_stiffness:
            return
        names = self.joint_names
        device = self.perm_lab_to_canon.device
        self.spring_ks = torch.tensor([float(spec.hand_spring_stiffness.get(n, 0.0)) for n in names], device=device)
        self.spring_ref = torch.tensor([float(spec.hand_spring_ref.get(n, 0.0)) for n in names], device=device)
        hand = self.hand_joint_ids
        kp = to_torch(robot.data.joint_stiffness)[0, hand]
        ratio = self.spring_ks[hand] / kp
        print(
            f"[spring] passive hand spring on {int((self.spring_ks > 0).sum())} joints as a joint effort: ks "
            f"{float(self.spring_ks[hand].min()):.4f}-{float(self.spring_ks[hand].max()):.4f} N m/rad, "
            f"ks / kp {float(ratio.min()):.3f}-{float(ratio.max()):.3f}",
            flush=True,
        )

    def spring_effort(self, joint_pos: torch.Tensor) -> torch.Tensor | None:
        """The hand's passive spring as a joint effort, ``ks (ref - q)`` (N, J) in Lab order, from
        this physics step's ``q``; ``None`` without a spring.

        Applied as a joint effort target every physics step: an implicit actuator passes it through
        unclipped (PhysX ``set_dof_actuation_forces``, Newton ``joint_f``), so the drive's maxForce
        limits only the PD, as the firmware's current limit does on the real hand.
        """
        if self.spring_ks is None:
            return None
        return self.spring_ks * (self.spring_ref - joint_pos)
