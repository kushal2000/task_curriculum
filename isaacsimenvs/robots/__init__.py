"""The robot the tasks run: everything about it that the task code reads.

One robot, the Flexiv Rizon 4s + right Wuji Hand 2 as the real robot is set up
(``FLEXIV_WUJI_RIGHT``, ``flexiv_right.py``), optionally with the real hand's identified joint
dynamics (``with_sysid``; the env's ``hand_sysid``). Its joint limits are the task's URDF
(``assets.robot_urdf``), and the env checks that the URDF's joints are exactly the spec's.

Kit-free on purpose (stdlib only), so tests and scripts can read the spec without booting Isaac.

The action pipeline treats the first 7 joints in Isaac Lab's order as the arm and the rest as the
hand; the envs assert that ordering.
"""

from __future__ import annotations

from .flexiv_right import FLEXIV_WUJI_RIGHT, WUJI_HAND_CURRENT_LIMIT_A, with_sysid
from .spec import RobotSpec

__all__ = ["FLEXIV_WUJI_RIGHT", "RobotSpec", "WUJI_HAND_CURRENT_LIMIT_A", "robot_spec", "with_sysid"]


def robot_spec(hand_sysid: bool = False) -> RobotSpec:
    """The robot of an env config: ``FLEXIV_WUJI_RIGHT``, with the identified hand if ``hand_sysid``."""
    return with_sysid(FLEXIV_WUJI_RIGHT) if hand_sysid else FLEXIV_WUJI_RIGHT
