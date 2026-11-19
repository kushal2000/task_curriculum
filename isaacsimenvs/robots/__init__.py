"""Robot embodiments the tasks can run: everything about the robot that the task code reads.

One robot, the Flexiv Rizon 4s + right Wuji Hand 2 as the real robot is set up, in four drive
variants (``flexiv_right.py``). A task's config names one by ``robot`` (set by a robot setup,
``configs/robot/``); ``assets.robot_urdf`` must be the matching URDF, and the env checks that its
joints are exactly the spec's.

Kit-free on purpose (stdlib only), so tests and scripts can read a spec without booting Isaac.

The action pipeline treats the first 7 joints in Isaac Lab's order as the arm and the rest as the
hand; the envs assert that ordering for whichever robot is loaded.
"""

from __future__ import annotations

from .flexiv_right import (
    FLEXIV_WUJI_RIGHT_HUMAN_LIKE,
    FLEXIV_WUJI_RIGHT_REAL,
    FLEXIV_WUJI_RIGHT_S2R_EFF,
    FLEXIV_WUJI_RIGHT_SHARPAGAIN,
    WUJI_HAND_CURRENT_LIMIT_A,
)
from .spec import RobotSpec

__all__ = [
    "FLEXIV_WUJI_RIGHT_HUMAN_LIKE",
    "FLEXIV_WUJI_RIGHT_REAL",
    "FLEXIV_WUJI_RIGHT_S2R_EFF",
    "FLEXIV_WUJI_RIGHT_SHARPAGAIN",
    "ROBOT_SPECS",
    "RobotSpec",
    "WUJI_HAND_CURRENT_LIMIT_A",
    "get_robot_spec",
]

ROBOT_SPECS: dict[str, RobotSpec] = {
    spec.name: spec for spec in (
        FLEXIV_WUJI_RIGHT_HUMAN_LIKE, FLEXIV_WUJI_RIGHT_REAL, FLEXIV_WUJI_RIGHT_S2R_EFF,
        FLEXIV_WUJI_RIGHT_SHARPAGAIN,
    )
}


def get_robot_spec(name: str) -> RobotSpec:
    try:
        return ROBOT_SPECS[name]
    except KeyError:
        raise ValueError(f"unknown robot {name!r}; one of {sorted(ROBOT_SPECS)}") from None
