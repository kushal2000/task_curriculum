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

from .flexiv_left import (
    FLEXIV_SHARPA,
    FLEXIV_SHARPA_WUJIGAIN,
    FLEXIV_WUJI,
    FLEXIV_WUJI_HUMAN_LIKE,
    FLEXIV_WUJI_HUMAN_LIKE_S2R,
    FLEXIV_WUJI_HUMAN_LIKE_SHARPAGAIN,
)
from .flexiv_right import (
    FLEXIV_WUJI_RIGHT_HUMAN_LIKE,
    FLEXIV_WUJI_RIGHT_REAL,
    FLEXIV_WUJI_RIGHT_S2R_EFF,
    FLEXIV_WUJI_RIGHT_SHARPAGAIN,
    WUJI_HAND_CURRENT_LIMIT_A,
)
from .kuka_sharpa import KUKA_SHARPA
from .spec import RobotSpec

__all__ = [
    "FLEXIV_SHARPA",
    "FLEXIV_SHARPA_WUJIGAIN",
    "FLEXIV_WUJI",
    "FLEXIV_WUJI_HUMAN_LIKE",
    "FLEXIV_WUJI_HUMAN_LIKE_S2R",
    "FLEXIV_WUJI_HUMAN_LIKE_SHARPAGAIN",
    "FLEXIV_WUJI_RIGHT_HUMAN_LIKE",
    "FLEXIV_WUJI_RIGHT_REAL",
    "FLEXIV_WUJI_RIGHT_S2R_EFF",
    "FLEXIV_WUJI_RIGHT_SHARPAGAIN",
    "KUKA_SHARPA",
    "ROBOT_SPECS",
    "RobotSpec",
    "WUJI_HAND_CURRENT_LIMIT_A",
    "get_robot_spec",
]

ROBOT_SPECS: dict[str, RobotSpec] = {
    spec.name: spec for spec in (
        KUKA_SHARPA, FLEXIV_WUJI, FLEXIV_WUJI_HUMAN_LIKE, FLEXIV_WUJI_HUMAN_LIKE_S2R, FLEXIV_SHARPA,
        FLEXIV_WUJI_HUMAN_LIKE_SHARPAGAIN, FLEXIV_SHARPA_WUJIGAIN, FLEXIV_WUJI_RIGHT_HUMAN_LIKE,
        FLEXIV_WUJI_RIGHT_REAL, FLEXIV_WUJI_RIGHT_S2R_EFF, FLEXIV_WUJI_RIGHT_SHARPAGAIN,
    )
}


def get_robot_spec(name: str) -> RobotSpec:
    try:
        return ROBOT_SPECS[name]
    except KeyError:
        raise ValueError(f"unknown robot {name!r}; one of {sorted(ROBOT_SPECS)}") from None
