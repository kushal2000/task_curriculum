"""PlayFlexivWuji.yaml must not drift from Play.yaml except in the robot.

The overlay loader reads one file with no inheritance, so the Flexiv task YAML is a full copy of
Play's. The point of the Flexiv task is to train SimToolReal's recipe on a different robot; if a
reward scale or reset range differed too, a gap between the two robots would no longer be about
the robot. Same approach as test_playnewton_overlay.py.

Kit-free by construction -- it reads YAML and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CFG = Path(__file__).resolve().parents[1] / "isaacsimenvs" / "cfg" / "task"

#: Top-level keys PlayFlexivWuji may add: the robot selector.
ALLOWED_EXTRA = {"robot"}
#: Leaves that must differ, and the value each must have.
EXPECTED_DIFFERENCES = {
    "action_space": 27,
    "assets.robot_urdf": "assets/flexiv_wuji/rizon4s_left_wuji.urdf",
}


def _flatten(node, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out[prefix] = node
    return out


@pytest.fixture(scope="module")
def cfgs() -> tuple[dict, dict]:
    play = yaml.safe_load((CFG / "Play.yaml").read_text())
    flexiv = yaml.safe_load((CFG / "PlayFlexivWuji.yaml").read_text())
    return play, flexiv


def test_selects_the_flexiv_robot(cfgs) -> None:
    _, flexiv = cfgs
    assert flexiv["robot"] == "flexiv_wuji"
    assert set(flexiv) - set(cfgs[0]) == ALLOWED_EXTRA


def test_task_definition_matches_play_outside_the_robot(cfgs) -> None:
    play, flexiv = cfgs
    flat_play = _flatten(play)
    flat_flexiv = _flatten({k: v for k, v in flexiv.items() if k not in ALLOWED_EXTRA})
    assert flat_play.keys() == flat_flexiv.keys(), (
        f"keys differ: only in Play {sorted(flat_play.keys() - flat_flexiv.keys())}, "
        f"only in PlayFlexivWuji {sorted(flat_flexiv.keys() - flat_play.keys())}"
    )
    differing = {k: (flat_play[k], flat_flexiv[k]) for k in flat_play if flat_play[k] != flat_flexiv[k]}
    assert set(differing) == set(EXPECTED_DIFFERENCES), (
        f"PlayFlexivWuji.yaml and Play.yaml differ outside the robot: "
        f"{ {k: v for k, v in differing.items() if k not in EXPECTED_DIFFERENCES} }"
    )
    for key, value in EXPECTED_DIFFERENCES.items():
        assert flat_flexiv[key] == value, key
