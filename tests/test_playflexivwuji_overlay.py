"""The Flexiv task YAMLs must not drift from Play.yaml except in the robot and its table.

The overlay loader reads one file with no inheritance, so the Flexiv task YAML is a full copy of
Play's. The point of the Flexiv task is to train SimToolReal's recipe on a different robot; if a
reward scale or reset range differed too, a gap between the two robots would no longer be about
the robot. The one other difference is the table, moved to where the real Flexiv setup has
it (and the goal volume with it). Same approach as test_playnewton_overlay.py.

Kit-free by construction -- it reads YAML and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CFG = Path(__file__).resolve().parents[1] / "isaacsimenvs" / "cfg" / "task"

#: Top-level keys a Flexiv task YAML may add: the robot selector.
ALLOWED_EXTRA = {"robot"}
#: The real Flexiv setup's table (every Flexiv task): where it is, and the goal volume moved with it.
REAL_TABLE = {
    "assets.table_urdf": "assets/urdf/table_narrow_flexiv.urdf",
    "reset.table_reset_z": 0.26,
    "reset.table_object_z_offset": 0.12,
    "reset.table_reset_xy": [0.0, 0.29],
    "reset.target_volume_mins": [-0.35, 0.0, 0.35],
    "reset.target_volume_maxs": [0.35, 0.49, 0.7],
    "reset.use_goal_table_clearance": True,
}
#: Per Flexiv task YAML: its robot, and the leaves that must differ from Play.yaml with their values.
VARIANTS = {
    "PlayFlexivWuji.yaml": ("flexiv_wuji", {
        "action_space": 27,
        "assets.robot_urdf": "assets/flexiv_wuji/rizon4s_left_wuji.urdf",
        **REAL_TABLE,
    }),
    "PlayFlexivSharpa.yaml": ("flexiv_sharpa", {
        "assets.robot_urdf": "assets/flexiv_wuji/rizon4s_left_sharpa.urdf",
        **REAL_TABLE,
    }),
    "PlayFlexivWujiRightHumanLike.yaml": ("flexiv_wuji_right_human_like", {
        "action_space": 27,
        "assets.robot_urdf": "assets/flexiv_wuji/rizon4s_right_wuji_human_like_joint_limits.urdf",
        **REAL_TABLE,
    }),
    "PlayFlexivWujiHumanLike.yaml": ("flexiv_wuji_human_like", {
        "action_space": 27,
        "assets.robot_urdf": "assets/flexiv_wuji/rizon4s_left_wuji_human_like_joint_limits.urdf",
        **REAL_TABLE,
    }),
}


def _flatten(node, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out[prefix] = node
    return out


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def cfgs(request) -> tuple[dict, dict, str]:
    play = yaml.safe_load((CFG / "Play.yaml").read_text())
    flexiv = yaml.safe_load((CFG / request.param).read_text())
    return play, flexiv, request.param


def test_selects_the_flexiv_robot(cfgs) -> None:
    play, flexiv, name = cfgs
    assert flexiv["robot"] == VARIANTS[name][0]
    assert set(flexiv) - set(play) == ALLOWED_EXTRA


def test_task_definition_matches_play_outside_the_robot(cfgs) -> None:
    play, flexiv, name = cfgs
    expected = VARIANTS[name][1]
    flat_play = _flatten(play)
    flat_flexiv = _flatten({k: v for k, v in flexiv.items() if k not in ALLOWED_EXTRA})
    assert flat_play.keys() == flat_flexiv.keys(), (
        f"keys differ: only in Play {sorted(flat_play.keys() - flat_flexiv.keys())}, "
        f"only in {name} {sorted(flat_flexiv.keys() - flat_play.keys())}"
    )
    differing = {k: (flat_play[k], flat_flexiv[k]) for k in flat_play if flat_play[k] != flat_flexiv[k]}
    assert set(differing) == set(expected), (
        f"{name} and Play.yaml differ outside the robot: "
        f"{ {k: v for k, v in differing.items() if k not in expected} }"
    )
    for key, value in expected.items():
        assert flat_flexiv[key] == value, key
