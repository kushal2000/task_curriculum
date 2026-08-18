"""The cloth viewer must not draw the spawned rigid tool.

`Cloth.yaml` still builds the rigid tool pool (`handle_head_types: ["hammer"]`) because the sheet
replaces the spawned object only *after* the scene is built. The Play viewer draws that tool at
`object_pose` and a green copy of it at `goal_pose`; for cloth both are hammers, and the green one
is close enough to the fold ghost's green to read as part of the goal. Regression test for that.
"""

from __future__ import annotations

import numpy as np
import pytest

from isaacsimenvs.tasks.play.pose_viewer import build_pose_viewer_html

# A hammer, as `handle_head_primitives` emits it: cylinder handle + box head.
HAMMER_URDF = """<?xml version="1.0"?>
<robot name="object">
  <link name="object_root">
    <visual><geometry><cylinder radius="0.0147" length="0.153"/></geometry></visual>
    <visual><geometry><box size="0.053 0.065 0.027"/></geometry></visual>
  </link>
</robot>"""

TABLE_URDF = """<?xml version="1.0"?>
<robot name="table">
  <link name="table_root">
    <visual><geometry><box size="1.0 1.0 0.3"/></geometry></visual>
  </link>
</robot>"""


def _frames(n: int = 3) -> list[dict]:
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return [
        {
            "robot_joint_names": ["j0"],
            "robot_joint_pos": np.zeros(1),
            "robot_base_pose": np.asarray(pose),
            "table_pose": np.asarray(pose),
            "object_pose": np.asarray(pose),
            "goal_pose": np.asarray(pose),
        }
        for _ in range(n)
    ]


def _html(**kwargs) -> str:
    return build_pose_viewer_html(
        frames=_frames(),
        object_urdf_text=HAMMER_URDF,
        table_urdf_text=TABLE_URDF,
        url_check="skip",
        **kwargs,
    )


def test_play_still_draws_object_and_goal():
    """Default behaviour is unchanged -- this fix must not regress the Play viewer."""
    html = _html()
    assert '"name":"object"' in html.replace(" ", "")
    assert '"name":"goal"' in html.replace(" ", "")
    assert "cylinder" in html


def test_cloth_opt_out_drops_both_hammer_glyphs():
    html = _html(draw_rigid_object=False)
    compact = html.replace(" ", "")
    assert '"name":"object"' not in compact, "inert rigid tool still drawn"
    assert '"name":"goal"' not in compact, "green rigid tool still drawn at goal_viz"
    # The table is embedded the same way, so its survival proves we dropped the right two.
    assert '"name":"table"' in compact
    assert '"name":"robot"' in compact


def test_cloth_opt_out_removes_the_hammer_geometry_entirely():
    """Not just the names -- the handle/head primitives must be gone from the payload."""
    html = _html(draw_rigid_object=False)
    assert "cylinder" not in html, "hammer handle still present in viewer payload"
    assert "0.153" not in html, "hammer handle length still present"


def test_deformables_survive_the_opt_out():
    """The cloth's own three channels are what replace the suppressed glyphs."""
    verts = np.zeros((3, 4, 3))
    html = _html(
        draw_rigid_object=False,
        deformables={
            "cloth_moving": {"indices": [0, 1, 2], "vertices": verts, "color": (1.0, 0.3, 0.1)},
            "fold_goal": {"indices": [0, 1, 2], "vertices": verts, "color": (0.15, 0.85, 0.35),
                          "opacity": 0.45},
        },
    )
    compact = html.replace(" ", "")
    assert "cloth_moving" in compact
    assert "fold_goal" in compact
