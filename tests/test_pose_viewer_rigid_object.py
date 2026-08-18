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
    """Not just the names -- the handle/head primitives must be gone from the payload.

    Assert on URDF MARKUP (`<cylinder`, `radius=`, `length=`), never on a bare dimension like
    "0.153". A real capture is megabytes of float arrays, so "0.153" occurs by chance inside
    coordinate data -- it appeared 26 times in the first production capture, every one of them a
    substring of a vertex or joint value, while the geometry was correctly absent. A test that
    checks a bare number passes on the synthetic fixture here and then cries wolf on real output.
    """
    html = _html(draw_rigid_object=False)
    assert "<cylinder" not in html, "hammer handle still present in viewer payload"
    assert "radius=" not in html
    assert "length=" not in html


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


# --------------------------------------------------------------------------------------------
# The tests above exercise `build_pose_viewer_html` directly, which proves the FUNCTION honours
# the flag. It does not prove `ClothPoseViewerWrapper` passes it. This one drives the real
# wrapper method, which is the wiring that actually shipped.
# --------------------------------------------------------------------------------------------

def _cloth_wrapper_html(resolution: int = 9, size: float = 0.10) -> str:
    import types

    from isaacsimenvs.tasks.cloth.pose_viewer import ClothPoseViewerWrapper

    n_particles = resolution * resolution
    n_half = resolution * ((resolution + 1) // 2)
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    frames = [
        {
            "robot_joint_names": ["j0"],
            "robot_joint_pos": np.zeros(1),
            "robot_base_pose": np.asarray(pose),
            "table_pose": np.asarray(pose),
            "object_pose": np.asarray(pose),
            "goal_pose": np.asarray(pose),
            "cloth_vertices": np.zeros((n_particles, 3)).tolist(),
            "goal_vertices": np.zeros((n_half, 3)).tolist(),
            "fold_err": 0.08,
            "footprint": 1.0,
        }
        for _ in range(3)
    ]

    inner = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            cloth=types.SimpleNamespace(size=size, resolution=resolution, fold_axis="x")
        )
    )
    # `object.__new__` skips gym.Wrapper.__init__, which would need a real env. `_build_html`
    # touches only the attributes set below.
    wrapper = object.__new__(ClothPoseViewerWrapper)
    wrapper.env = types.SimpleNamespace(unwrapped=inner)
    wrapper._object_urdf_text = HAMMER_URDF
    wrapper._table_urdf_text = TABLE_URDF
    wrapper._hole_urdf_text = None
    wrapper._object_urdf_path = None
    wrapper._table_urdf_path = None
    wrapper._hole_urdf_path = None
    wrapper.github_raw_base = "https://raw.githubusercontent.com/kushal2000/task_curriculum/main/"
    wrapper.url_check = "skip"
    return ClothPoseViewerWrapper._build_html(wrapper, frames)


def test_real_cloth_wrapper_emits_no_hammer():
    html = _cloth_wrapper_html()
    compact = html.replace(" ", "")
    assert '"name":"object"' not in compact
    assert '"name":"goal"' not in compact
    assert "<cylinder" not in html, "hammer handle survived the real wrapper path"
    assert "radius=" not in html


def test_real_cloth_wrapper_still_emits_the_three_cloth_channels():
    compact = _cloth_wrapper_html().replace(" ", "")
    for channel in ("cloth_moving", "cloth_stationary", "fold_goal"):
        assert channel in compact, f"{channel} missing"
    # Robot and table must survive, or the opt-out dropped more than the two tool glyphs.
    assert '"name":"table"' in compact
    assert '"name":"robot"' in compact


def test_the_hammer_strings_are_a_discriminating_check():
    """Guard against the assertions above going vacuous.

    Every 'not in' assertion passes trivially if the strings stop appearing for an unrelated
    reason (a renamed key, a changed serialisation). Drive the same builder with the flag ON and
    require the hammer to come back, so the checks are known to still discriminate.
    """
    html = _html()  # draw_rigid_object defaults True
    assert "<cylinder" in html
    assert "radius=" in html


# --------------------------------------------------------------------------------------------
# train.py decides which kwargs a viewer takes. It used to string-match the module name
# ("play.pose_viewer" in module_name), which is FALSE for the cloth viewer's module even though
# ClothPoseViewerWrapper subclasses PlayPoseViewerWrapper and needs both kwargs -- so cloth ran
# with url_check forced to "skip", disabling the very check that reports an unpushed branch.
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_name, class_name, expected",
    [
        ("isaacsimenvs.tasks.cloth.pose_viewer", "ClothPoseViewerWrapper", True),
        ("isaacsimenvs.tasks.play.pose_viewer", "PlayPoseViewerWrapper", True),
        ("isaacsimenvs.tasks.multilink_cartpole.pose_viewer", "CartpolePoseViewerWrapper", False),
    ],
)
def test_network_kwargs_are_offered_by_signature_not_module_name(module_name, class_name, expected):
    import importlib
    import inspect

    viewer_cls = getattr(importlib.import_module(module_name), class_name)
    accepted = inspect.signature(viewer_cls.__init__).parameters
    assert ("github_raw_base" in accepted) is expected
    assert ("url_check" in accepted) is expected


def test_cloth_viewer_module_name_defeats_the_old_substring_test():
    """Pin the exact reason the old gate failed, so it cannot be reintroduced."""
    module_name = "isaacsimenvs.tasks.cloth.pose_viewer"
    assert "play.pose_viewer" not in module_name
