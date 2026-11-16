"""The cloth tasks' hand bodies, per robot: what the VBD cloth couples to and the pierce detector watches.

Checked against the robot's URDF, as bodies after the importer merges fixed joints: every finger
segment the spec names must be a real body, and together they must be every finger body.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from isaacsimenvs.tasks.cloth.backends.robot import ROBOT

REPO = Path(__file__).resolve().parents[4]
URDFS = {ROBOT.name: "assets/urdf/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf"}
SPECS = [ROBOT]


def _bodies(spec) -> set[str]:
    """Links that stay bodies: the root and every child of a non-fixed joint."""
    root = ET.parse(REPO / URDFS[spec.name]).getroot()
    children = {j.find("child").get("link"): j.get("type") for j in root.findall("joint")}
    return {l.get("name") for l in root.findall("link") if children.get(l.get("name"), "root") != "fixed"}


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_finger_segments_are_the_finger_bodies(spec):
    bodies = _bodies(spec)
    pattern = re.compile(spec.finger_links_regex("distal", "middle", "proximal"))
    fingers = {b for b in bodies if pattern.fullmatch(b)}
    for segment, regex in spec.finger_segment_regex.items():
        assert any(re.fullmatch(regex, b) for b in bodies), f"{spec.name}: no {segment} body"
    hand = {b for b in bodies if b.startswith(("left_", "l_", "r_"))}
    palm_side = {spec.palm_body_name, *spec.palm_side_link_names}
    assert palm_side <= bodies, f"{spec.name}: palm-side bodies {palm_side - bodies} are not bodies"
    # Every hand body is a finger segment or palm-side, bar the Sharpa's knuckle-joint links (*_VL),
    # which carry no collision.
    left = {b for b in hand - fingers - palm_side if not b.endswith("_VL")}
    assert not left, f"{spec.name}: hand bodies in no segment: {sorted(left)}"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_cloth_proxies_follow_the_robot(spec):
    from isaacsimenvs.tasks.cloth.backends.vbd.config import DeformableClothEnvCfg

    cfg = DeformableClothEnvCfg()
    physics = cfg.deformable_cloth.build_physics(cfg.newton, 32, spec)
    proxy = [re.compile(b) for p in physics.solver_cfg.proxies for b in p.bodies]
    matched = {b for b in _bodies(spec) if any(p.fullmatch(f"/World/envs/env_7/Robot/{b}") for p in proxy)}
    fingers = re.compile(spec.finger_links_regex("distal", "middle", "proximal"))
    expected = {b for b in _bodies(spec) if fingers.fullmatch(b)} | {spec.palm_body_name, *spec.palm_side_link_names}
    assert matched == expected
