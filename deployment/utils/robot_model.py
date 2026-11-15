"""Forward kinematics of the robot URDF: where the palm and the fingertips are, for given joint angles.

    model = load_robot_model(spec)
    pos, quat = link_pose(model, "link7", q)       # env frame, quat wxyz
    palm = palm_center(spec, model, q)             # (3,)
    tips = fingertips(spec, model, q)              # (5, 3), in the observation's order

Composes quaternions link by link, as PhysX does, so it matches the sim to ~1e-6 m.
"""

import xml.etree.ElementTree as ET

import numpy as np

from .quat import quat_apply, quat_from_angle_axis, quat_from_rpy, quat_mul


def load_robot_model(spec: dict) -> dict:
    """Read the URDF's joints.
    Returns {"joints": {child link: joint}, "joint_index": {joint name: index in q},
             "root_pos": ..., "root_quat": ...}; a joint is a dict (see _read_joint)."""
    root = ET.parse(spec["urdf_path"]).getroot()
    joints = {}
    for j in root.iter("joint"):
        joint = _read_joint(j)
        joints[joint["child"]] = joint
    return {
        "joints": joints,
        "joint_index": {name: i for i, name in enumerate(spec["joint_names"])},
        "root_pos": spec["robot_root_pos_env"],
        "root_quat": spec["robot_root_quat_wxyz"],
    }


def link_pose(model: dict, link: str, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pos, quat wxyz) of a link frame in the env frame."""
    p, r = model["root_pos"].copy(), model["root_quat"].copy()
    for j in _chain(model, link):
        p = p + quat_apply(r, j["xyz"])          # move to the joint origin
        r = quat_mul(r, j["quat"])               # turn to the joint origin's orientation
        if j["type"] in ("revolute", "continuous"):
            r = quat_mul(r, quat_from_angle_axis(q[model["joint_index"][j["name"]]], j["axis"]))
        elif j["type"] == "prismatic":
            p = p + quat_apply(r, j["axis"] * q[model["joint_index"][j["name"]]])
    return p, r


def palm_center(spec: dict, model: dict, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(palm centre position, palm body quat wxyz): the palm body frame moved by palm_center_offset."""
    p, r = link_pose(model, spec["palm_body"], q)
    return p + quat_apply(r, spec["palm_center_offset"]), r


def fingertips(spec: dict, model: dict, q: np.ndarray) -> np.ndarray:
    """(5, 3) fingertip pad positions, in the order the observation lists them.

    That order is PhysX's body order (index, middle, pinky, ring, thumb), with the pad offsets
    applied by position, so the ring gets the thumb's 22.9 mm and the thumb 19 mm. The policy was
    trained like this: keep fingertip_obs_bodies / fingertip_obs_offsets as the spec has them.
    """
    tips = []
    for link, offset in zip(spec["fingertip_obs_bodies"], spec["fingertip_obs_offsets"]):
        p, r = link_pose(model, link, q)
        tips.append(p + quat_apply(r, offset))
    return np.stack(tips)


def _read_joint(j: ET.Element) -> dict:
    origin = j.find("origin")
    xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
    rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
    axis_el = j.find("axis")
    axis = np.array([float(v) for v in (axis_el.get("xyz") if axis_el is not None else "1 0 0").split()])
    return {
        "name": j.get("name"),
        "type": j.get("type"),                 # revolute | prismatic | fixed | ...
        "parent": j.find("parent").get("link"),
        "child": j.find("child").get("link"),
        "xyz": np.array([float(v) for v in xyz.split()]),
        "quat": quat_from_rpy([float(v) for v in rpy.split()]),
        "axis": axis / np.linalg.norm(axis),
    }


def _chain(model: dict, link: str) -> list[dict]:
    """The joints from the root down to `link`."""
    chain = []
    while link in model["joints"]:
        j = model["joints"][link]
        chain.append(j)
        link = j["parent"]
    return chain[::-1]
