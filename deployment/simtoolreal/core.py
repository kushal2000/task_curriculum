"""The Play policy's observation, action, goal and success logic, in numpy, without Isaac Sim.

Mirrors, step for step, the training env of branch kushal/2026-10-30_pretraining_real
(isaacsimenvs/tasks/play/utils/{obs_utils,action_utils,goal_sampling,goal_table_clearance,
reward_utils,termination_utils}.py) with every DR term off, for the Flexiv Rizon 4s + right
human-like Wuji Hand 2. Every number comes from spec_flexiv_wuji_right_hl.json (export_spec.py).

Frames. "env" is the training env's frame: the robot base sits at robot_root_pos_env (0, 0.8, 0)
with no rotation, the table top at z = 0.28. Poses the real loop measures in the robot base frame
go through base_to_env first. Quaternions are wxyz internally; the policy observes xyzw.

check_offline.py checks this module against a rollout recorded in the sim (record_sim_rollout.py).
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SPEC = HERE / "spec_flexiv_wuji_right_hl.json"
DEFAULT_CKPT = REPO / "pretrained_policy" / "nov6_dr.pth"
NUM_OBS = 134
NUM_ACTIONS = 27


# ------------------------------------------------------------------------------------ quaternions


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions, broadcasting over leading axes."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v by wxyz quaternions q (Isaac Lab's quat_apply)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_from_angle_axis(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(angle)[..., None]
    return np.concatenate([np.cos(half), np.sin(half) * axis], axis=-1)


def quat_from_rpy(rpy) -> np.ndarray:
    """URDF rpy (fixed-axis x, then y, then z) as wxyz: q = qz * qy * qx."""
    r, p, y = rpy
    qx = quat_from_angle_axis(r, np.array([1.0, 0, 0]))
    qy = quat_from_angle_axis(p, np.array([0, 1.0, 0]))
    qz = quat_from_angle_axis(y, np.array([0, 0, 1.0]))
    return quat_mul(qz, quat_mul(qy, qx))


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:], q[..., :3]], axis=-1)


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation, wxyz (Isaac Lab random_orientation: a normalized Gaussian 4-vector)."""
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


# -------------------------------------------------------------------------------------------- spec


@dataclass
class Spec:
    d: dict
    joint_names: list[str]
    lower: np.ndarray
    upper: np.ndarray
    default_q: np.ndarray
    n_arm: int

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SPEC) -> "Spec":
        d = json.loads(Path(path).read_text())
        a = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
        return cls(d=d, joint_names=list(d["joint_names"]), lower=a("joint_lower"),
                   upper=a("joint_upper"), default_q=a("default_q"), n_arm=int(d["num_arm_joints"]))

    def __getitem__(self, key: str):
        return self.d[key]

    @property
    def fingertip_obs(self) -> list[tuple[str, list[float]]]:
        """(body, pad offset) in the order the observation lists the fingertips.

        The env takes the bodies in PhysX's find_bodies order and applies robot_spec's offsets by
        position, which need not be fingertip_links' order; record_sim_rollout.py reads the real one
        and export_spec.py --rollout_meta writes it as fingertip_obs_bodies / fingertip_obs_offsets.
        """
        bodies = self.d.get("fingertip_obs_bodies", self.d["fingertip_links"])
        offsets = self.d.get("fingertip_obs_offsets", self.d["fingertip_offsets"])
        return list(zip(bodies, offsets))

    @property
    def urdf_path(self) -> Path:
        return HERE / self.d["robot_urdf"]

    def base_to_env(self, pos_base: np.ndarray, quat_base_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A pose measured in the robot base frame, in the training env's frame."""
        root_q = np.asarray(self.d["robot_root_quat_wxyz"], dtype=np.float64)
        root_p = np.asarray(self.d["robot_root_pos_env"], dtype=np.float64)
        return root_p + quat_apply(root_q, pos_base), quat_mul(root_q, quat_base_wxyz)


# --------------------------------------------------------------------------------------------- FK


@dataclass
class _Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    quat: np.ndarray  # wxyz, the joint origin's rotation
    axis: np.ndarray


class RobotFK:
    """Forward kinematics of the robot URDF, by composing quaternions link by link as PhysX does.

    Link poses are in the env frame (robot root at robot_root_pos_env). Only the links on the paths
    to the palm body and the fingertips are computed.
    """

    def __init__(self, spec: Spec) -> None:
        root = ET.parse(spec.urdf_path).getroot()
        self.joints: dict[str, _Joint] = {}
        for j in root.iter("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
            jt = _Joint(j.get("name"), j.get("type"), j.find("parent").get("link"),
                        j.find("child").get("link"), xyz, quat_from_rpy(rpy), axis / np.linalg.norm(axis))
            self.joints[jt.child] = jt
        self.root_pos = np.asarray(spec["robot_root_pos_env"], dtype=np.float64)
        self.root_quat = np.asarray(spec["robot_root_quat_wxyz"], dtype=np.float64)
        self.index = {n: i for i, n in enumerate(spec.joint_names)}
        targets = [spec["palm_body"], *spec["fingertip_links"], *(b for b, _ in spec.fingertip_obs)]
        self.chains = {t: self._chain(t) for t in dict.fromkeys(targets)}

    def _chain(self, link: str) -> list[_Joint]:
        chain = []
        while link in self.joints:
            j = self.joints[link]
            chain.append(j)
            link = j.parent
        return chain[::-1]

    def link_pose(self, link: str, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pos, quat wxyz) of one link frame in the env frame, for canonical-order joint angles q."""
        p, r = self.root_pos.copy(), self.root_quat.copy()
        for j in self.chains[link]:
            p = p + quat_apply(r, j.xyz)
            r = quat_mul(r, j.quat)
            if j.type in ("revolute", "continuous"):
                r = quat_mul(r, quat_from_angle_axis(q[self.index[j.name]], j.axis))
            elif j.type == "prismatic":
                p = p + quat_apply(r, j.axis * q[self.index[j.name]])
        return p, r


# -------------------------------------------------------------------------------------- keypoints


def keypoint_offsets(spec: Spec, object_scales: np.ndarray) -> np.ndarray:
    """(4, 3) observation keypoints in the object frame (reset_utils._keypoint_offsets)."""
    c = np.asarray(spec["keypoint_corners"], dtype=np.float64)
    return c * (np.asarray(object_scales) * spec["object_base_size"] * spec["keypoint_scale"] * 0.5)


def fixed_keypoint_offsets(spec: Spec) -> np.ndarray:
    """(4, 3) success-test keypoints: one fixed size for every object (fixed_size_keypoint_reward)."""
    c = np.asarray(spec["keypoint_corners"], dtype=np.float64)
    return c * (0.5 * spec["keypoint_scale"] * np.asarray(spec["fixed_size"]))


def keypoints_world(pos: np.ndarray, quat_wxyz: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return pos + quat_apply(np.broadcast_to(quat_wxyz, offsets.shape[:-1] + (4,)), offsets)


# ------------------------------------------------------------------------------------ observation


@dataclass
class ObsInputs:
    """What the real loop measures each step. Poses in the env frame, quaternions wxyz."""

    q: np.ndarray            # (27,) joint angles, canonical order
    qd: np.ndarray           # (27,) joint velocities
    prev_targets: np.ndarray  # (27,) the joint targets sent last step
    object_pos: np.ndarray   # (3,)
    object_quat: np.ndarray  # (4,) wxyz
    goal_pos: np.ndarray     # (3,)
    goal_quat: np.ndarray    # (4,) wxyz
    object_scales: np.ndarray  # (3,) grasp-box scale (DexToolBench Object.scale), not metres


def build_obs(spec: Spec, fk: RobotFK, x: ObsInputs, parts: bool = False):
    """The 134-dim policy observation (obs_utils.build_observations, DR off), float32.

    parts=True also returns each field by name, for check_offline.py.
    """
    q = np.asarray(x.q, dtype=np.float64)
    joint_pos = 2.0 * (q - spec.lower) / (spec.upper - spec.lower) - 1.0

    palm_body_pos, palm_rot = fk.link_pose(spec["palm_body"], q)
    palm_pos = palm_body_pos + quat_apply(palm_rot, np.asarray(spec["palm_center_offset"]))

    ft = []
    for link, off in spec.fingertip_obs:
        p, r = fk.link_pose(link, q)
        ft.append(p + quat_apply(r, np.asarray(off)))
    fingertip_rel_palm = np.stack(ft) - palm_pos

    kp = keypoint_offsets(spec, x.object_scales)
    obj_kp = keypoints_world(np.asarray(x.object_pos), np.asarray(x.object_quat), kp)
    goal_kp = keypoints_world(np.asarray(x.goal_pos), np.asarray(x.goal_quat), kp)

    fields = {
        "joint_pos": joint_pos,
        "joint_vel": np.asarray(x.qd, dtype=np.float64),
        "prev_action_targets": np.asarray(x.prev_targets, dtype=np.float64),
        "palm_pos": palm_pos,
        "palm_rot": wxyz_to_xyzw(palm_rot),
        "object_rot": wxyz_to_xyzw(np.asarray(x.object_quat, dtype=np.float64)),
        "fingertip_pos_rel_palm": fingertip_rel_palm.reshape(-1),
        "keypoints_rel_palm": (obj_kp - palm_pos).reshape(-1),
        "keypoints_rel_goal": (obj_kp - goal_kp).reshape(-1),
        "object_scales": np.asarray(x.object_scales, dtype=np.float64),
    }
    obs = np.concatenate([fields[k] for k in spec["obs_list"]])
    assert obs.shape == (NUM_OBS,), obs.shape
    clip = spec["obs_clip"]
    obs = np.clip(obs, -clip, clip).astype(np.float32)
    return (obs, fields) if parts else obs


# ---------------------------------------------------------------------------------------- actions


class ActionPipeline:
    """Policy action in [-1, 1] -> joint position targets (action_utils.apply_action_pipeline, no delay).

    Arm: a velocity command integrated onto the previous target, then smoothed. Hand: an absolute
    position across the joint range, then smoothed. Both clamped to the joint limits.
    """

    def __init__(self, spec: Spec, start_targets: np.ndarray) -> None:
        self.spec = spec
        self.prev = np.asarray(start_targets, dtype=np.float64).copy()
        self.n = spec.n_arm
        self.dt = float(spec["policy_dt"])

    def step(self, action: np.ndarray) -> np.ndarray:
        s, n, prev = self.spec, self.n, self.prev
        a = np.asarray(action, dtype=np.float64)
        lo, hi = s.lower, s.upper
        arm_raw = np.clip(prev[:n] + s["dof_speed_scale"] * self.dt * a[:n], lo[:n], hi[:n])
        ma = s["arm_moving_average"]
        arm = np.clip(ma * arm_raw + (1.0 - ma) * prev[:n], lo[:n], hi[:n])
        hand_raw = lo[n:] + 0.5 * (a[n:] + 1.0) * (hi[n:] - lo[n:])
        mh = s["hand_moving_average"]
        hand = np.clip(mh * hand_raw + (1.0 - mh) * prev[n:], lo[n:], hi[n:])
        self.prev = np.concatenate([arm, hand])
        return self.prev.copy()


# ------------------------------------------------------------------------------- goals and success


def object_collision_corners(urdf_path: str | Path) -> np.ndarray:
    """Bounding-box corners of every collision shape of a one-link object URDF, in its frame.

    goal_table_clearance.collision_corners for boxes and cylinders (as capsules); mesh collisions
    (DexToolBench objects) are bounded by the mesh's vertices' box, read with trimesh.
    """
    import itertools

    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    out = []
    for col in root.iter("collision"):
        o = col.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        g = col.find("geometry")
        box, cyl, mesh = g.find("box"), g.find("cylinder"), g.find("mesh")
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
        if box is not None:
            local = signs * (np.array([float(v) for v in box.get("size").split()]) / 2)
        elif cyl is not None:
            r, length = float(cyl.get("radius")), float(cyl.get("length"))
            local = signs * np.array([r, r, length / 2 + r])
        elif mesh is not None:
            import trimesh

            scale = np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
            m = trimesh.load(urdf_path.parent / mesh.get("filename"), force="mesh")
            lo, hi = m.bounds * scale
            local = np.where(signs > 0, hi, lo)
        else:
            raise ValueError(f"{urdf_path}: unsupported collision geometry")
        out.append(xyz + quat_apply(np.broadcast_to(quat_from_rpy(rpy), (8, 4)), local))
    return np.concatenate(out)


class GoalSampler:
    """Training's goal sequence (reset_utils._reset_goal_pose): the first goal uniform in the target
    volume with a uniform random orientation, every next one the previous goal moved by up to
    delta_goal_distance per axis and turned by up to delta_rotation_degrees about a random axis,
    clamped to the volume, then raised until the object clears the table by goal_table_clearance_m."""

    def __init__(self, spec: Spec, object_corners: np.ndarray, seed: int = 0,
                 table_top_z: float | None = None) -> None:
        self.spec, self.corners = spec, np.asarray(object_corners, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        mins, maxs = np.asarray(spec["target_volume_mins"]), np.asarray(spec["target_volume_maxs"])
        c, h = 0.5 * (mins + maxs), 0.5 * (maxs - mins) * spec["target_volume_region_scale"]
        self.lo, self.hi = c - h, c + h
        self.table_top_z = spec["table_top_z_env"] if table_top_z is None else table_top_z
        self.pos: np.ndarray | None = None
        self.quat: np.ndarray | None = None

    def min_z(self, quat: np.ndarray) -> float:
        lowest = quat_apply(np.broadcast_to(quat, self.corners.shape[:-1] + (4,)), self.corners)[:, 2].min()
        return self.table_top_z + self.spec["goal_table_clearance_m"] - lowest

    def _finish(self, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.spec["use_goal_table_clearance"]:
            pos = pos.copy()
            pos[2] = max(pos[2], self.min_z(quat))
        self.pos, self.quat = pos, quat
        return pos.copy(), quat.copy()

    def first(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.lo + (self.hi - self.lo) * self.rng.random(3)
        return self._finish(pos, random_quat(self.rng))

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.pos is not None, "call first() before next()"
        d = self.spec["delta_goal_distance"]
        pos = np.clip(self.pos + (self.rng.random(3) * 2 - 1) * d, self.lo, self.hi)
        axis = self.rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = (self.rng.random() * 2 - 1) * math.radians(self.spec["delta_rotation_degrees"])
        return self._finish(pos, quat_mul(quat_from_angle_axis(angle, axis), self.quat))


@dataclass
class SuccessTracker:
    """Training's goal test (obs_utils.compute_intermediate_values + termination_utils): the object is
    near the goal when its four fixed-size keypoints are all within tolerance * keypoint_scale of the
    goal's; the goal counts as reached once that has held for success_steps steps (in total, not
    necessarily in a row) since the goal was set."""

    spec: Spec
    tolerance: float | None = None  # metres before keypoint_scale; default: training's final 0.01
    near_steps: int = 0
    kp: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.kp = fixed_keypoint_offsets(self.spec)
        if self.tolerance is None:
            self.tolerance = float(self.spec["success_tolerance_final"])

    def max_dist(self, obj_pos, obj_quat, goal_pos, goal_quat) -> float:
        a = keypoints_world(np.asarray(obj_pos), np.asarray(obj_quat), self.kp)
        b = keypoints_world(np.asarray(goal_pos), np.asarray(goal_quat), self.kp)
        return float(np.linalg.norm(a - b, axis=-1).max())

    def update(self, obj_pos, obj_quat, goal_pos, goal_quat) -> bool:
        """One policy step; True when the goal is reached (the caller then sets the next goal)."""
        near = self.max_dist(obj_pos, obj_quat, goal_pos, goal_quat) <= self.tolerance * self.spec["keypoint_scale"]
        self.near_steps += int(near)
        if self.near_steps >= self.spec["success_steps"]:
            self.near_steps = 0
            return True
        return False


# ----------------------------------------------------------------------------------------- policy


def load_policy(checkpoint: str | Path = DEFAULT_CKPT, expl_coef: float = 0.0, device: str = "cpu"):
    """The SAPG LSTM checkpoint (deploy/joint_reach/core.Policy), one env, deterministic actions.

    expl_coef picks the SAPG block: 0.0 is the last block (task reward only), as viser_play loads it.
    """
    import sys

    sys.path.insert(0, str(REPO))
    from deploy.joint_reach.core import Policy

    return Policy(checkpoint, num_obs=NUM_OBS, num_actions=NUM_ACTIONS, expl_coef=expl_coef, device=device)
