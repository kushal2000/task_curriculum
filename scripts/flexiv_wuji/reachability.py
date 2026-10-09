"""Can each robot put its palm where the Play task needs it? Kuka + Sharpa vs Flexiv + Wuji.

Grid-samples the task's goal volume (``reset.target_volume_*``) and a layer just above the table
where objects rest, and solves IK for the policy's palm centre (``RobotSpec.palm_body_name`` +
``palm_center_offset``) at each point, with the robot base where the env puts it. Two conditions:

  * ``position``   palm centre within 5 mm, any orientation
  * ``palm_down``  the same, with the canonical palm normal (hand_frames.py) within 30 deg of -z,
                   the top-grasp case

Reports the reachable fraction per region and condition, for both robots, from several IK starts
(the default pose plus random ones). Unreachable points are listed for the Flexiv.

    .venv_isaaclab3/bin/python scripts/flexiv_wuji/reachability.py [--out report.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yourdfpy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flexiv_wuji as fw  # noqa: E402
from hand_frames import HAND_KEYPOINTS, hand_frame, hand_points  # noqa: E402

sys.path.insert(0, str(fw.REPO_ROOT / "isaacsimenvs" / "tasks" / "play"))
import robots  # noqa: E402  (stdlib-only; imported by path so Isaac Lab is not needed)

#: Play.yaml reset.* values.
TARGET_MINS, TARGET_MAXS = (-0.35, -0.2, 0.6), (0.35, 0.2, 0.95)
#: Objects spawn at the table centre +- reset_position_noise_{x,y} = 0.1 and drop onto the top
#: (z 0.53); a palm centre about 4 cm above it is where a grasp starts.
TABLE_LAYER = dict(x=(-0.1, 0.1), y=(-0.1, 0.1), z=0.57)
POS_TOL, ANGLE_TOL_DEG = 0.005, 30.0


class ArmChain:
    """Vectorisable FK from the robot base to the palm body, plus constant tool offsets."""

    def __init__(self, urdf: yourdfpy.URDF, spec: robots.RobotSpec, hand_key: str):
        self.names = list(spec.arm_joint_names)
        chain, link = [], spec.palm_body_name
        while link != urdf.base_link:
            joint = next(j for j in urdf.robot.joints if j.child == link)
            chain.append(joint)
            link = joint.parent
        self.joints = chain[::-1]
        self.lo = np.array([urdf.joint_map[n].limit.lower for n in self.names])
        self.hi = np.array([urdf.joint_map[n].limit.upper for n in self.names])
        self.q_default = np.array([spec.arm_default_joint_pos[n] for n in self.names])
        # Constant offsets in the palm body frame, measured with the hand open (all zeros).
        urdf.update_cfg(np.array([spec.arm_default_joint_pos.get(n, 0.0) for n in urdf.actuated_joint_names]))
        T_body = urdf.get_transform(spec.palm_body_name, urdf.base_link)
        self.palm = np.array([*spec.palm_center_offset, 1.0])[:3]
        H = np.linalg.inv(T_body) @ hand_frame(hand_points(urdf, HAND_KEYPOINTS[hand_key]))
        self.palm_normal = H[:3, 2]  # canonical z: out of the palm
        self.base = np.array(fw.ROBOT_BASE_POS)

    def fk(self, q: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        qi = dict(zip(self.names, q))
        for j in self.joints:
            T = T @ j.origin
            if j.type != "fixed":
                R = np.eye(4)
                R[:3, :3] = Rotation.from_rotvec(np.asarray(j.axis) * qi[j.name]).as_matrix()
                T = T @ R
        return T

    def palm_pose(self, q):
        T = self.fk(q)
        return self.base + T[:3, :3] @ self.palm + T[:3, 3], T[:3, :3] @ self.palm_normal


def solve(chain: ArmChain, target: np.ndarray, palm_down: bool, starts) -> bool:
    down = np.array([0.0, 0.0, -1.0])

    def residual(q):
        p, n = chain.palm_pose(q)
        r = [p - target]
        if palm_down:
            # Penalise only the part of the tilt beyond the tolerance cone.
            excess = max(0.0, np.arccos(np.clip(n @ down, -1, 1)) - np.radians(ANGLE_TOL_DEG))
            r.append([0.1 * excess])
        return np.concatenate(r)

    for q0 in starts:
        sol = least_squares(residual, q0, bounds=(chain.lo, chain.hi), max_nfev=200)
        p, n = chain.palm_pose(sol.x)
        ok = np.linalg.norm(p - target) < POS_TOL
        if palm_down:
            ok &= np.degrees(np.arccos(np.clip(n @ down, -1, 1))) <= ANGLE_TOL_DEG + 1.0
        if ok:
            return True
    return False


def grid() -> dict[str, list[np.ndarray]]:
    axes = [np.linspace(a, b, n) for a, b, n in zip(TARGET_MINS, TARGET_MAXS, (7, 5, 4))]
    table = [
        np.array([x, y, TABLE_LAYER["z"]])
        for x, y in itertools.product(np.linspace(*TABLE_LAYER["x"], 5), np.linspace(*TABLE_LAYER["y"], 5))
    ]
    return {"goal_volume": [np.array(p) for p in itertools.product(*axes)], "table_layer": table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--starts", type=int, default=4, help="IK starts: default pose + random")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    chains = {
        "kuka_sharpa": ArmChain(yourdfpy.URDF.load(str(fw.KUKA_SHARPA_URDF)), robots.KUKA_SHARPA, "kuka_sharpa"),
        "flexiv_wuji": ArmChain(yourdfpy.URDF.load(str(fw._require(fw.COMPOSED_URDF))), robots.FLEXIV_WUJI, "flexiv_wuji"),
    }
    points = grid()
    report: dict = {"pos_tol_m": POS_TOL, "palm_down_tol_deg": ANGLE_TOL_DEG, "robots": {}}
    for name, chain in chains.items():
        starts = [chain.q_default] + [rng.uniform(chain.lo, chain.hi) for _ in range(args.starts - 1)]
        out = {}
        for region, pts in points.items():
            for cond in ("position", "palm_down"):
                ok = [solve(chain, p, cond == "palm_down", starts) for p in pts]
                out[f"{region}/{cond}"] = {
                    "reachable": f"{sum(ok)}/{len(ok)}",
                    "unreachable": [[round(float(v), 3) for v in p] for p, k in zip(pts, ok) if not k],
                }
                print(f"{name:12s} {region:12s} {cond:10s} {sum(ok)}/{len(ok)}", flush=True)
        report["robots"][name] = out
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
