"""Check utils/fold.py against the fold-pool training code itself, on the CPU, with no Isaac Sim.

The training methods that turn the sheet into the observation and the success test are pulled out of
the training branch (git show), run with torch on a hand-built sheet, and compared with what
utils/fold.py computes from the two plates' poses:

  object   ClothAsRigidObject.fit_rotation_wxyz (+ _mat_to_wxyz), the corners' centroid
  goal     FoldPoolTaskMixin._stationary_frame / _folded_w, ClothTaskMixin.fold_targets_w /
           _drive_goal_marker, the PhysX _goal_quat_from_matrix (relabelled), Play's keypoints
  success  ClothTaskMixin.fold_error, FoldPoolTaskMixin.footprint_ratio

on random sheets (position, yaw, a few degrees of tilt, the moving plate turned 0-180 deg, the two
plates' poses disturbed independently, as two trackers would give them). Then PlateFrames: the plates'
mesh poses through random box symmetries must come back as the right sheet frames, at the start and
frame to frame. Exits non-zero on any mismatch.

    .venv_deploy/bin/python deploy/simtoolreal/checks/check_fold.py
"""

from __future__ import annotations

import ast
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # deploy/simtoolreal
sys.path.insert(0, str(HERE))
from export_spec import git_show                                                         # noqa: E402
from utils.config import repo_path, load_config                                          # noqa: E402
from utils.fold import (BOX_SYMMETRIES, FoldGeometry, PlateFrames, mat_to_wxyz, mesh_to_sheet,   # noqa: E402
                        quat_to_mat, rotation_angle)
from utils.observation import keypoints_world                                            # noqa: E402
from utils.spec import load_spec                                                         # noqa: E402

REF = "origin/yiboc_wuji"
CLOTH = "isaacsimenvs/tasks/cloth"


def method(path: str, cls: str, name: str, glb: dict):
    """The source of cls.name in the branch's file, as a plain function (globals glb)."""
    src = git_show(REF, path)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    code = textwrap.dedent(ast.get_source_segment(src, item))
                    ns = dict(glb)
                    exec(compile(code, f"{REF}:{path}:{cls}.{name}", "exec"), ns)
                    return ns[name]
    raise KeyError(f"{path}: {cls}.{name}")


def module_function(path: str, name: str, glb: dict):
    src = git_show(REF, path)
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = dict(glb)
            exec(compile(ast.get_source_segment(src, node), f"{REF}:{path}:{name}", "exec"), ns)
            return ns[name]
    raise KeyError(f"{path}: {name}")


T = {"torch": torch}
MAT_TO_WXYZ = module_function(f"{CLOTH}/utils/cloth_adapter.py", "_mat_to_wxyz", T)


def quat_from_matrix_wxyz(r):
    """Stand-in for Isaac Lab 2.3's quat_from_matrix (wxyz). Its sign never reaches the observation:
    the goal enters only through keypoints, and q and -q place them alike."""
    return MAT_TO_WXYZ(r)


class TrainingSheet:
    """The training env's fold quantities for one env, from its particles, by the branch's own methods."""

    def __init__(self, geom: FoldGeometry, spec: dict):
        f = spec["fold"]
        g = {"torch": torch, "quat_from_matrix": quat_from_matrix_wxyz}
        self.device = "cpu"
        self.num_envs = 1
        self.cfg = types.SimpleNamespace(cloth=types.SimpleNamespace(fold_axis=f["fold_axis"]))
        # the rest geometry, as ClothTaskMixin._init_fold_targets + FoldPoolTaskMixin._init_fold_targets
        # leave it (per env, scaled in-plane); geom's are built the same way, checked in check_rest()
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)                    # noqa: E731
        self._stationary_idx = torch.tensor(f["stationary_idx"])
        self._kp_idx_t = torch.tensor(f["corner_idx"])
        st = geom.grid[np.asarray(f["stationary_idx"])]
        self._st_rest_centroid = t(st.mean(axis=0))[None]
        self._st_rest_centred = t(st - st.mean(axis=0))[None]
        self._kp_folded_rest = t(geom.kp_folded_rest)[None]
        self._env_size = torch.tensor([geom.size])
        self._corner_rest = t(geom.grid[np.asarray(f["corner_idx"])])
        self._corner_idx = self._kp_idx_t
        self.goal_viz = types.SimpleNamespace(write_root_pose_to_sim=self._capture)
        bind = lambda fn: types.MethodType(fn, self)                                       # noqa: E731
        self._stationary_frame = bind(method(f"{CLOTH}/fold_pool_task.py", "FoldPoolTaskMixin", "_stationary_frame", g))
        self._folded_w = bind(method(f"{CLOTH}/fold_pool_task.py", "FoldPoolTaskMixin", "_folded_w", g))
        self.footprint_ratio = bind(method(f"{CLOTH}/fold_pool_task.py", "FoldPoolTaskMixin", "footprint_ratio", g))
        self.fold_targets_w = bind(method(f"{CLOTH}/cloth_task.py", "ClothTaskMixin", "fold_targets_w", g))
        self.fold_error = bind(method(f"{CLOTH}/cloth_task.py", "ClothTaskMixin", "fold_error", g))
        self._drive_goal_marker = bind(method(f"{CLOTH}/cloth_task.py", "ClothTaskMixin", "_drive_goal_marker", g))
        self._goal_quat_from_matrix = bind(method(f"{CLOTH}/physx_envs.py", "_PhysxClothMixin", "_goal_quat_from_matrix", g))
        g_adapter = {"torch": torch, "_mat_to_wxyz": MAT_TO_WXYZ}
        self.fit_rotation_wxyz = bind(method(f"{CLOTH}/utils/cloth_adapter.py", "ClothAsRigidObject",
                                             "fit_rotation_wxyz", g_adapter))

    def _capture(self, pose):
        self.goal_pose = pose[0].double().numpy()

    def set(self, particles: np.ndarray):
        self.parts = torch.tensor(particles, dtype=torch.float32)[None]

    # what the methods read
    def _particles_w(self):
        return self.parts

    def _particles(self):
        return self.parts

    def cloth_keypoints_w(self):
        return self.parts[:, self._kp_idx_t, :]


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def small_rotation(rng, deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.radians(deg) * rng.uniform(-1, 1)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


class Checker:
    def __init__(self):
        self.failures = 0

    def close(self, what, a, b, tol):
        err = float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
        if err > tol:
            self.failures += 1
            print(f"FAIL {what}: max |diff| {err:.2e} > {tol:.0e}\n  deploy   {np.round(a, 5).tolist()}\n  training {np.round(b, 5).tolist()}")
        return err


def main() -> None:
    cfg = load_config(["task=fold"])
    spec = load_spec(repo_path(cfg.spec_path))
    m2s = mesh_to_sheet(cfg.fold.hinge_axis_mesh)
    rng = np.random.default_rng(0)
    ck = Checker()

    # 1. _mat_to_wxyz, branch for branch (object_rot carries its sign)
    worst = 0.0
    for i in range(2000):
        r = quat_to_mat(rng.normal(size=4))
        if i % 4 == 0:   # near 180 deg (a finished fold), where the trace branch switches
            r = r @ np.diag([-1.0, 1.0, -1.0]) @ small_rotation(rng, 2)
        ref = MAT_TO_WXYZ(torch.tensor(r, dtype=torch.float64)[None])[0].numpy()
        worst = max(worst, ck.close("mat_to_wxyz", mat_to_wxyz(r), ref, 1e-9))
    print(f"mat_to_wxyz: 2000 rotations, max |diff| {worst:.1e}")

    # 2. the observation and the success test on random sheets, against the training methods
    for size, hinge_z in ((cfg.fold.sheet_size, 0.0), (cfg.fold.sheet_size, cfg.fold.hinge_z), (0.16, 0.0)):
        geom = FoldGeometry(spec, size, hinge_z)
        train = TrainingSheet(geom, spec)
        worst = {}
        for _ in range(300):
            origin = np.asarray(spec["fold"]["sheet_centre_env"]) + np.r_[rng.uniform(-0.1, 0.1, 2), 0.0025]
            r_sheet = rot_z(rng.uniform(-np.pi, np.pi)) @ small_rotation(rng, 3)
            fold = rng.uniform(0, np.pi)
            plates = geom.plate_poses(origin, r_sheet, fold, m2s)
            # two trackers: each plate a little off, independently
            frames = {}
            for which, (c, q) in plates.items():
                frames[which] = (c + rng.normal(scale=0.002, size=3),
                                 quat_to_mat(q) @ small_rotation(rng, 2) @ m2s)
            o = geom.observe(frames["moving"], frames["stationary"])
            sc = geom.score(o, frames["stationary"])
            train.set(o["particles"])
            # object: the corners' centroid and the Kabsch fit
            obj_quat = train.fit_rotation_wxyz()[0].double().numpy()
            kp = train.cloth_keypoints_w()[0].double().numpy()
            worst["object_pos"] = max(worst.get("object_pos", 0), ck.close("object_pos", o["object_pos"], kp.mean(0), 1e-5))
            worst["object_quat"] = max(worst.get("object_quat", 0), ck.close("object_quat", o["object_quat"], obj_quat, 1e-4))
            # goal: the marker _drive_goal_marker writes (relabelled quaternion), then Play's keypoints
            train._drive_goal_marker()
            g = train.goal_pose
            offsets = geom.corner_offsets
            goal_kp = keypoints_world(g[:3], g[3:7] / np.linalg.norm(g[3:7]), offsets)
            obj_kp = keypoints_world(kp.mean(0), obj_quat, offsets)
            worst["goal_pos"] = max(worst.get("goal_pos", 0), ck.close("goal_pos", o["goal_pos"], g[:3], 1e-5))
            worst["keypoints_rel_goal"] = max(worst.get("keypoints_rel_goal", 0), ck.close(
                "keypoints_rel_goal", o["object_kp"] - o["goal_kp"], obj_kp - goal_kp, 1e-4))
            worst["object_kp"] = max(worst.get("object_kp", 0), ck.close("object_kp", o["object_kp"], obj_kp, 1e-5))
            # success: training's fold_error (its target) and footprint
            worst["fold_error"] = max(worst.get("fold_error", 0), ck.close(
                "fold_error", sc["fold_error"], train.fold_error()[0].item(), 1e-5))
            worst["footprint"] = max(worst.get("footprint", 0), ck.close(
                "footprint", sc["footprint"], train.footprint_ratio()[0].item(), 1e-5))
        print(f"sheet {100 * size:.0f} cm, hinge_z {1000 * hinge_z:.1f} mm, 300 random sheets: "
              + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))

    # 3. the folds the success test must call: flat (no), folded at the real hinge (yes)
    geom = FoldGeometry(spec, cfg.fold.sheet_size, cfg.fold.hinge_z)
    origin = np.asarray(spec["fold"]["sheet_centre_env"]) + [0.0, 0.0, 0.0025]
    for deg, want in ((0, False), (90, False), (160, False), (170, True), (180, True)):   # 1.5 cm: 170 is in
        plates = geom.plate_poses(origin, rot_z(0.7), np.radians(deg), m2s)
        frames = {w: (c, quat_to_mat(q) @ m2s) for w, (c, q) in plates.items()}
        sc = geom.score(geom.observe(frames["moving"], frames["stationary"]), frames["stationary"])
        ok = sc["folded"] == want
        ck.failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} fold {deg:3d} deg: folded={sc['folded']} (want {want}), error at the real "
              f"hinge {100 * sc['fold_error_real']:.2f} cm, at training's {100 * sc['fold_error']:.2f} cm, "
              f"footprint {sc['footprint']:.2f}")

    # 4. PlateFrames: mesh poses through random box symmetries -> the right sheet frames
    starts = updates = 0
    for _ in range(200):
        r_sheet = rot_z(rng.uniform(-np.pi, np.pi)) @ small_rotation(rng, 3)
        plates = geom.plate_poses(origin, r_sheet, 0.0, m2s)
        truth = {w: quat_to_mat(q) @ m2s for w, (c, q) in plates.items()}

        def seen(ps):   # each tracker reports any of the box's symmetric poses
            return {w: (c, mat_to_wxyz(quat_to_mat(q) @ BOX_SYMMETRIES[rng.integers(4)])) for w, (c, q) in ps.items()}

        pf = PlateFrames(geom, m2s, cfg.fold.max_hinge_gap_m)
        s = seen(plates)
        f = pf.start(s["moving"], s["stationary"])
        bad = max(rotation_angle(f[w][1], truth[w]) for w in truth)
        starts += bad < 1e-6
        # then fold it over in 2 deg steps, the trackers flipping at random
        for deg in range(0, 181, 2):
            plates = geom.plate_poses(origin, r_sheet, np.radians(deg), m2s)
            truth = {w: quat_to_mat(q) @ m2s for w, (c, q) in plates.items()}
            s = seen(plates)
            f = pf.update(s["moving"], s["stationary"])
            bad = max(bad, max(rotation_angle(f[w][1], truth[w]) for w in truth))
        updates += bad < 1e-6
    ok = starts == 200 and updates == 200
    ck.failures += not ok
    print(f"{'ok  ' if ok else 'FAIL'} PlateFrames: start right {starts}/200, through a 0-180 deg fold with "
          f"random symmetric flips {updates}/200")

    # 5. PlateFrames.start refuses plates that are not side by side
    plates = geom.plate_poses(origin, np.eye(3), 0.0, m2s)
    far = (plates["stationary"][0] + [0.0, 0.08, 0.0], plates["stationary"][1])
    try:
        PlateFrames(geom, m2s, cfg.fold.max_hinge_gap_m).start(plates["moving"], far)
        ck.failures += 1
        print("FAIL PlateFrames.start accepted plates 8 cm apart")
    except SystemExit as e:
        print(f"ok   PlateFrames.start refuses plates 8 cm apart: {e}")

    print("all fold checks passed" if not ck.failures else f"{ck.failures} FAILED")
    sys.exit(1 if ck.failures else 0)


if __name__ == "__main__":
    main()
