"""View checks/record_lift.py's rollouts in viser, side by side, before sending one to the robot.

    .venv_deploy/bin/python deploy/simtoolreal/checks/view_lifts.py <out dir> [<out dir> ...] \
        [--measured T_base_obj.npy] [--port 8092]

Each <out dir> holds an openloop.npz. Pick one in the panel, then what to play:
  - "open loop replay": the saved targets sent with no policy, in sim, from the same start. This is
    what robot.mode=openloop does on the real robot (the arm as the sim had it, the eraser as the sim
    moved it).
  - "policy rollout": the closed-loop policy run the targets came from.
Robot: solid = measured sim joints; green ghost = the targets sent. Eraser: grey = the sim's,
orange = the goal, red = where FoundationPose saw the real one (--measured, a base-frame 4x4).
--camera (default: the Nov 7 extrinsics) draws the RealSense's 640 x 480 view (the centre 4:3 of its
1280 x 720 calibration) and says how much of the rollout the goal and the eraser stay inside it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from utils.config import REPO                    # noqa: E402
from utils.spec import base_to_env, load_spec    # noqa: E402

MESH = HERE.parent / "objects" / "flat_eraser" / "flat_eraser.obj"
MESH_DIR = REPO / "assets" / "flexiv_wuji"
CAMERA = REPO / "deploy" / "calibration" / "extrinsics" / "left_camera_nov7.json"
STREAM_WH = (640, 480)        # perception's stream: the 4:3 centre of the calibrated 16:9 image


def camera_view(extrinsics: Path, spec: dict):
    """(T_env_cam 4x4, vertical fov rad, aspect, in_view(p_env (N,3), margin_px) -> (N,) bool)."""
    from utils.spec import env_to_base
    e = json.loads(extrinsics.read_text())
    T_base_cam, K = np.array(e["T_base_cam"]), e["intrinsics"]
    T_cam_base = np.linalg.inv(T_base_cam)
    s = K["height"] / STREAM_WH[1]                       # 720 -> 480: the stream is scaled, then cropped
    half_w = 0.5 * STREAM_WH[0] * s                      # px of the calibrated image, about cx

    def in_view(p_env: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        pb = np.stack([env_to_base(spec, p, np.array([1.0, 0, 0, 0]))[0] for p in p_env])
        pc = pb @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
        u = K["fx"] * pc[:, 0] / pc[:, 2] + K["cx"]
        v = K["fy"] * pc[:, 1] / pc[:, 2] + K["cy"]
        m = margin_px * s
        return ((pc[:, 2] > 0.1) & (np.abs(u - K["cx"]) < half_w - m)
                & (v > m) & (v < K["height"] - m))

    pos, q = base_to_env(spec, T_base_cam[:3, 3], mat_to_wxyz(T_base_cam[:3, :3]))
    T = np.eye(4)
    from scipy.spatial.transform import Rotation
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = pos
    fov = 2 * np.arctan(0.5 * K["height"] / K["fy"])
    return T, fov, STREAM_WH[0] / STREAM_WH[1], in_view


def mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def load(d: Path) -> dict:
    z = np.load(d / "openloop.npz")
    meta = json.loads(str(z["meta"]))
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = meta
    out["name"] = d.name
    return out


def main() -> None:
    import trimesh
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--measured", type=Path, help="FoundationPose T_base_obj (4x4 .npy) of the real eraser")
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--camera", type=Path, default=CAMERA, help="extrinsics json (T_base_cam, intrinsics); 'none': off")
    args = p.parse_args()

    spec = load_spec(REPO / "deploy/simtoolreal/spec/spec_flexiv_wuji_right_hl.json")
    rolls = {}
    for d in args.dirs:
        if (d / "openloop.npz").exists():
            name = f"{d.parent.name}/{d.name}" if d.name.startswith("top") else d.name   # record_lift --save_top
            rolls[name] = load(d)
            rolls[name]["name"] = name
        else:
            print(f"[view] {d}: no openloop.npz (no clean lift), skipped", flush=True)
    if not rolls:
        raise SystemExit("[view] nothing to show")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=2.0, height=2.0, cell_size=0.25)
    server.scene.add_box("/table", dimensions=spec["table_size"], position=spec["table_pos_env"],
                         color=(0.55, 0.45, 0.35))
    root = server.scene.add_frame("/robot_root", position=spec["robot_root_pos_env"],
                                  wxyz=spec["robot_root_quat_wxyz"], show_axes=False)

    def urdf():
        return yourdfpy.URDF.load(spec["urdf_path"], mesh_dir=str(MESH_DIR), build_collision_scene_graph=False)

    robot = ViserUrdf(server, urdf(), root_node_name=root.name + "/q")
    ghost = ViserUrdf(server, urdf(), root_node_name=root.name + "/targets",
                      mesh_color_override=(0.15, 0.85, 0.35, 0.3))
    order = [list(spec["joint_names"]).index(j) for j in robot.get_actuated_joint_names()]
    mesh = trimesh.load(MESH, force="mesh")
    obj = server.scene.add_mesh_simple("/sim_object", vertices=mesh.vertices, faces=mesh.faces, color=(0.6, 0.6, 0.6))
    goal = server.scene.add_mesh_simple("/goal", vertices=mesh.vertices, faces=mesh.faces,
                                        color=(0.95, 0.6, 0.1), opacity=0.4)
    if args.measured is not None:
        T = np.load(args.measured)
        pos, q = base_to_env(spec, T[:3, 3], mat_to_wxyz(T[:3, :3]))
        server.scene.add_mesh_simple("/real_eraser", vertices=mesh.vertices, faces=mesh.faces,
                                     color=(0.9, 0.15, 0.15), opacity=0.5, position=pos, wxyz=q)
    in_view = None
    if str(args.camera) != "none":
        T_cam, fov, aspect, in_view = camera_view(args.camera, spec)
        server.scene.add_camera_frustum("/camera", fov=fov, aspect=aspect, scale=0.15, color=(0.2, 0.5, 1.0),
                                        position=T_cam[:3, 3], wxyz=mat_to_wxyz(T_cam[:3, :3]))
        for r in rolls.values():      # every 5th step, eraser centre at least 40 px (stream) inside the edge
            r["goal_in_view"] = in_view(r["goals_pos"][::5] if "goals_pos" in r
                                        else np.asarray(r["goal_pos"])[None], 40).mean()
            r["obj_in_view"] = {k: in_view(r[k][::5], 40).mean() for k in ("replay_object_pos", "object_pos")}
    pad = server.scene.add_box("/pad", dimensions=(0.30, 0.40, 0.01), position=(0, 0, 0), color=(0.15, 0.15, 0.15))

    which = server.gui.add_dropdown("rollout", tuple(rolls))
    mode = server.gui.add_dropdown("play", ("open loop replay", "policy rollout"))
    t_slider = server.gui.add_slider("step", 0, 599, 1, 0)
    playing = server.gui.add_checkbox("playing", True)
    speed = server.gui.add_slider("speed", 0.1, 2.0, 0.1, 1.0)
    info = server.gui.add_markdown("")

    def current():
        r = rolls[which.value]
        if mode.value == "open loop replay":
            return r, r["replay_q"], r["replay_object_mesh_pos"], r["replay_object_quat"]
        return r, r["q"], r["object_mesh_pos"], r["object_quat"]

    def refresh_info(_=None) -> None:
        r, q, _, _ = current()
        m = r["meta"]
        scr = next(s for s in m["open_loop_screen"] if s["env"] == m["chosen_env"])
        t_slider.max = len(q) - 1
        top = m.get("table_top_env", 0.28)
        pad.position = (r["start_pos"][0], r["start_pos"][1], top - 0.005)
        pad.visible = top > 0.285
        lift = r["replay_object_pos"][:, 2] - r["start_pos"][2]
        goals = ""
        if "goals_pos" in r:
            n_goals = 1 + int(np.any(np.diff(r["goals_pos"], axis=0) != 0, axis=1).sum())
            goals = (f"  \ntraining goals: {scr.get('goals_reached', '?')} reached, {n_goals} shown; "
                     f"lowest goal {100 * (r['goals_pos'][:, 2].min() - r['start_pos'][2]):.1f} cm above the start; "
                     f"{len(q) * float(r['dt']):.0f} s")
        view = ""
        if in_view is not None:
            k = "replay_object_pos" if mode.value == "open loop replay" else "object_pos"
            view = (f"  \nin camera view (blue): goal {100 * r['goal_in_view']:.0f} %, eraser "
                    f"{100 * r['obj_in_view'][k]:.0f} % of the time; highest goal "
                    f"{100 * (np.max(r['goals_pos'][:, 2]) - r['start_pos'][2]) if 'goals_pos' in r else 0:.0f} cm "
                    f"above the start")
        info.content = (
            f"**{r['name']}** — {Path(m['checkpoint']).stem}, delays {'on' if m.get('delays') else 'off'}  \n"
            f"policy: {m['envs_clean']}/{m['num_envs']} envs lift and hold without a drop "
            f"({m['envs_holding']} hold)  \n"
            f"chosen env {m['chosen_env']}: lifted at {scr['first_lift_s']:.1f} s; open loop in sim holds "
            f"{scr['open_loop_holds']}/{scr['of']}  \n"
            f"shown replay: {'holds' if bool(r['replay_holds']) else 'DOES NOT hold'}, "
            f"lift at end {100 * lift[-1]:.1f} cm (goal {100 * m['lift_m']:.0f})" + goals + view)

    which.on_update(refresh_info)
    mode.on_update(refresh_info)
    refresh_info()
    print(f"[view] http://localhost:{args.port}  ({len(rolls)} rollouts: {', '.join(rolls)})", flush=True)

    last = time.time()
    t_float = 0.0
    while True:
        r, q, opos, oquat = current()
        dt = float(r["dt"])
        now = time.time()
        if playing.value:
            t_float = (t_float + (now - last) * speed.value / dt) % len(q)
            t_slider.value = int(t_float)
        else:
            t_float = float(t_slider.value)
        last = now
        k = min(int(t_slider.value), len(q) - 1)
        robot.update_cfg(q[k][order])
        ghost.update_cfg(r["targets"][k][order])
        obj.position, obj.wxyz = opos[k], oquat[k]
        if "goals_mesh_pos" in r:
            goal.position, goal.wxyz = r["goals_mesh_pos"][k], r["goals_quat"][k]
        else:
            goal.position, goal.wxyz = r["goal_mesh_pos"], r["goal_quat"]
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
