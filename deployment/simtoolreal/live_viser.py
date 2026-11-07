"""Live viser view of a run.py run, in the training env frame: the measured robot, its joint targets as
a see-through ghost, the table, the object's grasp box and the goal's grasp box.

Runs in its own process, so rendering never takes the GIL from aioflexiv's torque loop or the policy
step. The control loop only writes the latest state into shared memory (publish); the viewer polls it.

    viewer = LiveViewer(spec, scales, port=8091)
    viewer.publish(q, targets, obj_pos, obj_quat, goal_pos, goal_quat, t_s, goals, status)  # ~µs
    viewer.close()
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MESH_DIR = HERE.parents[1] / "assets" / "flexiv_wuji"
STATUS = ("moving to the start pose", "running", "done")
# buffer: [version, t_s, goals, status, q (n), targets (n), obj pos 3, obj quat 4 (wxyz), goal pos 3, goal quat 4,
#          hand clearance (m), lowest hand point 3]
HEAD = 4
TAIL = 14 + 4


class LiveViewer:
    def __init__(self, spec, scales: np.ndarray, port: int = 8091, hz: float = 30.0, guard=None,
                 min_hand_height: float = 0.01) -> None:
        self.n = len(spec.joint_names)
        static = dict(
            urdf=str(HERE / spec["robot_urdf"]), names=list(spec.joint_names),
            root_pos=list(spec["robot_root_pos_env"]), root_quat=list(spec["robot_root_quat_wxyz"]),
            table_size=list(spec["table_size"]), table_pos=list(spec["table_pos_env"]),
            box=(np.asarray(scales, float) * float(spec["object_base_size"])).tolist(),
            real_table=None if guard is None else (guard.box_size, guard.box_pos),
            min_hand_height=float(min_hand_height),
        )
        ctx = mp.get_context("spawn")
        self._buf = ctx.Array("d", HEAD + 2 * self.n + TAIL, lock=False)
        self._stop = ctx.Event()
        self._proc = ctx.Process(target=_serve, daemon=True, args=(self._buf, self._stop, static, port, hz))
        self._proc.start()
        self._version = 0

    def publish(self, q, targets, obj_pos, obj_quat, goal_pos, goal_quat, t_s: float, goals: int,
                status: int, clearance: float = np.inf, low_point=(np.nan,) * 3) -> None:
        b, n = self._buf, self.n
        b[1], b[2], b[3] = t_s, goals, status
        b[HEAD:HEAD + 2 * n + TAIL] = np.concatenate(
            [q, targets, obj_pos, obj_quat, goal_pos, goal_quat, [clearance], low_point]).astype(float).tolist()
        self._version += 1
        b[0] = self._version  # written last; a half-written frame is harmless for display

    def close(self) -> None:
        self._stop.set()
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()


def _mesh_files(urdf: str) -> list[str]:
    import xml.etree.ElementTree as ET
    return sorted({m.get("filename") for m in ET.parse(urdf).getroot().iter("mesh")})


def _serve(buf, stop, s: dict, port: int, hz: float) -> None:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    names, n = s["names"], len(s["names"])
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=2.0, height=2.0, cell_size=0.25)
    server.scene.add_frame("/env", axes_length=0.1, axes_radius=0.003)
    root = server.scene.add_frame("/env/robot_root", position=s["root_pos"], wxyz=s["root_quat"], show_axes=False)
    # the URDF's mesh paths are relative to assets/flexiv_wuji/ (scripts/flexiv_wuji/fetch_assets.sh)
    load = lambda: yourdfpy.URDF.load(s["urdf"], mesh_dir=str(MESH_DIR), build_collision_scene_graph=False)
    missing = [m for m in _mesh_files(s["urdf"]) if not (MESH_DIR / m).is_file()]
    if missing:
        print(f"[viser] {len(missing)} robot meshes missing under {MESH_DIR}, e.g. {missing[0]}: "
              "run scripts/flexiv_wuji/fetch_assets.sh", flush=True)
    robot_vis = ViserUrdf(server, load(), root_node_name=root.name + "/robot")
    ghost_vis = ViserUrdf(server, load(), root_node_name=root.name + "/targets",
                          mesh_color_override=(0.15, 0.85, 0.35, 0.3))
    order = [names.index(j) for j in robot_vis.get_actuated_joint_names()]
    if s["real_table"] is not None:  # the real table (guarded) and training's table on it
        server.scene.add_box("/env/real_table", dimensions=s["real_table"][0], position=s["real_table"][1],
                             color=(0.75, 0.72, 0.68), opacity=0.6)
        server.scene.add_box("/env/table", dimensions=s["table_size"], position=np.add(s["table_pos"], [0, 0, 0.001]),
                             color=(0.55, 0.45, 0.35))
    else:
        server.scene.add_box("/env/table", dimensions=s["table_size"], position=s["table_pos"], color=(0.55, 0.45, 0.35))
    low = server.scene.add_icosphere("/env/lowest_hand_point", radius=0.008, color=(0.1, 0.8, 0.2), visible=False)
    obj = server.scene.add_box("/env/object", dimensions=s["box"], color=(0.2, 0.45, 0.9))
    server.scene.add_frame("/env/object/axes", axes_length=0.06, axes_radius=0.002)
    goal = server.scene.add_box("/env/goal", dimensions=s["box"], color=(0.95, 0.6, 0.1), opacity=0.35)
    server.scene.add_frame("/env/goal/axes", axes_length=0.06, axes_radius=0.002)
    with server.gui.add_folder("View"):
        g_ghost = server.gui.add_checkbox("show joint targets (green ghost)", True)
    hud = server.gui.add_markdown("waiting for run.py…")
    print(f"[viser] http://localhost:{port}", flush=True)

    seen, period = -1.0, 1.0 / hz
    while not stop.is_set():
        t0 = time.perf_counter()
        a = np.frombuffer(buf, dtype=np.float64).copy()
        if a[0] != seen and a[0] > 0:
            seen = a[0]
            t_s, goals, status = a[1], int(a[2]), int(a[3])
            q, tg = a[HEAD:HEAD + n], a[HEAD + n:HEAD + 2 * n]
            o = a[HEAD + 2 * n:]
            with server.atomic():
                robot_vis.update_cfg(q[order])
                ghost_vis.show_visual = g_ghost.value
                if g_ghost.value:
                    ghost_vis.update_cfg(tg[order])
                obj.position, obj.wxyz = o[0:3], o[3:7]
                goal.position, goal.wxyz = o[7:10], o[10:14]
                clear, lp = o[14], o[15:18]
                if np.isfinite(clear) and np.all(np.isfinite(lp)):
                    low.position, low.visible = lp, True
                    low.color = (0.9, 0.1, 0.1) if clear < 2 * s["min_hand_height"] else (0.1, 0.8, 0.2)
                else:
                    low.visible = False
            dist = np.linalg.norm(o[0:3] - o[7:10])
            clear_txt = (f"lowest hand point over the table: **{100 * clear:.1f} cm** above it "
                         f"(abort below {100 * s['min_hand_height']:.1f} cm)\n\n") if np.isfinite(clear) else ""
            hud.content = (f"**{STATUS[status]}** · t = {t_s:.1f} s · goals reached **{goals}**\n\n"
                           f"object → goal centre distance {100 * dist:.1f} cm\n\n" + clear_txt +
                           "blue: object grasp box · orange: goal · green ghost: joint targets · "
                           "grey: real table (guarded) · brown: training's table · dot: lowest hand point")
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))
