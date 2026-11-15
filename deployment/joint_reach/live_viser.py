"""Live viser view of a real-robot joint-reach run: the measured robot, its goal pose as a see-through
ghost, and the goal fingertip positions with a red line to the measured ones.

Runs in its own process, so rendering never takes the GIL from aioflexiv's torque loop or the policy
step. The control loop only writes the latest joints into shared memory (publish); the viewer polls it.

    viewer = LiveViewer(urdf, spec.joint_names, n_arm, arm_tol, hand_tol, n_goals, port=8090)
    viewer.publish(q, goal, k, phase)   # every step, ~µs
    viewer.close()
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

GHOST_RGBA = (0.15, 0.85, 0.35, 0.35)
TIP_LINKS = ("l_thumb_tip", "l_index_finger_tip", "l_middle_finger_tip", "l_ring_finger_tip", "l_pinky_tip")
PHASES = ("moving to the start pose", "running the policy", "done")


class LiveViewer:
    def __init__(self, urdf: str | Path, joint_names: list[str], n_arm: int, arm_tol: float, hand_tol: float,
                 n_goals: int, port: int, hz: float = 30.0, ignored: list[int] = ()) -> None:
        self.n = len(joint_names)
        ctx = mp.get_context("spawn")
        # [version, goal index, phase, reached, scored, q (n), goal (n)]
        self._buf = ctx.Array("d", 5 + 2 * self.n, lock=False)
        self._stop = ctx.Event()
        self._proc = ctx.Process(
            target=_serve, daemon=True,
            args=(self._buf, self._stop, str(urdf), list(joint_names), n_arm, arm_tol, hand_tol, n_goals, port, hz,
                  list(ignored)))
        self._proc.start()
        self._version = 0

    def publish(self, q: np.ndarray, goal: np.ndarray, goal_idx: int, phase: int,
                reached: int = 0, scored: int = 0) -> None:
        b, n = self._buf, self.n
        b[1], b[2], b[3], b[4] = goal_idx, phase, reached, scored
        b[5:5 + n] = q.tolist()
        b[5 + n:5 + 2 * n] = goal.tolist()
        self._version += 1
        b[0] = self._version  # written last: the viewer may still see a half-written frame, harmless for display

    def close(self) -> None:
        self._stop.set()
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()


def _serve(buf, stop, urdf: str, names: list[str], n_arm: int, arm_tol: float, hand_tol: float, n_goals: int,
           port: int, hz: float, ignored: list[int]) -> None:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    n = len(names)
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=2.0, height=2.0, cell_size=0.25)
    fk = yourdfpy.URDF.load(urdf, build_scene_graph=True, load_meshes=False)
    robot_vis = ViserUrdf(server, Path(urdf), root_node_name="/robot")
    ghost_vis = ViserUrdf(server, Path(urdf), root_node_name="/goal", mesh_color_override=GHOST_RGBA)
    order = [names.index(j) for j in robot_vis.get_actuated_joint_names()]
    fk_order = [names.index(j) for j in fk.actuated_joint_names]
    tip_goal = [server.scene.add_icosphere(f"/tips/goal_{t}", radius=0.008, color=(0.1, 0.8, 0.3))
                for t in TIP_LINKS]
    tip_line = server.scene.add_line_segments(
        "/tips/err", points=np.zeros((5, 2, 3), np.float32), colors=(1.0, 0.2, 0.2), line_width=3.0)
    with server.gui.add_folder("View"):
        g_ghost = server.gui.add_checkbox("show goal ghost", True)
        g_tips = server.gui.add_checkbox("show fingertip goals", True)
    hud = server.gui.add_markdown("waiting for the robot…")
    print(f"[jr_viser] http://localhost:{port}", flush=True)

    seen, period = -1.0, 1.0 / hz
    while not stop.is_set():
        t0 = time.perf_counter()
        a = np.frombuffer(buf, dtype=np.float64).copy()
        if a[0] != seen and a[0] > 0:
            seen = a[0]
            k, phase, reached, scored = int(a[1]), int(a[2]), int(a[3]), int(a[4])
            q, goal = a[5:5 + n], a[5 + n:5 + 2 * n]
            err = np.abs(goal - q)
            err[ignored] = 0.0  # disabled joints are not scored
            arm_e, hand_e = err[:n_arm].max(), err[n_arm:].max()
            worst = int(np.argmax(err))
            with server.atomic():
                robot_vis.update_cfg(q[order])
                ghost_vis.show_visual = g_ghost.value
                if g_ghost.value:
                    ghost_vis.update_cfg(goal[order])
                tips = []
                for cfg in (goal, q):
                    fk.update_cfg(cfg[fk_order])
                    tips.append([fk.get_transform(t, fk.base_link)[:3, 3] for t in TIP_LINKS])
                for h, p in zip(tip_goal, tips[0]):
                    h.position, h.visible = p, g_tips.value
                tip_line.points = np.stack([tips[1], tips[0]], axis=1).astype(np.float32)
                tip_line.visible = g_tips.value
            in_tol = arm_e < arm_tol and hand_e < hand_tol
            hud.content = (
                f"**{PHASES[phase]}** · goal **{k + 1}/{n_goals}**\n\n"
                f"arm max err **{math.degrees(arm_e):.2f}°** (tol {math.degrees(arm_tol):.1f}) · "
                f"hand max err **{math.degrees(hand_e):.2f}°** (tol {math.degrees(hand_tol):.1f}) · "
                f"{'✅ in tol' if in_tol else '… moving'}\n\n"
                f"worst joint: {names[worst]} {math.degrees(err[worst]):.2f}°\n\n"
                f"reached {reached}/{scored} goals so far"
            )
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))
