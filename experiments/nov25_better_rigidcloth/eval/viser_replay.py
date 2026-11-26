"""Replay recorded VBD rollouts (render_vbd.py --record) in a browser with viser: orbit, zoom, scrub.

    .venv/isaaclab3/bin/python experiments/nov25_better_rigidcloth/eval/viser_replay.py outputs [--port 9091]

Takes .npz files or directories of them; pick the episode in the GUI. Draws the robot and table (their
collision geometry, as the solver sees them) and the cloth (moving half orange, stationary half blue).
Only numpy + viser are needed: no Isaac Sim, Isaac Lab or Newton.
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import viser

import os

STAGE = int(os.environ.get("REPLAY_STAGE", "9"))  # debugging: 1 GUI only, 2 + meshes, 3 + cloth, 4+ playback
MOVING_RGB = STATIONARY_RGB = (88, 150, 214)  # one colour for the whole sheet


TABLE_RGB, FLOOR_RGB = (176, 132, 94), (205, 207, 212)


def catmull_rom_weights(n: int, k: int) -> np.ndarray:
    """(M, n) weights sampling a 1-D Catmull-Rom spline through n points at k samples per interval."""
    ts = np.linspace(0.0, n - 1.0, k * (n - 1) + 1)
    W = np.zeros((len(ts), n))
    for a, t in enumerate(ts):
        i = min(int(np.floor(t)), n - 2)
        u = t - i
        c = (-0.5 * u ** 3 + u ** 2 - 0.5 * u, 1.5 * u ** 3 - 2.5 * u ** 2 + 1.0,
             -1.5 * u ** 3 + 2.0 * u ** 2 + 0.5 * u, 0.5 * u ** 3 - 0.5 * u ** 2)
        for m, w in zip((i - 1, i, i + 1, i + 2), c):
            W[a, min(max(m, 0), n - 1)] += w
    return W


def smooth_cloth(ep: dict, k: int):
    """The cloth as a k-times-finer Catmull-Rom surface: (W, faces of the moving half, of the stationary half)."""
    P = len(ep["cloth"][0])
    res = int(round(P ** 0.5))
    W = catmull_rom_weights(res, k)
    M = W.shape[0]
    # The halves meet at the middle grid line. Which axis and which side is the moving half: from the
    # particles only the moving / only the stationary triangles use (the recorded triangle split alternates
    # along the crease with the checkerboard diagonals, so it cannot colour the crease cells itself).
    tm = set(np.asarray(ep["tri_moving"]).ravel().tolist()) if len(ep["tri_moving"]) else set()
    ts = set(np.asarray(ep["tri_stationary"]).ravel().tolist()) if len(ep["tri_stationary"]) else set()
    m_only = np.array(sorted(tm - ts)) if tm - ts else np.array([res * res - 1])
    s_only = np.array(sorted(ts - tm)) if ts - tm else np.array([0])
    c = (res - 1) / 2.0
    di = np.mean(m_only // res) - np.mean(s_only // res)
    dj = np.mean(m_only % res) - np.mean(s_only % res)
    axis, sign = (0, np.sign(di)) if abs(di) >= abs(dj) else (1, np.sign(dj))
    faces = {"moving": [], "stationary": []}
    for a in range(M - 1):
        for b in range(M - 1):
            centre = ((a + 0.5) / k, (b + 0.5) / k)[axis]  # in particle-grid units
            half = "moving" if sign * (centre - c) > 0 else "stationary"
            v00, v01, v10, v11 = a * M + b, a * M + b + 1, (a + 1) * M + b, (a + 1) * M + b + 1
            faces[half] += [(v00, v10, v11), (v00, v11, v01)]
    return W, res, {h: np.asarray(f, np.uint32) for h, f in faces.items() if f}


def cloth_surface(pts: np.ndarray, W: np.ndarray, res: int) -> np.ndarray:
    G = pts.reshape(res, res, 3)
    return np.einsum("ai,bj,ijc->abc", W, W, G).reshape(-1, 3).astype(np.float32)


def load_visuals(path):
    if not path or not Path(path).exists():
        return []
    d = np.load(path, allow_pickle=True)
    return [(str(k).split("|")[0], d[f"v{i}"], d[f"f{i}"].astype(np.uint32), tuple(int(c) for c in d["rgb"][i]))
            for i, k in enumerate(d["keys"])]


def load(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)  # our own recordings: shapes are object arrays
    return {k: d[k] for k in d.files}


def episodes(paths: list[str]) -> list[Path]:
    """Every recording under the given files / directories (recursively): .npz files with a ``cloth`` array."""
    out = []
    for p in map(Path, paths):
        for f in (sorted(p.rglob("*.npz")) if p.is_dir() else [p]):
            try:
                with np.load(f, allow_pickle=True) as d:
                    if "cloth" in d.files and "body_q" in d.files:
                        out.append(f)
            except Exception:  # noqa: BLE001 - not a recording
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--visual", default=str(Path(__file__).resolve().parents[3] / "outputs/renders/visual_flexiv_wuji.npz"),
                    help="visual-mesh cache (build_visual_cache.py); empty: draw the collision shapes")
    ap.add_argument("--smooth", type=int, default=4, help="cloth drawn k x finer (Catmull-Rom through the particles)")
    args = ap.parse_args()
    visuals = load_visuals(args.visual)
    print(f"[replay] {len(visuals)} visual meshes from {args.visual}" if visuals else "[replay] no visual cache: collision shapes", flush=True)
    files = episodes(args.paths)
    if not files:
        raise SystemExit(f"no recordings under {args.paths} (record with render_vbd.py --record)")
    roots = [Path(p) for p in args.paths]

    def name(f: Path) -> str:
        # Relative to the folder ABOVE each root: several roots hold the same file names, and duplicate
        # dropdown options crash the viser client (a blank page).
        for r in roots:
            if r.is_dir() and f.is_relative_to(r):
                return str(f.relative_to(r.parent))
        return str(f)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    state = {"ep": None, "bodies": [], "cloth": {}, "frame": 0}
    # Loading (a GUI callback thread) and drawing (the play loop, the slider) take turns: switching
    # episodes removes the old scene's handles, which a draw in flight would otherwise write to.
    lock = threading.RLock()

    g_toggle = server.gui.add_button("Play")
    g_ep = server.gui.add_dropdown("episode", [name(f) for f in files], initial_value=name(files[0]))
    g_rescan = server.gui.add_button("rescan for new recordings")
    g_frame = server.gui.add_slider("step", 0, 1, 1, 0)
    g_play = server.gui.add_checkbox("playing", False)  # paused: a slow tunnel loads the scene first

    @g_toggle.on_click
    def _(_e) -> None:
        g_play.value = not bool(g_play.value)

    @g_play.on_update
    def _(_e) -> None:
        g_toggle.label = "Pause" if g_play.value else "Play"
    g_speed = server.gui.add_slider("speed (x real time)", 0.1, 4.0, 0.1, 1.0)
    g_cloth_op = server.gui.add_slider("cloth opacity", 0.1, 1.0, 0.05, 1.0)
    g_robot_op = server.gui.add_slider("robot opacity", 0.1, 1.0, 0.05, 1.0)
    g_collision = server.gui.add_checkbox("show collision shapes", not visuals)
    g_info = server.gui.add_markdown("")
    g_status = server.gui.add_markdown("")

    def load_episode(stem: str) -> None:
        ep = load(next(f for f in files if name(f) == stem))
        with lock:
            _install(stem, ep)

    def _install(stem: str, ep: dict) -> None:
        state["ep"] = None  # no draws against the old handles from here on
        server.scene.reset()
        server.scene.set_up_direction("+z")
        try:
            server.scene.configure_environment_map("studio", environment_intensity=0.35)
            server.scene.enable_default_lights(True, cast_shadow=True)
        except Exception:  # noqa: BLE001 - cosmetics only
            pass
        origin = np.asarray(ep["env_origin"], float) if "env_origin" in ep else np.zeros(3)
        server.scene.add_grid("/floor", width=3.0, height=3.0, cell_size=0.1, section_size=0.5,
                              position=(float(origin[0]), float(origin[1]), 0.0), plane="xy")
        bodies = []
        server.scene.add_frame("/static", show_axes=False)  # the parent of static shapes must exist
        labels = [str(x).rsplit("/", 1)[-1] for x in ep["body_label"]]
        for b in range(len(ep["body_label"])):
            bodies.append(server.scene.add_frame(f"/b{b}", show_axes=False))
        robot_meshes, collision = [], []
        for i in (range(len(ep["shape_kind"])) if STAGE >= 2 else []):
            key = int(ep["shape_key"][i])
            parent = f"/b{key}" if key >= 0 else "/static"
            kind = str(ep["shape_kind"][i])
            rgb = TABLE_RGB if kind == "table" else tuple(int(c) for c in ep["shape_rgb"][i])
            h = server.scene.add_mesh_simple(
                f"{parent}/s{i}", vertices=np.asarray(ep["shape_verts"][i], np.float32),
                faces=np.asarray(ep["shape_faces"][i], np.uint32), color=rgb,
                position=tuple(float(x) for x in ep["shape_pos"][i]), wxyz=tuple(float(x) for x in ep["shape_wxyz"][i]))
            if kind == "robot":
                collision.append(h)
                h.visible = bool(g_collision.value) or not visuals
        for k, (body, v, f_, rgb) in enumerate(visuals):
            if body in labels:
                robot_meshes.append(server.scene.add_mesh_simple(f"/b{labels.index(body)}/v{k}", vertices=v, faces=f_, color=rgb))
        state.update(bodies=bodies, robot_meshes=robot_meshes or collision, collision=collision, cloth={}, frame=0)
        state["smooth"] = smooth_cloth(ep, max(1, int(args.smooth)))
        n = len(ep["cloth"])
        g_frame.max = n - 1
        g_frame.value = 0
        state["ep"] = ep
        g_info.content = (f"**{stem}**: {str(ep['outcome'])}, {n} steps ({n * float(ep['policy_dt']):.1f} s)  \n"
                          f"policy `{Path(str(ep['checkpoint'])).name}`, world `{Path(str(ep['protocol'])).name}`")
        _draw(ep, 0)

    def draw(f: int) -> None:
        with lock:
            ep = state["ep"]
            if ep is None:
                return
            _draw(ep, min(f, len(ep["cloth"]) - 1))

    def _draw(ep: dict, f: int) -> None:
        q = ep["body_q"][f]
        for b, h in enumerate(state["bodies"]):
            h.position = tuple(float(x) for x in q[b, :3])
            h.wxyz = (float(q[b, 6]), float(q[b, 3]), float(q[b, 4]), float(q[b, 5]))  # xyzw -> wxyz
        W, res, faces = state["smooth"]
        pts = cloth_surface(np.asarray(ep["cloth"][f], np.float64), W, res)
        for half, rgb in [] if STAGE < 3 else (("moving", MOVING_RGB), ("stationary", STATIONARY_RGB)):
            if half not in faces:
                continue
            h = state["cloth"].get(half)
            if h is None:
                state["cloth"][half] = server.scene.add_mesh_simple(
                    f"/cloth_{half}", vertices=pts, faces=faces[half], color=rgb, side="double",
                    opacity=None if float(g_cloth_op.value) >= 1.0 else float(g_cloth_op.value))
            else:
                h.vertices = pts
        state["frame"] = f
        err = float(ep["fold_err_mm"][f])
        if f % 10 == 0 or f == len(ep["cloth"]) - 1:
            g_status.content = f"step {f + 1}/{len(ep['cloth'])}, fold error {err:.1f} mm"

    @g_ep.on_update
    def _(_e) -> None:
        load_episode(g_ep.value)

    @g_rescan.on_click
    def _(_e) -> None:
        found = episodes(args.paths)
        files[:] = found
        current = g_ep.value
        g_ep.options = [name(f) for f in files]
        if current in g_ep.options:
            g_ep.value = current
        print(f"[replay] rescan: {len(files)} recording(s)", flush=True)

    @g_frame.on_update
    def _(_e) -> None:
        if state["ep"] is not None and int(g_frame.value) != state["frame"]:
            draw(int(g_frame.value))

    @g_cloth_op.on_update
    def _(_e) -> None:
        for h in state["cloth"].values():
            h.opacity = float(g_cloth_op.value)

    @g_collision.on_update
    def _(_e) -> None:
        for h in state.get("collision", []):
            h.visible = bool(g_collision.value)

    @g_robot_op.on_update
    def _(_e) -> None:
        for h in state.get("robot_meshes", []):
            h.opacity = float(g_robot_op.value)

    load_episode(name(files[0]))
    print(f"[replay] {len(files)} episode(s); open http://localhost:{args.port}", flush=True)
    while True:
        ep = state["ep"]
        dt = (float(ep["policy_dt"]) if ep is not None else 0.05) / max(float(g_speed.value), 0.05)
        try:
            if STAGE >= 4 and g_play.value and ep is not None:
                g_frame.value = (state["frame"] + 1) % len(ep["cloth"])  # triggers draw
        except Exception:  # noqa: BLE001 - a bad frame must not take the server down
            traceback.print_exc()
        time.sleep(dt)


if __name__ == "__main__":
    main()
