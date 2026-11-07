"""Viser replay of cloth_plasticity_probe.py, seen from the side: one row of sheets, one per fabric,
at the chosen yield threshold and press force, from just before the release until 3 s after it.
Shows whether each fold stays down or springs back. No GPU needed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser

ROOT = Path(__file__).resolve().parents[2]
RES = 7
NP = RES * RES
DX = 0.17  # sheet spacing along x (the fold axis), so the side view looks along the crease
CELL_MM = 16.7
COLORS = dict(base=(150, 150, 160), chiffon=(200, 160, 230), silk=(230, 220, 200), cotton=(235, 170, 190),
              napkin=(120, 200, 140), towel=(240, 210, 120), denim=(90, 120, 190))
T_RELEASE = 3.6  # run_fold: settle 0.5 + fold 1.5 + hold 0.3 + press 1.2 + gap 0.1
WINDOW = (T_RELEASE - 0.3, T_RELEASE + 3.0)


def parse(label):
    k, p = label.split("_")
    return float(k[5:]), float(p[5:])


def kappa_name(k):
    return "no plasticity" if np.isinf(k) else f"{k * CELL_MM / 1000 * 180 / np.pi:.0f} deg per cell (kappa {k:g})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="docs/results/cloth_plasticity_7fab_pin")
    ap.add_argument("--port", type=int, default=8092)
    a = ap.parse_args()
    d = np.load(ROOT / a.data / "fold.npz")
    rows = json.loads((ROOT / a.data / "fold.json").read_text())
    faces, times, labels = d["faces"], d["times"], [str(s) for s in d["labels"]]
    spacing = float(d["sheet_spacing_y"])
    fabrics = [k[: -len("_frames")] for k in d.files if k.endswith("_frames")]
    frames = {f: d[f"{f}_frames"] for f in fabrics}
    kappas = list(dict.fromkeys(parse(s)[0] for s in labels))
    presses = list(dict.fromkeys(parse(s)[1] for s in labels))
    res = {(r["fabric"], r["kappa"], r["press_N"]): r for r in rows}
    tw = np.array([i for i, t in enumerate(times) if WINDOW[0] <= t <= WINDOW[1]])

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    cx = DX * (len(fabrics) - 1) / 2
    server.initial_camera.position = (cx, -1.05, 0.06)
    server.initial_camera.look_at = (cx, 0.0, 0.02)
    server.initial_camera.fov = 0.6
    server.scene.add_box("/ground", dimensions=(DX * len(fabrics) + 0.1, 0.25, 0.004), color=(205, 205, 205),
                         position=(cx, 0.0, -0.002))

    names = {kappa_name(k): k for k in kappas}
    default_k = 100.0 if 100.0 in kappas else kappas[0]
    with server.gui.add_folder("Show"):
        k_dd = server.gui.add_dropdown("threshold", list(names), initial_value=kappa_name(default_k))
        p_dd = server.gui.add_dropdown("press [N]", [f"{p:g}" for p in presses],
                                       initial_value="10" if 10.0 in presses else f"{presses[0]:g}")
        t_slider = server.gui.add_slider("t after release [s]", min=round(WINDOW[0] - T_RELEASE, 2),
                                         max=round(WINDOW[1] - T_RELEASE, 2), step=0.05, initial_value=-0.3)
        play = server.gui.add_checkbox("play", True)
        phase_txt = server.gui.add_markdown("")
        server.gui.add_markdown("Side view along the crease. Each sheet is folded right over left, then released "
                                "at t = 0. Label: result 3 s after release (open = flap angle, 0 folded, 180 flat).")

    labels_h = []

    def draw_labels(k, p):
        for h in labels_h:
            h.remove()
        labels_h.clear()
        for i, fab in enumerate(fabrics):
            r = res.get((fab, k, p))
            txt = fab if r is None else f"{fab}: {'HELD' if r['held_1p5cm'] else 'springs back'}, open {r['free_open_deg']:.0f}°"
            labels_h.append(server.scene.add_label(f"/label/{i}", txt, position=(i * DX, 0.0, 0.08)))

    def draw(k, p, ti):
        sheet = labels.index(f"kappa{k:g}_press{p:g}")
        for i, fab in enumerate(fabrics):
            v = frames[fab][ti, sheet * NP:(sheet + 1) * NP] + np.array([i * DX, -spacing * sheet, 0.0])
            server.scene.add_mesh_simple(f"/sheet/{i}", vertices=v.astype(np.float32), faces=faces,
                                         color=COLORS.get(fab, (180, 180, 180)), side="double", flat_shading=False)
        t = times[ti] - T_RELEASE
        phase_txt.content = f"**t = {t:+.2f} s — {'held by the edge' if t < 0 else 'RELEASED'}**"

    shown_kp, last = None, None
    t_wall = time.time()
    while True:
        now = time.time()
        if play.value:
            t = t_slider.value + (now - t_wall) * 0.5  # half speed
            t_slider.value = float(t if t <= t_slider.max else t_slider.min)
        t_wall = now
        k, p = names[k_dd.value], float(p_dd.value)
        ti = int(tw[min(int(np.searchsorted(times[tw], t_slider.value + T_RELEASE - 1e-9)), len(tw) - 1)])
        if (k, p) != shown_kp:
            draw_labels(k, p)
            shown_kp = (k, p)
        if (k, p, ti) != last:
            draw(k, p, ti)
            last = (k, p, ti)
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
