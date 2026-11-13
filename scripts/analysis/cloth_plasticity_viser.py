"""Viser replay of cloth_plasticity_probe.py, seen from the side: one row of sheets, one per fabric,
at the chosen yield threshold and press force, from just before the release until 3 s after it.
Shows whether each fold stays down or springs back. No GPU needed.

`View` is also used by fabric_results_viser.py, which shows every probe in one server.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fabric_textures import add_fabric_mesh  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# The 11-04 run's --out, else the probe's own default --out.
DEFAULT_DATA = next((d for d in ("docs/results/cloth_plasticity_7fab_pin", "docs/results/cloth_plasticity")
                     if (ROOT / d / "fold.npz").exists()), "docs/results/cloth_plasticity_7fab_pin")
RES = 7
NP = RES * RES
DX = 0.17  # sheet spacing along x (the fold axis), so the side view looks along the crease
CELL_MM = 16.7
# The real swatches' colours, except towel: really white, drawn light blue to tell it from big napkin and silk.
COLORS = dict(base=(150, 150, 160), chiffon=(245, 140, 40), silk=(245, 245, 240), cotton=(205, 45, 45),
              napkin=(245, 210, 50), big_napkin=(245, 245, 240), towel=(165, 205, 240), denim=(50, 85, 165))
ALL = "all"
FABRIC_DIR = ROOT / "isaacsimenvs/cfg/fabric"
FITTED = "fitted κ per fabric (preset plastic_yield_curvature)"


def color(fab):
    return COLORS.get(base_name(fab), (180, 180, 180))


def base_name(fab):
    """chiffon_measured_r25 -> chiffon"""
    return fab.removesuffix("_r25").removesuffix("_measured")


def preset_kappa(fab):
    """The preset's fitted plastic_yield_curvature (0 = off -> inf), or None without a preset."""
    import yaml
    path = FABRIC_DIR / f"{fab}.yaml"
    if not path.is_file():
        return None
    k = float(yaml.safe_load(path.read_text()).get("plastic_yield_curvature", 0.0))
    return float("inf") if k <= 0 else k
T_RELEASE = 3.6  # run_fold: settle 0.5 + fold 1.5 + hold 0.3 + press 1.2 + gap 0.1
WINDOW = (T_RELEASE - 0.3, T_RELEASE + 3.0)
#: run_fold's phases (its T dict), as start times [s] from the start of the run.
PHASES = [(0.0, "settling"), (0.5, "folding: edge carried over"), (2.0, "folded, edge held"),
          (2.3, "weight on"), (3.5, "weight off, edge still held"), (T_RELEASE, "RELEASED"),
          (T_RELEASE + 3.0, "unfolding: edge carried back to flat"), (T_RELEASE + 4.5, "let go, unfolded")]
#: span name -> window around the (first) release [s], None = the whole run
SPANS = {"full run: fold → weight → release → unfold (all cycles)": None, "first release only (−0.3 … +3 s)": (-0.3, 3.0)}


def parse(label):
    k, p = label.split("_")
    return float(k[5:]), float(p[5:])


def kappa_name(k):
    return "no plasticity" if np.isinf(k) else f"{k * CELL_MM / 1000 * 180 / np.pi:.0f} deg per cell (kappa {k:g})"


class View:
    """The fold-press-release replay under scene prefix `prefix`, its controls in one GUI folder."""

    title = "Plasticity"

    def __init__(self, server: viser.ViserServer, data: str = DEFAULT_DATA, prefix: str = "", title: str | None = None,
                 press: float | None = None):
        self.server, self.p = server, prefix
        self.title = title or type(self).title
        d = np.load(ROOT / data / "fold.npz")
        rows = json.loads((ROOT / data / "fold.json").read_text())
        self.faces, self.times, self.labels = d["faces"], d["times"], [str(s) for s in d["labels"]]
        self.np = int(self.faces.max()) + 1  # particles per sheet (49 on the env's 7x7)
        self.spacing = float(d["sheet_spacing_y"])
        self.fabrics = fabrics = [k[: -len("_frames")] for k in d.files if k.endswith("_frames")]
        self.frames = {f: d[f"{f}_frames"] for f in fabrics}
        kappas = list(dict.fromkeys(parse(s)[0] for s in self.labels))
        presses = list(dict.fromkeys(parse(s)[1] for s in self.labels))
        self.res = {(r["fabric"], r["kappa"], r["press_N"]): r for r in rows}
        # Phase log saved by the probe (multi-cycle runs); older runs: the fixed single-cycle timeline.
        if "phase_t" in d.files:
            self.phases = [(float(t0), str(nm)) for t0, nm in zip(d["phase_t"], d["phase_names"])]
        else:
            self.phases = PHASES
        self.t_rel = next(t0 for t0, nm in self.phases if nm.startswith("RELEASED"))
        self.cycles = int(d["cycles"]) if "cycles" in d.files else 1
        times = self.times
        self.spans = {name: np.array([i for i, t in enumerate(times) if w is None or w[0] <= t - self.t_rel <= w[1]])
                      for name, w in SPANS.items()}

        self.cx = cx = DX * (len(fabrics) - 1) / 2
        self._cam_all = dict(position=(cx, -1.05, 0.06), look_at=(cx, 0.0, 0.02), fov=0.6)
        self._cam_one = dict(position=(cx, -0.42, 0.05), look_at=(cx, 0.0, 0.015), fov=0.6)
        # mid grey, so the white swatches (silk, big napkin, towel) stand out
        self.static = [server.scene.add_box(f"{prefix}/ground", dimensions=(DX * len(fabrics) + 0.1, 0.25, 0.004),
                                            color=(135, 135, 140), position=(cx, 0.0, -0.002))]

        self.names = {kappa_name(k): k for k in kappas}
        self.fitted = {f: preset_kappa(f) for f in fabrics}
        has_fitted = all(k in kappas for k in self.fitted.values())
        if has_fitted:
            self.names = {FITTED: None, **self.names}
        default_name = FITTED if has_fitted else kappa_name(100.0 if 100.0 in kappas else kappas[0])
        nonzero = [q for q in presses if q > 0] or presses
        default_p = 10.0 if 10.0 in presses else min(nonzero, key=lambda q: abs(q - 1.4))
        if press is not None and press in presses:
            default_p = press
        self.folder = server.gui.add_folder(self.title)
        with self.folder:
            self.pretty = {base_name(f).replace("_", " "): f for f in fabrics}
            self.fab_dd = server.gui.add_dropdown("fabric", [ALL, *self.pretty], initial_value=ALL)
            self.k_dd = server.gui.add_dropdown("threshold", list(self.names), initial_value=default_name)
            self.p_dd = server.gui.add_dropdown("press [N]", [f"{p:g}" for p in presses], initial_value=f"{default_p:g}")
            self.span_dd = server.gui.add_dropdown("show", list(SPANS), initial_value=next(iter(SPANS)))
            lo, hi = self.span_range(self.span_dd.value)
            self.t_slider = server.gui.add_slider("t after release [s]", min=lo, max=hi, step=0.05, initial_value=lo)
            self.play = server.gui.add_checkbox("play", True)
            self.speed = server.gui.add_slider("speed", min=0.1, max=2.0, step=0.05, initial_value=0.75)
            self.phase_txt = server.gui.add_markdown("")
            self.show_labels = server.gui.add_checkbox("labels above sheets", False)
            self.textured = server.gui.add_checkbox("fabric textures (when paused)", True)
            self.table = server.gui.add_markdown("")
            server.gui.add_markdown("Side view along the crease. Each sheet is folded right over left, weighted, "
                                    "released at t = 0, then 3 s later unfolded back to flat and let go. Label: open = "
                                    "flap angle 3 s after release (0 folded, 180 flat); crease = what is left 2 s after "
                                    "the unfold. Sheets in the table are listed left to right.")

        self.labels_h, self.meshes = [], []
        self.shown_span = self.span_dd.value
        self.shown_fab = self.fab_dd.value
        self.active, self.shown_kp, self.last, self.t_wall = True, None, None, time.time()

    @property
    def camera(self):
        return self._cam_all if self.fab_dd.value == ALL else self._cam_one

    def shown(self):
        """(index, fabric, x offset) of the sheets on screen: all in a row, or one centred."""
        if self.fab_dd.value == ALL:
            return [(i, f, i * DX) for i, f in enumerate(self.fabrics)]
        f = self.pretty[self.fab_dd.value]
        return [(self.fabrics.index(f), f, self.cx)]

    def clear_scene(self):
        for h in self.labels_h + self.meshes + getattr(self, "_tex_pending", []):
            h.remove()
        self.labels_h.clear()
        self.meshes.clear()
        self._tex_pending = []

    def promote_textured(self):
        """Show the frame built hidden last time and drop the one on screen. By now the browser has
        decoded the hidden GLB, so the swap is one step with no empty frame in between."""
        if not getattr(self, "_tex_pending", None):
            return
        for h in self._tex_pending:
            h.visible = True
        for h in self.meshes:
            h.remove()
        self.meshes, self._tex_pending = self._tex_pending, []

    def span_range(self, name):
        tw = self.spans[name]
        return round(float(self.times[tw[0]]) - self.t_rel, 2), round(float(self.times[tw[-1]]) - self.t_rel, 2)

    def phase(self, t_run, press=None):
        name = [nm for t0, nm in self.phases if t_run >= t0 - 1e-9][-1]
        if press == 0.0:  # the probe logs the press phases whatever the force
            name = name.replace("weight off, edge still held", "edge held on the other half (no weight)")
            name = name.replace("RELEASED (weight off)", "RELEASED (no weight)")
            name = name.replace("weight on", "lying folded, no weight")
        return name

    def set_visible(self, on: bool):
        self.active = on
        self.folder.visible = on
        for h in self.static:
            h.visible = on
        if not on:
            self.clear_scene()
        self.shown_kp, self.last, self.t_wall = None, None, time.time()

    def draw_labels(self, k, p):
        """Results table in the panel; text labels in the scene only when "labels above sheets" is on."""
        for h in self.labels_h:
            h.remove()
        self.labels_h.clear()
        multi = any("free_open_deg_cycles" in r for r in self.res.values())
        if multi:
            rows = ["| left → right | κ | open after release ° (per cycle) | crease after unfold ° |", "|---|---|---|---|"]
        else:
            rows = ["| left → right | κ | after release | open ° | crease ° |", "|---|---|---|---|---|"]
        for i, fab in enumerate(self.fabrics):
            kf = self.fitted[fab] if k is None else k
            r = self.res.get((fab, kf, p))
            name = fab.removesuffix("_measured").replace("_", " ")
            if r is None:
                rows.append(f"| {name} | {kf:g} | – | – |" + ("" if multi else " – |"))
                txt = name
            elif multi:
                o = " ".join(f"{abs(v):.0f}" for v in r["free_open_deg_cycles"])
                c = " ".join(f"{180 - abs(v):.0f}" for v in r["reopen_open_deg_cycles"])
                rows.append(f"| {name} | {kf:g} | {o} | {c} |")
                txt = f"{name}: open {o}°, crease {c}°"
            else:
                held = "held" if r["held_1p5cm"] else "springs back"
                o, c = abs(r["free_open_deg"]), 180 - abs(r["reopen_open_deg"])
                rows.append(f"| {name} | {kf:g} | {held} | {o:.0f} | {c:.0f} |")
                txt = f"{name}: {held}, open {o:.0f}°, crease {c:.0f}°"
            on_screen = {j: x for j, _, x in self.shown()}
            if self.show_labels.value and i in on_screen:
                self.labels_h.append(self.server.scene.add_label(f"{self.p}/label/{i}", txt, position=(on_screen[i], 0.0, 0.08)))
        self.table.content = "\n".join(rows)

    def draw(self, k, p, ti):
        # A textured sheet is a GLB rebuilt and re-sent per frame: the browser drops the old node
        # before the new one decodes, so it blinks. Flat sheets keep their node and only get new
        # vertices, which is smooth; they are recreated only when the set on screen changes.
        # Textured: double-buffered. Each frame is added hidden under the other of two names and
        # shown on the next draw or tick (promote_textured), so one is always on screen.
        shown = self.shown()
        # Textures only on a still frame: a textured sheet is a GLB rebuilt per frame and blinks
        # however it is swapped, so playback uses flat sheets updated in place.
        flat = not (self.textured.value and not self.play.value)
        if getattr(self, "_flat_meshes", None) is not None and self._flat_meshes != flat:
            self.clear_scene()  # mode switch: the two modes use different node names
        if not flat:
            if getattr(self, "_tex_pending", None) and time.time() - self._tex_t < 0.06:
                return False  # the last frame is still decoding: skip this one rather than blink
            self.promote_textured()
            self._flip = 1 - getattr(self, "_flip", 0)
        reuse = flat and getattr(self, "_flat_meshes", False) and len(self.meshes) == len(shown)
        meshes = []
        for j, (i, fab, x) in enumerate(shown):
            kf = self.fitted[fab] if k is None else k
            sheet = self.labels.index(f"kappa{kf:g}_press{p:g}")
            v = self.frames[fab][ti, sheet * self.np:(sheet + 1) * self.np] + np.array([x, -self.spacing * sheet, 0.0])
            if reuse:
                self.meshes[j].vertices = np.asarray(v, np.float32)
                meshes.append(self.meshes[j])
            elif flat:
                meshes.append(add_fabric_mesh(self.server, f"{self.p}/sheet/{i}", v, self.faces, fab, color(fab),
                                              textured=False))
            else:
                meshes.append(add_fabric_mesh(self.server, f"{self.p}/tsheet/{i}/{self._flip}", v, self.faces, fab,
                                              color(fab), textured=True, visible=False))
        if flat:
            self.meshes = meshes
        else:
            self._tex_pending, self._tex_t = meshes, time.time()
        self._flat_meshes = flat
        t_run = float(self.times[ti])
        self.phase_txt.content = f"**t = {t_run - self.t_rel:+.2f} s — {self.phase(t_run, p)}**"

    def tick(self):
        if not self.active:
            return
        now = time.time()
        if getattr(self, "_tex_pending", None) and now - self._tex_t > 0.08:
            self.promote_textured()  # paused, or the last frame of a scrub: show it
        if self.play.value:
            t = self.t_slider.value + (now - self.t_wall) * self.speed.value
            self.t_slider.value = float(t if t <= self.t_slider.max else self.t_slider.min)
        self.t_wall = now
        k, p = self.names[self.k_dd.value], float(self.p_dd.value)
        if self.span_dd.value != self.shown_span:
            self.shown_span = self.span_dd.value
            lo, hi = self.span_range(self.shown_span)
            self.t_slider.min, self.t_slider.max = lo, hi
            self.t_slider.value = float(min(max(self.t_slider.value, lo), hi))
        if self.fab_dd.value != self.shown_fab:  # one fabric <-> all: redraw and move the camera
            self.shown_fab = self.fab_dd.value
            self.clear_scene()
            self.shown_kp, self.last = None, None
            for client in self.server.get_clients().values():
                with client.atomic():
                    for key, val in self.camera.items():
                        setattr(client.camera, key, val)
        times, tw = self.times, self.spans[self.shown_span]
        ti = int(tw[min(int(np.searchsorted(times[tw], self.t_slider.value + self.t_rel - 1e-9)), len(tw) - 1)])
        if (k, p, self.show_labels.value) != self.shown_kp:
            self.draw_labels(k, p)
            self.shown_kp = (k, p, self.show_labels.value)
        look = self.textured.value and not self.play.value
        if (k, p, ti, look) != self.last:
            if self.draw(k, p, ti) is not False:
                self.last = (k, p, ti, look)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--port", type=int, default=8092)
    a = ap.parse_args()
    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    view = View(server, a.data)
    for k, v in view.camera.items():
        setattr(server.initial_camera, k, v)
    while True:
        view.tick()
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
