"""All the VBD fabric probe viewers in one viser server, switched with a "Result" dropdown:
cantilever (fabric_cantilever_viser), droop lineup (fabric_droop_viser) and fold-press-release
plasticity (cloth_plasticity_viser), for the measured presets (cfg/fabric/*_measured.yaml) and the
literature ones. Each view keeps its own controls and camera. A view whose data folder is missing is
skipped. No GPU needed (replays the saved .npz files).

    .venv_isaaclab3/bin/python scripts/analysis/fabric_results_viser.py [--port 8090]
"""

import argparse
import importlib.util
import os
import time
from pathlib import Path

# A replay viewer needs no BLAS threads; OpenBLAS would otherwise start one per core (64 on a login
# node), which counts against the per-user process limit there.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import viser  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    cant, droop, plast = _load("fabric_cantilever_viser"), _load("fabric_droop_viser"), _load("cloth_plasticity_viser")
    # (title, viewer module, data folder): the dropdown order.
    datasets = [
        ("Plasticity — napkin & silk retune, 7x7 touch fold (kappa 26-35; presets 30 / 32)", plast, "outputs/results/cloth_plasticity_measured_r7_touch_v3_ns"),
        ("Plasticity — measured 7x7, touch fold (napkin/silk here at the old 150)", plast, "outputs/results/cloth_plasticity_measured_r7_touch_v3"),
        ("Plasticity — 25x25 reference, touch fold (0.5 mm contact)", plast, "outputs/results/cloth_plasticity_measured_r25_v3a"),
        ("Plasticity — measured, 5 fold cycles", plast, "outputs/results/cloth_plasticity_measured_cycles"),
        ("Cantilever — measured", cant, "outputs/results/fabric_cantilever_measured"),
        ("Plasticity — measured (weight on top)", plast, "outputs/results/cloth_plasticity_measured_weight"),
        ("Plasticity — measured (crease press)", plast, "outputs/results/cloth_plasticity_measured"),
        ("Cantilever — literature", cant, cant.DEFAULT_DATA),
        ("Droop lineup — literature", droop, droop.DEFAULT_DATA),
        ("Plasticity — literature", plast, plast.DEFAULT_DATA),
    ]
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--start", default=datasets[0][0], help="title of the view shown first")
    ap.add_argument("--press", type=float, default=None, help="plasticity views open on this press force [N] (0 = no weight)")
    a = ap.parse_args()

    present = [d for d in datasets if (ROOT / d[2]).is_dir()]
    for title, _, data in datasets:
        if not (ROOT / data).is_dir():
            print(f"[skip] {title}: no data at {data}")
    titles = [t for t, _, _ in present]

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    picker = server.gui.add_dropdown("Result", titles, initial_value=a.start if a.start in titles else titles[0])
    views = {t: (m.View(server, d, f"/v{i}", title=t, press=a.press) if m is plast else m.View(server, d, f"/v{i}", title=t))
             for i, (t, m, d) in enumerate(present)}

    def activate(title):
        for t, v in views.items():
            v.set_visible(t == title)
        cam = views[title].camera
        for k, val in cam.items():
            setattr(server.initial_camera, k, val)
        for client in server.get_clients().values():
            with client.atomic():
                for k, val in cam.items():
                    setattr(client.camera, k, val)

    # Switching happens in this loop (not in a GUI callback) so it never races a view's redraw.
    shown = None
    while True:
        if picker.value != shown:
            shown = picker.value
            activate(shown)
        views[shown].tick()
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
