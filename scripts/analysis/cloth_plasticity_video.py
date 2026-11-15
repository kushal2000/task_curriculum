"""MP4 per fabric of the plasticity probe's scripted fold sequence, from its saved frames (no sim, CPU).

The probe (cloth_plasticity_probe.py) saves every sheet's particles to <data>/fold.npz. With
``--fold touch --cycles 2 --press-per-cycle 0 0.93`` the sequence on one sheet is:
fold, let go with no weight -> unfold -> fold, 0.93 N weight on, let go -> unfold.

Each fabric's npz holds one sheet per (kappa, press) label; the video shows the sheet whose kappa is
that fabric's preset ``plastic_yield_curvature`` (isaacsimenvs/cfg/fabric/<fabric>.yaml), or --kappa.

    .venv_isaaclab3/bin/python -I scripts/analysis/cloth_plasticity_video.py \
        --data outputs/results/cloth_plasticity_measured_current_seq --out videos/plasticity_seq

Per frame: the sheet (gold = stationary half, vermilion = the flap, as render_newton.py), the phase,
the time, and the flap's opening angle (0 = folded flat, 180 = flat sheet) from the crease to the far
edge's midpoint, measured against the stationary half.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GOLD, VERMILION = (0.98, 0.78, 0.15), (0.95, 0.30, 0.10)


def preset_kappa(fabric: str) -> float | None:
    p = ROOT / "isaacsimenvs/cfg/fabric" / f"{fabric}.yaml"
    if not p.exists():
        return None
    m = re.search(r"^\s*plastic_yield_curvature:\s*([0-9.eE+-]+|inf)", p.read_text(), re.M)
    return float(m.group(1)) if m else None


def opening_deg(sheet: np.ndarray, res: int) -> float:
    """Angle between the stationary half (crease -> near edge) and the flap (crease -> far edge)."""
    g = sheet.reshape(res, res, 3)
    c = res // 2
    crease, near, far = g[c].mean(0), g[0].mean(0), g[-1].mean(0)
    a, b = near - crease, far - crease
    cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def render(frames, faces, times, phases, res, title, out: Path, fps: int, elev: float, azim: float):
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rows = (faces // res).mean(1)
    colors = [VERMILION if r > res // 2 else GOLD for r in rows]
    lo, hi = frames.reshape(-1, 3).min(0), frames.reshape(-1, 3).max(0)
    mid, half = (lo + hi) / 2, max((hi - lo)[:2].max() / 2, 0.03) * 1.05
    phase_t = np.array([t for t, _ in phases])

    fig = plt.figure(figsize=(7.2, 5.4), dpi=120)
    ax = fig.add_axes([0, 0, 1, 0.86], projection="3d")
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7, macro_block_size=8)
    for i, (pts, t) in enumerate(zip(frames, times)):
        ax.cla()
        ax.add_collection3d(Poly3DCollection(pts[faces], facecolors=colors, edgecolors=(0, 0, 0, 0.25),
                                             linewidths=0.4, shade=True))
        ax.plot([mid[0] - half, mid[0] + half], [mid[1] - half] * 2, [0, 0], color="0.6", lw=0.8)
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(0.0, 2 * half * 0.6)
        ax.set_box_aspect((1, 1, 0.6), zoom=1.25)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        k = int(np.searchsorted(phase_t, t + 1e-9) - 1)
        phase = phases[max(k, 0)][1]
        fig.suptitle(f"{title}\n{phase}\nt = {t:5.2f} s   opening {opening_deg(pts, res):5.1f} deg   "
                     f"(0 = folded flat)", fontsize=10, y=0.985)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)
    writer.close()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/results/cloth_plasticity_measured_current_seq")
    ap.add_argument("--out", default="videos/plasticity_seq")
    ap.add_argument("--fabrics", nargs="*", default=None, help="default: every fabric in the npz")
    ap.add_argument("--kappa", type=float, default=None, help="override the preset kappa for every fabric")
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    a = ap.parse_args()

    d = np.load(ROOT / a.data / "fold.npz")
    res, faces, times = int(d["res"]), np.asarray(d["faces"]), np.asarray(d["times"])
    labels = [str(x) for x in d["labels"]]
    phases = list(zip(d["phase_t"].tolist(), [str(x) for x in d["phase_names"]]))
    dt = float(np.median(np.diff(times)))
    fps = max(1, round(1.0 / dt))  # real time
    n = res * res
    fabrics = a.fabrics or sorted(k[: -len("_frames")] for k in d.files if k.endswith("_frames"))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for fab in fabrics:
        fab = fab if fab.endswith("_measured") or f"{fab}_frames" in d.files else f"{fab}_measured"
        kap = a.kappa if a.kappa is not None else preset_kappa(fab)
        cand = [(i, lab) for i, lab in enumerate(labels)
                if (m := re.match(r"kappa([0-9.]+|inf)", lab)) and kap is not None and float(m.group(1)) == kap]
        if not cand:
            print(f"[video] {fab}: no sheet at kappa={kap} in {labels}; skipped", file=sys.stderr)
            continue
        i, lab = cand[0]
        frames = np.asarray(d[f"{fab}_frames"])[:, i * n:(i + 1) * n]
        frames = frames - np.array([0.0, frames[0, :, 1].mean(), 0.0])  # sheets sit side by side in y
        path = out / f"{fab.replace('_measured', '')}_fold_press_seq.mp4"
        name = fab.replace("_measured", "")
        m = re.match(r"kappa([0-9.]+|inf)_press([0-9.]+)", lab)
        title = (f"{name}   kappa {m.group(1)}   (cycle 1: no weight, cycle 2: {m.group(2)} N weight)"
                 if m else f"{name}  {lab}")
        render(frames, faces, times, phases, res, title, path, fps, a.elev, a.azim)
        print(f"[video] wrote {path.relative_to(ROOT)}  ({len(frames)} frames, {len(frames) / fps:.1f} s, {lab})",
              flush=True)


if __name__ == "__main__":
    main()
