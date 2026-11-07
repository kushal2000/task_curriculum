"""Static PNG of fabric_droop_lineup.py: one panel per sheet, name on top, settled droop below."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#9696a0", "#eba9be", "#f0d278", "#78b4e6", "#e6965a", "#78c88c", "#4664aa"]


def main():
    data = ROOT / "docs/results/fabric_droop"
    d = np.load(data / "droop.npz")
    info = json.loads((data / "angles.json").read_text())
    frames, faces, names = d["frames"], d["faces"], [str(n) for n in d["names"]]
    clamp_x = float(d["clamp_x"])
    sheets = {s[0]: s for s in info["sheets"]}
    n = len(names)
    fig = plt.figure(figsize=(2.3 * n, 2.8), dpi=200)
    for k, name in enumerate(names):
        ax = fig.add_subplot(1, n, k + 1, projection="3d", computed_zorder=False)
        q = frames[-1, k]
        # clamp block under the pinned rows
        bx0, bx1, by, bz0, bz1 = -0.05, clamp_x, 0.055, -0.013, -0.001
        box = [[(bx0, -by, bz1), (bx1, -by, bz1), (bx1, by, bz1), (bx0, by, bz1)],
               [(bx0, -by, bz0), (bx1, -by, bz0), (bx1, -by, bz1), (bx0, -by, bz1)],
               [(bx1, -by, bz0), (bx1, by, bz0), (bx1, by, bz1), (bx1, -by, bz1)]]
        ax.add_collection3d(Poly3DCollection(box, facecolor="#5a4636", edgecolor="#3a2c22", linewidth=0.3, zorder=1))
        tris = q[faces]
        ax.add_collection3d(Poly3DCollection(tris, facecolor=COLORS[k % len(COLORS)], edgecolor="#333333",
                                             linewidth=0.25, alpha=0.95, zorder=2))
        ax.set_xlim(-0.05, 0.05)
        ax.set_ylim(-0.05, 0.05)
        ax.set_zlim(-0.065, 0.005)
        ax.set_box_aspect((1, 1, 0.7), zoom=1.35)
        ax.view_init(elev=12, azim=-75)
        ax.set_axis_off()
        _, rho, _, ke, _ = sheets[name]
        ax.set_title(f"{name}\n{info['droop_deg'][name]:.0f}° droop", fontsize=11, pad=-8)
        ax.text2D(0.5, 0.06, f"ρ {rho * 1000:.0f} g/m²  ke {ke:.1e}", transform=ax.transAxes, ha="center", fontsize=7, color="#555")
    fig.subplots_adjust(left=0, right=1, top=0.82, bottom=0.0, wspace=-0.05)
    out = data / "fabric_droop.png"
    fig.savefig(out, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
