"""Side views (x-z) of vbd_pinch_test.py --record runs: end of squeeze and end of lift, per sheet.

    python vbd_pinch_plot.py out.png run1/frames.npz [run2/frames.npz ...]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAD_R = 0.008


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    runs = [np.load(p) for p in paths]
    nS = max(sum(1 for k in r.files if k.endswith("_frames")) for r in runs)
    fig, axes = plt.subplots(nS, 2 * len(runs), figsize=(3.2 * 2 * len(runs), 1.5 * nS), squeeze=False)
    for c, r in enumerate(runs):
        ph = r["phases"]
        i_sq = np.nonzero(ph == "hold")[0][-1]
        i_up = len(ph) - 1
        t = float(r["thickness"])
        for k in range(nS):
            fr, pads, faces = r[f"s{k}_frames"], r[f"s{k}_pads"], r[f"s{k}_faces"]
            for j, (i, name) in enumerate(((i_sq, "squeezed"), (i_up, "lifted"))):
                ax = axes[k, 2 * c + j]
                q = fr[i]
                ax.triplot(q[:, 0] * 1e3, q[:, 2] * 1e3, faces, lw=0.4, color="0.6")
                strip = np.abs(fr[0][:, 1]) < 0.01
                ax.plot(q[strip, 0] * 1e3, q[strip, 2] * 1e3, ".", ms=3, color="C0")
                for p in pads[i]:
                    ax.add_patch(plt.Circle((p[0] * 1e3, p[2] * 1e3), PAD_R * 1e3, color="C3", alpha=0.5))
                ax.axhline(0, color="k", lw=0.5)
                ax.set_xlim(-55, 55); ax.set_ylim(-2, 60 if name == "lifted" else 35); ax.set_aspect("equal")
                ax.tick_params(labelsize=6)
                ax.set_title(f"{r[f's{k}_label']} | t {t*1e3:.0f} mm tri x{float(r['tri_scale']):g} | {name}",
                             fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
