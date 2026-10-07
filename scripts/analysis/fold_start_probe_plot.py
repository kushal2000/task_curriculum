"""Draw and summarise a ``fold_start_probe.py`` dump: what the partly folded starts look like and how
they behave over the probe's zero-action seconds.

    .venv_isaaclab3/bin/python scripts/analysis/fold_start_probe_plot.py docs/results/fold_start_probe.pt \\
        --png docs/results/fold_start_probe.png
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("dump")
p.add_argument("--png", required=True)
p.add_argument("--cols", type=int, default=6)
p.add_argument("--rows", type=int, default=5)
a = p.parse_args()

d = torch.load(a.dump, weights_only=False)
P, tips = d["particles"], d["tips"]  # (T, N, M, 3), (T, N, 5, 3)
T, N = P.shape[0], P.shape[1]
shapes, active = d["shapes"], d["active"]
dt = d["dt_policy"]


def start_deg(e: int) -> float:
    q = d["start_q"][e]
    return 0.0 if q is None else math.degrees(float(q.abs().max()))


def hinge_deg(t: int, e: int) -> float:
    return math.degrees(float(d["hinges"][t][e].abs().max()))


# Flat reference height per env: the lowest particle of a flat sheet sits at the sheet's rest height.
z0 = P[0, :, :, 2]
print(f"{'env':>3} {'shape':<14} {'size':>5} {'start':>6} {'t=0':>6} {f't={T - 1}':>6} "
      f"{'tip-sheet min [mm]':>18} {'sheet top [mm]':>14}")
rows = []
for e in range(N):
    dist = (tips[0, e].unsqueeze(1) - P[0, e].unsqueeze(0)).norm(dim=-1).min()
    top = float((z0[e].max() - z0[e].min()))
    rows.append((e, shapes[int(active[e])], float(d["env_size"][e]), start_deg(e), hinge_deg(0, e),
                 hinge_deg(T - 1, e), float(dist) * 1e3, top * 1e3))
    print(f"{e:>3} {rows[-1][1]:<14} {rows[-1][2] * 100:>4.1f} {rows[-1][3]:>6.1f} {rows[-1][4]:>6.1f} "
          f"{rows[-1][5]:>6.1f} {rows[-1][6]:>18.1f} {rows[-1][7]:>14.1f}")
landed = [abs(r[4] - r[3]) for r in rows]
drift = [r[5] - r[4] for r in rows]
print(f"\nstart angle landed within {max(landed):.2f} deg (max over envs)")
print(f"hinge change over {(T - 1) * dt:.2f} s of zero actions: min {min(drift):+.1f}, "
      f"max {max(drift):+.1f} deg")
print(f"fingertip-to-sheet distance at reset: min {min(r[6] for r in rows):.1f} mm")

k = min(N, a.cols * a.rows)
fig = plt.figure(figsize=(3.2 * a.cols, 3.0 * a.rows))
for i in range(k):
    ax = fig.add_subplot(a.rows, a.cols, i + 1, projection="3d")
    def ref(t):  # each cloud on its own centroid (xy) and lowest point (z): the sheet drops 15 cm
        c = P[t, i].mean(dim=0).clone()
        c[2] = P[t, i, :, 2].min()
        return c

    for t, col, al in ((0, "tab:blue", 0.9), (T - 1, "tab:orange", 0.5)):
        q = P[t, i] - ref(t)
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=3, c=col, alpha=al)
    tp = tips[0, i] - ref(0)
    ax.scatter(tp[:, 0], tp[:, 1], tp[:, 2], s=12, c="k", marker="^")
    r = rows[i]
    ax.set_title(f"{r[1]} {r[2] * 100:.0f}cm\nstart {r[3]:.0f}°, after {(T - 1) * dt:.1f}s {r[5]:.0f}°",
                 fontsize=8)
    lim = 0.13
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.01, 0.15)
    ax.view_init(elev=22, azim=-60)
    ax.tick_params(labelsize=5)
fig.suptitle("partly folded starts on PhysX: blue = at reset, orange = after landing + zero actions, ▲ = fingertips at reset",
             fontsize=11)
fig.tight_layout()
fig.savefig(a.png, dpi=90)
print(f"wrote {a.png}")
