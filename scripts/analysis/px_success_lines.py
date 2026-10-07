"""Line charts of the three generic-prior PhysX policies: each on its own rigid task (PhysX, Newton)
and on the VBD cloth, pinned and randomised starts. One figure per cutoff.

    .venv_isaaclab3/bin/python scripts/analysis/px_success_lines.py

Reads docs/results/px_matrix (bos14_px_eval_matrix.sh). A point is the held-fold share pooled over
the three training seeds (equal episodes per seed, so = their mean). Writes
docs/results/px_success_lines_{4cm,1p5cm}.png.
"""

from __future__ import annotations

import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POLICIES = [  # tag, label, own rigid task, colour, marker
    ("one_flat", "pool-flat", "pool", "#2a78d6", "o"),
    ("one_initfold", "pool-initfold", "pool", "#eb6834", "s"),
    ("box3", "box3", "box3", "#1baf7a", "^"),
]
COLS = [  # (target kind, start, tick label)
    ("physx", "pinned", "PhysX\npinned"),
    ("physx", "rand", "PhysX\nrandomized"),
    ("newton", "pinned", "Newton\npinned"),
    ("newton", "rand", "Newton\nrandomized"),
    ("vbd", "pinned", "Newton\npinned"),
    ("vbd", "rand", "Newton\nrandomized"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def rate(tag: str, target: str, start: str, cut: str) -> float:
    """Held-fold %, pooled over the three training seeds and both eval seeds."""
    held = n = 0
    for seed in (1, 2, 3):
        files = glob.glob(f"docs/results/px_matrix/{tag}_s{seed}__on_{target}__{start}__{cut}_s*.json")
        assert len(files) == 2, (tag, seed, target, start, cut, files)
        for path in files:
            d = json.load(open(path))
            held += int(d["termination_reasons"].get("fold_held", 0))
            n += int(d["num_envs"])
    return 100 * held / n


def plot(cut: str, cut_label: str) -> None:
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=200)
    xs = list(range(len(COLS)))
    cloth = {4: [], 5: []}  # VBD pinned / rand values, labelled left / right of the points
    for tag, label, own, colour, marker in POLICIES:
        mean = [rate(tag, "vbd" if kind == "vbd" else f"{kind}_{own}", start, cut)
                for kind, start, _ in COLS]
        for seg in (slice(0, 4), slice(4, 6)):  # rigid and cloth are separate panels: no bridge
            ax.plot(xs[seg], mean[seg], color=colour, lw=2, solid_joinstyle="round",
                    solid_capstyle="round", marker=marker, ms=7, mec="white", mew=1.5,
                    label=label if seg.start == 0 else None, zorder=3)
        for x in cloth:
            cloth[x].append(mean[x])
    # value labels, pushed apart where two points sit within a label's height of each other
    for x, values in cloth.items():
        order = sorted(range(len(values)), key=lambda i: values[i])
        label_y = [0.0] * len(values)
        for k, i in enumerate(order):
            label_y[i] = values[i] if k == 0 else max(values[i], label_y[order[k - 1]] + 5.0)
        dx, ha = (-0.12, "right") if x == 4 else (0.12, "left")
        for value, y in zip(values, label_y):
            ax.annotate(f"{value:.0f}%", (x, value), xytext=(x + dx, y), textcoords="data",
                        ha=ha, va="center", color=INK, fontsize=9)
    ax.axvline(3.5, color=MUTED, lw=1)
    for x, text in ((1.5, "Rigid body (each policy's own task)"), (4.5, "VBD cloth")):
        ax.text(x, 104, text, ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(xs, [c[2] for c in COLS], color=INK)
    ax.set_xlim(-0.4, 5.6)
    ax.set_ylim(0, 103)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)], color=INK)
    ax.set_ylabel("Success rate", color=INK)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0)
    ax.legend(loc="lower left", frameon=False, labelcolor=INK)
    fig.suptitle(f"Held-fold success at the {cut_label} cutoff", x=0.06, ha="left", color=INK,
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = f"docs/results/px_success_lines_{cut}.png"
    fig.savefig(out, facecolor="white")
    print(out)


if __name__ == "__main__":
    plot("1p5cm", "1.5 cm")
    plot("4cm", "4 cm")
