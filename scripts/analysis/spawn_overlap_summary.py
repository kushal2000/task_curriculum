"""Does the sheet start inside / against the hand? From fold_start_probe.py dumps.

    .venv_isaaclab3/bin/python scripts/analysis/spawn_overlap_summary.py docs/results/fold_start_probe_overlap_*.pt

The sheet spawns cloth.start_height (15 cm) above the table and falls. Until it lands nothing but the
robot can touch it, so any motion beyond free fall in the first steps is contact with the hand.
Per env: fingertip-to-sheet clearance at spawn (fingertip POINTS to grid points -- a lower bound on
how close the hand's geometry is), and the largest deviation from free fall over the first policy
steps (semi-implicit Euler: after n physics steps from rest the drop is g dt^2 n(n+1)/2).
"""

from __future__ import annotations

import sys

import torch

G = 9.81


def summarise(path: str, steps: int = 6, tol: float = 2e-3) -> None:
    d = torch.load(path, weights_only=False)
    P, tips = d["particles"], d["tips"]  # (T, N, M, 3), (T, N, 5, 3)
    dt, dec = d.get("dt_sim", 1 / 120), d.get("decimation", 2)
    n_env = P.shape[1]
    clear = (tips[0].unsqueeze(2) - P[0].unsqueeze(1)).norm(dim=-1).amin(dim=(1, 2)) * 1e3  # mm
    dev = torch.zeros(n_env)
    for t in range(1, min(steps, P.shape[0] - 1) + 1):
        n = t * dec
        drop = G * dt * dt * n * (n + 1) / 2
        want = P[0].clone()
        want[..., 2] -= drop
        dev = torch.maximum(dev, (P[t] - want).norm(dim=-1).amax(dim=1))
    hit = dev > tol
    tag = "PINNED" if d.get("pinned") else "training resets"
    folded = d.get("start_folded")
    nf = int(folded.sum()) if folded is not None else 0
    print(f"{path.split('/')[-1]}  [{tag}, {nf}/{n_env} folded starts]")
    print(f"  fingertip-sheet clearance at spawn [mm]: min {clear.min():.0f}, p10 {clear.quantile(0.1):.0f}, "
          f"median {clear.median():.0f}; envs < 20 mm: {int((clear < 20).sum())}/{n_env}")
    print(f"  sheet pushed during its first {steps} steps (> {tol * 1e3:.0f} mm off free fall): "
          f"{int(hit.sum())}/{n_env} envs; worst {dev.max() * 1e3:.1f} mm")
    if hit.any():
        idx = hit.nonzero(as_tuple=True)[0][:8].tolist()
        print("  e.g. env " + ", ".join(f"{e} (clear {clear[e]:.0f} mm, off {dev[e] * 1e3:.0f} mm)" for e in idx))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarise(p)
