"""Compare how the VBD cloth, the Newton chain and the PhysX chain respond to the same hand motion.

    python3 scripts/analysis/contact_fingerprint_compare.py --source newton [--tag _x]

Reads ``docs/results/fingerprint_<source>_on_{vbd,newton,physx}<tag>.pt`` (bos14_contact_fingerprint.sh)
and prints, every few steps until an env's episode first resets, the env-median of: sheet centroid
drift in the table plane, lowest and highest particle above the table, and fold error. The
centroid/height traces are what the policy's contact strategy depends on -- a sheet that slides
away under a push, or tilts and sinks under a fingertip, rewards a different motion.
"""

from __future__ import annotations

import argparse

import torch

TABLE_TOP = 0.53


def first_reset(p: torch.Tensor) -> torch.Tensor:
    """(N,) the first step at which each env's sheet is back at its spawn height (a reset)."""
    z = p[..., 2].mean(dim=-1)  # (T, N)
    spawn = z[0]
    hit = (z[1:] - spawn).abs() < 0.002
    t = torch.full((z.shape[1],), z.shape[0], dtype=torch.long)
    for n in range(z.shape[1]):
        idx = hit[:, n].nonzero()
        later = idx[idx[:, 0] > 20]
        if later.numel():
            t[n] = int(later[0, 0]) + 1
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="newton")
    ap.add_argument("--tag", default="")
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()
    runs = {}
    for target in ("vbd", "newton", "physx"):
        path = f"docs/results/fingerprint_{args.source}_on_{target}{args.tag}.pt"
        try:
            runs[target] = torch.load(path)
        except FileNotFoundError:
            print(f"(missing {path})")
    horizon = min(int(first_reset(r["particles"]).min()) for r in runs.values())
    print(f"source actions: {args.source}; compared up to step {horizon} (first reset in any run)")
    cols = "".join(f"{t:>34s}" for t in runs)
    print(f"{'t':>4s}{cols}")
    print(f"{'':>4s}" + "".join(f"{'drift  zmin  zmax  fold':>34s}" for _ in runs))
    for t in range(0, horizon, args.every):
        line = f"{t:4d}"
        for r in runs.values():
            p = r["particles"][t]  # (N, P, 3), env-relative
            c = p.mean(dim=1)
            drift = (c[:, :2] - r["particles"][0].mean(dim=1)[:, :2]).norm(dim=-1).median()
            zmin = (p[..., 2].amin(dim=1).median() - TABLE_TOP) * 1e3
            zmax = (p[..., 2].amax(dim=1).median() - TABLE_TOP) * 1e3
            fe = r["fold_err"][t].median()
            line += f"{float(drift) * 1e3:13.1f}mm{float(zmin):6.1f}{float(zmax):6.1f}{float(fe):7.3f}"
        print(line)


if __name__ == "__main__":
    main()
