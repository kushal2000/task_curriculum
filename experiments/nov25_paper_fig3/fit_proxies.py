"""Fit the hinge angles of a proxy with one added hinge to a VBD frame: least squares on the sheet's grid points,
each attached to the plate that holds it. -> <dir>/fit.npz entries used by render.py.

    python experiments/nov25_paper_fig3/fit_proxies.py <dir> <frame>
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from proxies import PROXIES  # noqa: E402


def fit(name, d, frame):
    z = np.load(d / f"{name}.npz")
    rest, q = z["rest"], z["frames"][frame]
    z0 = rest[:, 2].mean()
    s = PROXIES[f"{name}_flap"]
    own = s.owners(rest[:, :2])
    names = list(s.children)

    def pts(a):
        tf = s.transforms(dict(zip(names, a))); out = np.empty_like(q)
        for p in s.plates:
            k = own == p
            r, t = tf[p]; out[k] = (np.c_[rest[k, :2], np.zeros(k.sum())] @ r.T) + t + [0, 0, z0]
        return out

    def cost(a):
        p = pts(a)
        return np.mean(np.sum((p - q) ** 2, 1)) + 10 * np.mean(np.clip(-p[:, 2], 0, None) ** 2) + 1e-2 * (max(0, -a[0], a[0] - np.pi) ** 2 + sum(max(0, abs(x) - np.pi) ** 2 for x in a[1:]))

    best = None
    for init in np.ndindex(*([7] * len(names))):
        a0 = np.array(init) / 6 * 2 * np.pi - np.pi
        r = minimize(cost, a0, method="Nelder-Mead")
        if best is None or r.fun < best.fun:
            best = r
    ang = dict(zip(names, best.x))
    plain = np.deg2rad(np.linspace(0, 180, 361))
    b = PROXIES["box2"]; own2 = b.owners(rest[:, :2])
    def rms2(t):
        tf = b.transforms({"E": t}); out = np.empty_like(q)
        for p in b.plates:
            k = own2 == p; r, tt = tf[p]; out[k] = (np.c_[rest[k, :2], np.zeros(k.sum())] @ r.T) + tt + [0, 0, z0]
        return np.sqrt(np.mean(np.sum((out - q) ** 2, 1)))
    print(f"{name} frame {frame}: rms {1e3 * best.fun ** .5:.1f} mm with the added hinge, {1e3 * min(map(rms2, plain)):.1f} mm with"
          f" the single hinge; angles {({k: round(float(np.rad2deg(v))) for k, v in ang.items()})}", flush=True)
    return ang


if __name__ == "__main__":
    d, frame = Path(sys.argv[1]), int(sys.argv[2])
    np.savez(d / "fit.npz", **{f"{n}/{k}": v for n in ("strip", "corner") for k, v in fit(n, d, frame).items()})
