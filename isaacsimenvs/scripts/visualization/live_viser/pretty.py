"""Looks shared with the recorded-rollout viewer (experiments/nov25_better_rigidcloth/eval/viser_replay.py):
the robot's URDF visual meshes (a cache built by build_visual_cache.py), the sheet as a smooth surface
through its particles, the colours."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CLOTH_RGB = (88, 150, 214)  # one colour for the whole sheet
MOVING_RGB = STATIONARY_RGB = CLOTH_RGB
TABLE_RGB = (176, 132, 94)
VISUAL_CACHE = Path(__file__).resolve().parents[4] / "outputs/renders/visual_flexiv_wuji.npz"


def load_visuals(path=VISUAL_CACHE):
    """[(body link name, verts, faces, rgb)], or [] without a cache."""
    if not path or not Path(path).exists():
        return []
    d = np.load(path, allow_pickle=True)
    return [(str(k).split("|")[0], d[f"v{i}"], d[f"f{i}"].astype(np.uint32), tuple(int(c) for c in d["rgb"][i]))
            for i, k in enumerate(d["keys"])]


def catmull_rom_weights(n: int, k: int) -> np.ndarray:
    ts = np.linspace(0.0, n - 1.0, k * (n - 1) + 1)
    W = np.zeros((len(ts), n))
    for a, t in enumerate(ts):
        i = min(int(np.floor(t)), n - 2)
        u = t - i
        c = (-0.5 * u ** 3 + u ** 2 - 0.5 * u, 1.5 * u ** 3 - 2.5 * u ** 2 + 1.0,
             -1.5 * u ** 3 + 2.0 * u ** 2 + 0.5 * u, 0.5 * u ** 3 - 0.5 * u ** 2)
        for m, w in zip((i - 1, i, i + 1, i + 2), c):
            W[a, min(max(m, 0), n - 1)] += w
    return W


class SmoothSheet:
    """A (res x res, row-major) particle grid drawn k x finer, split at the middle line into its two halves."""

    def __init__(self, res: int, moving_local, stationary_local, k: int = 4) -> None:
        self.res, self.W = res, catmull_rom_weights(res, k)
        M = self.W.shape[0]
        mv, st = set(int(i) for i in moving_local), set(int(i) for i in stationary_local)
        m_only = np.array(sorted(mv - st)) if mv - st else np.array([res * res - 1])
        s_only = np.array(sorted(st - mv)) if st - mv else np.array([0])
        di = np.mean(m_only // res) - np.mean(s_only // res)
        dj = np.mean(m_only % res) - np.mean(s_only % res)
        axis, sign = (0, np.sign(di)) if abs(di) >= abs(dj) else (1, np.sign(dj))
        c = (res - 1) / 2.0
        faces = {"moving": [], "stationary": []}
        for a in range(M - 1):
            for b in range(M - 1):
                centre = ((a + 0.5) / k, (b + 0.5) / k)[axis]
                half = "moving" if sign * (centre - c) > 0 else "stationary"
                v00, v01, v10, v11 = a * M + b, a * M + b + 1, (a + 1) * M + b, (a + 1) * M + b + 1
                faces[half] += [(v00, v10, v11), (v00, v11, v01)]
        self.faces = {h: np.asarray(f, np.int32) for h, f in faces.items()}

    def surface(self, pts: np.ndarray) -> np.ndarray:
        G = np.asarray(pts, np.float64).reshape(self.res, self.res, 3)
        return np.einsum("ai,bj,ijc->abc", self.W, self.W, G).reshape(-1, 3).astype(np.float32)
