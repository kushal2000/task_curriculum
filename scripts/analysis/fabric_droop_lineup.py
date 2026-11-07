"""Side-by-side cantilever droop of the default VBD cloth and the 6 tuned fabric presets
(bos14_fabric_eval.sh), for a screenshot. Reuses fabric_cantilever.simulate; GPU job.
Output: docs/results/fabric_droop/droop.npz (read by fabric_droop_viser.py / fabric_droop_render.py).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fc", ROOT / "scripts/analysis/fabric_cantilever.py")
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)

# name, rho [kg/m^2], tri_ke = tri_ka, edge_ke, edge_kd: the presets in isaacsimenvs/cfg/fabric.
SHEETS = [
    ("default VBD" if n == "base" else n, p["density"], p["tri_ke"], p["edge_ke"], p["edge_kd"])
    for n, p in ((n, fc.fabrics.cloth_params(n)) for n in fc.fabrics.names())
]

def main():
    out = ROOT / "docs/results/fabric_droop"
    out.mkdir(parents=True, exist_ok=True)
    specs = [dict(rho=r, tri=t, ke=ke, kd=kd) for _, r, t, ke, kd in SHEETS]
    frames, times, vmax = fc.simulate(specs, 8.0, record_every=2)
    angles = {n: fc.droop(frames[-1, k]) for k, (n, *_) in enumerate(SHEETS)}
    for n, a in angles.items():
        print(f"{n:12s} droop {a:.1f} deg", flush=True)
    print(f"max |v| at end {vmax:.2e} m/s", flush=True)
    verts, faces = fc._geo.grid_mesh(fc.SIZE, fc.RES)
    np.savez_compressed(
        out / "droop.npz",
        frames=frames.astype(np.float32),
        times=times,
        names=np.array([s[0] for s in SHEETS]),
        faces=np.array(faces, dtype=np.int32).reshape(-1, 3),
        clamp_x=-fc.SIZE / 2 + (fc.N_CLAMP - 1) * fc.H,
        res=fc.RES,
        n_clamp=fc.N_CLAMP,
    )
    (out / "angles.json").write_text(json.dumps({"sheets": [list(s) for s in SHEETS], "droop_deg": angles}, indent=1))


if __name__ == "__main__":
    main()
