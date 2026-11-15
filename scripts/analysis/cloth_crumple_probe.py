"""Does the env's VBD sheet still crumple and deform freely with plasticity on? (beyond the fold test)

Three scripted handlings of the 10 cm 7x7 sheet on a ground plane, each run on three copies of the
sheet side by side (one model per fabric and scenario):
    elastic   plasticity off (yield inf)
    current   single-stage, the preset's committed yield
    twostage  two-stage, the proposed yield with set curvature --set (default 80 /m)

scrunch     rows 0 and 6 held on the ground, row 6 slid toward row 0 to 40 % of the width (1.5 s),
            held 1 s, let go, 3 s free. The middle has to buckle up into wrinkles.
pinch_lift  the centre particle lifted 12 cm (1.5 s, the sheet hangs like a handkerchief), held 1 s,
            lowered to 1 cm (1 s), let go, 3 s free: the sheet collapses into a heap.
gather      the four corners carried to one point 4 cm above the centre (1.5 s), held 1 s, let go,
            3 s free: a pouch that falls into a crumpled pile.

Measured at the end of the hold (deformed) and at the end (settled): height above the ground,
convex-hull footprint (flat sheet = 100 cm^2), hinges bent past 30 deg, max hinge angle, and the
plastic set left (max rest angle, edges with rest > 10 deg); scrunch also the recovered width.
Output: --out/crumple.json and crumple.npz (frames, for a viewer / snapshots).
"""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cloth_plasticity_probe", ROOT / "scripts/analysis/cloth_plasticity_probe.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

RES, NP, SIZE = P.RES, P.NP, P.SIZE
Z0 = P.PARTICLE_RADIUS  # resting particle height on the ground

# Committed single-stage yields (cfg/fabric/*_measured.yaml) and the proposed two-stage ones.
CURRENT = dict(big_napkin=25, chiffon=42, cotton=20, napkin=30, silk=32, towel=60, denim=100)
PROPOSED = dict(big_napkin=20, chiffon=30, cotton=20, napkin=30, silk=30, towel=100, denim=100)
MODES = ["elastic", "current", "twostage"]


def hull_area(xy):
    p = sorted(map(tuple, np.round(xy, 6)))
    if len(p) < 3:
        return 0.0

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    h = half(p)[:-1] + half(p[::-1])[:-1]
    x, y = np.array(h).T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def run(fab, scenario, set_kappa, record_every):
    preset = f"{fab}_measured"
    kappas = [math.inf, float(CURRENT[fab]), float(PROPOSED[fab])]
    P.SET_KAPPA = 0.0
    sim = P.Sim(preset, len(MODES), kappas, ground=True, clamp_rows=0, z0=Z0 + 0.002, graph=False)
    m = sim.model
    sets = np.array([0.0, 0.0, set_kappa])[sim.sheet_of_edge]
    sim.plast.set_set_curvature(sets)
    verts = np.array(P._geo.grid_mesh(SIZE, RES)[0])
    origins = np.array([[0.0, 0.25 * s, 0.0] for s in range(len(MODES))])
    centre = verts.mean(0)

    if scenario == "scrunch":
        local = [P.pid(0, 0, j) for j in range(RES)] + [P.pid(0, RES - 1, j) for j in range(RES)]
        axis = verts[P.pid(0, RES - 1, 0)] - verts[P.pid(0, 0, 0)]
        span = np.linalg.norm(axis)
        axis /= span
    elif scenario == "pinch_lift":
        local = [P.pid(0, RES // 2, RES // 2)]
    else:
        local = [P.pid(0, 0, 0), P.pid(0, 0, RES - 1), P.pid(0, RES - 1, 0), P.pid(0, RES - 1, RES - 1)]
    nl = len(local)
    idx = np.array([sim.starts[s] + i for s in range(len(MODES)) for i in local])
    sim.drive_idx_np = idx
    sim.drive_idx = wp.array(idx, dtype=int)
    sim.drive_target = wp.zeros(len(idx), dtype=wp.vec3)
    sim.s0.particle_q.assign(m.particle_q)
    sim.s0.particle_qd.zero_()
    sim.plast.reset()
    if wp.get_device().is_cuda:
        with wp.ScopedCapture() as cap:
            sim._frame()
        sim.graph = cap.graph
        sim.s0.particle_q.assign(m.particle_q)
        sim.s0.particle_qd.zero_()
        sim.plast.reset()

    rest0 = verts[local] + np.array([0.0, 0.0, Z0])  # driven points at rest, sheet frame

    def targets(u_move, u_lower=0.0):
        """Driven-point positions per sheet for move progress u_move (and lower progress for pinch)."""
        out = []
        for s in range(len(MODES)):
            p = rest0.copy()
            if scenario == "scrunch":
                p[RES:] -= 0.6 * span * P.smooth(u_move) * axis
            elif scenario == "pinch_lift":
                p[:, 2] = Z0 + 0.12 * P.smooth(u_move) - (0.12 - 0.01) * P.smooth(u_lower)
            else:
                goal = np.array([centre[0], centre[1], Z0 + 0.04])
                p = p + (goal - p) * P.smooth(u_move)
            out.append(p + origins[s])
        return np.concatenate(out)

    fr = lambda sec: int(round(sec * 60))
    rec, times, t = [], [], 0.0
    phase_t, phase_names = [], []

    def step():
        nonlocal t
        sim.frame()
        t += 1.0 / 60.0
        if int(round(t * 60)) % record_every == 0:
            rec.append(sim.q().astype(np.float32)), times.append(t)

    def mark(name):
        phase_t.append(t), phase_names.append(name)

    def measure():
        q = sim.q()
        th = np.abs(P.plast.dihedral_angles(sim.s0.particle_q, m.edge_indices))
        rest = np.abs(m.edge_rest_angle.numpy())
        res = []
        for s in range(len(MODES)):
            sq = q[sim.starts[s]:sim.starts[s] + NP] - origins[s]
            e = sim.sheet_of_edge == s
            r = dict(height_cm=100 * float(sq[:, 2].max() - Z0), footprint_cm2=1e4 * hull_area(sq[:, :2]),
                     n_bent30=int((th[e] > math.radians(30)).sum()), theta_max_deg=float(np.degrees(th[e].max())),
                     set_max_deg=float(np.degrees(rest[e].max())), n_set10=int((rest[e] > math.radians(10)).sum()),
                     finite=bool(np.isfinite(sq).all()))
            if scenario == "scrunch":
                w = np.linalg.norm(sq[[P.pid(0, RES - 1, j) for j in range(RES)]].mean(0)
                                   - sq[[P.pid(0, 0, j) for j in range(RES)]].mean(0))
                r["width_frac"] = float(w / span)
            res.append(r)
        return res

    t0 = time.time()
    mark("settle")
    for _ in range(fr(0.5)):
        step()
    mark("handle")
    n = fr(1.5)
    for k in range(1, n + 1):
        sim.set_drive(targets(k / n))
        step()
    mark("hold")
    for _ in range(fr(1.0)):
        step()
    held = measure()
    if scenario == "pinch_lift":
        mark("lower")
        n = fr(1.0)
        for k in range(1, n + 1):
            sim.set_drive(targets(1.0, k / n))
            step()
    mark("let go")
    sim.set_drive(None)
    for _ in range(fr(3.0)):
        step()
    final = measure()
    rows = []
    for s, mode in enumerate(MODES):
        rows.append(dict(fabric=fab, scenario=scenario, mode=mode, yield_kappa=kappas[s],
                         set_kappa=float([0.0, 0.0, set_kappa][s]), held=held[s], final=final[s]))
        h, f = held[s], final[s]
        extra = f" width {h.get('width_frac', 0):.2f}->{f.get('width_frac', 0):.2f}" if scenario == "scrunch" else ""
        print(f"[crumple {fab:10s} {scenario:10s} {mode:8s}] held: h {h['height_cm']:4.1f} cm  area {h['footprint_cm2']:5.1f}  "
              f"bent30 {h['n_bent30']:3d}  | final: h {f['height_cm']:4.1f}  area {f['footprint_cm2']:5.1f}  "
              f"bent30 {f['n_bent30']:3d}  set max {f['set_max_deg']:5.1f} n>10 {f['n_set10']:3d}{extra}  finite {f['finite']}",
              flush=True)
    print(f"[crumple {fab} {scenario}] {time.time() - t0:.1f} s", flush=True)
    return rows, np.stack(rec), np.array(times), (phase_t, phase_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="debug_outputs/train_logs/results/cloth_crumple")
    ap.add_argument("--fabrics", nargs="+", default=list(CURRENT))
    ap.add_argument("--scenarios", nargs="+", default=["scrunch", "pinch_lift", "gather"])
    ap.add_argument("--set", type=float, default=80.0, help="two-stage set curvature [1/m]")
    ap.add_argument("--record-every", type=int, default=2)
    ap.add_argument("--tri-ke", type=float, default=None, help="override stretch stiffness tri_ke = tri_ka")
    a = ap.parse_args()
    if a.tri_ke is not None:
        for f in P.FABRICS.values():
            f["tri"] = a.tri_ke
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows, npz = [], {}
    for fab in a.fabrics:
        for sc in a.scenarios:
            r, frames, times, (pt, pn) = run(fab, sc, a.set, a.record_every)
            rows += r
            npz[f"{fab}__{sc}_frames"] = frames
            npz[f"{sc}_times"] = times
            npz[f"{sc}_phase_t"], npz[f"{sc}_phase_names"] = np.array(pt), np.array(pn)
    faces = np.array(P.grid_mesh(SIZE, RES)[1], dtype=np.int32).reshape(-1, 3)
    np.savez_compressed(out / "crumple.npz", faces=faces, modes=np.array(MODES), sheet_spacing_y=0.25, res=RES, size=SIZE, **npz)
    (out / "crumple.json").write_text(json.dumps(rows, indent=1))
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
