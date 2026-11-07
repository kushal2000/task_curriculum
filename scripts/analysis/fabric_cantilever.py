"""Sim cantilever test for the VBD fabric presets (bos14_fabric_eval.sh).

Each preset's literature bending rigidity B [N*m] and areal density rho [kg/m^2] fix a bending
length c = (B / (rho g))^(1/3). Clamp the env's own 10 cm 7x7 sheet along four rows, let 3 cells
(5 cm) hang off, settle under gravity, and compare the tip-line droop angle with the exact
inextensible elastica for that c. edge_ke is then rescaled, ke <- ke (c_target / c_sim)^3, until
they match. Statics do not depend on edge_kd, so tuning runs at density-scaled damping (settles
fast); the final pass records drops at the current edge_kd 1e-3 and at the scaled one.

Pure Newton, one model holding every sheet. Output: an .npz for fabric_cantilever_viser.py and a
JSON summary.
"""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.optimize import brentq

import newton

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "cloth_geometry", ROOT / "isaacsimenvs/tasks/cloth/utils/cloth_geometry.py"
)
_geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geo)

G = 9.81
SIZE, RES = 0.10, 7
H = SIZE / (RES - 1)
N_CLAMP = 4  # rows i = 0..3 pinned (also fixes the slope); the free 3 cells are the fold flap
L = (RES - N_CLAMP) * H  # overhang, 5 cm = the half sheet a fold lifts
SUBSTEPS_PER_FRAME = 24  # 60 Hz frames, cloth dt 1/1440 s as in the env
DT = 1.0 / 60.0 / SUBSTEPS_PER_FRAME
ITERS = 12  # cloth_cfg.vbd_iterations
KD_NOW = 1.0e-3  # cloth_cfg.edge_kd
RHO_BASE = 2.0  # cloth_cfg.density, the sheet KD_NOW was set on

# The presets (isaacsimenvs/cfg/fabric): literature B and density, tri = tri_ke = tri_ka, the 10-31
# tensile ranges (--tri lo|hi) and the 10-30 tuned edge_ke (--ke-start tuned).
_fspec = importlib.util.spec_from_file_location("fabrics", ROOT / "isaacsimenvs/tasks/cloth/utils/fabrics.py")
fabrics = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(fabrics)
_PRESETS = {n: fabrics.load(n) for n in fabrics.names() if n != "base"}
FABRICS = {
    n: dict(rho=d["cloth"]["density"], B=d["literature"]["bending_rigidity"], tri=d["cloth"]["tri_ke"])
    for n, d in _PRESETS.items()
}
TRI_RANGE = {n: tuple(d["literature"]["tri_range"]) for n, d in _PRESETS.items()}
TUNED_KE = {n: d["cloth"]["edge_ke"] for n, d in _PRESETS.items()}

# --- exact elastica --------------------------------------------------------------------------
def elastica(c, n=200):
    """Clamped inextensible strip of length L under its own weight, bending length c.

    phi(s) is the slope below horizontal. The bending moment at s is the weight beyond s times its
    lever arm, so phi'(s) = (1/c^3) * integral_s^L (x(t) - x(s)) dt with phi(0) = 0. Solved on n
    segments by relaxed fixed-point iteration (a shooting method picks curled-back branches once a
    soft strip passes vertical). Returns node (x, z), z negative downward.
    """
    ds = L / n
    phi = np.zeros(n)
    alpha = min(0.3, 1.5 * (c / L) ** 3)  # the map's gain grows as (L/c)^3
    for _ in range(200000):
        x = np.concatenate([[0], np.cumsum(np.cos(phi) * ds)])
        xm = 0.5 * (x[1:] + x[:-1])  # segment midpoints carry the weight
        # moment at node j from segments i >= j
        lever = np.cumsum((xm * ds)[::-1])[::-1] - x[:-1] * (n - np.arange(n)) * ds
        kappa = lever / c**3
        new = np.cumsum(kappa * ds) - 0.5 * kappa * ds
        step = new - phi
        phi += alpha * step
        if np.abs(step).max() < 1e-10:
            break
    x = np.concatenate([[0], np.cumsum(np.cos(phi) * ds)])
    z = -np.concatenate([[0], np.cumsum(np.sin(phi) * ds)])
    return x, z


def tip_angle(c):
    x, z = elastica(c)
    return math.degrees(math.atan2(-z[-1], x[-1]))


def c_from_angle(theta_deg):
    """Invert tip_angle (monotone decreasing in c); a droop past the range clamps to L/8."""
    if theta_deg >= tip_angle(L / 8):
        return L / 8
    return brentq(lambda c: tip_angle(c) - theta_deg, L / 8, 50 * L)


# --- simulation ------------------------------------------------------------------------------
def simulate(specs, seconds, record_every=2, iterations=ITERS):
    """specs: list of dict(rho, tri, ke, kd). Returns (frames [T, n_sheets, RES*RES, 3], times)."""
    verts, idx = _geo.grid_mesh(SIZE, RES)
    builder = newton.ModelBuilder()
    starts = []
    for k, sp in enumerate(specs):
        starts.append(builder.particle_count)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.25 * k, 1.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=verts,
            indices=idx,
            density=sp["rho"],
            tri_ke=sp["tri"],
            tri_ka=sp["tri"],
            tri_kd=1.5e-6,
            edge_ke=sp["ke"],
            edge_kd=sp["kd"],
            particle_radius=0.008,
        )
        for i in range(N_CLAMP):
            for j in range(RES):
                p = starts[-1] + i * RES + j
                builder.particle_mass[p] = 0.0
                builder.particle_flags[p] = builder.particle_flags[p] & ~newton.ParticleFlags.ACTIVE
    builder.color(include_bending=True)
    model = builder.finalize()
    solver = newton.solvers.SolverVBD(model=model, iterations=iterations)
    s0, s1 = model.state(), model.state()
    control = model.control()

    def frame():
        nonlocal s0, s1
        for _ in range(SUBSTEPS_PER_FRAME):
            s0.clear_forces()
            solver.step(s0, s1, control, None, DT)
            s0, s1 = s1, s0

    frame()  # warm-up compile; state is reset below
    s0.particle_q.assign(model.particle_q)
    s0.particle_qd.zero_()
    if wp.get_device().is_cuda:
        with wp.ScopedCapture() as cap:
            frame()
        graph = cap.graph
        s0.particle_q.assign(model.particle_q)
        s0.particle_qd.zero_()
        step = lambda: wp.capture_launch(graph)
    else:
        step = frame

    n_frames = int(round(seconds * 60))
    out, times = [], []
    q = s0.particle_q.numpy()
    out.append(q.copy()), times.append(0.0)
    for f in range(1, n_frames + 1):
        step()
        if f % record_every == 0 or f == n_frames:
            out.append(s0.particle_q.numpy().copy())
            times.append(f / 60.0)
    n = RES * RES
    frames = np.stack([np.stack([o[s : s + n] for s in starts]) for o in out])
    origin = np.array([[0.0, 0.25 * k, 1.0] for k in range(len(specs))])
    frames = frames - origin[None, :, None, :]  # sheet-local
    return frames, np.array(times), float(np.abs(s0.particle_qd.numpy()).max())


def droop(sheet_q):
    """Tip-line angle [deg] from the clamp edge (row 1) to the tip row, averaged across width."""
    clamp = sheet_q[(N_CLAMP - 1) * RES : N_CLAMP * RES]
    tip = sheet_q[(RES - 1) * RES : RES * RES]
    d = (tip - clamp).mean(0)
    return math.degrees(math.atan2(-d[2], d[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/fabric_cantilever")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--drop", type=float, default=8.0)
    ap.add_argument("--tri", choices=["now", "lo", "hi"], default="now", help="tri_ke = tri_ka: preset or TRI_RANGE end")
    ap.add_argument("--iters", type=int, default=ITERS, help="VBD iterations (env: 12)")
    ap.add_argument("--ke-start", choices=["lit", "tuned"], default="lit", help="first-round edge_ke: B/h or TUNED_KE")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    names = list(FABRICS)
    for n in names:
        f = FABRICS[n]
        f["c"] = (f["B"] / (f["rho"] * G)) ** (1 / 3)
        f["theta_target"] = tip_angle(f["c"])
        f["kd_scaled"] = KD_NOW * f["rho"] / RHO_BASE
        f["ke_old"] = 100.0 * f["B"]
        f["ke"] = f["B"] / H if a.ke_start == "lit" else TUNED_KE[n]  # starting guess
        if a.tri != "now":
            f["tri"] = TRI_RANGE[n][0 if a.tri == "lo" else 1]
        print(f"[target] {n:8s} c {100 * f['c']:.2f} cm  droop {f['theta_target']:.1f} deg", flush=True)

    history = {n: [] for n in names}
    for r in range(a.rounds):
        t0 = time.time()
        specs = [dict(rho=FABRICS[n]["rho"], tri=FABRICS[n]["tri"], ke=FABRICS[n]["ke"], kd=FABRICS[n]["kd_scaled"]) for n in names]
        frames, _, vmax = simulate(specs, a.settle, record_every=60, iterations=a.iters)
        for k, n in enumerate(names):
            f = FABRICS[n]
            th = droop(frames[-1, k])
            th_prev = droop(frames[-2, k])
            c_sim = c_from_angle(th)
            history[n].append(dict(ke=f["ke"], theta=th, c_sim=c_sim, drift_last_s=th - th_prev))
            print(f"[round {r}] {n:8s} ke {f['ke']:.3e} ({f['ke'] / f['B']:.1f} B)  droop {th:.2f} "
                  f"(target {f['theta_target']:.2f}, drift {th - th_prev:+.3f}/s)  c_sim {100 * c_sim:.2f} cm", flush=True)
            if r < a.rounds - 1:
                f["ke"] *= (f["c"] / c_sim) ** 3
        print(f"[round {r}] max |v| at end {vmax:.2e} m/s, {time.time() - t0:.1f} s", flush=True)

    # Final pass for the viewer: old mapping (100 B) and tuned ke, each at current and scaled kd.
    variants = [("old_ke", "ke_old"), ("tuned_ke", "ke")]
    dampings = [("kd_now", None), ("kd_scaled", "kd_scaled")]
    specs, labels = [], []
    for vk, kkey in variants:
        for dk, dkey in dampings:
            for n in names:
                f = FABRICS[n]
                specs.append(dict(rho=f["rho"], tri=f["tri"], ke=f[kkey], kd=f[dkey] if dkey else KD_NOW))
                labels.append((n, vk, dk))
    frames, times, vmax = simulate(specs, a.drop, record_every=2, iterations=a.iters)
    print(f"[final] max |v| at end {vmax:.2e} m/s", flush=True)

    summary = {"tri_set": a.tri, "L_m": L, "h_m": H, "dt": DT, "iterations": a.iters, "kd_now": KD_NOW, "fabrics": {}}
    for n in names:
        f = FABRICS[n]
        row = {k: f[k] for k in ("rho", "B", "tri", "c", "theta_target", "ke_old", "ke", "kd_scaled")}
        row["ke_over_B"] = f["ke"] / f["B"]
        row["rounds"] = history[n]
        for (ln, vk, dk), k in zip(labels, range(len(labels))):
            if ln != n:
                continue
            th = np.array([droop(frames[t, k]) for t in range(len(times))])
            final = th[-1]
            t90 = float(times[np.argmax(th >= 0.9 * final)]) if final > 0 else float("nan")
            row[f"{vk}/{dk}"] = dict(theta_final=float(final), t90_s=t90)
        summary["fabrics"][n] = row
        print(f"[final] {n:8s} " + "  ".join(f"{vk}/{dk}: {row[f'{vk}/{dk}']['theta_final']:.1f} deg t90 {row[f'{vk}/{dk}']['t90_s']:.2f}s"
                                            for vk, _ in variants for dk, _ in dampings), flush=True)

    ex = {n: np.stack(elastica(FABRICS[n]["c"])) for n in names}
    _, faces = _geo.grid_mesh(SIZE, RES)
    np.savez_compressed(
        out / "cantilever.npz",
        frames=frames.astype(np.float32),
        times=times,
        labels=np.array(["|".join(l) for l in labels]),
        faces=np.array(faces, dtype=np.int32).reshape(-1, 3),
        clamp_x=-SIZE / 2 + (N_CLAMP - 1) * H,
        **{f"elastica_{n}": ex[n] for n in names},
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
