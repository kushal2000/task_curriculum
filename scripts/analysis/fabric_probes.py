"""Sim probes for the 10-31 literature ranges of the VBD fabric presets (fabric_cantilever.py has the
bending one).

stretch: the env's 10 cm 7x7 sheet, no gravity, row 0 pinned, the far row pulled with a line load
    [N/m] that ramps over 1 s and then holds. The gauge strain between rows 1 and 5 gives the sim's
    tensile stiffness E_sim = load / strain, compared with the continuum value (8/3) tri_ke for
    tri_ka = tri_ke. Run at 12 (env) and 48 iterations to separate under-convergence from the mesh.

ply: two plies per fabric. The lower sheet is pinned flat at z = 0; the upper one starts at
    z = self_contact_radius, shifted half a cell so its vertices sit over triangle interiors, with
    its x-min row pinned. A 3x3 patch at its centre is pressed down, then dragged along +x at half
    the press force. A ply has crossed when an upper vertex inside the lower sheet's footprint is
    below z = 0. One model per (radius, press force), since the radius is a solver-global scalar.

Pure Newton, same dt / iterations / contact settings as the env. Output: JSON summaries (+ the ply
frames as .npz) in --out.
"""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fabric_cantilever", ROOT / "scripts/analysis/fabric_cantilever.py")
cant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cant)

SIZE, RES, H, DT, SUB = cant.SIZE, cant.RES, cant.H, cant.DT, cant.SUBSTEPS_PER_FRAME
FABRICS, TRI_RANGE, TUNED_KE = cant.FABRICS, cant.TRI_RANGE, cant.TUNED_KE
NAMES = list(FABRICS)

# Env contact settings (cloth_cfg.py): self-contact uses the model's soft_contact_* too.
SOFT_KE, SOFT_KD = 5.0e2, 5.0e-3
SOFT_MU = {n: cant.fabrics.cloth_params(n)["soft_contact_mu"] for n in NAMES}
BOUND_RELAX = 0.42
# 10-31 literature thickness ranges [m]; the ply probe sweeps radii across them.
THICK_RANGE = {
    "chiffon": (0.08e-3, 0.2e-3),
    "silk": (0.05e-3, 0.12e-3),
    "cotton": (0.15e-3, 0.3e-3),
    "napkin": (0.3e-3, 0.5e-3),
    "towel": (2.0e-3, 5.0e-3),
    "denim": (0.6e-3, 1.0e-3),
}


def add_sheet(builder, origin, rho, tri, ke, kd):
    verts, idx = cant._geo.grid_mesh(SIZE, RES)
    start = builder.particle_count
    builder.add_cloth_mesh(
        pos=wp.vec3(*origin), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
        vertices=verts, indices=idx, density=rho, tri_ke=tri, tri_ka=tri, tri_kd=1.5e-6,
        edge_ke=ke, edge_kd=kd, particle_radius=0.008,
    )
    return start


def pin(builder, particles):
    for p in particles:
        builder.particle_mass[p] = 0.0
        builder.particle_flags[p] = builder.particle_flags[p] & ~newton.ParticleFlags.ACTIVE


def pid(start, i, j):
    """Particle of row i (along x) and column j (along y)."""
    return start + i * RES + j


class Runner:
    """One model + solver, stepped in 60 Hz frames with a per-frame external force array."""

    def __init__(self, builder, iterations, **solver_kw):
        builder.color(include_bending=True)
        self.model = builder.finalize()
        self.model.soft_contact_ke = SOFT_KE
        self.model.soft_contact_kd = SOFT_KD
        self.solver = newton.solvers.SolverVBD(model=self.model, iterations=iterations, **solver_kw)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.f_ext = wp.zeros(self.model.particle_count, dtype=wp.vec3)
        self._frame()  # compile
        self.s0.particle_q.assign(self.model.particle_q)
        self.s0.particle_qd.zero_()
        self.graph = None
        if wp.get_device().is_cuda and not solver_kw.get("particle_enable_self_contact", False):
            with wp.ScopedCapture() as cap:
                self._frame()
            self.graph = cap.graph
            self.s0.particle_q.assign(self.model.particle_q)
            self.s0.particle_qd.zero_()

    def _frame(self):
        for _ in range(SUB):
            wp.copy(self.s0.particle_f, self.f_ext)
            self.solver.step(self.s0, self.s1, self.control, None, DT)
            self.s0, self.s1 = self.s1, self.s0

    def frame(self, f_ext_np):
        self.f_ext.assign(f_ext_np.astype(np.float32))
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._frame()

    def q(self):
        return self.s0.particle_q.numpy()


# --- stretch ---------------------------------------------------------------------------------
def stretch(a):
    loads = [20.0, 100.0]  # N/m across the width: ~2 N and ~10 N finger pulls on a 10 cm sheet
    w = np.ones(RES)
    w[[0, -1]] = 0.5
    w /= w.sum()  # consistent nodal weights of a uniform edge load
    specs = []
    for n in NAMES:
        f = FABRICS[n]
        for tag, tri in (("now", f["tri"]), ("lo", TRI_RANGE[n][0]), ("hi", TRI_RANGE[n][1])):
            for load in loads:
                specs.append(dict(fabric=n, tri_tag=tag, tri=tri, load=load))
    out = {"loads_N_per_m": loads, "seconds": a.seconds, "results": {}}
    for iters in a.iters:
        t0 = time.time()
        b = newton.ModelBuilder(gravity=0.0)
        starts = []
        for k, sp in enumerate(specs):
            f = FABRICS[sp["fabric"]]
            s = add_sheet(b, (0.0, 0.25 * k, 1.0), f["rho"], sp["tri"], TUNED_KE[sp["fabric"]], 1e-3 * f["rho"] / 2.0)
            pin(b, [pid(s, 0, j) for j in range(RES)])
            starts.append(s)
        r = Runner(b, iters)
        full = np.zeros((r.model.particle_count, 3))
        for s, sp in zip(starts, specs):
            for j in range(RES):
                full[pid(s, RES - 1, j), 0] = sp["load"] * SIZE * w[j]
        n_frames = int(round(a.seconds * 60))
        hist = []
        for fr in range(1, n_frames + 1):
            ramp = min(1.0, fr / 60.0)
            r.frame(full * ramp)
            if fr % 15 == 0:
                hist.append((fr / 60.0, r.q()))
        q_end = hist[-1][1]
        q_1s = next(q for t, q in hist if t >= a.seconds - 1.0 - 1e-9)

        def gauge(q, s):
            x1 = q[[pid(s, 1, j) for j in range(1, RES - 1)], 0]
            x5 = q[[pid(s, RES - 2, j) for j in range(1, RES - 1)], 0]
            return float(np.mean(x5 - x1) / ((RES - 3) * H) - 1.0)

        res = []
        for s, sp in zip(starts, specs):
            eps = gauge(q_end, s)
            eps_prev = gauge(q_1s, s)
            width = float(np.ptp(q_end[[pid(s, 3, j) for j in range(RES)], 1]))
            e_sim = sp["load"] / eps if eps > 0 else float("nan")
            e_cont = 8.0 / 3.0 * sp["tri"]
            row = dict(sp, iterations=iters, strain=eps, strain_drift_last_s=eps - eps_prev,
                       mid_width_ratio=width / SIZE, E_sim=e_sim, E_cont=e_cont, E_ratio=e_sim / e_cont,
                       finite=bool(np.isfinite(q_end).all()))
            res.append(row)
            print(f"[stretch it{iters}] {sp['fabric']:8s} tri {sp['tri_tag']:3s} {sp['tri']:7.0f}  load {sp['load']:5.0f} N/m  "
                  f"strain {100 * eps:7.3f}%  (drift {100 * (eps - eps_prev):+.4f}%/s)  E_sim {e_sim:9.1f}  "
                  f"E_cont {e_cont:9.1f}  ratio {e_sim / e_cont:.3f}  width {width / SIZE:.3f}", flush=True)
        out["results"][f"it{iters}"] = res
        print(f"[stretch it{iters}] {time.time() - t0:.1f} s", flush=True)
    return out


# --- ply -------------------------------------------------------------------------------------
def ply(a):
    out = {"radii_m": a.radii, "press_N": a.press, "results": []}
    frames_all = {}
    n_frames = int(round(2.5 * 60))
    patch_ij = [(i, j) for i in range(2, 5) for j in range(2, 5)]
    for r_c in a.radii:
        for press in a.press:
            t0 = time.time()
            b = newton.ModelBuilder()  # gravity on
            low, up = [], []
            for k, n in enumerate(NAMES):
                f = FABRICS[n]
                tri = math.sqrt(TRI_RANGE[n][0] * TRI_RANGE[n][1])
                kd = 1e-3 * f["rho"] / 2.0
                s_lo = add_sheet(b, (0.0, 0.25 * k, 0.0), f["rho"], tri, TUNED_KE[n], kd)
                pin(b, range(s_lo, s_lo + RES * RES))
                s_up = add_sheet(b, (H / 2, 0.25 * k + H / 2, r_c), f["rho"], tri, TUNED_KE[n], kd)
                pin(b, [pid(s_up, 0, j) for j in range(RES)])
                low.append(s_lo), up.append(s_up)
            # self-contact friction is model-global: use the median preset
            r = Runner(
                b, cant.ITERS,
                particle_enable_self_contact=True,
                particle_self_contact_radius=r_c,
                particle_self_contact_margin=1.5 * r_c,
                particle_collision_detection_interval=-1,
                particle_conservative_bound_relaxation=BOUND_RELAX,
            )
            r.model.soft_contact_mu = float(np.median(list(SOFT_MU.values())))
            full_press = np.zeros((r.model.particle_count, 3))
            full_drag = np.zeros_like(full_press)
            for s in up:
                for i, j in patch_ij:
                    full_press[pid(s, i, j), 2] = -press / len(patch_ij)
                    full_drag[pid(s, i, j), 0] = 0.5 * press / len(patch_ij)
            rec, times = [], []
            t_run = time.time()
            for fr in range(1, n_frames + 1):
                t = fr / 60.0
                p = np.clip((t - 0.5) / 0.5, 0.0, 1.0)
                d = np.clip((t - 1.5) / 0.5, 0.0, 1.0)
                r.frame(full_press * p + full_drag * d)
                if fr % 2 == 0:
                    rec.append(r.q().copy()), times.append(t)
            t_run = time.time() - t_run
            rec = np.stack(rec)
            times = np.array(times)
            phases = {"settle": (0.3, 0.5), "press": (1.0, 1.5), "drag": (2.0, 2.5)}
            for k, n in enumerate(NAMES):
                qu = rec[:, up[k] : up[k] + RES * RES]
                ql = rec[:, low[k] : low[k] + RES * RES]
                cx, cy = ql[0, :, 0].mean(), ql[0, :, 1].mean()
                inside = (np.abs(qu[..., 0] - cx) < SIZE / 2 - 1e-3) & (np.abs(qu[..., 1] - cy) < SIZE / 2 - 1e-3)
                z = qu[..., 2]
                crossed = inside & (z < -1e-5)
                row = dict(fabric=n, radius=r_c, press=press, finite=bool(np.isfinite(rec[:, up[k]:up[k] + RES * RES]).all()),
                           crossed_any=int(crossed.any(axis=0).sum()), crossed_end=int(crossed[-1].sum()),
                           first_cross_t=float(times[crossed.any(axis=1).argmax()]) if crossed.any() else None,
                           patch_slide_mm=float(1e3 * (qu[-1, [i * RES + j for i, j in patch_ij], 0].mean()
                                                       - qu[0, [i * RES + j for i, j in patch_ij], 0].mean())))
                for ph, (t_a, t_b) in phases.items():
                    m = (times >= t_a) & (times <= t_b)
                    zin = np.where(inside[m], z[m], np.nan)
                    row[f"gap_med_{ph}_mm"] = float(1e3 * np.nanmedian(zin))
                    row[f"gap_min_{ph}_mm"] = float(1e3 * np.nanmin(zin))
                out["results"].append(row)
                print(f"[ply r {1e3 * r_c:.2f} mm F {press:4.0f} N] {n:8s} gap med/min settle {row['gap_med_settle_mm']:.3f}/"
                      f"{row['gap_min_settle_mm']:.3f}  press {row['gap_med_press_mm']:.3f}/{row['gap_min_press_mm']:.3f}  "
                      f"drag {row['gap_med_drag_mm']:.3f}/{row['gap_min_drag_mm']:.3f} mm  crossed {row['crossed_any']} "
                      f"(end {row['crossed_end']})  slide {row['patch_slide_mm']:.1f} mm  finite {row['finite']}", flush=True)
            frames_all[f"r{r_c:.5f}_F{press:g}"] = rec.astype(np.float32)
            print(f"[ply r {1e3 * r_c:.2f} mm F {press:g} N] build+run {time.time() - t0:.1f} s, sim {t_run:.1f} s "
                  f"({n_frames / t_run:.1f} frames/s)", flush=True)
    return out, frames_all, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["stretch", "ply"])
    ap.add_argument("--out", default="docs/results/fabric_probes")
    ap.add_argument("--iters", type=int, nargs="+", default=[12, 48])
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--radii", type=float, nargs="+", default=[5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3])
    ap.add_argument("--press", type=float, nargs="+", default=[10.0, 40.0])
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.probe == "stretch":
        (out / "stretch.json").write_text(json.dumps(stretch(a), indent=1))
    else:
        summary, frames, times = ply(a)
        (out / "ply.json").write_text(json.dumps(summary, indent=1))
        _, faces = cant._geo.grid_mesh(SIZE, RES)
        np.savez_compressed(out / "ply.npz", times=times, faces=np.array(faces, dtype=np.int32).reshape(-1, 3), **frames)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
