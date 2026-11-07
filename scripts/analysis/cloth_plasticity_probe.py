"""Scripted fold-press-release of the env's VBD sheet, with and without bending plasticity.

fold: the 10 cm 7x7 sheet lies on a ground plane, rows 0-1 of the bottom ply held (--pin-rows). Its far row (row 6, all 7 particles) is held
    kinematically and carried along a half circle about the crease (row 3) onto the other half,
    the plies one ``2.5 x self_contact_radius`` apart at the end. Optionally a press force pushes
    rows 4-5 of the top ply down next to the crease, then the press and the edge are released and
    the sheet settles for 3 s. Measured: the flap's opening angle (0 = folded flat, 180 = flat
    sheet), the env's fold error (worst of its four keypoints against the folded target, on the
    stationary half's live frame, lift one self-contact radius; success at 1.5 cm), and the plastic
    rest angles. Then the edge is carried back to flat and released (reopen): with plasticity the
    sheet must open again, leaving at most a yield-angle crease.

drape: rows 0-3 clamped, the 5 cm flap hangs under gravity for 6 s (fabric_cantilever.py's test).
    Plasticity must not change it much: a yield angle below the drape's own hinge angles would make
    a hanging sheet set permanently.

One model per fabric (contact and self-contact settings are model-global), one sheet per
(yield curvature, press force). Pure Newton, the env's dt / iterations / contact settings.
Output: --out/{fold,drape}.json and fold.npz (frames for a viewer).
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


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cant = _load("fabric_cantilever", "scripts/analysis/fabric_cantilever.py")
plast = _load("plasticity", "isaacsimenvs/tasks/cloth/utils/plasticity.py")
_geo = cant._geo

SIZE, RES, H, DT, SUB, ITERS = cant.SIZE, cant.RES, cant.H, cant.DT, cant.SUBSTEPS_PER_FRAME, cant.ITERS
NP = RES * RES
CREASE = RES // 2  # row 3

# The presets in isaacsimenvs/cfg/fabric (base = the env defaults).
FABRICS = {
    n: dict(rho=p["density"], tri=p["tri_ke"], ke=p["edge_ke"], kd=p["edge_kd"], mu=p["soft_contact_mu"],
            r_self=p["self_contact_radius"])
    for n, p in ((n, cant.fabrics.cloth_params(n)) for n in cant.fabrics.names())
}
SOFT_KE, SOFT_KD = 5.0e2, 5.0e-3
PARTICLE_RADIUS = 0.008
BOUND_RELAX = 0.42


def pid(start, i, j):
    return start + i * RES + j


@wp.kernel
def _drive(q: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3), idx: wp.array(dtype=int),
           target: wp.array(dtype=wp.vec3), on: wp.array(dtype=int)):
    k = wp.tid()
    if on[0] == 1:
        q[idx[k]] = target[k]
        qd[idx[k]] = wp.vec3(0.0)


class Sim:
    """One fabric's model: sheets, ground (optional), VBD with self-contact, plasticity per edge."""

    def __init__(self, fab, n_sheets, kappas, ground=True, clamp_rows=0, z0=0.0, graph=True):
        f = FABRICS[fab]
        verts, idx = _geo.grid_mesh(SIZE, RES)
        b = newton.ModelBuilder()
        b.default_shape_cfg.mu = 1.0  # effective particle-shape mu = sqrt(soft_contact_mu * 1.0)
        self.starts = []
        for k in range(n_sheets):
            self.starts.append(b.particle_count)
            b.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.25 * k, z0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
                vertices=verts, indices=idx, density=f["rho"], tri_ke=f["tri"], tri_ka=f["tri"], tri_kd=1.5e-6,
                edge_ke=f["ke"], edge_kd=f["kd"], particle_radius=PARTICLE_RADIUS,
            )
            for i in range(clamp_rows):
                for j in range(RES):
                    p = pid(self.starts[-1], i, j)
                    b.particle_mass[p] = 0.0
                    b.particle_flags[p] = b.particle_flags[p] & ~newton.ParticleFlags.ACTIVE
        b.color(include_bending=True)
        if ground:
            b.add_ground_plane()
        self.model = m = b.finalize()
        m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = SOFT_KE, SOFT_KD, f["mu"]
        self.solver = newton.solvers.SolverVBD(
            model=m, iterations=ITERS,
            particle_enable_self_contact=True,
            particle_self_contact_radius=f["r_self"],
            particle_self_contact_margin=1.5 * f["r_self"],
            particle_collision_detection_interval=-1,
            particle_conservative_bound_relaxation=BOUND_RELAX,
        )
        self.pipe = newton.CollisionPipeline(m, broad_phase="nxn", soft_contact_margin=0.01) if ground else None
        self.contacts = self.pipe.contacts() if ground else None
        # Per-edge yield curvature from the sheet each edge belongs to.
        sheet_of_edge = (m.edge_indices.numpy()[:, 2] // NP).astype(int)
        self.plast = plast.BendingPlasticity(m, np.asarray(kappas, dtype=np.float64)[sheet_of_edge])
        self.sheet_of_edge = sheet_of_edge
        self.s0, self.s1, self.control = m.state(), m.state(), m.control()
        self.f_ext = wp.zeros(m.particle_count, dtype=wp.vec3)
        # Driven particles: row 6 of every sheet.
        self.drive_idx_np = np.array([pid(s, RES - 1, j) for s in self.starts for j in range(RES)])
        self.drive_idx = wp.array(self.drive_idx_np, dtype=int)
        self.drive_target = wp.zeros(len(self.drive_idx_np), dtype=wp.vec3)
        self.drive_on = wp.zeros(1, dtype=int)
        self.flags0 = m.particle_flags.numpy().copy()
        self._frame()  # compile
        self.s0.particle_q.assign(m.particle_q)
        self.s0.particle_qd.zero_()
        self.plast.reset()
        self.graph = None
        if graph and wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self._frame()
            self.graph = cap.graph
            self.s0.particle_q.assign(m.particle_q)
            self.s0.particle_qd.zero_()
            self.plast.reset()

    def _frame(self):
        for _ in range(SUB):
            wp.launch(_drive, dim=len(self.drive_idx_np), inputs=[self.s0.particle_q, self.s0.particle_qd,
                                                                  self.drive_idx, self.drive_target, self.drive_on])
            wp.copy(self.s0.particle_f, self.f_ext)
            if self.pipe is not None:
                self.pipe.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, DT)
            self.s0, self.s1 = self.s1, self.s0
            self.plast.apply(self.s0)
        # The graph records an even number of swaps (SUB is even), so s0 is s0 again after a launch.

    def set_drive(self, target_np):
        """target_np: (n_sheets*RES, 3) positions, or None to release (particles active again)."""
        flags = self.flags0.copy()
        if target_np is None:
            self.drive_on.assign(np.array([0], dtype=np.int32))
            # Released at rest: the gripper opens, it does not throw.
            qd = self.s0.particle_qd.numpy()
            qd[self.drive_idx_np] = 0.0
            self.s0.particle_qd.assign(qd)
        else:
            self.drive_target.assign(target_np.astype(np.float32))
            self.drive_on.assign(np.array([1], dtype=np.int32))
            flags[self.drive_idx_np] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(flags)

    def frame(self, f_ext_np=None):
        if f_ext_np is not None:
            self.f_ext.assign(f_ext_np.astype(np.float32))
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._frame()

    def q(self):
        return self.s0.particle_q.numpy().copy()  # on CPU numpy() is a view of a ping-pong buffer


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def arc(rest_row, origin, z_base, lift, ang):
    """Rest positions of row 6 (local, z = 0) turned by ``ang`` about the crease line, the pivot
    lift/2 above the bottom ply (cloth_adapter.fold_rest)."""
    r = 0.5 * lift
    a = rest_row[:, 0]  # distance past the crease (crease at x = 0)
    out = np.empty_like(rest_row)
    out[:, 0] = origin[0] + a * math.cos(ang) + r * math.sin(ang)
    out[:, 1] = origin[1] + rest_row[:, 1]
    out[:, 2] = z_base + a * math.sin(ang) + r * (1.0 - math.cos(ang))
    return out


def kabsch(p, q):
    """R, t with q ~ R p + t (least squares, proper rotation)."""
    pc, qc = p.mean(0), q.mean(0)
    hm = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(hm)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, qc - r @ pc


def fold_metrics(q, start, rest_local, origin, r_self, stat_idx, kp_idx, kp_target_local):
    sq = q[start:start + NP]
    rest_w = rest_local + origin
    r, t = kabsch(rest_w[stat_idx], sq[stat_idx])
    tgt = (r @ (kp_target_local + origin).T).T + t
    err = float(np.linalg.norm(sq[kp_idx] - tgt, axis=1).max())
    # Far edge in the stationary half's frame, about the REST crease line (x = 0): row 3 itself
    # bulges and slides with the crease loop, so it is no reference. 0 = folded flat, 180 = flat;
    # negative = the edge below the bottom ply's plane (over-folded / tucked).
    v = r.T @ (sq[[pid(0, RES - 1, j) for j in range(RES)]].mean(0) - t) - origin
    return err, math.degrees(math.atan2(v[2], -v[0]))


def run_fold(a, fab):
    f = FABRICS[fab]
    kappas, presses = a.kappa, a.press
    combos = [(k, p) for k in kappas for p in presses]
    t0 = time.time()
    # Spawn 2 mm above the resting height so the sheet starts essentially on the ground.
    # Rows 0..pin_rows-1 of the bottom ply are held in place (a hand or table friction keeping the
    # sheet put). Unpinned, the drive that follows the bottom ply's frame drags the sheet: it slid
    # 10-80 cm across the ground in job 259460 and could run into the next sheet. Pinned rows sit at
    # the contact height; free sheets spawn 2 mm above it so they start essentially on the ground.
    z0 = PARTICLE_RADIUS if a.pin_rows else PARTICLE_RADIUS + 0.002
    sim = Sim(fab, len(combos), [k for k, _ in combos], ground=True, clamp_rows=a.pin_rows, z0=z0, graph=not a.no_graph)
    verts, _ = _geo.grid_mesh(SIZE, RES)
    rest_local = np.array(verts)
    origins = np.array([[0.0, 0.25 * k, 0.0] for k in range(len(combos))])
    stat_idx = sorted(_geo.half_indices(RES, "x", positive=False))
    kp_idx = _geo.corner_indices(RES, "x")
    kp_target = np.array(_geo.folded_targets(verts, kp_idx, "x", f["r_self"]))
    row6 = rest_local[[pid(0, RES - 1, j) for j in range(RES)]]
    lift = 2.5 * f["r_self"]
    press_idx = [pid(0, i, j) for i in (CREASE + 1, CREASE + 2) for j in range(RES)]
    m = sim.model

    T = dict(settle=0.5, fold=1.5, hold=0.3, press_ramp=0.2, press=0.8, release_gap=0.1, free=3.0,
             reopen=1.5, reopen_free=2.0)
    fr = lambda s: int(round(s * 60))
    rec, times, phase_log = [], [], []
    t = 0.0

    def step(f_ext=None):
        nonlocal t
        sim.frame(f_ext)
        t += 1.0 / 60.0
        if int(round(t * 60)) % 2 == 0:
            rec.append(sim.q().astype(np.float32)), times.append(t)

    for _ in range(fr(T["settle"])):
        step()
    n_s = len(combos)
    row6_idx = [pid(0, RES - 1, j) for j in range(RES)]

    def frames_now():
        """The stationary half's live rigid frame per sheet: the drive follows it the way a policy
        does, so a bottom ply dragged or pushed by the fold does not turn into fold error."""
        q = sim.q()
        return [kabsch(rest_local[stat_idx] + origins[s], q[sim.starts[s]:sim.starts[s] + NP][stat_idx]) for s in range(n_s)]

    def drive_local(local_rows):
        """local_rows: (n_s, RES, 3) row-6 targets in each sheet's rest frame."""
        fr_now = frames_now()
        sim.set_drive(np.concatenate([(r @ local_rows[s].T).T + t_ for s, (r, t_) in enumerate(fr_now)]))

    def arc_local(ang):
        return np.stack([arc(row6, origins[s], 0.0, lift, ang) for s in range(n_s)])

    # fold
    n = fr(T["fold"])
    for k in range(1, n + 1):
        drive_local(arc_local(math.pi * smooth(k / n)))
        step()
    folded = arc_local(math.pi)
    for _ in range(fr(T["hold"])):
        drive_local(folded)
        step()
    # press rows 4-5 of the top ply, ramped
    full = np.zeros((m.particle_count, 3))
    for s, (_, p) in enumerate(combos):
        for i in press_idx:
            full[sim.starts[s] + i, 2] = -p / len(press_idx)
    nr = fr(T["press_ramp"])
    for k in range(1, nr + 1):
        drive_local(folded)
        step(full * (k / nr))
    for _ in range(fr(T["press"])):
        drive_local(folded)
        step()
    for k in range(nr - 1, -1, -1):
        drive_local(folded)
        step(full * (k / nr))
    q_pressed = sim.q()
    for _ in range(fr(T["release_gap"])):
        drive_local(folded)
        step()
    sim.set_drive(None)
    t_release = t
    for _ in range(fr(T["free"])):
        step()
    q_free = sim.q()
    rest_free = m.edge_rest_angle.numpy().copy()
    # reopen: carry row 6 from where it lies back to flat, over the fold, in the live frame
    q = sim.q()
    cur = np.stack([(r.T @ (q[sim.starts[s]:sim.starts[s] + NP][row6_idx] - t_).T).T
                    for s, (r, t_) in enumerate(frames_now())])
    flat = arc_local(0.0)
    n = fr(T["reopen"])
    for k in range(1, n + 1):
        u = smooth(k / n)
        tgt = (1.0 - u) * cur + u * flat
        # Lift the path over the fold so the edge does not drag through the bottom ply.
        tgt[..., 2] += 0.5 * SIZE * math.sin(math.pi * u)
        drive_local(tgt)
        step()
    sim.set_drive(None)
    for _ in range(fr(T["reopen_free"])):
        step()
    q_end = sim.q()
    rest_end = m.edge_rest_angle.numpy().copy()

    theta_free = plast.dihedral_angles(wp.array(q_free, dtype=wp.vec3), m.edge_indices)
    results = []
    for s, (kappa, p) in enumerate(combos):
        e_mask = sim.sheet_of_edge == s
        err_p, open_p = fold_metrics(q_pressed, sim.starts[s], rest_local, origins[s], f["r_self"], stat_idx, kp_idx, kp_target)
        err_f, open_f = fold_metrics(q_free, sim.starts[s], rest_local, origins[s], f["r_self"], stat_idx, kp_idx, kp_target)
        _, open_e = fold_metrics(q_end, sim.starts[s], rest_local, origins[s], f["r_self"], stat_idx, kp_idx, kp_target)
        rf = np.degrees(np.abs(rest_free[e_mask]))
        th = np.degrees(np.abs(theta_free[e_mask]))
        ya = plast.BendingPlasticity.yield_angle_deg(kappa, H) if math.isfinite(kappa) else float("inf")
        row = dict(fabric=fab, kappa=kappa, yield_deg_cell=ya, press_N=p,
                   pressed_err_cm=100 * err_p, pressed_open_deg=open_p,
                   free_err_cm=100 * err_f, free_open_deg=open_f, held_1p5cm=bool(err_f <= 0.015),
                   reopen_open_deg=open_e,
                   rest_max_deg=float(rf.max()), rest_sum_deg=float(rf.sum()), n_set_10deg=int((rf > 10).sum()),
                   theta_max_deg=float(th.max()),
                   rest_end_max_deg=float(np.degrees(np.abs(rest_end[e_mask])).max()),
                   finite=bool(np.isfinite(q_end[sim.starts[s]:sim.starts[s] + NP]).all()))
        results.append(row)
        print(f"[fold {fab:6s}] kappa {kappa:6.1f} (yield {ya:5.1f} deg/cell)  press {p:4.1f} N  "
              f"pressed err {row['pressed_err_cm']:5.2f} cm open {open_p:5.1f}  ->  free err {row['free_err_cm']:5.2f} cm "
              f"open {open_f:5.1f} held {row['held_1p5cm']}  | set max {row['rest_max_deg']:5.1f} n>10 {row['n_set_10deg']:3d}  "
              f"| reopen open {open_e:5.1f} rest max {row['rest_end_max_deg']:5.1f}  finite {row['finite']}", flush=True)
    print(f"[fold {fab}] {time.time() - t0:.1f} s (graph {sim.graph is not None}), release at {t_release:.2f} s", flush=True)
    frames = np.stack(rec)
    return results, frames, np.array(times), sim.starts, origins


def run_drape(a, fab):
    kappas = a.kappa
    t0 = time.time()
    sim = Sim(fab, len(kappas), kappas, ground=False, clamp_rows=cant.N_CLAMP, z0=1.0, graph=not a.no_graph)
    for _ in range(int(6.0 * 60)):
        sim.frame()
    q = sim.q()
    rest = sim.model.edge_rest_angle.numpy().copy()
    out = []
    for s, kappa in enumerate(kappas):
        sq = q[sim.starts[s]:sim.starts[s] + NP] - np.array([0.0, 0.25 * s, 1.0])
        th = cant.droop(sq)
        rmax = float(np.degrees(np.abs(rest[sim.sheet_of_edge == s])).max())
        out.append(dict(fabric=fab, kappa=kappa, droop_deg=th, rest_max_deg=rmax))
        print(f"[drape {fab:6s}] kappa {kappa:6.1f}  droop {th:5.1f} deg  set max {rmax:5.1f} deg", flush=True)
    print(f"[drape {fab}] {time.time() - t0:.1f} s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/cloth_plasticity")
    ap.add_argument("--fabrics", nargs="+", default=list(FABRICS))
    ap.add_argument("--kappa", type=float, nargs="+", default=[math.inf, 100.0, 50.0, 25.0, 12.0, 6.0],
                    help="yield curvature [1/m]; inf = elastic. Yield angle per grid cell = kappa * 16.7 mm")
    ap.add_argument("--press", type=float, nargs="+", default=[0.0, 1.0, 10.0, 40.0], help="total press force [N]")
    ap.add_argument("--pin-rows", type=int, default=2, help="fold test: bottom-ply rows held in place (0 = none)")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--skip-drape", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    fold_all, drape_all, npz = [], [], {}
    for fab in a.fabrics:
        res, frames, times, starts, origins = run_fold(a, fab)
        fold_all += res
        npz[f"{fab}_frames"] = frames
        npz["times"] = times
        if not a.skip_drape:
            drape_all += run_drape(a, fab)
    _, faces = _geo.grid_mesh(SIZE, RES)
    labels = [f"kappa{k:g}_press{p:g}" for k in a.kappa for p in a.press]
    np.savez_compressed(out / "fold.npz", faces=np.array(faces, dtype=np.int32).reshape(-1, 3),
                        labels=np.array(labels), sheet_spacing_y=0.25, pin_rows=a.pin_rows, **npz)
    (out / "fold.json").write_text(json.dumps(fold_all, indent=1))
    (out / "drape.json").write_text(json.dumps(drape_all, indent=1))
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
