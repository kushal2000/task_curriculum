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
CREASE = RES // 2  # row 3 on the 7x7 sheet

# The presets in isaacsimenvs/cfg/fabric (base = the env defaults).
FABRICS = {
    n: dict(rho=p["density"], tri=p["tri_ke"], ke=p["edge_ke"], kd=p["edge_kd"], mu=p["soft_contact_mu"],
            r_self=p["self_contact_radius"])
    for n, p in ((n, cant.fabrics.cloth_params(n)) for n in cant.fabrics.names())
}
SOFT_KE, SOFT_KD = 5.0e2, 5.0e-3
PARTICLE_RADIUS = 0.008
BOUND_RELAX = 0.42
# Self-contact margin [m]; None = 1.5 x self_contact_radius. VBD caps particle speed at
# 0.5 * BOUND_RELAX * margin per substep, so a thin-radius margin also slows the whole sheet.
SELF_MARGIN = None
#: Set from --mesh / --self-mu in main (read by Sim, like ITERS).
MESH = "alternate"
SELF_MU = None
SET_KAPPA = 0.0  # --set-kappa (two-stage plasticity gate), read by Sim


def grid_mesh(size, res):
    """The sheet with --mesh's cell diagonals (``cloth_geometry.grid_mesh``; alternate = the env)."""
    return _geo.grid_mesh(size, res, MESH)


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
        verts, idx = grid_mesh(SIZE, RES)
        b = newton.ModelBuilder()
        b.default_shape_cfg.mu = 1.0  # effective particle-shape mu = sqrt(soft_contact_mu * 1.0)
        # --self-mu: cloth-cloth friction is model.soft_contact_mu alone (solver_vbd self-contact
        # kernel), cloth-ground is sqrt(soft_contact_mu * shape mu). Lower the first and raise the
        # shape mu so the ground still sees sqrt(f["mu"] * 1.0).
        soft_mu = f["mu"] if SELF_MU is None else SELF_MU
        if SELF_MU is not None:
            b.default_shape_cfg.mu = f["mu"] / SELF_MU
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
        m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = SOFT_KE, SOFT_KD, soft_mu
        self.solver = newton.solvers.SolverVBD(
            model=m, iterations=ITERS,
            particle_enable_self_contact=True,
            particle_self_contact_radius=f["r_self"],
            particle_self_contact_margin=SELF_MARGIN if SELF_MARGIN else 1.5 * f["r_self"],
            particle_collision_detection_interval=-1,
            particle_conservative_bound_relaxation=BOUND_RELAX,
        )
        self.pipe = newton.CollisionPipeline(m, broad_phase="nxn", soft_contact_margin=0.01) if ground else None
        self.contacts = self.pipe.contacts() if ground else None
        # Per-edge yield curvature from the sheet each edge belongs to.
        sheet_of_edge = (m.edge_indices.numpy()[:, 2] // NP).astype(int)
        self.plast = plast.BendingPlasticity(m, np.asarray(kappas, dtype=np.float64)[sheet_of_edge],
                                             set_curvature=SET_KAPPA)
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
    # forced: the edge is laid exactly over its mirror point, one 2.5 x radius gap up, which pulls the
    # crease into a tight loop and holds it there. touch: the edge comes over --touch-gap above the
    # other half and is let go at once, so the crease keeps whatever loop the cloth makes.
    lift = a.touch_gap if a.fold == "touch" else 2.5 * f["r_self"]
    press_idx = [pid(0, i, j) for i in a.press_rows for j in range(RES)]
    m = sim.model

    T = dict(settle=0.5, fold=1.5, hold=0.3, press_ramp=0.2, press=0.8, release_gap=0.1, free=3.0,
             reopen=1.5, reopen_free=2.0, land=0.5)
    fr = lambda s: int(round(s * 60))
    rec, times, phase_log = [], [], []
    t = 0.0

    def step(f_ext=None):
        nonlocal t
        sim.frame(f_ext)
        t += 1.0 / 60.0
        if int(round(t * 60)) % a.record_every == 0:
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

    phase_t, phase_names = [0.0], ["settling"]

    def mark(name):
        phase_t.append(t), phase_names.append(name)

    def live_row6():
        """Row 6 where it lies now, in each sheet's rest frame (the stationary half's live frame)."""
        q = sim.q()
        return np.stack([(r.T @ (q[sim.starts[s]:sim.starts[s] + NP][row6_idx] - t_).T).T
                         for s, (r, t_) in enumerate(frames_now())])

    full = np.zeros((m.particle_count, 3))
    for s, (_, p) in enumerate(combos):
        for i in press_idx:
            full[sim.starts[s] + i, 2] = -p / len(press_idx)
    nr = fr(T["press_ramp"])
    # --release-ramp: frames to take the press off (default = the on-ramp; 0 = all at once, as a hand
    # that leaves within one policy step).
    nrr = nr if getattr(a, "release_ramp", None) is None else fr(a.release_ramp)

    def off_ramp():
        return [k * nr / max(nrr, 1) / nr for k in range(nrr - 1, -1, -1)] if nrr > 0 else []
    flat = arc_local(0.0)
    cyc = []  # per cycle: q after the free settle, q after the reopen settle, rest angles after each
    for c in range(a.cycles):
        tag = f" (cycle {c + 1})" if a.cycles > 1 else ""
        # --press-per-cycle: this cycle's total press force for every sheet (replaces its --press).
        f_c, w_on = full, "weight"
        if a.press_per_cycle:
            pc = a.press_per_cycle[min(c, len(a.press_per_cycle) - 1)]
            f_c = np.zeros_like(full)
            for s_ in range(len(combos)):
                f_c[[sim.starts[s_] + i for i in press_idx], 2] = -pc / len(press_idx)
            w_on = f"weight {pc:g} N" if pc > 0 else "no weight"
        # fold; after the first cycle the edge eases in from wherever the reopen left it
        mark("folding: edge carried over" + tag)
        n = fr(T["fold"])
        start_rows = live_row6() if c else None
        for k in range(1, n + 1):
            tgt = arc_local(math.pi * smooth(k / n))
            if start_rows is not None:
                w = smooth(min(1.0, k / (0.2 * n)))
                tgt = (1.0 - w) * start_rows + w * tgt
            drive_local(tgt)
            step()
        folded = arc_local(math.pi)
        if a.fold == "touch":
            # The edge stays where it touched (the loop is not pulled tight) while the weight comes down
            # on the whole flap, then the hand lets go under it, as with a phone laid on a fold. A springy
            # sheet let go before the weight lands is already open by then. 0 N sheets get the same short
            # hold and are let go: fold, touch, let go.
            mark(f"{w_on} on, edge held where it touched" + tag if w_on != "no weight" else "edge held where it touched, no weight" + tag)
            for k in range(1, nr + 1):
                drive_local(folded)
                step(f_c * (k / nr))
            sim.set_drive(None)
            mark(f"{w_on} on, edge let go" + tag if w_on != "no weight" else "edge let go, no weight" + tag)
            for _ in range(fr(T["press"])):
                step()
            for w in off_ramp():
                step(f_c * w)
            if c == 0:
                q_pressed, t_release = sim.q(), t
            mark(("RELEASED (weight off)" if w_on != "no weight" else "RELEASED (no weight)") + tag)
        else:
            mark("folded, edge held" + tag)
            for _ in range(fr(T["hold"])):
                drive_local(folded)
                step()
            # press the --press-rows of the top ply (default 4-5, next to the crease), ramped
            mark("weight on" + tag)
            for k in range(1, nr + 1):
                drive_local(folded)
                step(full * (k / nr))
            for _ in range(fr(T["press"])):
                drive_local(folded)
                step()
            for k in range(nr - 1, -1, -1):
                drive_local(folded)
                step(full * (k / nr))
            if c == 0:
                q_pressed = sim.q()
            mark("weight off, edge still held" + tag)
            for _ in range(fr(T["release_gap"])):
                drive_local(folded)
                step()
            sim.set_drive(None)
            if c == 0:
                t_release = t
            mark("RELEASED" + tag)
        for _ in range(fr(T["free"])):
            step()
        q_free_c, rest_free_c = sim.q(), m.edge_rest_angle.numpy().copy()
        # reopen: carry row 6 from where it lies back to flat, over the fold, in the live frame
        mark("unfolding: edge carried back to flat" + tag)
        cur = live_row6()
        n = fr(T["reopen"])
        if a.fold == "touch":
            # The fold run backwards: the edge swings back about the crease from each sheet's own
            # opening angle to flat and is laid down there. A sheet that already sprang open barely
            # moves, so the unfold adds no bend of its own (the arch below lifts every sheet 5 cm).
            q_now = sim.q()
            ang0 = [math.radians(min(max(180.0 - abs(fold_metrics(q_now, sim.starts[s], rest_local, origins[s], f["r_self"],
                                                                     stat_idx, kp_idx, kp_target)[1]), 0.0), 180.0))
                    for s in range(n_s)]
            for k in range(1, n + 1):
                u = smooth(k / n)
                back = np.stack([arc(row6, origins[s], 0.0, lift, ang0[s] * (1.0 - u)) for s in range(n_s)])
                w = smooth(min(1.0, k / (0.2 * n)))  # ease in from where the edge lies
                drive_local((1.0 - w) * cur + w * back)
                step()
        else:
            for k in range(1, n + 1):
                u = smooth(k / n)
                tgt = (1.0 - u) * cur + u * flat
                # Lift the path over the fold so the edge does not drag through the bottom ply.
                tgt[..., 2] += 0.5 * SIZE * math.sin(math.pi * u)
                drive_local(tgt)
                step()
        sim.set_drive(None)
        mark("let go, unfolded" + tag)
        for _ in range(fr(T["reopen_free"])):
            step()
        cyc.append((q_free_c, rest_free_c, sim.q(), m.edge_rest_angle.numpy().copy()))
    # Cycle 1 keeps the single-cycle metrics (comparable with earlier runs); the last cycle ends the run.
    q_free, rest_free = cyc[0][0], cyc[0][1]
    q_end, rest_end = cyc[0][2], cyc[0][3]
    q_last = cyc[-1][2]

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
        row = dict(fabric=fab, kappa=kappa, set_kappa=SET_KAPPA, yield_deg_cell=ya, press_N=p,
                   pressed_err_cm=100 * err_p, pressed_open_deg=open_p,
                   free_err_cm=100 * err_f, free_open_deg=open_f, held_1p5cm=bool(err_f <= 0.015),
                   reopen_open_deg=open_e,
                   rest_max_deg=float(rf.max()), rest_sum_deg=float(rf.sum()), n_set_10deg=int((rf > 10).sum()),
                   theta_max_deg=float(th.max()),
                   rest_end_max_deg=float(np.degrees(np.abs(rest_end[e_mask])).max()),
                   finite=bool(np.isfinite(q_last[sim.starts[s]:sim.starts[s] + NP]).all()))
        if a.cycles > 1:
            per = [(fold_metrics(qf, sim.starts[s], rest_local, origins[s], f["r_self"], stat_idx, kp_idx, kp_target),
                    fold_metrics(qe, sim.starts[s], rest_local, origins[s], f["r_self"], stat_idx, kp_idx, kp_target)[1],
                    float(np.degrees(np.abs(re[e_mask])).max()))
                   for qf, _, qe, re in cyc]
            row.update(free_open_deg_cycles=[pf[1] for pf, _, _ in per], held_cycles=[bool(pf[0] <= 0.015) for pf, _, _ in per],
                       reopen_open_deg_cycles=[pe for _, pe, _ in per], rest_end_max_deg_cycles=[r for _, _, r in per])
            print(f"[cycles {fab:6s}] kappa {kappa:6.1f} press {p:4.1f} N  open after release "
                  + " ".join(f"{abs(v):5.1f}" for v in row["free_open_deg_cycles"]) + "  | crease after unfold "
                  + " ".join(f"{180 - abs(v):5.1f}" for v in row["reopen_open_deg_cycles"]), flush=True)
        results.append(row)
        print(f"[fold {fab:6s}] kappa {kappa:6.1f} (yield {ya:5.1f} deg/cell)  press {p:4.1f} N  "
              f"pressed err {row['pressed_err_cm']:5.2f} cm open {open_p:5.1f}  ->  free err {row['free_err_cm']:5.2f} cm "
              f"open {open_f:5.1f} held {row['held_1p5cm']}  | set max {row['rest_max_deg']:5.1f} n>10 {row['n_set_10deg']:3d}  "
              f"| reopen open {open_e:5.1f} rest max {row['rest_end_max_deg']:5.1f}  finite {row['finite']}", flush=True)
    print(f"[fold {fab}] {time.time() - t0:.1f} s (graph {sim.graph is not None}), release at {t_release:.2f} s", flush=True)
    frames = np.stack(rec)
    return results, frames, np.array(times), sim.starts, origins, (np.array(phase_t), np.array(phase_names))


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
        out.append(dict(fabric=fab, kappa=kappa, set_kappa=SET_KAPPA, droop_deg=th, rest_max_deg=rmax))
        print(f"[drape {fab:6s}] kappa {kappa:6.1f}  droop {th:5.1f} deg  set max {rmax:5.1f} deg", flush=True)
    print(f"[drape {fab}] {time.time() - t0:.1f} s", flush=True)
    return out


def main():
    global ITERS, MESH, SELF_MU, SET_KAPPA, PARTICLE_RADIUS, SOFT_KE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/results/cloth_plasticity")
    ap.add_argument("--fabrics", nargs="+", default=list(FABRICS))
    ap.add_argument("--kappa", type=float, nargs="+", default=[math.inf, 100.0, 50.0, 25.0, 12.0, 6.0],
                    help="yield curvature [1/m]; inf = elastic. Yield angle per grid cell = kappa * 16.7 mm")
    ap.add_argument("--press", type=float, nargs="+", default=[0.0, 1.0, 10.0, 40.0], help="total press force [N]")
    ap.add_argument("--press-rows", type=int, nargs="+", default=[CREASE + 1, CREASE + 2],
                    help="rows the press force is spread over (default 4 5; 3 4 5 = a weight resting on the whole folded "
                         "flap, row 6 is held by the drive during the press)")
    ap.add_argument("--press-per-cycle", type=float, nargs="+", default=None,
                    help="touch fold only: total press force [N] per cycle for every sheet, overriding --press "
                         "(e.g. --cycles 2 --press-per-cycle 0 0.93: fold let go unpressed, then fold, press, let go)")
    ap.add_argument("--cycles", type=int, default=1,
                    help="fold -> press -> release -> unfold -> let go, repeated on the same sheet (plasticity accumulates)")
    ap.add_argument("--pin-rows", type=int, default=None,
                    help="fold test: bottom-ply rows held in place (0 = none; default a 16.7 mm strip: 2 rows on 7x7)")
    ap.add_argument("--res", type=int, default=7, help="particles per side (env: 7; read at import)")
    ap.add_argument("--size", type=float, default=0.10, help="sheet side [m] (env: 0.10; read at import)")
    ap.add_argument("--fold", choices=["forced", "touch"], default="forced",
                    help="forced: edge laid on its mirror point and held through the press (the original test); "
                         "touch: edge brought over, let go at once, weight (if any) rests on the whole flap")
    ap.add_argument("--touch-gap", type=float, default=0.01, help="touch: edge height above the other half when let go [m]")
    ap.add_argument("--release-ramp", type=float, default=None,
                    help="seconds to take the press off (default: the 0.2 s on-ramp; 0 = instantly)")
    ap.add_argument("--record-every", type=int, default=2, help="keep every n-th 60 Hz frame")
    ap.add_argument("--self-contact-radius", type=float, default=None,
                    help="override every preset's self_contact_radius [m] (margin follows at 1.5x)")
    ap.add_argument("--iters", type=int, default=ITERS,
                    help="VBD iterations (env: 12; a 25x25 sheet needs ~96 to settle at all)")
    ap.add_argument("--mesh", choices=["alternate", "uniform", "flipped"], default="alternate",
                    help="cell diagonals: alternate (checkerboard, the env), uniform (the env before 2026-11-09), "
                         "flipped (uniform's mirror image)")
    ap.add_argument("--self-mu", type=float, default=None,
                    help="override cloth-cloth friction only (cloth-ground friction kept at the preset's)")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--set-kappa", type=float, default=0.0,
                    help="two-stage plasticity: set grows only past this curvature [1/m] (0 = single-stage)")
    ap.add_argument("--skip-drape", action="store_true")
    ap.add_argument("--edge-ke", type=float, default=None, help="override every preset's bending stiffness edge_ke")
    ap.add_argument("--particle-radius", type=float, default=None, help="particle radius = half the thickness [m] (default 0.008)")
    ap.add_argument("--soft-ke", type=float, default=None, help="soft_contact_ke [N/m] (default 500)")
    ap.add_argument("--tri-ke", type=float, default=None,
                    help="override every preset's stretch stiffness tri_ke = tri_ka")
    a = ap.parse_args()
    SET_KAPPA = a.set_kappa
    ITERS = a.iters  # read by Sim
    MESH, SELF_MU = a.mesh, a.self_mu
    if a.tri_ke is not None:
        for f in FABRICS.values():
            f["tri"] = a.tri_ke
    if a.particle_radius is not None:
        PARTICLE_RADIUS = a.particle_radius
    if a.soft_ke is not None:
        SOFT_KE = a.soft_ke
    if a.edge_ke is not None:
        for f in FABRICS.values():
            f["ke"] = a.edge_ke
    if a.self_contact_radius is not None:
        for f in FABRICS.values():
            f["r_self"] = a.self_contact_radius
    if a.pin_rows is None:
        a.pin_rows = round(0.0167 / H) + 1
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    fold_all, drape_all, npz = [], [], {}
    for fab in a.fabrics:
        res, frames, times, starts, origins, (phase_t, phase_names) = run_fold(a, fab)
        fold_all += res
        npz[f"{fab}_frames"] = frames
        npz["times"] = times
        npz["phase_t"], npz["phase_names"] = phase_t, phase_names  # the same for every fabric
        if not a.skip_drape:
            drape_all += run_drape(a, fab)
    _, faces = grid_mesh(SIZE, RES)
    labels = [f"kappa{k:g}_press{p:g}" for k in a.kappa for p in a.press]
    np.savez_compressed(out / "fold.npz", faces=np.array(faces, dtype=np.int32).reshape(-1, 3),
                        labels=np.array(labels), sheet_spacing_y=0.25, pin_rows=a.pin_rows, cycles=a.cycles, res=RES, size=SIZE, fold=a.fold, iters=a.iters, mesh=MESH, self_mu=-1.0 if SELF_MU is None else SELF_MU, r_self_override=-1.0 if a.self_contact_radius is None else a.self_contact_radius, **npz)
    (out / "fold.json").write_text(json.dumps(fold_all, indent=1))
    (out / "drape.json").write_text(json.dumps(drape_all, indent=1))
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
