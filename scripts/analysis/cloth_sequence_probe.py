"""One handling sequence of the env's VBD sheet, per fabric, current presets vs proposed ones.

    1. fold, no press   row 6 carried over the crease and let go 1 cm above the other half
                        (the touch fold of cloth_plasticity_probe.py), rows 0-1 held, 2.5 s free
    2. lift             row 6 picked up where it lies and raised 13 cm: the sheet hangs from that
                        edge (a kept crease shows as a kink), held 1.5 s, then laid back down flat
                        (row 0 touches first, row 6 swung down to its flat place) and let go
    3. fold, press      the same fold, then a 0.93 N weight on rows 4-5 of the top half (about the
                        213 g swatch weight) for 1 s, taken off, 2.5 s free
    4. lift             as 2
    5. crumple          the four corners carried to one point 4 cm above the middle, held 1 s,
                        let go, 3 s free

Variants (one model each): current = the committed preset, single-stage plasticity at its yield;
new = tri_ke = tri_ka 100 for the soft-membrane fabrics (chiffon, cotton, silk), chiffon's edge_ke
refit (--chiffon-ke), two-stage plasticity (set --set /m) at the proposed yields.
Output: --out/sequence.npz (frames for the viewer) and sequence.json (per-phase metrics).
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
spec = importlib.util.spec_from_file_location("cloth_plasticity_probe", ROOT / "scripts/analysis/cloth_plasticity_probe.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

RES, NP, SIZE, SUB, DT = P.RES, P.NP, P.SIZE, P.SUB, P.DT
Z0 = P.PARTICLE_RADIUS
RADIUS_DEFAULT, SOFT_KE_DEFAULT = P.PARTICLE_RADIUS, P.SOFT_KE
# Realistic thicknesses [m] (thin swatches): --variants thin uses them (see main).
THIN = dict(chiffon=0.0001, silk=0.0003, napkin=0.00015, big_napkin=0.0003, cotton=0.0003, denim=0.00035, towel=0.001)
CURRENT = dict(big_napkin=25, chiffon=42, cotton=20, napkin=30, silk=32, towel=60, denim=100)
NEW = dict(big_napkin=20, chiffon=30, cotton=20, napkin=30, silk=30, towel=100, denim=100)
SOFT = ("chiffon", "cotton", "silk")
THIN_YIELD, THIN_OVER = {}, {}
CRUMPLE_SPREAD = 0.0  # [m], --crumple-spread
PRESS_N, PRESS_ROWS, PIN_ROWS, GAP = 0.93, (4, 5), 2, 0.01


@wp.kernel
def _drive_mask(q: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3), idx: wp.array(dtype=int),
                target: wp.array(dtype=wp.vec3), on: wp.array(dtype=int)):
    k = wp.tid()
    if on[k] == 1:
        q[idx[k]] = target[k]
        qd[idx[k]] = wp.vec3(0.0)


class SeqSim(P.Sim):
    """P.Sim with a per-particle drive mask (some driven particles held, others free)."""

    def setup(self, idx):
        self.drive_idx_np = np.asarray(idx)
        self.drive_idx = wp.array(self.drive_idx_np, dtype=int)
        self.drive_target = wp.zeros(len(idx), dtype=wp.vec3)
        self.mask = wp.zeros(len(idx), dtype=int)
        m = self.model
        self.s0.particle_q.assign(m.particle_q)
        self.s0.particle_qd.zero_()
        self.plast.reset()
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self._frame()
            self.graph = cap.graph
            self.s0.particle_q.assign(m.particle_q)
            self.s0.particle_qd.zero_()
            self.plast.reset()

    def _frame(self):
        if not hasattr(self, "mask"):
            return super()._frame()
        for _ in range(SUB):
            wp.launch(_drive_mask, dim=len(self.drive_idx_np),
                      inputs=[self.s0.particle_q, self.s0.particle_qd, self.drive_idx, self.drive_target, self.mask])
            wp.copy(self.s0.particle_f, self.f_ext)
            self.pipe.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, DT)
            self.s0, self.s1 = self.s1, self.s0
            self.plast.apply(self.s0)

    def drive(self, targets, mask):
        """targets: (n_drive, 3) (rows with mask 0 ignored); mask: (n_drive,) bool."""
        mask = np.asarray(mask, dtype=bool)
        if targets is not None:
            self.drive_target.assign(np.asarray(targets, dtype=np.float32))
        newly_free = (~mask) & (self.mask.numpy() == 1)
        if newly_free.any():  # let go at rest: the hand opens, it does not throw
            qd = self.s0.particle_qd.numpy()
            qd[self.drive_idx_np[newly_free]] = 0.0
            self.s0.particle_qd.assign(qd)
        self.mask.assign(mask.astype(np.int32))
        flags = self.flags0.copy()
        flags[self.drive_idx_np[mask]] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(flags)


def run(fab, variant, yield_k, set_k, overrides, rec_every, only=None):
    """overrides: preset keys (tri, ke, r_self, ...) plus ``radius`` (particle radius = half the
    thickness, cloth-vs-table) and ``soft_ke`` (soft_contact_ke), both model-level."""
    global Z0
    preset = f"{fab}_measured"
    overrides = dict(overrides)
    radius = overrides.pop("radius", RADIUS_DEFAULT)
    soft_ke = overrides.pop("soft_ke", SOFT_KE_DEFAULT)
    saved = dict(P.FABRICS[preset])
    P.FABRICS[preset].update(overrides)
    P.SET_KAPPA = set_k
    P.PARTICLE_RADIUS, P.SOFT_KE, Z0 = radius, soft_ke, radius
    try:
        sim = SeqSim(preset, 1, [yield_k], ground=True, clamp_rows=0, z0=Z0 + 0.002, graph=False)
        r_self = P.FABRICS[preset]["r_self"]
    finally:
        P.FABRICS[preset].clear()
        P.FABRICS[preset].update(saved)
        P.PARTICLE_RADIUS, P.SOFT_KE = RADIUS_DEFAULT, SOFT_KE_DEFAULT
    m = sim.model
    verts = np.array(P._geo.grid_mesh(SIZE, RES)[0])  # crease (row 3) at x = 0, row i at x = (i-3) h
    row = lambda i: [P.pid(0, i, j) for j in range(RES)]
    pin = sum((row(i) for i in range(PIN_ROWS)), [])
    r6 = row(RES - 1)
    corners = [P.pid(0, 0, 0), P.pid(0, 0, RES - 1), P.pid(0, RES - 1, 0), P.pid(0, RES - 1, RES - 1)]
    drive_ids = pin + r6  # holds all four corners too (two in row 0, two in row 6)
    sim.setup(drive_ids)
    D = {p: k for k, p in enumerate(drive_ids)}
    nd = len(drive_ids)
    stat_idx = sorted(P._geo.half_indices(RES, "x", positive=False))
    kp_idx = P._geo.corner_indices(RES, "x")
    kp_target = np.array(P._geo.folded_targets(verts.tolist(), kp_idx, "x", r_self))
    press_ids = sum((row(i) for i in PRESS_ROWS), [])
    edges = m.edge_indices.numpy()
    # hinges in the crease band (both endpoints in rows 2-4): where a fold's loop sits
    er = edges[:, 2:4] // RES
    crease_edges = np.where((edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (er >= 2).all(1) & (er <= 4).all(1))[0]

    fr = lambda s: int(round(s * 60))
    rec, times, phases, metrics = [], [], [], []
    t = 0.0
    tgt = np.zeros((nd, 3))
    mask = np.zeros(nd, dtype=bool)
    f_ext = np.zeros((m.particle_count, 3))

    def step(force=None):
        nonlocal t
        sim.frame(force)
        t += 1.0 / 60.0
        if int(round(t * 60)) % rec_every == 0:
            rec.append(sim.q().astype(np.float32)), times.append(t)

    def mark(name):
        phases.append((t, name))

    def set_drive(ids, pos):
        for p, x in zip(ids, pos):
            tgt[D[p]] = x
            mask[D[p]] = True

    def free(ids):
        for p in ids:
            mask[D[p]] = False

    def apply():
        sim.drive(tgt, mask)

    def frame_now():
        q = sim.q()
        return P.kabsch(verts[stat_idx], q[stat_idx])

    def hinge_deg():
        th = np.degrees(np.abs(P.plast.dihedral_angles(sim.s0.particle_q, m.edge_indices)))
        rest = np.degrees(np.abs(m.edge_rest_angle.numpy()))
        return th, rest

    def record(name, **extra):
        q = sim.q()
        th, rest = hinge_deg()
        r = dict(phase=name, t=round(t, 3), height_cm=100 * float(q[:, 2].max() - Z0),
                 hinge_max_deg=float(th.max()), set_max_deg=float(rest.max()),
                 set_crease_max_deg=float(rest[crease_edges].max()) if len(crease_edges) else 0.0, **extra)
        if not np.isfinite(q).all():
            r["finite"] = False
        metrics.append(r)
        return r

    def fold(press):
        """Rows 0-1 held where they lie, row 6 over the crease in the bottom half's live frame."""
        R, T = frame_now()
        q = sim.q()
        set_drive(pin, q[pin])
        row6_local = verts[r6]
        start = q[r6]
        n = fr(1.5)
        mark("fold: edge carried over" + (" (then pressed)" if press else ", no press"))
        for k in range(1, n + 1):
            loc = P.arc(row6_local, np.zeros(3), 0.0, GAP, math.pi * P.smooth(k / n))
            w = P.smooth(min(1.0, k / (0.2 * n)))
            set_drive(r6, (1 - w) * start + w * ((R @ loc.T).T + T))
            apply()
            step()
        folded = (R @ P.arc(row6_local, np.zeros(3), 0.0, GAP, math.pi).T).T + T
        force = np.zeros_like(f_ext)
        if press:
            force[press_ids, 2] = -PRESS_N / len(press_ids)
            mark("weight 0.93 N on the top half")
            nr = fr(0.2)
            for k in range(1, nr + 1):
                set_drive(r6, folded)
                apply()
                step(force * k / nr)
            free(r6)
            apply()
            for _ in range(fr(0.8)):
                step()
            for k in range(nr - 1, -1, -1):
                step(force * k / nr)
            mark("weight off, let go")
        else:
            for _ in range(fr(0.2)):
                set_drive(r6, folded)
                apply()
                step()
            free(r6)
            apply()
            mark("let go, no press")
        for _ in range(fr(2.5)):
            step()
        err, ang = P.fold_metrics(sim.q(), 0, verts, np.zeros(3), r_self, stat_idx, kp_idx, kp_target)
        return record("folded" + (" + pressed" if press else ", no press"), fold_err_cm=100 * err, open_deg=abs(ang),
                      held=bool(err < 0.015))

    def lift(tag):
        """Row 6 picked up where it lies, raised so the sheet hangs from it, laid back down flat."""
        R, T = frame_now()
        free(pin)
        q = sim.q()
        start = q[r6]
        # Local frame of the bottom half: row 0 at x = -L/2. Hang above row 0, then lay down.
        L = SIZE
        x0 = verts[row(0)][0, 0]
        ys = verts[r6][:, 1]

        def world(x, z):
            loc = np.stack([np.full(RES, x), ys, np.full(RES, z - Z0)], axis=1)
            return (R @ loc.T).T + T

        top = world(x0, Z0 + 0.13)
        mark(f"lift {tag}: edge picked up")
        n = fr(1.2)
        for k in range(1, n + 1):
            u = P.smooth(k / n)
            p = (1 - u) * start + u * top
            p[:, 2] += 0.03 * math.sin(math.pi * u)
            set_drive(r6, p)
            apply()
            step()
        mark(f"lift {tag}: hanging")
        for _ in range(fr(1.5)):
            set_drive(r6, top)
            apply()
            step()
        th, _ = hinge_deg()
        hang = dict(hang_hinge_max_deg=float(th.max()), hang_crease_max_deg=float(th[crease_edges].max()))
        mark(f"lift {tag}: laid back down")
        n = fr(0.4)
        low = world(x0, Z0 + L)
        for k in range(1, n + 1):
            u = P.smooth(k / n)
            set_drive(r6, (1 - u) * top + u * low)
            apply()
            step()
        n = fr(1.5)
        for k in range(1, n + 1):
            psi = 0.5 * math.pi * P.smooth(k / n)
            set_drive(r6, world(x0 + L * math.sin(psi), Z0 + L * math.cos(psi)))
            apply()
            step()
        free(r6)
        apply()
        for _ in range(fr(1.5)):
            step()
        return record(f"lifted and laid down ({tag})", **hang)

    def crumple():
        q = sim.q()
        c = q.mean(0)
        goal = np.array([c[0], c[1], Z0 + 0.04])
        start = q[corners]
        # each corner goes to its own fingertip spot CRUMPLE_SPREAD apart around the goal (corners
        # driven to one exact point overlap, which self-contact cannot resolve while they are held)
        goal = goal + CRUMPLE_SPREAD * 0.5 * np.array([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]], dtype=float)
        # corners 0, 1 are in rows 0-1 (pin), corners 2, 3 in row 6; drive only the corners
        mask[:] = False
        mark("crumple: corners gathered")
        n = fr(1.5)
        for k in range(1, n + 1):
            u = P.smooth(k / n)
            set_drive(corners, start + (goal - start) * u)
            apply()
            step()
        for _ in range(fr(1.0)):
            step()
        mark("crumple: let go")
        free(corners)
        apply()
        for _ in range(fr(3.0)):
            step()
        q = sim.q()
        xy = q[:, :2]
        return record("crumpled", footprint_cm2=1e4 * float(np.ptp(xy[:, 0]) * np.ptp(xy[:, 1])))

    t0 = time.time()
    mark("flat")
    for _ in range(fr(0.5)):
        step()
    if only is not None:  # yield sweep: one fold on a fresh flat sheet ("fold0" no press, "fold1" pressed)
        seq = [crumple()] if only == "crumple" else [fold(only == "fold1")]
    else:
        seq = [fold(False), lift("after unpressed fold"), fold(True), lift("after pressed fold"), crumple()]
    for r in seq:
        r.update(fabric=fab, variant=variant)
        extra = " ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()
                         if k not in ("fabric", "variant", "phase", "t"))
        print(f"[seq {fab:10s} {variant:7s}] {r['phase']:38s} {extra}", flush=True)
    print(f"[seq {fab} {variant}] {time.time() - t0:.1f} s", flush=True)
    return seq, (np.stack(rec) if rec else None), np.array(times), phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/cloth_sequence")
    ap.add_argument("--fabrics", nargs="+", default=list(CURRENT))
    ap.add_argument("--set", type=float, default=80.0)
    ap.add_argument("--soft-tri", type=float, default=100.0, help="new tri_ke = tri_ka for chiffon, cotton, silk")
    ap.add_argument("--chiffon-ke", type=float, default=None, help="new chiffon edge_ke (refit at --soft-tri)")
    ap.add_argument("--new-yield", nargs="*", default=[], help="fab=kappa overrides of the proposed yields")
    ap.add_argument("--record-every", type=int, default=2)
    ap.add_argument("--variants", nargs="+", default=["current", "new"], choices=["current", "new", "thin"])
    ap.add_argument("--thin-ke", type=float, default=500.0, help="thin: soft_contact_ke")
    ap.add_argument("--thin-self", type=float, default=1.0, help="thin: self_contact_radius / thickness")
    ap.add_argument("--self-margin", type=float, default=None, help="self-contact margin [m] (default 1.5 x radius)")
    ap.add_argument("--thin-yield", nargs="*", default=[], help="thin: fab=kappa yield overrides")
    ap.add_argument("--thin-set", nargs="*", default=[], help="thin: fab:key=value preset overrides (e.g. cotton:ke=0.004)")
    ap.add_argument("--sweep-yield", nargs="*", type=float, default=None,
                    help="thin only: instead of the sequence, one unpressed and one pressed fold on fresh sheets per "
                         "yield; writes sweep.json (the yield refit on the sequence's own fold)")
    ap.add_argument("--crumple-spread", type=float, default=0.0, help="crumple: corner targets this far apart [m]")
    ap.add_argument("--only", choices=["crumple"], default=None, help="run just this phase from flat")
    a = ap.parse_args()
    if a.self_margin:
        P.SELF_MARGIN = a.self_margin
    global CRUMPLE_SPREAD
    CRUMPLE_SPREAD = a.crumple_spread
    for s in a.new_yield:
        k, v = s.split("=")
        NEW[k] = float(v)
    for s in a.thin_yield:
        k, v = s.split("=")
        THIN_YIELD[k] = float(v)
    for s in a.thin_set:
        fk, v = s.split("=")
        fab_, key = fk.split(":")
        THIN_OVER.setdefault(fab_, {})[key] = float(v)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows, npz, cfgs = [], {}, {}
    for fab in a.fabrics:
        new_over = {}
        if fab in SOFT:
            new_over["tri"] = a.soft_tri
        if fab == "chiffon" and a.chiffon_ke is not None:
            new_over["ke"] = a.chiffon_ke
        thin_over = dict(new_over, radius=0.5 * THIN[fab], r_self=a.thin_self * THIN[fab], soft_ke=a.thin_ke)
        thin_over.update({k: v for k, v in THIN_OVER.get(fab, {}).items()})
        vdef = dict(current=(CURRENT[fab], 0.0, {}), new=(NEW[fab], a.set, new_over),
                    thin=(THIN_YIELD.get(fab, NEW[fab]), a.set, thin_over))
        if a.sweep_yield:
            for yk in a.sweep_yield:
                for only in ("fold0", "fold1"):
                    seq, *_ = run(fab, "thin", float(yk), a.set, thin_over, 1000, only=only)
                    r = seq[0]
                    r.update(fabric=fab, yield_kappa=float(yk), press=only == "fold1")
                    rows.append(r)
                    print(f"[sweep {fab:10s}] yield {yk:6.1f} {'pressed' if only == 'fold1' else 'no press'}  "
                          f"err {r['fold_err_cm']:5.2f} cm open {r['open_deg']:6.1f} held {r['held']}", flush=True)
            continue
        for variant in a.variants:
            yk, sk, over = vdef[variant]
            seq, frames, times, phases = run(fab, variant, float(yk), sk, over, a.record_every, only=a.only)
            rows += seq
            npz[f"{fab}__{variant}"] = frames
            npz["times"] = times
            npz["phase_t"] = np.array([p[0] for p in phases])
            npz["phase_names"] = np.array([p[1] for p in phases])
            base = P.FABRICS[f"{fab}_measured"]
            cfgs[f"{fab}__{variant}"] = dict(yield_kappa=float(yk), set_kappa=sk, tri_ke=over.get("tri", base["tri"]),
                                             edge_ke=over.get("ke", base["ke"]), density=base["rho"],
                                             thickness_mm=1e3 * 2 * over.get("radius", RADIUS_DEFAULT),
                                             self_contact_radius_mm=1e3 * over.get("r_self", base["r_self"]),
                                             soft_contact_ke=over.get("soft_ke", SOFT_KE_DEFAULT))
    if a.sweep_yield:
        (out / "sweep.json").write_text(json.dumps(rows, indent=1))
        print(f"[done] {out}", flush=True)
        return
    faces = np.array(P.grid_mesh(SIZE, RES)[1], dtype=np.int32).reshape(-1, 3)
    np.savez_compressed(out / "sequence.npz", faces=faces, res=RES, size=SIZE, **npz)
    (out / "sequence.json").write_text(json.dumps(dict(configs=cfgs, rows=rows), indent=1))
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
