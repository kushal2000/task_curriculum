"""A scripted fingertip against the env's VBD sheet: does it go THROUGH the cloth?

No policy. A kinematic capsule the size of a fingertip (radius 7.25 mm, 24 mm shaft, mu 1.5 like the
env's fingertips) is driven into the 10 cm sheet. Scenarios, most of them with nothing behind the
cloth, so the finger can push as far as it likes:

    suspended_slow/fast  sheet held flat 8 cm up by two opposite edges (it sags), the fingertip
                         pokes straight down into its middle 5 cm past the sheet's starting
                         height at 0.1 / 0.5 m/s, holds 0.5 s, pulls back out the same way
    hanging_slow/fast    sheet hanging from its top edge like a curtain, the fingertip pokes
                         horizontally into its middle 5 cm past the sheet's plane, holds, pulls back
    table                reference: sheet on the table, finger tilted 45 deg pressed down onto the
                         table through the cloth, dragged 3 cm, lifted 6 cm

One model per (full-surface contact, soft_contact_ke); one sheet + finger per (thickness, scenario).

Measured (dense cloth samples, 10 points per triangle, every frame):
    through_frames  frames in which the finger's AXIS (shaft top -> tip centre) crosses a cloth
                    triangle. Cloth that wraps the fingertip like a glove stays one finger radius
                    off the axis, so a crossing means part of the finger is on the far side of the
                    sheet. For the table case it is the cloth ending up ON TOP of the finger instead
                    (> half a radius above the axis, next to it): a pressed finger never has cloth up there.
    pen_mm          deepest a cloth point gets inside the finger (the soft contact's give)
    tent_cm         how far the cloth was pushed along the poke (max displacement along it)
    end_cm          after pulling back: cloth still displaced along the poke (carried / snagged)
    drag_frac / lift_max_cm (table case) sheet travel / finger travel in the drag; highest cloth
                    point 1 s after the lift (> 2 cm = hanging on the finger)
Output: --out/finger.json and finger.npz (frames).
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

RES, NP, SIZE = P.RES, P.NP, P.SIZE
FPS, SUB = 60, 24  # 1/1440 s cloth step, as the env
DT = 1.0 / (FPS * SUB)
R_F, HH_F, MU_F, MU_TABLE = 0.00725, 0.012, 1.5, 0.5
SPACING = 0.3
SETTLE = 1.0
SCENARIOS = ["suspended_slow", "suspended_fast", "hanging_slow", "hanging_fast", "table"]


def bary_samples(n=4):
    out = []
    for i in range(n):
        for j in range(n - i):
            out.append((i / (n - 1), j / (n - 1), 1 - i / (n - 1) - j / (n - 1)))
    return np.array(out)


def seg_tri_cross(p0, p1, tri):
    """Does segment p0->p1 cross any triangle (T, 3, 3)? Moller-Trumbore, vectorised."""
    d = p1 - p0
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-12
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = p0 - tri[:, 0]
    u = f * np.einsum("ij,ij->i", s, h)
    qv = np.cross(s, e1)
    v = f * (qv @ d)
    t = f * np.einsum("ij,ij->i", e2, qv)
    return bool((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)).any())


def quat_between(a, b):
    """Rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    c = np.cross(a, b)
    w = 1.0 + float(a @ b)
    if w < 1e-8:  # opposite: any perpendicular axis
        ax = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0.0, 1.0, 0.0])
        ax /= np.linalg.norm(ax)
        return np.array([ax[0], ax[1], ax[2], 0.0])
    q = np.array([c[0], c[1], c[2], w])
    return q / np.linalg.norm(q)


class Path_:
    """Fingertip-centre path for one sheet: list of (t_end, fn(u) -> tip) after the settle."""

    def __init__(self, segs):
        self.segs, t = [], SETTLE
        for dur, fn in segs:
            self.segs.append((t, t + dur, fn))
            t += dur
        self.end = t
        self.first = segs[0][1](0.0)

    def __call__(self, t):
        if t <= SETTLE:
            return self.first, "settle"
        for t0, t1, fn in self.segs:
            if t <= t1:
                return fn((t - t0) / (t1 - t0)), fn.__name__
        return self.segs[-1][2](1.0), "end"


def scenario(kind, thick, k):
    """(sheet pos, sheet rot quat xyzw-as-warp, pin rule, finger axis-down dir, Path_, push dir)."""
    rc = 0.5 * thick
    y0 = SPACING * k
    sm = P.smooth
    if kind == "table":
        z_lo = 0.0 + R_F  # tip centre with the finger bottom on the table
        z_hi = thick + 0.03 + R_F
        tilt = math.radians(45.0)
        ad = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
        x0, x1 = -0.015, 0.015

        def down(u): return np.array([x0, y0, z_hi + (z_lo - z_hi) * sm(u)])
        def hold(u): return np.array([x0, y0, z_lo])
        def drag(u): return np.array([x0 + (x1 - x0) * sm(u), y0, z_lo])
        def hold2(u): return np.array([x1, y0, z_lo])
        def lift(u): return np.array([x1, y0, z_lo + 0.06 * sm(u)])
        def after(u): return np.array([x1, y0, z_lo + 0.06])
        path = Path_([(0.6, down), (0.5, hold), (1.0, drag), (0.3, hold2), (0.8, lift), (1.0, after)])
        return dict(pos=(0.0, y0, rc + 0.0005), rot=wp.quat_identity(), pin=None, axis_down=ad, path=path,
                    push=np.array([0.0, 0.0, -1.0]))
    speed = 0.1 if kind.endswith("slow") else 0.5
    depth, gap = 0.05, 0.03  # past the sheet's start plane; start this far before it
    dist = gap + depth
    dur = dist / speed
    if kind.startswith("suspended"):
        zc = 0.08
        push = np.array([0.0, 0.0, -1.0])
        start = np.array([0.0, y0, zc + gap + R_F])
        pos, rot, pin = (0.0, y0, zc), wp.quat_identity(), "x_edges"
    else:
        zc = 0.08  # sheet centre height; top edge at zc + 5 cm
        push = np.array([0.0, 1.0, 0.0])
        start = np.array([0.0, y0 - gap - R_F, zc])
        pos, rot, pin = (0.0, y0, zc), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi), "top"
    tip_in = start + push * dist

    def poke(u): return start + push * dist * u
    def hold(u): return tip_in
    def retract(u): return tip_in - push * dist * u
    def after(u): return start
    path = Path_([(dur, poke), (0.5, hold), (dur, retract), (0.5, after)])
    return dict(pos=pos, rot=rot, pin=pin, axis_down=push.copy(), path=path, push=push)


def run_model(fab, fs, ke, thicks, scen, iters, rec_every):
    f = P.FABRICS[f"{fab}_measured"]
    verts, idx = P._geo.grid_mesh(SIZE, RES, "alternate")
    faces = np.array(idx).reshape(-1, 3)
    combos = [(t, s) for t in thicks for s in scen]
    b = newton.ModelBuilder()
    b.default_shape_cfg.mu = MU_TABLE
    b.add_ground_plane()
    starts, bodies, specs = [], [], []
    for k, (thick, kind) in enumerate(combos):
        sp = scenario(kind, thick, k)
        specs.append(sp)
        st = b.particle_count
        starts.append(st)
        b.add_cloth_mesh(pos=wp.vec3(*sp["pos"]), rot=sp["rot"], scale=1.0, vel=wp.vec3(0.0),
                         vertices=verts, indices=idx, density=f["rho"], tri_ke=f["tri"], tri_ka=f["tri"], tri_kd=1.5e-6,
                         edge_ke=f["ke"], edge_kd=f["kd"], particle_radius=0.5 * thick)
        q = np.array(b.particle_q[st:st + NP])
        if sp["pin"] == "x_edges":
            pins = np.where((q[:, 0] < q[:, 0].min() + 1e-6) | (q[:, 0] > q[:, 0].max() - 1e-6))[0]
        elif sp["pin"] == "top":
            pins = np.where(q[:, 2] > q[:, 2].max() - 1e-6)[0]
        else:
            pins = []
        for p in pins:
            b.particle_mass[st + p] = 0.0
            b.particle_flags[st + p] = b.particle_flags[st + p] & ~newton.ParticleFlags.ACTIVE
        tip0, _ = sp["path"](0.0)
        ad = sp["axis_down"]
        qd = quat_between(np.array([0.0, 0.0, -1.0]), ad)
        body = b.add_body(xform=wp.transform(wp.vec3(*(tip0 - ad * HH_F)), wp.quat(*qd)), is_kinematic=True,
                          label=f"finger_{k}")
        b.add_shape_capsule(body, radius=R_F, half_height=HH_F, cfg=newton.ModelBuilder.ShapeConfig(mu=MU_F))
        bodies.append((body, qd))
    b.color(include_bending=True)
    m = b.finalize()
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = ke, P.SOFT_KD, f["mu"]
    solver = newton.solvers.SolverVBD(model=m, iterations=iters, particle_enable_self_contact=True,
                                      particle_self_contact_radius=f["r_self"],
                                      particle_self_contact_margin=1.5 * f["r_self"],
                                      particle_collision_detection_interval=-1,
                                      particle_conservative_bound_relaxation=P.BOUND_RELAX)
    pipe = newton.CollisionPipeline(m, broad_phase="nxn", soft_contact_margin=0.01,
                                    enable_rigid_soft_full_surface_contact=fs)
    contacts = pipe.contacts()
    s0, s1, ctrl = m.state(), m.state(), m.control()
    n_s = len(combos)
    total = max(sp["path"].end for sp in specs)
    bw = bary_samples()

    def set_bodies(t):
        bq = s0.body_q.numpy()
        bqd = np.zeros((m.body_count, 6), dtype=np.float32)
        for k, (bk, qd) in enumerate(bodies):
            ad = specs[k]["axis_down"]
            tip, _ = specs[k]["path"](t)
            tip2, _ = specs[k]["path"](t + DT)
            bq[bk, :3], bq[bk, 3:] = tip - ad * HH_F, qd
            bqd[bk, :3] = (tip2 - tip) / DT
        for st in (s0, s1):
            st.body_q.assign(bq)
            st.body_qd.assign(bqd)

    q_init = None
    per = [dict(through=0, first=None, pen=0.0, tent=0.0) for _ in range(n_s)]
    drag0 = [None] * n_s
    rec_q, rec_b, times = [], [], []
    t, frame, t0w = 0.0, 0, time.time()
    while t < total - 1e-9:
        for _ in range(SUB):
            set_bodies(t)
            pipe.collide(s0, contacts)
            solver.step(s0, s1, ctrl, contacts, DT)
            s0, s1 = s1, s0
            t += DT
        frame += 1
        q = s0.particle_q.numpy()
        if q_init is None and t >= SETTLE - 1e-9:
            q_init = q.copy()  # settled (sagged / hanging) sheet, before the finger arrives
        for k in range(n_s):
            sp = specs[k]
            sq = q[starts[k]:starts[k] + NP]
            if not np.isfinite(sq).all():
                per[k]["finite"] = False
                continue
            tip, ph = sp["path"](t)
            ad = sp["axis_down"]
            top = tip - 2 * HH_F * ad
            pts = np.einsum("sb,fbd->fsd", bw, sq[faces]).reshape(-1, 3)
            seg = tip - top
            w = np.clip(((pts - top) @ seg) / (seg @ seg), 0.0, 1.0)
            near = top + w[:, None] * seg
            dvec = pts - near
            d = np.linalg.norm(dvec, axis=1)
            per[k]["pen"] = max(per[k]["pen"], 1e3 * max(0.0, float((R_F - d).max())))
            if combos[k][1] == "table":
                hit = bool(((d < R_F + 0.003) & (dvec[:, 2] > 0.5 * R_F)).any())
            else:
                hit = seg_tri_cross(top, tip, sq[faces])
            if hit:
                per[k]["through"] += 1
                per[k]["first"] = per[k]["first"] or ph
            if q_init is not None:
                disp = (sq - q_init[starts[k]:starts[k] + NP]) @ sp["push"]
                per[k]["tent"] = max(per[k]["tent"], 100 * float(disp.max()))
            if combos[k][1] == "table":
                if ph == "drag" and drag0[k] is None:
                    drag0[k] = sq.mean(0)
                if ph == "hold2" and drag0[k] is not None:
                    per[k]["drag_frac"] = float((sq.mean(0)[0] - drag0[k][0]) / 0.03)
        if frame % rec_every == 0:
            rec_q.append(q.astype(np.float32)), rec_b.append(s0.body_q.numpy().astype(np.float32)), times.append(t)
    q = s0.particle_q.numpy()
    rows = []
    for k, (thick, kind) in enumerate(combos):
        sq = q[starts[k]:starts[k] + NP]
        end = 100 * float(((sq - q_init[starts[k]:starts[k] + NP]) @ specs[k]["push"]).max()) if q_init is not None else None
        r = dict(fabric=fab, full_surface=fs, soft_contact_ke=ke, thickness_mm=1e3 * thick, scenario=kind,
                 through_frames=per[k]["through"], through_first_phase=per[k]["first"], pen_mm=round(per[k]["pen"], 2),
                 tent_cm=round(per[k]["tent"], 2), end_cm=None if end is None else round(end, 2),
                 finite=per[k].get("finite", True))
        if kind == "table":
            r["drag_frac"] = None if per[k].get("drag_frac") is None else round(per[k]["drag_frac"], 2)
            r["lift_max_cm"] = round(100 * float(sq[:, 2].max()), 2)
        rows.append(r)
        extra = f"  drag {r['drag_frac']} lift max {r['lift_max_cm']} cm" if kind == "table" else ""
        print(f"[finger fs={int(fs)} ke={ke:g}] {1e3 * thick:4.1f} mm {kind:15s} through {r['through_frames']:3d} frames "
              f"({r['through_first_phase']})  pen {r['pen_mm']:5.2f} mm  tent {r['tent_cm']:5.2f} cm  end {r['end_cm']} cm"
              f"{extra}  finite {r['finite']}", flush=True)
    print(f"[finger fs={int(fs)} ke={ke:g}] {time.time() - t0w:.1f} s", flush=True)
    return rows, np.stack(rec_q), np.stack(rec_b), np.array(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/cloth_finger_probe")
    ap.add_argument("--fabric", default="cotton")
    ap.add_argument("--thickness", type=float, nargs="+", default=[0.016, 0.002, 0.0005])
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--ke", type=float, nargs="+", default=[500.0, 5000.0])
    ap.add_argument("--fs", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--tri-ke", type=float, default=None)
    ap.add_argument("--record-every", type=int, default=2)
    a = ap.parse_args()
    if a.tri_ke is not None:
        P.FABRICS[f"{a.fabric}_measured"]["tri"] = a.tri_ke
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows, npz = [], {}
    for fs in a.fs:
        for ke in a.ke:
            r, fq, fb, times = run_model(a.fabric, bool(fs), ke, a.thickness, a.scenarios, a.iters, a.record_every)
            rows += r
            npz[f"fs{fs}_ke{ke:g}_q"], npz[f"fs{fs}_ke{ke:g}_body_q"], npz["times"] = fq, fb, times
    faces = np.array(P._geo.grid_mesh(SIZE, RES, "alternate")[1], dtype=np.int32).reshape(-1, 3)
    combos = [(t, s) for t in a.thickness for s in a.scenarios]
    np.savez_compressed(out / "finger.npz", faces=faces, res=RES, finger_radius=R_F, finger_half_height=HH_F,
                        labels=np.array([f"{1e3 * t:g}mm_{s}" for t, s in combos]), **npz)
    (out / "finger.json").write_text(json.dumps(rows, indent=1))
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
