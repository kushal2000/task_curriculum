"""Pinch test: can two finger pads gather the flat VBD sheet into a ridge and lift it?

    .venv_isaaclab3/bin/python isaacsimenvs/tasks/cloth/tools/vbd_pinch_test.py --out x.json \
        [--thickness 0.008] [--edges 0.017 0.010 0.005] [--tri-scale 1] [--fabrics denim cotton chiffon]

Physics as vbd_press_test.py (same sheet builder, hinge rheology, full-surface contact, cloth dt 1/1440 s),
defaults set to the CURRENT env (8 mm thickness, contact ke 500, friction_epsilon 0.1, table mu 0.6).
Per sheet, material from isaacsimenvs/cfg/fabric/{fab}_vbd_coarse.yaml for the 17 mm mesh (the env's) and
{fab}_vbd.yaml for finer meshes (no preset was calibrated below 10 mm; B is a per-length rigidity, so it transfers).

Script per sheet: settle 0.5 s; two sphere pads (radius 8 mm, the press test's finger pad) at x = +-start_half on
the sheet's centre line, placed 1 mm above the top face and pressed down with --force N each (z dynamic, x/y
scripted); after 0.4 s they slide toward each other at 2 cm/s until their surfaces are --final-gap apart (default
2 x thickness: room for exactly two cloth layers); hold 0.3 s; then both are carried up 4 cm at 5 cm/s, x held.

Measured:
  * ridge_mm      -- highest mid-surface point between the pads above its rest height, end of squeeze
                     (a real fabric buckles into a wrinkle here);
  * grip          -- mean x travel of the material that started under each pad / pad travel (1 = carried, 0 = slid);
  * area_ratio    -- current/rest area of the triangles that started between the pads (<1 = the squeeze went
                     into in-plane compression instead of buckling; real woven fabric barely compresses);
  * pierced       -- a particle ended inside a pad beyond its mid-surface (pad on both sides of the sheet);
  * lifted_mm     -- highest particle above rest after the lift; held_frac = share of the sheet above 1 cm;
  * pinched       -- ridge >= 5 mm, not pierced, and lifted_mm >= 20 mm.
--record saves particle/pad positions at 60 fps for rendering (see vbd_pinch_plot.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
import yaml

import newton

ROOT = Path(__file__).resolve().parents[4]


def _load_probe():
    spec = importlib.util.spec_from_file_location("cloth_probe", ROOT / "isaacsimenvs/tasks/cloth/tools/cloth_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cloth_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


cp = _load_probe()
PAD_R = 0.008
G = 9.81
SOFT_MU = 0.265


def fabric_params(fab: str, edge: float) -> dict:
    name = f"{fab}_vbd_coarse" if abs(edge - 0.017) < 1e-6 else f"{fab}_vbd"
    c = yaml.safe_load((ROOT / f"isaacsimenvs/cfg/fabric/{name}.yaml").read_text())["cloth"]
    inf = float("inf")
    return dict(
        preset=name, density=float(c["density"]), tri=float(c["tri_ke"]), tri_ka=float(c.get("tri_ka", c["tri_ke"])),
        ke=0.0, kd=float(c["edge_kd"]), B=float(c["bending_rigidity"]),
        stiffness_scale=float(c.get("bending_stiffness_scale", 1.0)),
        friction_ratio=float(c.get("friction_ratio", 0.0)), friction_slip=float(c.get("friction_slip_curvature", 0.0)),
        yield_curvature=float(c.get("yield_curvature") or inf), hardening=float(c.get("yield_hardening", 0.0)),
        creep_curvature=float(c.get("creep_curvature") or inf), creep_tau=float(c.get("creep_tau") or inf),
        natural_curvature=float(c.get("natural_curvature", 0.0)),
    )


def build(sheets, fsc, iters, pad_mass, pad_mu, table_mu):
    b = newton.ModelBuilder()
    for s in sheets:
        p = s.params
        s.p0 = b.particle_count
        b.add_cloth_mesh(
            pos=wp.vec3(*[float(v) for v in s.origin]), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
            vertices=[tuple(v) for v in s.rest], indices=[int(i) for i in s.faces.reshape(-1)],
            density=p["density"], tri_ke=p["tri"], tri_ka=p["tri_ka"], tri_kd=1.5e-6,
            edge_ke=p["ke"], edge_kd=p["kd"], particle_radius=cp.PARTICLE_RADIUS,
        )
        s.p1 = b.particle_count
    I = 0.4 * pad_mass * PAD_R ** 2
    b.default_shape_cfg.mu = pad_mu ** 2 / SOFT_MU          # effective cloth-pad mu = sqrt(0.265 * shape mu)
    pads = []
    for k, s in enumerate(sheets):
        row = []
        for side in (-1, 1):
            start = s.origin + np.array([side * 0.03, 0.0, 0.3])
            body = b.add_body(xform=wp.transform(wp.vec3(*[float(v) for v in start]), wp.quat_identity()),
                              mass=pad_mass, inertia=wp.mat33(I, 0, 0, 0, I, 0, 0, 0, I), lock_inertia=True,
                              label=f"pad_{k}_{side}")
            b.add_shape_sphere(body, radius=PAD_R)
            row.append(body)
        pads.append(row)
    b.color(include_bending=True, target_max_min_color_ratio=cp.COLOR_RATIO)
    b.default_shape_cfg.mu = table_mu ** 2 / SOFT_MU
    b.add_ground_plane()
    m = b.finalize()
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = cp.SOFT_KE, cp.SOFT_KD, SOFT_MU
    solver = newton.solvers.SolverVBD(
        model=m, iterations=iters, friction_epsilon=cp.FRICTION_EPS,
        particle_enable_self_contact=True, particle_self_contact_radius=cp.SELF_RADIUS,
        particle_self_contact_margin=cp.SELF_MARGIN, particle_collision_detection_interval=-1,
        particle_conservative_bound_relaxation=cp.BOUND_RELAX,
    )
    ei = m.edge_indices.numpy()
    starts = np.array([s.p0 for s in sheets])
    sheet_of_edge = np.searchsorted(starts, ei[:, 2], side="right") - 1
    ne = len(ei)
    arr = {k: np.zeros(ne) for k in ("bending_rigidity", "friction_ratio", "friction_moment", "yield_curvature",
                                     "hardening", "creep_curvature", "creep_tau", "natural_curvature")}
    for k, s in enumerate(sheets):
        sel = sheet_of_edge == k
        p = s.params
        arr["bending_rigidity"][sel] = p["B"] * p["stiffness_scale"]
        arr["friction_ratio"][sel] = p["friction_ratio"]
        arr["friction_moment"][sel] = p["friction_slip"] * p["friction_ratio"] * p["B"]
        arr["yield_curvature"][sel] = p["yield_curvature"]
        arr["hardening"][sel] = p["hardening"]
        arr["creep_curvature"][sel] = p["creep_curvature"]
        arr["creep_tau"][sel] = p["creep_tau"]
        arr["natural_curvature"][sel] = p["natural_curvature"]
    rheo = cp.rh.HingeRheology(m, arr, dt=cp.DT)
    pipe = newton.CollisionPipeline(m, broad_phase="nxn", soft_contact_margin=0.01,
                                    enable_rigid_soft_full_surface_contact=fsc)
    return m, solver, rheo, pipe, np.asarray(pads)


def tri_areas(q, faces):
    a, b_, c = q[faces[:, 0]], q[faces[:, 1]], q[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b_ - a, c - a), axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fabrics", nargs="+", default=["denim", "cotton", "chiffon"])
    ap.add_argument("--edges", type=float, nargs="+", default=[0.017, 0.010, 0.005])
    ap.add_argument("--thickness", type=float, default=0.008)
    ap.add_argument("--tri-scale", type=float, default=1.0, help="multiply stretch stiffness tri_ke/tri_ka")
    ap.add_argument("--force", type=float, default=1.0, help="downward force per pad [N]")
    ap.add_argument("--start-half", type=float, default=0.03, help="initial pad centre distance from sheet centre [m]")
    ap.add_argument("--final-gap", type=float, default=-1.0, help="pad surface gap at end of squeeze; <0 = 2 x thickness")
    ap.add_argument("--pad-mu", type=float, default=1.0, help="effective cloth-pad friction")
    ap.add_argument("--pad-mass", type=float, default=0.05)
    ap.add_argument("--sub", type=int, default=24)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--no-fsc", action="store_true")
    ap.add_argument("--soft-ke", type=float, default=500.0)
    ap.add_argument("--soft-kd", type=float, default=5e-3)
    ap.add_argument("--table-mu", type=float, default=0.6)
    ap.add_argument("--friction-eps", type=float, default=0.1)
    ap.add_argument("--mesh-seed", type=int, default=0)
    ap.add_argument("--record", type=Path)
    a = ap.parse_args()

    cp.PARTICLE_RADIUS = cp.Z0 = 0.5 * a.thickness
    cp.SUB, cp.DT = a.sub, 1.0 / cp.FPS / a.sub
    cp.SOFT_KE, cp.SOFT_KD, cp.FRICTION_EPS = a.soft_ke, a.soft_kd, a.friction_eps
    final_gap = 2 * a.thickness if a.final_gap < 0 else a.final_gap
    final_half = PAD_R + 0.5 * final_gap
    size = 0.10

    sheets, x = [], 0.0
    for fab in a.fabrics:
        for edge in a.edges:
            p = fabric_params(fab, edge)
            p["tri"] *= a.tri_scale
            p["tri_ka"] *= a.tri_scale
            s = cp.Sheet(fab, size, "new", "irregular", edge, (x, 0.0, 0.0), p, mesh_seed=a.mesh_seed)
            sheets.append(s)
            x += size + 0.15
    nS = len(sheets)
    print(f"[pinch] {nS} sheets, thickness {a.thickness*1e3:.0f} mm, tri x{a.tri_scale}, edges {a.edges}, "
          f"F {a.force} N, pad mu {a.pad_mu}, final gap {final_gap*1e3:.0f} mm", flush=True)
    m, solver, rheo, pipe, pads = build(sheets, not a.no_fsc, a.iters, a.pad_mass, a.pad_mu, a.table_mu)
    contacts = pipe.contacts()
    s0, s1, control = m.state(), m.state(), m.control()
    s0.particle_q.assign(m.particle_q)
    s0.particle_qd.zero_()
    rheo.reset()
    r_cloth = cp.PARTICLE_RADIUS
    ox = np.array([s.origin[0] for s in sheets])

    half = np.full(nS, a.start_half)       # pad centre offset from sheet centre along x
    vx = np.zeros(nS)                       # closing speed (>0 = pads moving inward)
    held = True
    z_cmd, vz_cmd = np.full(nS, 0.3), np.zeros(nS)
    force_z = 0.0

    def pin(state):
        bq = state.body_q.numpy(); bqd = state.body_qd.numpy()
        for k in range(nS):
            for j, side in zip(pads[k], (-1, 1)):
                bq[j, 0] = ox[k] + side * half[k]; bq[j, 1] = 0.0
                bq[j, 3:7] = (0.0, 0.0, 0.0, 1.0)
                bqd[j, 0] = -side * vx[k]; bqd[j, 1] = 0.0; bqd[j, 3:6] = 0.0
                if held:
                    bq[j, 2] = z_cmd[k]; bqd[j, 2] = vz_cmd[k]
        state.body_q.assign(bq); state.body_qd.assign(bqd)

    def frame():
        nonlocal s0, s1
        f = np.zeros((m.body_count, 6), dtype=np.float32)
        f[pads.reshape(-1), 2] = a.pad_mass * G - force_z
        for _ in range(cp.SUB):
            pin(s0)
            s0.body_f.assign(f)
            pipe.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, cp.DT)
            s0, s1 = s1, s0
            rheo.apply(s0)

    cloth_frames, pad_frames, phases = [], [], []

    def record(phase):
        phases.append(phase)
        if a.record:
            cloth_frames.append(s0.particle_q.numpy().copy())
            pad_frames.append(s0.body_q.numpy()[pads.reshape(-1), :3].copy())

    t0 = time.time()
    for _ in range(int(0.5 * cp.FPS)):
        frame(); record("settle")
    q_rest = s0.particle_q.numpy().copy()
    z_rest = np.array([q_rest[s.p0:s.p1, 2].mean() for s in sheets])
    # footprints: particles under each pad at rest; between: triangles whose rest centroid lies between the pads
    foot, between_tri, between_pts = [], [], []
    for k, s in enumerate(sheets):
        loc = q_rest[s.p0:s.p1] - s.origin
        foot.append([np.nonzero(np.hypot(loc[:, 0] - side * a.start_half, loc[:, 1]) < PAD_R + 0.5 * s.h)[0]
                     for side in (-1, 1)])
        cen = loc[s.faces].mean(1)
        between_tri.append(np.nonzero((np.abs(cen[:, 0]) < a.start_half - PAD_R) & (np.abs(cen[:, 1]) < 0.02))[0])
        between_pts.append(np.nonzero((np.abs(loc[:, 0]) < a.start_half) & (np.abs(loc[:, 1]) < 0.02))[0])
    area0 = [tri_areas(q_rest[s.p0:s.p1], s.faces)[between_tri[k]].sum() for k, s in enumerate(sheets)]

    z_cmd = z_rest + r_cloth + 0.001 + PAD_R
    frame(); record("place")
    held = False
    nr = int(0.4 * cp.FPS)
    for i in range(nr):
        force_z = a.force * (i + 1) / nr; frame(); record("press")
    foot_x0 = [[s0.particle_q.numpy()[s.p0 + f, 0].mean() for f in foot[k]] for k, s in enumerate(sheets)]
    vx[:] = 0.02
    travel = a.start_half - final_half
    for _ in range(int(round(travel / 0.02 * cp.FPS))):
        half[:] -= 0.02 / cp.FPS; frame(); record("squeeze")
    half[:] = final_half; vx[:] = 0.0
    for _ in range(int(0.3 * cp.FPS)):
        frame(); record("hold")
    q_sq = s0.particle_q.numpy().copy()
    bq_sq = s0.body_q.numpy()
    # lift: force off, pads carried up kinematically, x held
    force_z = 0.0; held = True
    z_cmd = bq_sq[pads[:, 0], 2].copy(); vz_cmd[:] = 0.05
    for _ in range(int(0.8 * cp.FPS)):
        z_cmd = z_cmd + 0.05 / cp.FPS; frame(); record("lift")
    q_end = s0.particle_q.numpy()
    bq_end = s0.body_q.numpy()
    finite = bool(np.isfinite(q_end).all())

    per = []
    for k, s in enumerate(sheets):
        seg = q_sq[s.p0:s.p1]
        loc = seg - s.origin
        # ridge: highest mid-surface point currently between the pads' inner surfaces, above rest
        inner = final_half - PAD_R
        mask = (np.abs(loc[:, 0]) < max(inner, 0.5 * s.h)) & (np.abs(loc[:, 1]) < 0.02)
        ridge = float(loc[mask, 2].max() - z_rest[k]) if mask.any() else float(loc[between_pts[k], 2].max() - z_rest[k])
        grips = []
        for side, f, x0 in zip((-1, 1), foot[k], foot_x0[k]):
            moved = -side * (seg[f, 0].mean() - x0)
            grips.append(moved / travel)
        area = tri_areas(seg, s.faces)[between_tri[k]].sum() / area0[k]
        # pierce: a particle inside a pad sphere by more than the pad radius minus the shell (past the mid-surface)
        pierced = False
        for qq, bq in ((q_sq, bq_sq), (q_end, bq_end)):
            sq = qq[s.p0:s.p1]
            for j in pads[k]:
                if (np.linalg.norm(sq - bq[j, :3], axis=1) < PAD_R - 1e-4).any():
                    pierced = True
        se = q_end[s.p0:s.p1]
        lifted = float(se[:, 2].max() - z_rest[k])
        held_frac = float((se[:, 2] - z_rest[k] > 0.01).mean())
        r = dict(fabric=s.fab, preset=s.params["preset"], edge_mm=round(s.h * 1e3, 1), n_particles=s.p1 - s.p0,
                 tri_ke=s.params["tri"], ridge_mm=round(ridge * 1e3, 1),
                 grip=[round(float(g), 2) for g in grips], area_ratio=round(float(area), 3), pierced=pierced,
                 lifted_mm=round(lifted * 1e3, 1), held_frac=round(held_frac, 2),
                 vmax=round(float(np.linalg.norm(s0.particle_qd.numpy()[s.p0:s.p1], axis=1).max()), 3))
        r["pinched"] = bool(ridge >= 0.005 and not pierced and lifted >= 0.02)
        per.append(r)
    res = dict(thickness=a.thickness, tri_scale=a.tri_scale, force=a.force, pad_mu=a.pad_mu, final_gap=final_gap,
               start_half=a.start_half, soft_ke=a.soft_ke, friction_eps=a.friction_eps, table_mu=a.table_mu,
               fsc=not a.no_fsc, iters=a.iters, sub=a.sub, finite=finite, wall_s=round(time.time() - t0, 1),
               sheets=per)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    if a.record:
        a.record.parent.mkdir(parents=True, exist_ok=True)
        cloth, padp = np.stack(cloth_frames), np.stack(pad_frames)
        arrays = {}
        for k, s in enumerate(sheets):
            arrays[f"s{k}_frames"] = cloth[:, s.p0:s.p1] - s.origin
            arrays[f"s{k}_pads"] = padp[:, 2 * k:2 * k + 2] - s.origin
            arrays[f"s{k}_faces"] = s.faces
            arrays[f"s{k}_label"] = np.array(f"{s.fab} {s.h*1e3:.0f}mm")
        np.savez_compressed(a.record, fps=cp.FPS, phases=np.array(phases), thickness=a.thickness,
                            tri_scale=a.tri_scale, **arrays)
    print(f"[pinch] finite {finite}  ({res['wall_s']} s)")
    for r in per:
        print(f"[pinch] {r['fabric']:8s} edge {r['edge_mm']:5.1f} mm (n={r['n_particles']:4d}) tri {r['tri_ke']:7.1f}  "
              f"ridge {r['ridge_mm']:5.1f} mm  grip {r['grip']}  area {r['area_ratio']:.2f}  pierced {r['pierced']!s:5s} "
              f"lifted {r['lifted_mm']:5.1f} mm  held {r['held_frac']:.2f}  vmax {r['vmax']:.2f}  "
              f"PINCHED {r['pinched']}", flush=True)


if __name__ == "__main__":
    main()
