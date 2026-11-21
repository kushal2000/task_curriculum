"""Press a finger pad into the VBD sheet with a set FORCE and measure penetration -- the VBD port of
kushal/2026-11-18_pc_experiments_scratch:isaacsimenvs/scripts/physics_test/press_test.py (rigid plates on PhysX).

    .venv_isaaclab3/bin/python isaacsimenvs/tasks/cloth/tools/vbd_press_test.py --out x.json --thickness 0.008 --fsc \
        [--forces 0.5 1 2 5] [--soft-ke 500 --soft-kd 5e-3 --iters 12 --sub 24 ...]

Physics as the cloth env: VBD sheets (legacy = the 7x7 `_measured` presets, new = `_vbd_coarse` via
experiments/nov18_physics_sweep/probe_c17.json), particle radius = thickness / 2, soft contact ke/kd/mu, self-contact
radius/margin, VBD iterations, cloth dt = 1 / (60 * --sub) (24 = the env's 1/1440 s), ``--fsc`` = full-surface
(edge/face) contact, table = ground plane. VBD integrates the pad as a rigid body (AVBD), two-way coupled to the cloth.

Pad: a sphere of radius 8 mm (a finger pad), mass --pad_mass, gravity cancelled, held over its site in x/y and
orientation. Script per sheet (one site x one force per sheet): settle 0.5 s; pad placed 1 mm above the sheet's top
face; downward force ramped 0 -> F over --ramp_s and held --hold_s; then the pad is lifted off at 5 cm/s.

Measured per sheet (footprint = the particles within the pad radius of the site AT REST, tracked by identity):
  * through     -- a footprint particle ends up ABOVE the pad centre: the finger is on both sides of the sheet;
  * squeezed    -- share of footprint particles pushed out from under the pad (x/y beyond the pad radius);
  * depth_mm    -- how far the pad bottom sank below the sheet's top face (steady = median of the last 0.5 s of hold);
  * table_mm    -- deepest particle below its table rest height (particle radius);
  * dragged_mm  -- footprint mean height change after the lift (sheet carried up on the finger);
  * nonfinite, max particle speed.
A sheet FAILS if the pad goes through, or the steady depth exceeds the contact shell (thickness/2: the pad reached the
sheet's mid-surface), or the sheet is dragged up > 15 mm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

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


def build(sheets, fsc: bool, iters: int, pad_mass: float, table_mu: float = 0.36):
    """cloth_probe.build + one dynamic pad sphere per sheet + full-surface contact in the pipeline."""
    b = newton.ModelBuilder()
    b.default_shape_cfg.mu = 1.0
    for s in sheets:
        p = s.params
        s.p0 = b.particle_count
        b.add_cloth_mesh(
            pos=wp.vec3(*[float(v) for v in s.origin]), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
            vertices=[tuple(v) for v in s.rest], indices=[int(i) for i in s.faces.reshape(-1)],
            density=p["density"], tri_ke=p["tri"], tri_ka=p.get("tri_ka", p["tri"]),
            tri_kd=p.get("tri_kd", 1.5e-6),
            edge_ke=p["ke"], edge_kd=p["kd"], particle_radius=cp.PARTICLE_RADIUS,
        )
        s.p1 = b.particle_count
    I = 0.4 * pad_mass * PAD_R ** 2
    pads = []
    for k, s in enumerate(sheets):
        start = s.origin + np.array([s.site_x, 0.0, 0.3])
        body = b.add_body(xform=wp.transform(wp.vec3(*[float(v) for v in start]), wp.quat_identity()),
                          mass=pad_mass, inertia=wp.mat33(I, 0, 0, 0, I, 0, 0, 0, I), lock_inertia=True,
                          label=f"pad_{k}")
        b.add_shape_sphere(body, radius=PAD_R)
        pads.append(body)
    b.color(include_bending=True, target_max_min_color_ratio=cp.COLOR_RATIO)
    # table friction as in the env: effective cloth-table mu = sqrt(soft_contact_mu * shape_mu) = table_mu
    b.default_shape_cfg.mu = table_mu ** 2 / 0.265
    b.add_ground_plane()
    m = b.finalize()
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = cp.SOFT_KE, cp.SOFT_KD, 0.265
    solver = newton.solvers.SolverVBD(
        model=m, iterations=iters, friction_epsilon=cp.FRICTION_EPS,
        particle_enable_self_contact=cp.SELF_CONTACT, particle_self_contact_radius=cp.SELF_RADIUS,
        particle_self_contact_margin=cp.SELF_MARGIN, particle_collision_detection_interval=-1,
        particle_conservative_bound_relaxation=cp.BOUND_RELAX,
    )
    ei = m.edge_indices.numpy()
    starts = np.array([s.p0 for s in sheets])
    sheet_of_edge = np.searchsorted(starts, ei[:, 2], side="right") - 1
    _, w = cp.rh.hinge_geometry(m.particle_q.numpy(), ei)
    ne = len(ei)
    arr = {k: np.zeros(ne) for k in ("bending_rigidity", "friction_ratio", "friction_moment", "yield_curvature",
                                     "hardening", "creep_curvature", "creep_tau", "natural_curvature")}
    for k, s in enumerate(sheets):
        sel = sheet_of_edge == k
        p = s.params
        if s.variant == "legacy":
            arr["bending_rigidity"][sel] = p["ke"] * w[sel]
            yl = p["yield_legacy"]
            arr["yield_curvature"][sel] = yl if yl > 0 else np.inf
            arr["creep_curvature"][sel] = np.inf
            arr["creep_tau"][sel] = np.inf
        else:
            arr["bending_rigidity"][sel] = p["B"] * p.get("stiffness_scale", 1.0)
            arr["friction_ratio"][sel] = p["friction_ratio"]
            arr["friction_moment"][sel] = p["friction_slip"] * p["friction_ratio"] * p["B"]
            arr["yield_curvature"][sel] = p["yield_curvature"]
            arr["hardening"][sel] = p.get("hardening", 0.0)
            arr["creep_curvature"][sel] = p.get("creep_curvature", np.inf)
            arr["creep_tau"][sel] = p.get("creep_tau", np.inf)
            arr["natural_curvature"][sel] = p.get("natural_curvature", 0.0)
    rheo = cp.rh.HingeRheology(m, arr, dt=cp.DT)
    pipe = newton.CollisionPipeline(m, broad_phase="nxn", soft_contact_margin=0.01,
                                    enable_rigid_soft_full_surface_contact=fsc)
    return m, solver, rheo, pipe, pads


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fabrics", nargs="+", default=["denim", "towel", "chiffon"])
    ap.add_argument("--variants", nargs="+", default=["legacy", "new"], help="legacy = 7x7 _measured, new = _vbd_coarse")
    ap.add_argument("--configs", default=str(ROOT / "experiments/nov18_physics_sweep/probe_c17.json"))
    ap.add_argument("--edge", type=float, default=0.017, help="irregular mesh target edge length in metres")
    ap.add_argument("--mesh-seed", type=int, default=0, help="seed for generating the irregular mesh")
    ap.add_argument("--sites", nargs="+", default=["moving_centre", "crease"])
    ap.add_argument("--forces", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--thickness", type=float, default=0.016)
    ap.add_argument("--sub", type=int, default=24, help="cloth substeps per 60 Hz frame (24 = 1/1440 s, the env's)")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--fsc", action="store_true", help="full-surface (edge/face) contact")
    ap.add_argument("--soft-ke", type=float, default=500.0)
    ap.add_argument("--soft-kd", type=float, default=5e-3)
    ap.add_argument("--self-radius", type=float, default=0.002)
    ap.add_argument("--self-margin", type=float, default=0.003)
    ap.add_argument("--pad_mass", type=float, default=0.05)
    ap.add_argument("--ramp_s", type=float, default=0.5)
    ap.add_argument("--hold_s", type=float, default=1.5)
    ap.add_argument("--color-ratio", type=float, default=1.3)
    ap.add_argument("--drag", type=float, default=0.0, help="after the hold, slide the pressed pad this far in +y [m] at 5 cm/s")
    ap.add_argument("--record", type=Path, help="save cloth and pad positions at 60 fps to an NPZ file")
    ap.add_argument("--tri-scale", type=float, default=1.0, help="multiply the preset's stretch stiffness tri_ke/tri_ka")
    ap.add_argument("--density-scale", type=float, default=1.0, help="multiply the preset's areal density")
    ap.add_argument("--table-mu", type=float, default=0.36, help="effective cloth-table friction (env default 0.36)")
    ap.add_argument("--friction-eps", type=float, default=1e-2, help="VBD friction_epsilon [m/s] (Newton default 1e-2)")
    a = ap.parse_args()

    cp.PARTICLE_RADIUS = cp.Z0 = 0.5 * a.thickness
    cp.SUB, cp.DT = a.sub, 1.0 / cp.FPS / a.sub
    cp.SOFT_KE, cp.SOFT_KD = a.soft_ke, a.soft_kd
    cp.FRICTION_EPS = a.friction_eps
    cp.SELF_RADIUS, cp.SELF_MARGIN, cp.COLOR_RATIO = a.self_radius, a.self_margin, a.color_ratio
    cfg = json.loads(Path(a.configs).read_text())["c17"]
    irregular_preset = "vbd_coarse" if a.edge == 0.017 else "vbd"

    size = 0.10
    site_x = {"moving_centre": 0.25 * size, "crease": 0.0, "stationary_centre": -0.25 * size}
    sheets, x = [], 0.0
    for fab in a.fabrics:
        for variant in a.variants:
            for site in a.sites:
                for F in a.forces:
                    p = cp.make_params(fab, variant, {})
                    if variant == "new":
                        p.update(cfg[fab])
                    p["tri"] *= a.tri_scale
                    if "tri_ka" in p:
                        p["tri_ka"] *= a.tri_scale
                    p["density"] *= a.density_scale
                    kind = "grid" if variant == "legacy" else "irregular"
                    s = cp.Sheet(fab, size, variant, kind, a.edge, (x, 0.0, 0.0), p, "c17" if variant == "new" else "",
                                 mesh_seed=a.mesh_seed)
                    s.site, s.site_x, s.force = site, site_x[site], F
                    sheets.append(s)
                    x += size + 0.15
    nS = len(sheets)
    print(f"[press] {nS} sheets, thickness {a.thickness*1e3:.0f} mm, dt 1/{60*a.sub}, iters {a.iters}, fsc {a.fsc}, "
          f"ke {a.soft_ke}, kd {a.soft_kd}, forces {a.forces} N", flush=True)
    m, solver, rheo, pipe, pads = build(sheets, a.fsc, a.iters, a.pad_mass, a.table_mu)
    contacts = pipe.contacts()
    s0, s1, control = m.state(), m.state(), m.control()
    s0.particle_q.assign(m.particle_q)
    s0.particle_qd.zero_()
    rheo.reset()
    r_cloth = cp.PARTICLE_RADIUS
    pads = np.asarray(pads)
    site_xy = np.array([[s.origin[0] + s.site_x, s.origin[1]] for s in sheets])
    foot = []
    for s in sheets:
        d = np.hypot(s.rest[:, 0] - s.site_x, s.rest[:, 1])
        foot.append(np.nonzero(d < PAD_R + 0.5 * s.h)[0] if (d < PAD_R + 0.5 * s.h).any() else np.argsort(d)[:1])

    force_z = np.zeros(nS)  # current downward force per pad [N]
    mode = {"held": np.ones(nS, bool), "z": np.full(nS, 0.3), "vz": np.zeros(nS)}  # held = kinematically placed

    def pin_pads(state):
        """Keep each pad over its site (x/y, upright, no spin); held pads also get their scripted z."""
        bq = state.body_q.numpy(); bqd = state.body_qd.numpy()
        for k in range(nS):
            j = pads[k]
            bq[j, 0:2] = site_xy[k]
            bq[j, 3:7] = (0.0, 0.0, 0.0, 1.0)
            bqd[j, 3:6] = 0.0          # angular (Newton spatial: linear first, angular last)
            bqd[j, 0:2] = 0.0
            if mode["held"][k]:
                bq[j, 2] = mode["z"][k]; bqd[j, 2] = mode["vz"][k]
        state.body_q.assign(bq); state.body_qd.assign(bqd)

    def frame():
        nonlocal s0, s1
        f = np.zeros((m.body_count, 6), dtype=np.float32)
        for k in range(nS):
            f[pads[k], 2] = a.pad_mass * G - force_z[k]   # cancel gravity, press down by F
        for _ in range(cp.SUB):
            pin_pads(s0)
            s0.body_f.assign(f)
            pipe.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, cp.DT)
            s0, s1 = s1, s0
            rheo.apply(s0)

    t0 = time.time()
    for _ in range(int(0.5 * cp.FPS)):     # settle, pads parked high
        frame()
    q = s0.particle_q.numpy()
    top = np.array([q[s.p0 + foot[k]][:, 2].max() + r_cloth for k, s in enumerate(sheets)])
    ref_mean = np.array([q[s.p0 + foot[k]][:, 2].mean() for k, s in enumerate(sheets)])
    q_settle = q.copy()
    # far half of each sheet: the half away from the pad's site (local x on the other side of the crease)
    far = [np.nonzero((s.rest[:, 0] - 0.0) * np.sign(s.site_x if s.site_x != 0 else 1.0) < -0.01)[0] for s in sheets]
    mode["z"] = top + 0.001 + PAD_R; mode["vz"][:] = 0.0      # 1 mm above the top face
    frame()
    mode["held"][:] = False                                    # from here the pads are dynamic in z
    rec = {k: [] for k in ("through", "squeezed", "depth", "table", "vmax", "padz")}
    phases = []
    cloth_frames, pad_frames = [], []

    def measure(phase):
        q = s0.particle_q.numpy(); v = np.linalg.norm(s0.particle_qd.numpy(), axis=1)
        bq = s0.body_q.numpy()
        if a.record:
            cloth_frames.append(q.copy())
            pad_frames.append(bq[pads, :3].copy())
        thr, sq, dep, tab, vm, pz = [], [], [], [], [], []
        for k, s in enumerate(sheets):
            centre = bq[pads[k], 2]
            fp = q[s.p0 + foot[k]]
            thr.append(bool((fp[:, 2] > centre + 1e-4).any()))
            sq.append(float((np.hypot(fp[:, 0] - site_xy[k, 0], fp[:, 1] - site_xy[k, 1]) > PAD_R + 0.5 * s.h).mean()))
            # local through: any particle under/around the pad's CURRENT x/y above its centre (catches drag pierces)
            seg_ = q[s.p0:s.p1]
            near = np.hypot(seg_[:, 0] - site_xy[k, 0], seg_[:, 1] - site_xy[k, 1]) < PAD_R
            thr[-1] = thr[-1] or bool((seg_[near, 2] > centre + 1e-4).any())
            dep.append(float(top[k] - (centre - PAD_R)))
            seg = q[s.p0:s.p1]
            tab.append(float(max(0.0, r_cloth - seg[:, 2].min())))
            vm.append(float(v[s.p0:s.p1].max())); pz.append(float(centre))
        for key, val in zip(("through", "squeezed", "depth", "table", "vmax", "padz"), (thr, sq, dep, tab, vm, pz)):
            rec[key].append(val)
        phases.append(phase)

    F_all = np.array([s.force for s in sheets])
    nr = max(1, int(a.ramp_s * cp.FPS))
    for i in range(nr):
        force_z = F_all * (i + 1) / nr; frame(); measure("ramp")
    for _ in range(int(a.hold_s * cp.FPS)):
        frame(); measure("hold")
    if a.drag > 0:   # slide the pressed pads in +y at 5 cm/s, force kept on
        nd = int(a.drag / 0.05 * cp.FPS)
        for _ in range(nd):
            site_xy[:, 1] += 0.05 / cp.FPS; frame(); measure("drag")
    q_pre_lift = s0.particle_q.numpy().copy()
    # lift: release the force and carry the pads up at 5 cm/s
    force_z = np.zeros(nS); mode["held"][:] = True; mode["vz"][:] = 0.05
    z = s0.body_q.numpy()[pads, 2].copy()
    for _ in range(int(0.6 * cp.FPS)):
        z = z + 0.05 / cp.FPS; mode["z"] = z; frame(); measure("lift")
    q = s0.particle_q.numpy()
    finite = bool(np.isfinite(q).all())
    R = {k: np.array(v) for k, v in rec.items()}; ph = np.array(phases)
    hold = ph == "hold"; last = np.zeros_like(hold); last[np.nonzero(hold)[0][-int(0.5 * cp.FPS):]] = True
    per = []
    for k, s in enumerate(sheets):
        dragged = float(q[s.p0 + foot[k]][:, 2].mean() - ref_mean[k])
        seg0, seg1, seg2 = q_settle[s.p0:s.p1, :2], q_pre_lift[s.p0:s.p1, :2], q[s.p0:s.p1, :2]
        slip_c = float(np.linalg.norm(seg1.mean(0) - seg0.mean(0)))                       # sheet centroid xy shift
        slip_far = float(np.linalg.norm(seg1[far[k]] - seg0[far[k]], axis=1).mean())     # far half, mean xy shift
        slip_end = float(np.linalg.norm(seg2.mean(0) - seg0.mean(0)))
        steady = float(np.median(R["depth"][last, k]))
        r = dict(fabric=s.fab, preset="7x7_measured" if s.variant == "legacy" else irregular_preset, site=s.site,
                 mesh_kind=s.kind, target_edge_m=s.edge_length if s.kind == "irregular" else s.h,
                 tri_ke=s.params["tri"], tri_ka=s.params.get("tri_ka", s.params["tri"]),
                 tri_kd=s.params.get("tri_kd", 1.5e-6),
                 force_N=s.force, through=bool(R["through"][:, k].any()),
                 through_frac_hold=round(float(R["through"][hold, k].mean()), 3),
                 squeezed_max=round(float(R["squeezed"][:, k].max()), 3),
                 depth_mm_steady=round(steady * 1e3, 2), depth_mm_max=round(float(R["depth"][:, k].max()) * 1e3, 2),
                 shell_mm=round(r_cloth * 1e3, 1), table_mm_max=round(float(R["table"][:, k].max()) * 1e3, 2),
                 dragged_mm=round(dragged * 1e3, 1), vmax=round(float(R["vmax"][:, k].max()), 3),
                 slip_centroid_mm=round(slip_c * 1e3, 2), slip_far_mm=round(slip_far * 1e3, 2),
                 slip_after_lift_mm=round(slip_end * 1e3, 2))
        r["failed"] = r["through"] or steady > r_cloth or dragged > 0.015
        per.append(r)
    res = dict(thickness=a.thickness, sub=a.sub, dt=cp.DT, iters=a.iters, fsc=a.fsc, soft_ke=a.soft_ke, soft_kd=a.soft_kd,
               self_radius=a.self_radius, self_margin=a.self_margin, pad_radius=PAD_R, pad_mass=a.pad_mass,
               forces=a.forces, drag=a.drag, tri_scale=a.tri_scale, density_scale=a.density_scale,
               table_mu=a.table_mu, friction_eps=a.friction_eps, mesh_seed=a.mesh_seed,
               finite=finite, wall_s=round(time.time() - t0, 1),
               failed=sum(r["failed"] for r in per), sheets=per)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.record:
        a.record.parent.mkdir(parents=True, exist_ok=True)
        cloth, pad_positions = np.stack(cloth_frames), np.stack(pad_frames)
        arrays = {}
        for k, sheet in enumerate(sheets):
            arrays[f"s{k}_frames"] = cloth[:, sheet.p0:sheet.p1] - sheet.origin
            arrays[f"s{k}_pad"] = pad_positions[:, k] - sheet.origin
            arrays[f"s{k}_faces"] = sheet.faces
        np.savez_compressed(a.record, fps=cp.FPS, phases=ph, **arrays)
    print(f"[press] failed {res['failed']}/{len(per)}  finite {finite}  ({res['wall_s']} s)")
    for r in per:
        print(f"[press] {r['preset']:13s} {r['fabric']:8s} {r['site']:14s} F {r['force_N']:4.1f} N  "
              f"through {str(r['through']):5s} squeezed {r['squeezed_max']:.2f}  depth {r['depth_mm_steady']:6.2f} "
              f"(shell {r['shell_mm']}) mm  table {r['table_mm_max']:.2f} mm  dragged {r['dragged_mm']:5.1f} mm  "
              f"vmax {r['vmax']:.2f}", flush=True)


if __name__ == "__main__":
    main()
