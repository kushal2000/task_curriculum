"""Scripted handling of VBD sheets, cloth only (no robot): the realism checks for the cloth model.

Many sheets (fabric x size x model variant) share one Newton model, laid out apart on a ground plane,
and run the same scripted scenario at once. Contact is the env's: particle radius 8 mm (cloth vs
table), self-contact radius 2 mm, margin 3 mm, cloth dt 1/1440 s, 12 VBD iterations.

Scenarios (``--scenario``):

  sequence    fold with no press -> open -> fold with a 0.93 N press -> open. The near edge strip is
              held (a hand keeping the sheet put); the far edge strip is carried over the midline and
              let go 1 cm above the other half; "open" picks the far edge up where it lies and lays
              it back flat along the same arc.
  foldline    one pressed fold about a line at offset (0, 1, 2 cm) and angle (0, 15, 45 deg) from
              the midline; measures where the crease actually formed. A mesh that prefers its own
              rows puts the crease there instead.
  shear       both side strips held, one slid sideways by 10% of the size: in-plane shear buckles
              out of plane into diagonal wrinkles; held, then released.
  hang        held by the two far corners, lifted and brought together to 70% of the width: drape
              folds; then laid down and released.
  touch       a fingertip (vertices within 6 mm) drags 3 cm across the flat sheet and lets go: the
              cloth bunches in front of it; afterwards a light touch should leave (almost) no set.
  cantilever  half the sheet clamped, the other half droops for 6 s (the swatch cantilever test).

Variants per sheet: ``legacy`` = the committed preset on the 7x7 grid (preset edge_ke, tri_ke and
single-stage yield), ``new`` = the irregular mesh with bending from measured rigidity B, literature
membrane stiffness and the hinge rheology (``utils/hinge_rheology.py``), parameters from
``--params`` (json) over the defaults below.

    python isaacsimenvs/tasks/cloth/tools/cloth_probe.py --scenario sequence --out outputs/cloth_probe/seq

Writes ``<out>/<scenario>.json`` (sheets, parameters, per-phase metrics) and ``<out>/<scenario>.npz``
(per-sheet frames at --fps for the viewer page).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

ROOT = Path(__file__).resolve().parents[4]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


geo = _load("cloth_geometry", "isaacsimenvs/tasks/cloth/utils/cloth_geometry.py")
rh = _load("hinge_rheology", "isaacsimenvs/tasks/cloth/utils/hinge_rheology.py")
fabrics = _load("fabrics", "isaacsimenvs/tasks/cloth/utils/fabrics.py")
_plast = _load("plasticity", "isaacsimenvs/tasks/cloth/utils/plasticity.py")

FPS = 60
SUB = 24  # cloth dt 1/1440 s as in the env
DT = 1.0 / FPS / SUB
ITERS = 12
PARTICLE_RADIUS = 0.008  # cloth vs table (env thickness 16 mm)
SELF_RADIUS = 0.002
SELF_MARGIN = 0.003
BOUND_RELAX = 0.42
SOFT_KE, SOFT_KD = 5.0e2, 5.0e-3
GAP = 0.01  # fold: far edge let go this far above the other half
#: Newton colour balancing target (--color-ratio). Warp's balancer never returns for some meshes at the
#: default 1.1 (the 17 mm irregular sheet with bending edges, 2 or 15 copies), and it runs in native code.
COLOR_RATIO = 1.1
PRESS_N = 0.93  # the 213 g swatch weight
Z0 = PARTICLE_RADIUS  # resting height of a flat sheet

FABRICS = ("chiffon", "silk", "napkin", "cotton", "big_napkin", "towel", "denim")
#: literature membrane stiffness (low end of each preset's tensile tri_range) [N/m]
TRI_NEW = dict(chiffon=600.0, silk=1100.0, napkin=1100.0, cotton=1100.0, big_napkin=1100.0, towel=600.0, denim=1100.0)
#: hinge rheology defaults per fabric (overridable with --params). friction_slip = slip curvature
#: friction_moment / k_f [1/m]; the moment is derived from it and B.
RHEO_DEFAULT = dict(
    chiffon=dict(friction_ratio=0.5, friction_slip=60.0, yield_curvature=900.0),
    silk=dict(friction_ratio=0.5, friction_slip=60.0, yield_curvature=450.0),
    napkin=dict(friction_ratio=0.5, friction_slip=60.0, yield_curvature=450.0),
    cotton=dict(friction_ratio=1.0, friction_slip=60.0, yield_curvature=150.0),
    big_napkin=dict(friction_ratio=1.0, friction_slip=60.0, yield_curvature=150.0),
    towel=dict(friction_ratio=1.0, friction_slip=60.0, yield_curvature=150.0),
    denim=dict(friction_ratio=1.0, friction_slip=60.0, yield_curvature=150.0),
)


# ---------------------------------------------------------------------------------- materials
def preset(fab: str) -> dict:
    d = fabrics.load(f"{fab}_measured")
    c = d["cloth"]
    return dict(
        density=float(c["density"]), tri=float(c["tri_ke"]), ke=float(c["edge_ke"]), kd=float(c["edge_kd"]),
        mu=float(c.get("soft_contact_mu", 0.25)), yield_legacy=float(d.get("plastic_yield_curvature", 0.0)),
        B=float(d["measured"]["bending_rigidity"]), cantilever_deg=float(d["measured"]["cantilever_deg"]),
    )


class Sheet:
    """One sheet in the shared model: its mesh, where it sits, its particle and edge ranges."""

    def __init__(self, fab, size, variant, kind, edge_length, origin, params, config=""):
        self.fab, self.size, self.variant, self.kind, self.edge_length = fab, size, variant, kind, edge_length
        self.config = config
        self.origin = np.asarray(origin, dtype=np.float64)
        self.mesh = geo.sheet_mesh(size, "x", kind, 7, edge_length, seed=0)
        self.rest = np.asarray(self.mesh.verts, dtype=np.float64)  # local, z = 0
        self.faces = np.asarray(self.mesh.indices, dtype=np.int32).reshape(-1, 3)
        self.params = params
        self.p0 = self.p1 = 0  # particle range in the model
        self.h = self.mesh.edge_length

    @property
    def key(self) -> str:
        v = self.variant + (f"-{self.config}" if self.config else "")
        return f"{self.fab}__{v}__{int(round(self.size * 100))}cm"

    def local_ids(self, mask) -> np.ndarray:
        return np.nonzero(mask)[0]

    def strip(self, side: float, axis: int = 0, width: float = 0.75) -> np.ndarray:
        """Vertices within ``width * h`` of the edge at ``side * size/2`` along ``axis``."""
        lim = 0.5 * self.size - width * self.h - 1e-9
        return self.local_ids(side * self.rest[:, axis] > lim)


FRICTION_EPS = 1e-2  # --friction-eps: SolverVBD friction_epsilon (velocity regularisation)
SELF_CONTACT = True  # --no-self: self-contact off (diagnostics only)
SHEAR, SQUEEZE = 0.10, 0.05  # --shear / --squeeze: shear scenario displacement fractions
ELASTIC = False  # --elastic: no friction, yield or creep on any sheet (calibration)


def build(sheets, legacy_yield_scale: float = 1.0):
    b = newton.ModelBuilder()
    b.default_shape_cfg.mu = 1.0
    for s in sheets:
        p = s.params
        s.p0 = b.particle_count
        b.add_cloth_mesh(
            pos=wp.vec3(*[float(v) for v in s.origin]), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
            vertices=[tuple(v) for v in s.rest], indices=[int(i) for i in s.faces.reshape(-1)],
            density=p["density"], tri_ke=p["tri"], tri_ka=p["tri"], tri_kd=1.5e-6,
            edge_ke=p["ke"], edge_kd=p["kd"], particle_radius=PARTICLE_RADIUS,
        )
        s.p1 = b.particle_count
    b.color(include_bending=True, target_max_min_color_ratio=COLOR_RATIO)
    b.add_ground_plane()
    m = b.finalize()
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = SOFT_KE, SOFT_KD, 0.265
    solver = newton.solvers.SolverVBD(
        model=m, iterations=ITERS, friction_epsilon=FRICTION_EPS,
        particle_enable_self_contact=SELF_CONTACT,
        particle_self_contact_radius=SELF_RADIUS,
        particle_self_contact_margin=SELF_MARGIN,
        particle_collision_detection_interval=-1,
        particle_conservative_bound_relaxation=BOUND_RELAX,
    )
    # Per-edge rheology parameters from the sheet each edge belongs to.
    ei = m.edge_indices.numpy()
    starts = np.array([s.p0 for s in sheets])
    sheet_of_edge = np.searchsorted(starts, ei[:, 2], side="right") - 1
    rq = m.particle_q.numpy()
    _, w = rh.hinge_geometry(rq, ei)
    ne = len(ei)
    arr = {k: np.zeros(ne) for k in ("bending_rigidity", "friction_ratio", "friction_moment", "yield_curvature",
                                     "hardening", "creep_curvature", "creep_tau", "natural_curvature")}
    for k, s in enumerate(sheets):
        sel = sheet_of_edge == k
        p = s.params
        if s.variant == "legacy":
            # The committed model exactly: uniform edge_ke (B_e = edge_ke * w_e) and single-stage yield
            # on the same hinge width as plasticity.py.
            arr["bending_rigidity"][sel] = p["ke"] * w[sel]
            arr["friction_ratio"][sel] = 0.0
            arr["friction_moment"][sel] = 0.0
            yl = p["yield_legacy"] * legacy_yield_scale
            arr["yield_curvature"][sel] = yl if yl > 0 else np.inf
            arr["hardening"][sel] = 0.0
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
    if ELASTIC:
        arr["friction_ratio"][:] = 0.0
        arr["friction_moment"][:] = 0.0
        arr["yield_curvature"][:] = np.inf
        arr["creep_curvature"][:] = np.inf
    rheo = rh.HingeRheology(m, arr, dt=DT)
    pipe = newton.CollisionPipeline(m, broad_phase="nxn", soft_contact_margin=0.01)
    return m, solver, rheo, pipe, sheet_of_edge, w


@wp.kernel
def _drive(q: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3), idx: wp.array(dtype=int),
           target: wp.array(dtype=wp.vec3), on: wp.array(dtype=int)):
    k = wp.tid()
    if on[k] == 1:
        q[idx[k]] = target[k]
        qd[idx[k]] = wp.vec3(0.0)


class Sim:
    """The shared model, a per-particle kinematic drive (any particle can be held or released) and
    per-particle external forces."""

    def __init__(self, sheets, graph=True):
        self.sheets = sheets
        self.m, self.solver, self.rheo, self.pipe, self.sheet_of_edge, self.w = build(sheets)
        m = self.m
        self.contacts = self.pipe.contacts()
        self.s0, self.s1, self.control = m.state(), m.state(), m.control()
        n = m.particle_count
        self.n = n
        self.idx = wp.array(np.arange(n), dtype=int)
        self.target_np = m.particle_q.numpy().copy()
        self.on_np = np.zeros(n, dtype=np.int32)
        self.target = wp.array(self.target_np, dtype=wp.vec3)
        self.on = wp.array(self.on_np, dtype=int)
        self.f_np = np.zeros((n, 3), dtype=np.float32)
        self.f_ext = wp.zeros(n, dtype=wp.vec3)
        self.flags0 = m.particle_flags.numpy().copy()
        self._frame()  # compile
        self.reset()
        self.graph = None
        if graph and wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self._frame()
            self.graph = cap.graph
            self.reset()
        self.t = 0.0

    def reset(self):
        self.s0.particle_q.assign(self.m.particle_q)
        self.s0.particle_qd.zero_()
        self.rheo.reset()

    def _frame(self):
        for _ in range(SUB):
            wp.launch(_drive, dim=self.n, inputs=[self.s0.particle_q, self.s0.particle_qd, self.idx, self.target, self.on])
            wp.copy(self.s0.particle_f, self.f_ext)
            self.pipe.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, DT)
            self.s0, self.s1 = self.s1, self.s0
            self.rheo.apply(self.s0)

    def commit(self):
        """Push the drive mask/targets and forces to the device (call after changing them)."""
        newly_free = (self.on_np == 0) & (self.on.numpy() == 1)
        if newly_free.any():  # let go at rest: the hand opens, it does not throw
            qd = self.s0.particle_qd.numpy()
            qd[newly_free] = 0.0
            self.s0.particle_qd.assign(qd)
        self.target.assign(self.target_np.astype(np.float32))
        self.on.assign(self.on_np)
        flags = self.flags0.copy()
        flags[self.on_np == 1] &= ~int(newton.ParticleFlags.ACTIVE)
        self.m.particle_flags.assign(flags)
        self.f_ext.assign(self.f_np)

    def frame(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._frame()
        self.t += 1.0 / FPS

    def q(self) -> np.ndarray:
        return self.s0.particle_q.numpy().astype(np.float64)

    def theta(self) -> np.ndarray:
        """Current per-edge dihedral angle [rad]."""
        return _plast.dihedral_angles(self.s0.particle_q, self.m.edge_indices).astype(np.float64)


# ---------------------------------------------------------------------------------- helpers
def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def fold_pose(rest_local, c, phi, ang, lift):
    """Rest points (local, z = 0) turned by ``ang`` about the line through (c, 0) at angle ``phi``
    from the y axis (normal (cos phi, sin phi)); pivot lift/2 above the bottom ply.

    Length-preserving: the crease wraps a half circle of radius lift/2, which takes ``r * ang`` of
    material, so the flap beyond it is shorter by that much. (A rigid turn about the raised pivot
    held the flap ~30% stretched, which a soft membrane released as a catapult when let go.)"""
    n = np.array([math.cos(phi), math.sin(phi), 0.0])
    s = (rest_local[:, :2] - np.array([c, 0.0])) @ n[:2]
    r = 0.5 * lift
    base = rest_local - s[:, None] * n
    sf = s - r * ang
    out = base + (sf * math.cos(ang) + r * math.sin(ang))[:, None] * n
    out[:, 2] = base[:, 2] + sf * math.sin(ang) + r * (1.0 - math.cos(ang))
    return out


def current_angle(q_local, c, phi):
    """Angle [rad] of points about the fold line: 0 = lying flat on their own side, pi = folded over."""
    n = np.array([math.cos(phi), math.sin(phi)])
    v = q_local.mean(0)
    return float(np.clip(math.atan2(max(v[2] - Z0, 0.0), (v[:2] - [c, 0.0]) @ n), 0.0, math.pi))


class Recorder:
    def __init__(self, sim: Sim, every: int):
        self.sim, self.every, self.k = sim, every, 0
        self.frames, self.times, self.phases = [], [], []

    def tick(self):
        if self.k % self.every == 0:
            q = self.sim.q()
            self.frames.append([q[s.p0:s.p1] - s.origin for s in self.sim.sheets])
            self.times.append(round(self.sim.t, 4))
        self.k += 1

    def mark(self, name):
        self.phases.append((round(self.sim.t, 4), name))


def run_frames(sim, rec, seconds, per_frame=None):
    n = int(round(seconds * FPS))
    for k in range(1, n + 1):
        if per_frame is not None:
            per_frame(k / n)
            sim.commit()
        sim.frame()
        rec.tick()


def sheet_edges(sim, k):
    return np.nonzero(sim.sheet_of_edge == k)[0]


def interior(sim, eids):
    ei = sim.m.edge_indices.numpy()
    return eids[(ei[eids, 0] >= 0) & (ei[eids, 1] >= 0)]


def curvature_stats(sim, k, theta, exclude_local=None):
    """|kappa| = |theta| / w over a sheet's interior edges [1/m], optionally excluding edges touching
    the given local vertex ids (held strips)."""
    s = sim.sheets[k]
    e = interior(sim, sheet_edges(sim, k))
    if exclude_local is not None and len(exclude_local):
        ei = sim.m.edge_indices.numpy()[e]
        bad = np.isin(ei, np.asarray(exclude_local) + s.p0).any(1)
        e = e[~bad]
    kap = np.abs(theta[e]) / np.maximum(sim.w[e], 1e-9)
    return dict(kappa_mean=float(kap.mean()), kappa_p90=float(np.percentile(kap, 90)),
                frac_over_50=float((kap > 50.0).mean()), frac_over_150=float((kap > 150.0).mean()))


def plastic_stats(sim, k):
    s_e = interior(sim, sheet_edges(sim, k))
    kp = np.abs(sim.rheo.plastic_curvature()[s_e])
    tp = np.degrees(np.abs(sim.rheo.theta_p.numpy()[s_e]))
    return dict(set_kappa_max=float(kp.max()), set_deg_max=float(tp.max()), set_frac_over_10deg=float((tp > 10).mean()))


def wrinkle_profile(local_q, rest, axis_dir, size, n=200):
    """Height profile along a line through the centre (direction ``axis_dir`` in rest coordinates),
    detrended with a quadratic, sampled at n points: (amplitude std [m], zero crossings, wavelength [m])."""
    from scipy.interpolate import griddata

    d = np.asarray(axis_dir, dtype=np.float64)
    d = d / np.linalg.norm(d)
    L = 0.8 * size / max(abs(d[0]), abs(d[1]))
    t = np.linspace(-0.5 * L, 0.5 * L, n)
    pts = np.stack([t * d[0], t * d[1]], 1)
    z = griddata(rest[:, :2], local_q[:, 2], pts, method="linear")
    ok = np.isfinite(z)
    if ok.sum() < 10:
        return 0.0, 0, 0.0
    t, z = t[ok], z[ok]
    z = z - np.polyval(np.polyfit(t, z, 2), t)
    amp = float(z.std())
    zc = int(((z[:-1] > 0) != (z[1:] > 0)).sum()) if amp > 2e-4 else 0
    lam = float(2.0 * (t[-1] - t[0]) / zc) if zc > 0 else 0.0
    return amp, zc, lam


# ---------------------------------------------------------------------------------- scenarios
def scenario_sequence(sim: Sim, rec: Recorder):
    sheets = sim.sheets
    out = {s.key: [] for s in sheets}
    pins = [s.strip(-1.0) for s in sheets]
    fars = [s.strip(+1.0) for s in sheets]
    corners = [s.mesh.corner_idx for s in sheets]

    def hold(k, local, pos_local):
        s = sheets[k]
        g = local + s.p0
        sim.target_np[g] = pos_local + s.origin
        sim.on_np[g] = 1

    def free(k, local):
        sim.on_np[np.asarray(local) + sheets[k].p0] = 0

    def press(on):
        sim.f_np[:] = 0.0
        if on:
            for k, s in enumerate(sheets):
                sel = s.local_ids((s.rest[:, 0] > 0.1 * s.size) & (s.rest[:, 0] < 0.4 * s.size))
                sim.f_np[sel + s.p0, 2] = -PRESS_N / len(sel)

    def fold_metrics(k):
        s = sheets[k]
        q = sim.q()[s.p0:s.p1] - s.origin
        tgt = s.rest[corners[k]].copy()
        tgt[:, 0] = -tgt[:, 0]
        tgt[:, 2] = Z0 + SELF_RADIUS
        err = float(np.linalg.norm(q[corners[k]] - tgt, axis=1).max())
        far = q[fars[k]].mean(0)
        ang = math.degrees(math.atan2(far[2] - Z0, -far[0]))  # 0 = folded flat, 180 = flat sheet
        return dict(fold_err_cm=100 * err, open_deg=ang, held=bool(err < 0.015), held_4cm=bool(err < 0.04))

    def flat_metrics(k):
        s = sheets[k]
        q = sim.q()[s.p0:s.p1] - s.origin
        th = sim.theta()
        return dict(max_height_mm=1e3 * float(q[:, 2].max() - Z0), **plastic_stats(sim, k),
                    **{f"now_{a}": b for a, b in curvature_stats(sim, k, th, pins[k]).items()})

    for k, s in enumerate(sheets):
        hold(k, pins[k], s.rest[pins[k]] + [0, 0, Z0])
    sim.commit()
    rec.mark("flat")
    run_frames(sim, rec, 0.5)

    def fold(pressed: bool):
        q0 = sim.q()
        starts = [q0[s.p0:s.p1][fars[k]] - s.origin for k, s in enumerate(sheets)]
        rec.mark("fold: far edge carried over" + (" (then pressed)" if pressed else ", no press"))

        def carry(u):
            for k, s in enumerate(sheets):
                arc = fold_pose(s.rest[fars[k]], 0.0, 0.0, math.pi * smooth(u), GAP) + [0, 0, Z0]
                wgt = smooth(min(1.0, u / 0.2))
                hold(k, fars[k], (1 - wgt) * starts[k] + wgt * arc)

        run_frames(sim, rec, 1.5, carry)
        if pressed:
            rec.mark("press 0.93 N on the top half")
            for i in range(1, 13):
                press(True)
                sim.f_np *= i / 12.0
                sim.commit()
                sim.frame()
                rec.tick()
            for k in range(len(sheets)):
                free(k, fars[k])
            press(True)
            sim.commit()
            run_frames(sim, rec, 0.8)
            for i in range(11, -1, -1):
                press(True)
                sim.f_np *= i / 12.0
                sim.commit()
                sim.frame()
                rec.tick()
            rec.mark("press off, let go")
        else:
            run_frames(sim, rec, 0.2)
            for k in range(len(sheets)):
                free(k, fars[k])
            sim.commit()
            rec.mark("let go, no press")
        # 180 policy steps (3 s at 60 Hz) untouched after letting go: the worst keypoint error over the
        # whole window is the cloth's own "holds 180 steps" (the eval's criterion, without a hand).
        worst = np.zeros(len(sheets))
        for _ in range(180):
            sim.frame()
            rec.tick()
            q = sim.q()
            for k, s in enumerate(sheets):
                ql = q[s.p0:s.p1] - s.origin
                tgt = s.rest[corners[k]].copy()
                tgt[:, 0] = -tgt[:, 0]
                tgt[:, 2] = Z0 + SELF_RADIUS
                worst[k] = max(worst[k], float(np.linalg.norm(ql[corners[k]] - tgt, axis=1).max()))
        tag = "folded + pressed" if pressed else "folded, no press"
        for k, s in enumerate(sheets):
            out[s.key].append(dict(phase=tag, t=round(sim.t, 2), **fold_metrics(k), **plastic_stats(sim, k),
                                   worst_err_180_cm=100 * worst[k], held_180_4cm=bool(worst[k] < 0.04),
                                   held_180_1p5cm=bool(worst[k] < 0.015)))

    def unfold(tag):
        q0 = sim.q()
        starts = [q0[s.p0:s.p1][fars[k]] - s.origin for k, s in enumerate(sheets)]
        # Pick the far edge up where it is and carry it back from its CURRENT angle: a sheet that
        # already sprang open is just held and let go, not dragged across itself.
        ang0 = [current_angle(st, 0.0, 0.0) for st in starts]
        rec.mark(f"open ({tag}): far edge laid back flat")

        def carry(u):
            for k, s in enumerate(sheets):
                arc = fold_pose(s.rest[fars[k]], 0.0, 0.0, ang0[k] * (1.0 - smooth(u)), GAP) + [0, 0, Z0]
                wgt = smooth(min(1.0, u / 0.15))
                hold(k, fars[k], (1 - wgt) * starts[k] + wgt * arc)

        run_frames(sim, rec, 1.8, carry)
        for k in range(len(sheets)):
            free(k, fars[k])
        sim.commit()
        rec.mark(f"open ({tag}): let go")
        run_frames(sim, rec, 2.0)
        for k, s in enumerate(sheets):
            out[s.key].append(dict(phase=f"opened ({tag})", t=round(sim.t, 2), **flat_metrics(k)))

    fold(False)
    unfold("after unpressed fold")
    fold(True)
    unfold("after pressed fold")
    return out


def scenario_foldline(sim: Sim, rec: Recorder, lines):
    """``lines``: one (offset_m, angle_deg) per sheet (sheets are replicated per line by main)."""
    sheets = sim.sheets
    out = {}
    infos = []
    for k, s in enumerate(sheets):
        c, deg = s.line
        phi = math.radians(deg)
        n2 = np.array([math.cos(phi), math.sin(phi)])
        sd = (s.rest[:, :2] - [c, 0.0]) @ n2
        far = s.local_ids(sd > sd.max() - 0.75 * s.h)
        # hold the opposite side: vertices farthest on the - side
        pin = s.local_ids(sd < sd.min() + 0.75 * s.h)
        top = s.local_ids((sd > 0.15 * (sd.max())) & (sd < 0.6 * sd.max()))
        infos.append((c, phi, sd, far, pin, top))
    for k, s in enumerate(sheets):
        c, phi, sd, far, pin, top = infos[k]
        g = pin + s.p0
        sim.target_np[g] = s.rest[pin] + [0, 0, Z0] + s.origin
        sim.on_np[g] = 1
    sim.commit()
    rec.mark("flat")
    run_frames(sim, rec, 0.4)
    q0 = sim.q()
    starts = [q0[s.p0:s.p1][infos[k][3]] - s.origin for k, s in enumerate(sheets)]
    rec.mark("fold about the given line")

    def carry(u):
        for k, s in enumerate(sheets):
            c, phi, sd, far, pin, top = infos[k]
            arc = fold_pose(s.rest[far], c, phi, math.pi * smooth(u), GAP) + [0, 0, Z0]
            wgt = smooth(min(1.0, u / 0.2))
            sim.target_np[far + s.p0] = (1 - wgt) * starts[k] + wgt * arc + s.origin
            sim.on_np[far + s.p0] = 1

    run_frames(sim, rec, 1.5, carry)
    rec.mark("press, let go")
    for k, s in enumerate(sheets):
        top = infos[k][5]
        sim.f_np[top + s.p0, 2] = -PRESS_N / max(1, len(top))
        sim.on_np[infos[k][3] + s.p0] = 0
    sim.commit()
    run_frames(sim, rec, 1.0)
    sim.f_np[:] = 0.0
    sim.commit()
    run_frames(sim, rec, 1.5)
    th = sim.theta()
    ei = sim.m.edge_indices.numpy()
    q = sim.q()
    for k, s in enumerate(sheets):
        c, phi, sd, far, pin, top = infos[k]
        e = interior(sim, sheet_edges(sim, k))
        mid = 0.5 * (s.rest[ei[e, 2] - s.p0] + s.rest[ei[e, 3] - s.p0])
        n2 = np.array([math.cos(phi), math.sin(phi)])
        sde = (mid[:, :2] - [c, 0.0]) @ n2
        a = np.abs(th[e])
        # The crease = where the sheet's bend is concentrated: weight each hinge by its angle, over the
        # hinges that carry most of the total fold (the 180 deg is shared by the loop's hinges).
        ordr = np.argsort(-a)
        cum = np.cumsum(a[ordr])
        sel = ordr[: max(1, int(np.searchsorted(cum, 0.6 * cum[-1])) + 1)]
        wts = a[sel]
        crease = float((sde[sel] * wts).sum() / wts.sum())
        # along-line spread of the crease (straightness): the crease position in bins along the line
        tdir = np.array([-n2[1], n2[0]])
        along = mid[sel, :2] @ tdir
        bins = np.linspace(along.min(), along.max(), 6)
        pos = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = (along >= lo) & (along <= hi)
            if mm.sum():
                pos.append(float((sde[sel][mm] * wts[mm]).sum() / wts[mm].sum()))
        ql = q[s.p0:s.p1] - s.origin
        out[s.key + f"__c{int(round(c * 1000))}mm_a{int(round(math.degrees(phi)))}"] = dict(
            fab=s.fab, variant=s.variant, size=s.size, offset_mm=1e3 * c, angle_deg=math.degrees(phi),
            crease_offset_err_mm=1e3 * crease, crease_wiggle_mm=1e3 * float(np.std(pos)) if pos else 0.0,
            folded_height_mm=1e3 * float(ql[:, 2].max() - Z0), **plastic_stats(sim, k))
    return out


def scenario_shear(sim: Sim, rec: Recorder):
    sheets = sim.sheets
    out = {s.key: [] for s in sheets}
    L = [s.strip(-1.0) for s in sheets]
    R = [s.strip(+1.0) for s in sheets]
    for k, s in enumerate(sheets):
        for ids in (L[k], R[k]):
            sim.target_np[ids + s.p0] = s.rest[ids] + [0, 0, Z0] + s.origin
            sim.on_np[ids + s.p0] = 1
    sim.commit()
    rec.mark("flat, both sides held")
    run_frames(sim, rec, 0.4)
    rec.mark(f"shear: right side slid {100 * SHEAR:.0f}% sideways, sides {100 * SQUEEZE:.0f}% closer")

    def move(u):
        for k, s in enumerate(sheets):
            d = np.array([-0.5 * SQUEEZE * s.size, SHEAR * s.size, 0.0]) * smooth(u)
            sim.target_np[R[k] + s.p0] = s.rest[R[k]] + [0, 0, Z0] + d + s.origin
            sim.target_np[L[k] + s.p0] = s.rest[L[k]] + [0, 0, Z0] + np.array([0.5 * SQUEEZE * s.size, 0, 0]) * smooth(u) + s.origin

    run_frames(sim, rec, 1.2, move)
    run_frames(sim, rec, 1.0)
    th = sim.theta()
    q = sim.q()
    for k, s in enumerate(sheets):
        ql = q[s.p0:s.p1] - s.origin
        amp, zc, lam = wrinkle_profile(ql, s.rest, (1.0, 1.0), s.size)
        out[s.key].append(dict(phase="sheared, held", t=round(sim.t, 2), amp_mm=1e3 * amp, crossings=zc,
                               wavelength_mm=1e3 * lam, max_height_mm=1e3 * float(ql[:, 2].max() - Z0),
                               **curvature_stats(sim, k, th, np.concatenate([L[k], R[k]]))))
    rec.mark("released")
    for k, s in enumerate(sheets):
        sim.on_np[np.concatenate([L[k], R[k]]) + s.p0] = 0
    sim.commit()
    run_frames(sim, rec, 2.0)
    th = sim.theta()
    q = sim.q()
    for k, s in enumerate(sheets):
        ql = q[s.p0:s.p1] - s.origin
        amp, zc, lam = wrinkle_profile(ql, s.rest, (1.0, 1.0), s.size)
        out[s.key].append(dict(phase="released", t=round(sim.t, 2), amp_mm=1e3 * amp, crossings=zc,
                               wavelength_mm=1e3 * lam, max_height_mm=1e3 * float(ql[:, 2].max() - Z0),
                               **curvature_stats(sim, k, th), **plastic_stats(sim, k)))
    return out


def scenario_hang(sim: Sim, rec: Recorder):
    sheets = sim.sheets
    out = {s.key: [] for s in sheets}
    grips = []
    for s in sheets:
        r = 0.9 * s.h
        c1 = s.local_ids(np.hypot(s.rest[:, 0] - s.size / 2, s.rest[:, 1] + s.size / 2) < r)
        c2 = s.local_ids(np.hypot(s.rest[:, 0] - s.size / 2, s.rest[:, 1] - s.size / 2) < r)
        grips.append((c1, c2))
    q0 = sim.q()
    rec.mark("flat")
    run_frames(sim, rec, 0.3)
    rec.mark("far corners lifted and brought together")
    starts = [(q0[s.p0:s.p1][g[0]] - s.origin, q0[s.p0:s.p1][g[1]] - s.origin) for s, g in zip(sheets, grips)]

    def lift(u):
        for k, s in enumerate(sheets):
            for side, ids in zip((-1.0, 1.0), grips[k]):
                st = starts[k][0 if side < 0 else 1]
                goal = st.copy()
                goal[:, 0] = -0.5 * s.size + 0.0  # swing the far edge over the near edge
                goal[:, 1] = st[:, 1] * 0.7
                goal[:, 2] = Z0 + 1.15 * s.size
                p = st + (goal - st) * smooth(u)
                p[:, 0] = st[:, 0] + (goal[:, 0] - st[:, 0]) * smooth(u)
                sim.target_np[ids + s.p0] = p + s.origin
                sim.on_np[ids + s.p0] = 1

    run_frames(sim, rec, 1.6, lift)
    run_frames(sim, rec, 1.5)
    th = sim.theta()
    for k, s in enumerate(sheets):
        out[s.key].append(dict(phase="hanging", t=round(sim.t, 2),
                               **curvature_stats(sim, k, th, np.concatenate(grips[k]))))
    rec.mark("let go")
    for k, s in enumerate(sheets):
        sim.on_np[np.concatenate(grips[k]) + s.p0] = 0
    sim.commit()
    run_frames(sim, rec, 2.0)
    th = sim.theta()
    q = sim.q()
    for k, s in enumerate(sheets):
        ql = q[s.p0:s.p1] - s.origin
        xy = ql[:, :2]
        out[s.key].append(dict(phase="dropped", t=round(sim.t, 2), **curvature_stats(sim, k, th), **plastic_stats(sim, k),
                               footprint_frac=float(np.ptp(xy[:, 0]) * np.ptp(xy[:, 1]) / s.size ** 2),
                               max_height_mm=1e3 * float(ql[:, 2].max() - Z0)))
    return out


def scenario_touch(sim: Sim, rec: Recorder):
    sheets = sim.sheets
    out = {s.key: [] for s in sheets}
    tips = []
    for s in sheets:
        cx = -0.15 * s.size
        tips.append(s.local_ids(np.hypot(s.rest[:, 0] - cx, s.rest[:, 1]) < max(0.008, 1.2 * s.h)))  # a fingertip pad, >= 1 vertex
    rec.mark("flat")
    run_frames(sim, rec, 0.3)
    q0 = sim.q()
    starts = [q0[s.p0:s.p1][tips[k]] - s.origin for k, s in enumerate(sheets)]
    rec.mark("fingertip drags 3 cm")

    def drag(u):
        for k, s in enumerate(sheets):
            p = starts[k].copy()
            p[:, 0] += 0.03 * smooth(u)
            p[:, 2] = Z0
            sim.target_np[tips[k] + s.p0] = p + s.origin
            sim.on_np[tips[k] + s.p0] = 1

    run_frames(sim, rec, 1.0, drag)
    run_frames(sim, rec, 0.3)
    th = sim.theta()
    q = sim.q()
    for k, s in enumerate(sheets):
        ql = q[s.p0:s.p1] - s.origin
        amp, zc, lam = wrinkle_profile(ql, s.rest, (1.0, 0.0), s.size)
        out[s.key].append(dict(phase="dragged, held", t=round(sim.t, 2), amp_mm=1e3 * amp, crossings=zc,
                               wavelength_mm=1e3 * lam, max_height_mm=1e3 * float(ql[:, 2].max() - Z0),
                               **curvature_stats(sim, k, th, tips[k])))
    rec.mark("let go")
    for k, s in enumerate(sheets):
        sim.on_np[tips[k] + s.p0] = 0
    sim.commit()
    run_frames(sim, rec, 2.0)
    th = sim.theta()
    q = sim.q()
    for k, s in enumerate(sheets):
        ql = q[s.p0:s.p1] - s.origin
        out[s.key].append(dict(phase="released", t=round(sim.t, 2), max_height_mm=1e3 * float(ql[:, 2].max() - Z0),
                               **curvature_stats(sim, k, th), **plastic_stats(sim, k)))
    return out


def scenario_cantilever(sim: Sim, rec: Recorder):
    """Clamp x < 0 flat at table height + 10 cm (no ground under the flap); report the tip-line angle
    at a 5.25 cm overhang, as the swatch test measured."""
    sheets = sim.sheets
    out = {}
    clamps = []
    for s in sheets:
        cl = s.local_ids(s.rest[:, 0] <= 0.0 + 1e-9)
        clamps.append(cl)
        sim.target_np[cl + s.p0] = s.rest[cl] + [0, 0, 0.10] + s.origin
        sim.on_np[cl + s.p0] = 1
    q = sim.s0.particle_q.numpy()
    for s in sheets:
        q[s.p0:s.p1] = s.rest + [0, 0, 0.10] + s.origin
    sim.s0.particle_q.assign(q)
    sim.commit()
    rec.mark("half clamped, flap droops")
    run_frames(sim, rec, 6.0)
    q = sim.q()
    for k, s in enumerate(sheets):
        ql = q[s.p0:s.p1] - s.origin
        # tip = free-edge vertices (x = size/2) near y = 0; the overhang is size/2
        tip = s.local_ids((s.rest[:, 0] > 0.5 * s.size - 0.6 * s.h) & (np.abs(s.rest[:, 1]) < 0.25 * s.size))
        root = s.local_ids((np.abs(s.rest[:, 0]) < 0.6 * s.h) & (np.abs(s.rest[:, 1]) < 0.25 * s.size) & (s.rest[:, 0] <= 0))
        dt = ql[tip].mean(0) - ql[root].mean(0)
        ang = math.degrees(math.atan2(-dt[2], math.hypot(dt[0], dt[1])))
        out[s.key] = dict(fab=s.fab, variant=s.variant, size=s.size, overhang_cm=50 * s.size, tip_angle_deg=ang,
                          measured_deg=s.params["cantilever_deg"], edge_length_mm=1e3 * s.h)
    return out


# ---------------------------------------------------------------------------------- main
def make_params(fab, variant, over, stiffness_scale=1.0):
    p = preset(fab)
    if variant == "new":
        p["tri"] = TRI_NEW[fab]
        p["stiffness_scale"] = stiffness_scale
        p.update(RHEO_DEFAULT[fab])
        # damping relative to bending: keep the preset's kd (it multiplies ke inside VBD)
    p.update(over.get(fab, {}))
    p.update(over.get(f"{fab}:{variant}", {}))
    return p


def main():
    global ITERS, ELASTIC, SELF_CONTACT, SUB, DT, FRICTION_EPS, SHEAR, SQUEEZE, GAP, COLOR_RATIO
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                    choices=["sequence", "foldline", "shear", "hang", "touch", "cantilever"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fabrics", nargs="+", default=list(FABRICS))
    ap.add_argument("--sizes", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    ap.add_argument("--variants", nargs="+", default=["legacy", "new"])
    ap.add_argument("--edge", type=float, default=0.006, help="irregular mesh edge length [m] (new variant)")
    ap.add_argument("--edge-per-size", nargs="*", default=[], help="size=edge overrides, e.g. 0.3=0.008")
    ap.add_argument("--params", default="", help="json: {fab or fab:variant: {param: value}}")
    ap.add_argument("--configs", default="", help="json file {name: {fab: {param: value}}}: one new-variant sheet "
                    "per (config, fabric) in the same model, keys fab__new-<name>__<size>")
    ap.add_argument("--fps", type=int, default=10, help="recorded frames per second")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--iters", type=int, default=ITERS, help="VBD iterations (env: 12)")
    ap.add_argument("--stiffness-scale", type=float, default=1.0, help="new variant: VBD stiffness / (B / w)")
    ap.add_argument("--tag", default="", help="suffix of the output file names")
    ap.add_argument("--tri", type=float, default=0.0, help="new variant: tri_ke = tri_ka for every fabric (0 = TRI_NEW)")
    ap.add_argument("--sub", type=int, default=SUB, help="cloth substeps per 60 Hz frame (env: 24 = 1/1440 s)")
    ap.add_argument("--gap", type=float, default=GAP, help="fold: far edge let go this far above the other half [m]")
    ap.add_argument("--shear", type=float, default=0.10, help="shear: sideways slide / size")
    ap.add_argument("--squeeze", type=float, default=0.05, help="shear: sides brought together / size")
    ap.add_argument("--friction-eps", type=float, default=1e-2, help="SolverVBD friction_epsilon")
    ap.add_argument("--no-self", action="store_true", help="self-contact off (diagnostics)")
    ap.add_argument("--color-ratio", type=float, default=1.1,
                    help="colour balancing target max/min group size (1.3 for the 17 mm mesh, which hangs at 1.1)")
    ap.add_argument("--elastic", action="store_true", help="no friction / yield / creep (stiffness calibration)")
    a = ap.parse_args()
    ITERS = a.iters
    ELASTIC = a.elastic
    SUB = a.sub
    FRICTION_EPS = a.friction_eps
    SHEAR, SQUEEZE = a.shear, a.squeeze
    GAP = a.gap
    COLOR_RATIO = a.color_ratio
    DT = 1.0 / FPS / SUB
    SELF_CONTACT = not a.no_self
    over = {}
    if a.params:
        over = json.loads(Path(a.params).read_text()) if Path(a.params).is_file() else json.loads(a.params)
    edge_size = {float(k): float(v) for k, v in (s.split("=") for s in a.edge_per_size)}

    sheets = []
    x = 0.0
    lines = [(0.0, 0.0), (0.01, 0.0), (0.02, 0.0), (0.0, 15.0), (0.0, 45.0)] if a.scenario == "foldline" else [None]
    configs = json.loads(Path(a.configs).read_text()) if a.configs else {"": {}}
    for size in a.sizes:
        for fab in a.fabrics:
            for variant in a.variants:
                kind = "grid" if variant == "legacy" else "irregular"
                for cname, cfg in (configs.items() if variant == "new" else [("", {})]):
                    if variant == "new" and a.configs and fab not in cfg:
                        continue
                    for line in lines:
                        p = make_params(fab, variant, over, a.stiffness_scale)
                        if variant == "new" and a.tri > 0:
                            p["tri"] = a.tri
                        p.update(cfg.get(fab, {}))
                        s = Sheet(fab, size, variant, kind, edge_size.get(size, a.edge), (x, 0.0, 0.0), p, cname)
                        s.line = line
                        sheets.append(s)
                        x += size + 0.12
    print(f"[probe] {a.scenario}: {len(sheets)} sheets, {sum(len(s.rest) for s in sheets)} particles", flush=True)
    t0 = time.time()
    sim = Sim(sheets, graph=not a.no_graph)
    print(f"[probe] built in {time.time() - t0:.1f} s", flush=True)
    rec = Recorder(sim, max(1, FPS // a.fps))
    t0 = time.time()
    fn = dict(sequence=scenario_sequence, shear=scenario_shear, hang=scenario_hang, touch=scenario_touch,
              cantilever=scenario_cantilever)
    res = scenario_foldline(sim, rec, lines) if a.scenario == "foldline" else fn[a.scenario](sim, rec)
    wall = time.time() - t0
    print(f"[probe] simulated {sim.t:.1f} s in {wall:.1f} s wall", flush=True)
    finite = np.isfinite(sim.q()).all()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(scenario=a.scenario, tag=a.tag, stiffness_scale=a.stiffness_scale, fps=a.fps, dt=DT, iterations=ITERS, particle_radius=PARTICLE_RADIUS,
                self_contact_radius=SELF_RADIUS, wall_s=wall, sim_s=sim.t, finite=bool(finite),
                sheets=[dict(key=s.key, fab=s.fab, size=s.size, variant=s.variant, config=s.config, mesh=s.kind, edge_length=s.h,
                             particles=len(s.rest), triangles=len(s.faces), line=s.line,
                             params={k: (v if np.isfinite(v) else None) if isinstance(v, float) else v
                                     for k, v in s.params.items()}) for s in sheets],
                phases=rec.phases, times=rec.times, results=res)
    (out / f"{a.scenario}{a.tag}.json").write_text(json.dumps(meta, indent=1))
    arrays = {}
    for k, s in enumerate(sheets):
        tag = f"s{k}"
        arrays[f"{tag}_faces"] = s.faces
        arrays[f"{tag}_frames"] = np.stack([f[k] for f in rec.frames]).astype(np.float16)
    np.savez_compressed(out / f"{a.scenario}{a.tag}.npz", times=np.array(rec.times), **arrays)
    for key, rows in res.items():
        if isinstance(rows, list):
            for r in rows:
                print(f"[{a.scenario}] {key:32s} " + " ".join(
                    f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items() if k != "t"), flush=True)
        else:
            print(f"[{a.scenario}] {key:44s} " + " ".join(
                f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in rows.items()
                if k not in ("fab", "variant", "size")), flush=True)
    print(f"[probe] done -> {out} (finite {finite})", flush=True)


if __name__ == "__main__":
    main()
