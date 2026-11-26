"""Rigid hinged plates on Newton/MJWarp (no Isaac Lab): settle, fold the half fold by driving the hinges,
release, and record it for viser_replay.py.

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/isaaclab3/bin/python \
        experiments/nov25_better_rigidcloth/newton_plates/fold_plates.py --shape box2-mid --size 0.15 \
        --out outputs/newton_plates/box2-mid.npz

The sheet is the PhysX pool's own USD (backends/rigid/assets.write_shape_usd from the design YAML), so
geometry, mass, hinge axes and limits are the same as on PhysX. Phases: settle on the ground, drive the
half fold's hinges 0 -> target over --fold_s with a PD drive, hold, then drop the drive (spring-back).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="box2-mid")
    ap.add_argument("--size", type=float, default=0.15, help="sheet side [m]")
    ap.add_argument("--thickness", type=float, default=0.002)
    ap.add_argument("--margin", type=float, default=0.007, help="contact shell on every plate shape [m] (PhysX rest offset)")
    ap.add_argument("--fold_deg", type=float, default=175.0)
    ap.add_argument("--ke", type=float, default=2.0, help="hinge drive stiffness [N m/rad] while folding")
    ap.add_argument("--kd", type=float, default=0.05, help="hinge damping [N m s/rad] (PhysX joint_damping 0.05)")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--settle_s", type=float, default=0.5)
    ap.add_argument("--fold_s", type=float, default=1.5)
    ap.add_argument("--hold_s", type=float, default=0.5)
    ap.add_argument("--release_s", type=float, default=1.5)
    ap.add_argument("--mode", choices=("finger", "drive"), default="finger",
                    help="finger: both plates free on the table, a spring at each moving plate's centroid pulls it along "
                         "the fold arc (a virtual finger; the stationary half stays down by gravity and friction). "
                         "drive: PD-drive the hinge (reacts on both plates: lifts the stationary half too)")
    ap.add_argument("--finger_k", type=float, default=600.0, help="virtual finger spring [N/m]")
    ap.add_argument("--finger_c", type=float, default=10.0, help="virtual finger damping [N s/m]")
    ap.add_argument("--palm_press", type=float, default=2.0,
                    help="virtual palm: downward force [N] on each stationary plate's centroid while folding (0: none)")
    ap.add_argument("--pin_root", action="store_true",
                    help="weld the root (stationary) plate to the world, at its rest height, so only the moving half folds")
    ap.add_argument("--out", default="outputs/newton_plates/box2-mid.npz")
    args = ap.parse_args()

    import newton
    import warp as wp

    from isaacsimenvs.tasks.cloth.backends.rigid import assets as fsa
    from isaacsimenvs.tasks.cloth.mdp.geometry import fold_shapes as fs

    design = ROOT / "isaacsimenvs/tasks/cloth/config/designs"
    path = next(design.rglob(f"{args.shape}.yaml"))
    shape = fs.load_design(path, size=args.size, thickness=args.thickness)
    usd = Path(args.out).with_suffix(".usda")
    fsa.write_shape_usd(shape, usd)

    b = newton.ModelBuilder()
    b.default_shape_cfg.gap = 0.002
    # Pinned: fixed base at the rest height a 7 mm shell gives (half thickness + margin), so the root
    # plate is where the free run settles; free: dropped from 5 cm.
    z0 = (0.5 * args.thickness + args.margin) if args.pin_root else 0.05
    b.add_usd(str(usd), xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()), floating=not args.pin_root)
    b.add_ground_plane()
    # The USD writes the hinge drive with zero gains, which Newton reads as EFFORT (no MuJoCo actuator):
    # make every revolute DOF a position+velocity actuator so the drive below can fold it.
    for j, jt in enumerate(b.joint_type):
        if int(jt) == int(newton.JointType.REVOLUTE):
            d = b.joint_qd_start[j]
            b.joint_target_mode[d] = int(newton.JointTargetMode.POSITION_VELOCITY)
            b.joint_target_ke[d] = 1e-6
            b.joint_target_kd[d] = args.kd
    model = b.finalize()
    # The contact shell on every plate shape (the ground's stays 0), as the old Newton env set it.
    margin = model.shape_margin.numpy()
    body = model.shape_body.numpy()
    margin[body >= 0] = args.margin
    model.shape_margin.assign(margin)
    print(f"[plates] shape margins {sorted(set(np.round(margin * 1e3, 3).tolist()))} mm, gaps "
          f"{sorted(set(np.round(model.shape_gap.numpy() * 1e3, 3).tolist()))} mm", flush=True)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, iterations=100, ls_iterations=15, njmax=2000, nconmax=2000)
    flags = int(newton.ModelFlags.JOINT_DOF_PROPERTIES) | int(newton.ModelFlags.ACTUATOR_PROPERTIES)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    contacts = model.collide(s0)

    # The half fold's hinges and their fold-up sign.
    labels = [str(x) for x in model.joint_label]
    qd_start = model.joint_qd_start.numpy()
    q_start = model.joint_q_start.numpy()
    tq_start = model.joint_target_q_start.numpy() if model.joint_target_q_start is not None else qd_start
    moving = set(shape.half_fold_moving())
    target_hinges = set(shape.folds[fs.HALF_FOLD][0])
    hinge_dofs = []
    for j, lab in enumerate(labels):
        name = lab.rsplit("/", 1)[-1]
        if name.startswith(fsa.JOINT_PREFIX):
            child = name[len(fsa.JOINT_PREFIX):]
            if child in target_hinges:
                hinge_dofs.append((int(qd_start[j]), int(q_start[j]), child, int(tq_start[j])))
    print(f"[plates] {args.shape} {args.size * 100:.0f} cm: {model.body_count} bodies, {model.joint_count} joints, "
          f"driving {[c for _a, _b, c, _d in hinge_dofs]} (moving half {sorted(moving)})", flush=True)
    if not hinge_dofs:
        raise SystemExit(f"no half-fold hinge among {labels}")

    # Virtual finger: every plate of the moving half is pulled at its centroid (sheet frame) toward that
    # centroid rotated about the half-fold axis by the scheduled angle, in the root plate's current frame.
    body_names = [str(x).rsplit("/", 1)[-1] for x in model.body_label]
    root_b = body_names.index(f"{fsa.BODY_PREFIX}{shape.root}")
    fold_child = sorted(target_hinges)[0]
    axis_pt, axis_dir = shape.axis(fold_child)
    axis_dir = np.asarray(axis_dir, float) / np.linalg.norm(axis_dir)
    pulled = []
    for pl in shape.plates:
        if pl.name in moving:
            c = np.asarray(pl.centroid, float)
            pulled.append((body_names.index(f"{fsa.BODY_PREFIX}{pl.name}"), np.array([c[0], c[1], shape.hinge_z])))
    com = model.body_com.numpy()
    pressed = [(body_names.index(f"{fsa.BODY_PREFIX}{pl.name}"), np.array([*np.asarray(pl.centroid, float)[:2], shape.hinge_z]))
               for pl in shape.plates if pl.name not in moving] if args.palm_press > 0 else []

    def rot(axis, ang):
        x, y, z = axis
        c, s_, C = math.cos(ang), math.sin(ang), 1 - math.cos(ang)
        return np.array([[c + x * x * C, x * y * C - z * s_, x * z * C + y * s_],
                         [y * x * C + z * s_, c + y * y * C, y * z * C - x * s_],
                         [z * x * C - y * s_, z * y * C + x * s_, c + z * z * C]])

    def xform(q, p):  # body_q row (p, quat xyzw) applied to a local point
        x, y, z, w = q[3:7]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        return R @ p + q[:3]

    prev_pts = {}
    root_ref = {}  # the root plate's pose when the fold starts: the finger's path is fixed in the world

    def finger_force(angle: float, dt_: float) -> np.ndarray:
        bq = s0.body_q.numpy()
        if "q" not in root_ref:
            root_ref["q"] = bq[root_b].copy()
        f = np.zeros((model.body_count, 6), np.float32)
        Rf = rot(axis_dir, angle)
        for bi, c_loc in pulled:
            target = xform(root_ref["q"], Rf @ (c_loc - axis_pt) + axis_pt)
            p = xform(bq[bi], c_loc)
            v = (p - prev_pts.get(bi, p)) / dt_
            prev_pts[bi] = p
            F = args.finger_k * (target - p) - args.finger_c * v
            r = p - xform(bq[bi], com[bi])
            f[bi, :3] = F
            f[bi, 3:] = np.cross(r, F)
        for bi, c_loc in pressed:
            p = xform(bq[bi], c_loc)
            F = np.array([0.0, 0.0, -args.palm_press])
            r = p - xform(bq[bi], com[bi])
            f[bi, :3] += F
            f[bi, 3:] += np.cross(r, F)
        return f

    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    for qd, _q, _c, _t in hinge_dofs:
        kd[qd] = args.kd
    model.joint_target_kd.assign(kd)
    solver.notify_model_changed(flags)

    dt = 1.0 / args.fps
    sub = dt / args.substeps
    phases = [("settle", args.settle_s), ("fold", args.fold_s), ("hold", args.hold_s), ("release", args.release_s)]
    frames, angles, phase_of = [], [], []
    t = 0.0
    for phase, dur in phases:
        n = int(round(dur * args.fps))
        for k in range(n):
            frac = (k + 1) / n
            tgt = ctrl.joint_target_q.numpy()
            sched = {"fold": math.radians(args.fold_deg) * 0.5 * (1 - math.cos(math.pi * frac)),
                     "hold": math.radians(args.fold_deg)}.get(phase)
            for qd, _q, _c, ti in hinge_dofs:
                if args.mode == "finger":
                    ke[qd], tgt[ti] = 0.0, 0.0  # passive hinge, damping only
                elif phase == "fold":
                    ke[qd], tgt[ti] = args.ke, math.radians(args.fold_deg) * 0.5 * (1 - math.cos(math.pi * frac))
                elif phase == "hold":
                    ke[qd], tgt[ti] = args.ke, math.radians(args.fold_deg)
                else:
                    ke[qd], tgt[ti] = 0.0, 0.0
            model.joint_target_ke.assign(ke)
            solver.notify_model_changed(flags)
            ctrl.joint_target_q.assign(tgt)
            for _ in range(args.substeps):
                s0.clear_forces()
                if args.mode == "finger" and sched is not None:
                    s0.body_f.assign(finger_force(sched, sub))
                contacts = model.collide(s0)
                solver.step(s0, s1, ctrl, contacts, sub)
                s0, s1 = s1, s0
            t += dt
            frames.append(s0.body_q.numpy().copy())
            q = s0.joint_q.numpy()
            angles.append([math.degrees(q[qi]) for _qd, qi, _c, _t in hinge_dofs])
            phase_of.append(phase)

    body_q = np.stack(frames).astype(np.float32)
    ang = np.array(angles)
    z = body_q[:, :, 2]
    print(f"[plates] after settle: plate z {z[int(args.settle_s * args.fps) - 1].round(4).tolist()} m, "
          f"hinge {ang[int(args.settle_s * args.fps) - 1].round(2).tolist()} deg", flush=True)
    i_hold = int((args.settle_s + args.fold_s + args.hold_s) * args.fps) - 1
    print(f"[plates] end of hold: hinge {ang[i_hold].round(1).tolist()} deg (target {args.fold_deg}); "
          f"end of release: {ang[-1].round(1).tolist()} deg", flush=True)
    print(f"[plates] max |body z| {float(np.abs(z).max()):.3f} m, finite {bool(np.isfinite(body_q).all())}", flush=True)
    rz = body_q[:, root_b, 2]
    rw = np.abs(body_q[:, root_b, 6]).clip(0, 1)
    i0 = int(args.settle_s * args.fps)
    print(f"[plates] stationary half during fold+hold+release: z {1e3 * rz[i0:].min():.1f}-{1e3 * rz[i0:].max():.1f} mm, "
          f"max tilt {float(np.degrees(2 * np.arccos(rw[i0:])).max()):.1f} deg, xy drift "
          f"{1e3 * float(np.linalg.norm(body_q[-1, root_b, :2] - body_q[i0, root_b, :2])):.1f} mm", flush=True)

    # Drawable shapes for viser_replay.py, captured from Newton's own viewer pipeline.
    from newton.viewer import ViewerNull

    class Capture(ViewerNull):
        def __init__(self):
            super().__init__(num_frames=1)
            self.meshes = {}

        def log_mesh(self, name, points, indices, *a, **k):
            p = points.numpy() if hasattr(points, "numpy") else np.asarray(points)
            ix = indices.numpy() if hasattr(indices, "numpy") else np.asarray(indices)
            self.meshes[name] = (np.asarray(p, np.float32).reshape(-1, 3), np.asarray(ix, np.int32).reshape(-1, 3))

    v = Capture()
    v.show_collision = True
    v.show_static = True
    v.set_model(model)
    body_labels = [str(x) for x in model.body_label]
    keys, sk, sp, sw, sv, sf, kind, rgb = [], [], [], [], [], [], [], []
    for batch in v._shape_instances.values():
        if int(batch.geo_type) == int(newton.GeoType.PLANE) or batch.mesh not in v.meshes:
            continue
        verts, faces = v.meshes[batch.mesh]
        for parent, xf, sc in zip(batch.parents.numpy(), batch.xforms.numpy(), batch.scales.numpy()):
            if parent < 0:
                continue
            if parent not in keys:
                keys.append(int(parent))
            name = body_labels[parent].rsplit("/", 1)[-1]
            root = name == f"{fsa.BODY_PREFIX}{shape.root}"
            mov = name[len(fsa.BODY_PREFIX):] in moving
            sk.append(keys.index(int(parent))); sp.append(xf[:3]); sw.append([xf[6], xf[3], xf[4], xf[5]])
            sv.append((verts * sc[None, :]).astype(np.float32)); sf.append(faces)
            kind.append("plate"); rgb.append((240, 150, 60) if mov else ((200, 80, 120) if root else (90, 160, 230)))
    # The table top (the ground plane, z = 0), drawn as a 60 cm square.
    h = 0.3
    sk.append(-1); sp.append([0.0, 0.0, 0.0]); sw.append([1.0, 0.0, 0.0, 0.0])
    sv.append(np.array([[-h, -h, -0.0005], [h, -h, -0.0005], [h, h, -0.0005], [-h, h, -0.0005]], np.float32))
    sf.append(np.array([[0, 1, 2], [0, 2, 3]], np.int32)); kind.append("table"); rgb.append((150, 155, 165))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, outcome=f"scripted fold ({args.mode}, finger {args.finger_k:g} N/m, palm {args.palm_press:g} N{', root pinned' if args.pin_root else ''}) to {args.fold_deg:g} deg, then release", seed=0, env=0, policy_dt=dt,
        checkpoint="(scripted)", protocol=f"newton mjwarp plates {args.shape} {args.size * 100:.0f} cm margin {args.margin * 1e3:g} mm",
        shape_key=np.array(sk), shape_kind=np.array(kind), shape_label=np.array([""] * len(sk)),
        shape_pos=np.array(sp), shape_wxyz=np.array(sw), shape_rgb=np.array(rgb),
        shape_verts=np.array(sv, dtype=object), shape_faces=np.array(sf, dtype=object),
        body_label=np.array([body_labels[k] for k in keys]), body_q=body_q[:, keys],
        cloth=np.zeros((len(body_q), 0, 3), np.float32), fold_err_mm=ang[:, 0],
        tri_moving=np.zeros((0, 3), np.int32), tri_stationary=np.zeros((0, 3), np.int32),
        env_origin=np.zeros(3), hinge_deg=ang, phase=np.array(phase_of))
    print(f"[plates] wrote {out} ({len(body_q)} frames; fold_err_mm field = hinge angle [deg])", flush=True)


if __name__ == "__main__":
    main()
