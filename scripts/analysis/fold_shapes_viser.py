"""Interactive viser view of the stage-2 fold-shape pool, simulated live in MuJoCo.

Every shape in ``fold_shapes.POOL`` is laid out on one table. Each named fold is played by
position motors on the hinges (smooth ramp, stage by stage), then the motors let go so you see
whether the fold stays. While the motors drive, the root plate is held in place (a weld to a
mocap "hand"), as a person folding holds the stationary piece down. Seams (the loop closures of
``cross4`` / ``corners8``) are MuJoCo ``connect`` constraints. Drag a shape's gizmo to pinch one
of its plates by hand instead.

    .venv_isaaclab3/bin/python scripts/analysis/fold_shapes_viser.py --port 8090
    .venv_isaaclab3/bin/python scripts/analysis/fold_shapes_viser.py --shapes cross4-mid corners8-mid

Plate-plate contact is excluded, as in ``RigidClothEnv`` (which drops every slat-slat pair);
plates touch only the table. MuJoCo CPU, not the env's MJWarp, and without its +7 mm contact margin.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _rigid_cloth_common import load_module, rigid_cloth  # noqa: E402

fs = load_module("isaacsimenvs/tasks/cloth/utils/fold_shapes.py", "fold_shapes")
rc = rigid_cloth()

ROOT_COLOR = (217, 89, 140)
PLATE_COLORS = [
    (74, 143, 194), (80, 170, 130), (242, 173, 56), (150, 110, 200), (90, 190, 200),
    (200, 110, 70), (120, 160, 60), (110, 120, 210),
]
SEAM_COLOR = (230, 120, 30)

#: Explicit pinch gains are bounded by the timestep; see ``rigid_cloth_viser.apply_pinch``.
PINCH_FMAX = 0.3


def _joint(name: str) -> str:
    return f"hinge_{name}"


def build_model(shape, args):
    """Compile ``shape`` into MuJoCo: free root, one hinge + position motor per child, seams as
    ``connect`` pairs, every plate-plate contact excluded."""
    import mujoco

    s = mujoco.MjSpec()
    s.compiler.degree = False  # MjSpec defaults to DEGREES: a +-pi range would be +-3.14 deg
    s.option.timestep = args.dt
    # Implicit in the motors' velocity gain: the smallest plate (a 2.5 g corner triangle) has a
    # hinge inertia ~5e-7 kg m^2, and an explicit kv that holds the 10 g half-plate would blow it up.
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mu = rc.cloth_effective_friction(rc.CLOTH_REFERENCE["shape_mu"]["table"])
    s.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.1],
        friction=[mu, 0.005, 0.0001],
    )
    t = shape.thickness
    bodies = {}
    for name in shape.tree_order():
        plate = shape.plate(name)
        if name == shape.root:
            body = s.worldbody.add_body(name=name, pos=[0.0, 0.0, 0.5 * t + 5e-4])
            body.add_freejoint()
        else:
            # Body frames coincide with the sheet frame at rest, so pos 0 relative to the parent.
            body = bodies[shape.hinge_of(name).parent].add_body(name=name)
            p, u = shape.axis(name)
            body.add_joint(
                name=_joint(name),
                type=mujoco.mjtJoint.mjJNT_HINGE,
                pos=p,
                axis=u,
                range=list(shape.joint_range()),
                limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                damping=args.damping,
                armature=args.armature,
            )
        verts = [(x, y, z) for z in (-0.5 * t, 0.5 * t) for x, y in plate.polygon]
        s.add_mesh(name=f"mesh_{name}", uservert=np.asarray(verts).ravel().tolist())
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"mesh_{name}",
            mass=args.density * plate.area,
            friction=[mu, 0.005, 0.0001],
        )
        bodies[name] = body

    names = [p.name for p in shape.plates]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            s.add_exclude(bodyname1=a, bodyname2=b)
    for seam in shape.seams:
        # Two connects on the seam line are a hinge: each pins a point, together they fix the axis.
        for q in (seam.a, seam.b):
            eq = s.add_equality(
                type=mujoco.mjtEq.mjEQ_CONNECT,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1=seam.plate1,
                name2=seam.plate2,
            )
            eq.data[0:3] = [q[0], q[1], shape.hinge_z]
    # The motors' torques are INTERNAL, so on their own they lift the root as readily as the flap:
    # box2's equal halves stood up on the crease like an open book and fell open on release. A
    # person folding holds the stationary piece down; this weld (to a mocap "hand", moved to the
    # root's live pose when switched on) is that hold, active only while the motors drive.
    hand = s.worldbody.add_body(name="hold", mocap=True, pos=[0.0, 0.0, 0.5 * t + 5e-4])
    weld = s.add_equality(
        name="hold",
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1=shape.root,
        name2=hand.name,
    )
    weld.active = False
    for h in shape.hinges:
        act = s.add_actuator(
            name=f"motor_{h.child}", trntype=mujoco.mjtTrn.mjTRN_JOINT, target=_joint(h.child)
        )
        act.set_to_position(kp=args.kp, dampratio=1.0)
    model = s.compile()
    return model, mujoco.MjData(model)


def _grasp_point(shape, plate: str) -> tuple[float, float]:
    """The plate vertex farthest from its hinge line, pulled 6 mm inside -- where a pinch goes."""
    pl = shape.plate(plate)
    c = pl.centroid
    if plate == shape.root:
        v = max(pl.polygon, key=lambda q: np.hypot(*(np.asarray(q))))
    else:
        h = shape.hinge_of(plate)
        a, b = np.asarray(h.a), np.asarray(h.b)
        d = (b - a) / np.linalg.norm(b - a)

        def dist(q):
            r = np.asarray(q) - a
            return abs(r[0] * d[1] - r[1] * d[0])

        v = max(pl.polygon, key=dist)
    v = np.asarray(v)
    w = c - v
    return tuple(v + 0.006 * np.sqrt(2.0) * w / np.linalg.norm(w))


@dataclass
class Sim:
    shape: object
    model: object
    data: object
    origin: np.ndarray
    body: dict
    qadr: dict
    act: dict
    gain0: np.ndarray
    bias0: np.ndarray
    cmd: dict = field(default_factory=dict)
    queue: list = field(default_factory=list)  # [(stage dict, record?)]
    current: tuple | None = None  # (t0, start, end, record)
    history: list = field(default_factory=list)  # start angles of each executed stage
    motors: bool = False
    release_at: float | None = None
    t: float = 0.0
    pinch: bool = False
    grasp: tuple = None  # (plate, local xyz)
    target: np.ndarray = None
    frames: dict = field(default_factory=dict)
    gizmo: object = None
    marker: object = None
    starts: dict = field(default_factory=dict)  # FoldShape.start_poses()
    start: str = ""  # readout of the last start pose put on

    def angle(self, child: str) -> float:
        return float(self.data.qpos[self.qadr[child]])

    def world(self, plate: str, local) -> np.ndarray:
        b = self.body[plate]
        return self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ np.asarray(local)

    def set_motors(self, on: bool) -> None:
        """Hinge motors on/off, with the root held in place while they drive."""
        import mujoco

        m, d = self.model, self.data
        if on and not self.motors:
            self.cmd = {k: self.angle(k) for k in self.act}
            root = self.body[self.shape.root]
            d.mocap_pos[0] = d.xpos[root]
            d.mocap_quat[0] = d.xquat[root]
        for a in self.act.values():
            m.actuator_gainprm[a] = self.gain0[a] if on else 0.0
            m.actuator_biasprm[a] = self.bias0[a] if on else 0.0
        d.eq_active[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "hold")] = on
        self.motors = on

    def seam_error(self) -> float:
        z = self.shape.hinge_z
        worst = 0.0
        for s in self.shape.seams:
            for q in (s.a, s.b):
                x = (q[0], q[1], z)
                worst = max(
                    worst, float(np.linalg.norm(self.world(s.plate1, x) - self.world(s.plate2, x)))
                )
        return worst


def make_sim(shape, args, origin) -> Sim:
    import mujoco

    model, data = build_model(shape, args)
    body = {
        p.name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p.name) for p in shape.plates
    }
    qadr, act = {}, {}
    for h in shape.hinges:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _joint(h.child))
        qadr[h.child] = model.jnt_qposadr[jid]
        act[h.child] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{h.child}")
    sim = Sim(
        shape=shape,
        model=model,
        data=data,
        origin=np.asarray(origin),
        body=body,
        qadr=qadr,
        act=act,
        gain0=model.actuator_gainprm.copy(),
        bias0=model.actuator_biasprm.copy(),
    )
    plate, xy = shape.grasp or (shape.plates[-1].name, _grasp_point(shape, shape.plates[-1].name))
    sim.grasp = (plate, np.array([xy[0], xy[1], 0.0]))
    sim.starts = shape.start_poses()
    reset(sim)
    return sim


def reset(sim: Sim) -> None:
    import mujoco

    mujoco.mj_resetData(sim.model, sim.data)
    sim.set_motors(False)
    sim.queue, sim.current, sim.history = [], None, []
    sim.release_at, sim.pinch, sim.t, sim.start = None, False, 0.0, ""
    sim.data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    sim.target = sim.world(*sim.grasp)


def put_start(sim: Sim, name: str, degs) -> None:
    """Teleport to start pose ``name`` with part ``i`` at ``degs[i]``, as a training reset writes it
    (``fold_pool.init_fold_starts: other``), then hold it with the motors (root held)."""
    import mujoco

    reset(sim)
    parts = sim.starts[name]
    for part, d in zip(parts, degs):
        for h in part:
            sim.data.qpos[sim.qadr[h]] = math.radians(d)
    mujoco.mj_forward(sim.model, sim.data)
    sim.set_motors(True)
    sim.target = sim.world(*sim.grasp)
    angles = ", ".join(f"{d:.0f}°" for d in degs[: len(parts)])
    sim.start = f"start **{name}** ({angles})"


def random_start(sim: Sim, lo: float, hi: float, name: str | None = None) -> None:
    """A start drawn as training draws it: a uniform pose, each part its own uniform angle."""
    if not sim.starts:
        reset(sim)
        sim.start = "start **flat** (its only fold is the target)"
        return
    name = name or list(sim.starts)[np.random.randint(len(sim.starts))]
    put_start(sim, name, np.random.uniform(lo, hi, size=len(sim.starts[name])))


def request(sim: Sim, stages, record: bool = True) -> None:
    sim.pinch = False
    sim.set_motors(True)
    sim.release_at = None
    sim.queue.extend((dict(st), record) for st in stages)


def unfold(sim: Sim) -> None:
    stages = list(reversed(sim.history))
    sim.history = []
    if not stages:
        stages = [{k: 0.0 for k in sim.act}]
    request(sim, stages, record=False)


def advance(sim: Sim, duration: float, hold: float, auto_release: bool) -> None:
    """Drive the motors one physics step along the current stage."""
    if sim.current is None and sim.queue:
        stage, record = sim.queue.pop(0)
        start = {k: sim.cmd.get(k, sim.angle(k)) for k in stage}
        if record:
            sim.history.append(start)
        sim.current = (sim.t, start, stage)
    if sim.current is not None:
        t0, start, end = sim.current
        s = min(max((sim.t - t0) / duration, 0.0), 1.0)
        w = s * s * (3.0 - 2.0 * s)
        for k in end:
            sim.cmd[k] = start[k] + w * (end[k] - start[k])
        if s >= 1.0:
            sim.current = None
            if not sim.queue:
                sim.release_at = sim.t + hold
    if sim.motors:
        for k, a in sim.act.items():
            sim.data.ctrl[a] = sim.cmd.get(k, 0.0)
        if auto_release and sim.release_at is not None and sim.t >= sim.release_at:
            sim.set_motors(False)
            sim.release_at = None


def apply_pinch(sim: Sim, k: float) -> None:
    import mujoco

    d = sim.data
    d.xfrc_applied[:] = 0.0
    if not sim.pinch:
        return
    plate, local = sim.grasp
    b = sim.body[plate]
    p = sim.world(plate, local)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(sim.model, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
    vp = vel[3:] + np.cross(vel[:3], p - d.xpos[b])
    # Damping scaled to the plate's mass so c*dt/m_eff stays ~0.3 for a 2.5 g triangle and a 10 g
    # half-sheet alike.
    c = 0.025 * sim.model.body_mass[b] / sim.model.opt.timestep
    f = k * (sim.target - p) - c * vp
    n = np.linalg.norm(f)
    if n > PINCH_FMAX:
        f *= PINCH_FMAX / n
    d.xfrc_applied[b, :3] = f
    d.xfrc_applied[b, 3:] = np.cross(p - d.xipos[b], f)


def readout(sim: Sim) -> str:
    sh = sim.shape
    ang = ", ".join(f"{k} {math.degrees(sim.angle(k)):.0f}°" for k in sim.act)
    state = "motors, root held" if sim.motors else ("pinched" if sim.pinch else "passive")
    seam = f", seam gap {sim.seam_error() * 1e3:.1f} mm" if sh.seams else ""
    start = f"<br>{sim.start}" if sim.start else ""
    return f"**{sh.name}** ({state}) — {ang}{seam}{start}"


def make_scene(server, sim: Sim) -> None:
    sh = sim.shape
    root = f"/{sh.name}"
    server.scene.add_frame(root, show_axes=False, position=tuple(sim.origin))
    L = sh.size / 2.0
    server.scene.add_label(f"{root}/label", sh.name, position=(-L, L + 0.012, 0.0))
    t = sh.thickness
    for i, p in enumerate(sh.plates):
        fr = server.scene.add_frame(f"{root}/{p.name}", show_axes=False)
        n = len(p.polygon)
        verts = np.array([(x, y, z) for z in (-0.5 * t, 0.5 * t) for x, y in p.polygon])
        faces = []
        for k in range(1, n - 1):
            faces.append((0, k + 1, k))  # bottom, facing -z
            faces.append((n, n + k, n + k + 1))  # top, facing +z
        for k in range(n):
            j = (k + 1) % n
            faces += [(k, j, n + j), (k, n + j, n + k)]
        color = ROOT_COLOR if p.name == sh.root else PLATE_COLORS[(i - 1) % len(PLATE_COLORS)]
        server.scene.add_mesh_simple(
            f"{root}/{p.name}/geom",
            vertices=verts,
            faces=np.array(faces),
            color=color,
            flat_shading=True,
            side="double",
        )
        sim.frames[p.name] = fr
    z = sh.hinge_z + 1e-4
    for h in sh.hinges:
        server.scene.add_line_segments(
            f"{root}/{h.child}/hinge",
            points=np.array([[[*h.a, z], [*h.b, z]]]),
            colors=(30, 30, 30),
            thickness=0.0012,
        )
    for k, s in enumerate(sh.seams):
        server.scene.add_line_segments(
            f"{root}/{s.plate1}/seam{k}",
            points=np.array([[[*s.a, z], [*s.b, z]]]),
            colors=SEAM_COLOR,
            thickness=0.0015,
        )
    sim.marker = server.scene.add_icosphere(f"{root}/grasp", radius=0.0025, color=(40, 180, 90))
    sim.gizmo = server.scene.add_transform_controls(
        f"{root}/gizmo", scale=0.035, disable_rotations=True, position=tuple(sim.target)
    )


def sync_scene(sim: Sim) -> None:
    d = sim.data
    for name, fr in sim.frames.items():
        b = sim.body[name]
        fr.position = tuple(d.xpos[b])
        fr.wxyz = tuple(d.xquat[b])
    p = sim.world(*sim.grasp)
    sim.marker.position = tuple(p)
    sim.marker.color = (230, 70, 50) if sim.pinch else (40, 180, 90)
    if not sim.pinch:
        sim.gizmo.position = tuple(p)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--shapes", nargs="+", default=list(fs.POOL))
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--density", type=float, default=2.0, help="kg/m^2, as ClothCfg.density")
    p.add_argument("--damping", type=float, default=1.0e-5, help="passive hinge damping")
    p.add_argument("--armature", type=float, default=1.0e-7)
    p.add_argument("--kp", type=float, default=0.1, help="fold motor gain [N m/rad]")
    p.add_argument("--dt", type=float, default=1.0e-3)
    p.add_argument("--cols", type=int, default=3)
    args = p.parse_args()

    import mujoco
    import viser

    spacing = 0.16
    sims = []
    for k, name in enumerate(args.shapes):
        r, c = divmod(k, args.cols)
        origin = ((c - (min(len(args.shapes), args.cols) - 1) / 2.0) * spacing, -r * spacing, 0.0)
        sims.append(make_sim(fs.shape(name), args, origin))
    rows = math.ceil(len(args.shapes) / args.cols)
    centre_y = -(rows - 1) * spacing / 2.0
    width = min(len(args.shapes), args.cols) * spacing

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/ground",
        width=1.0,
        height=1.0,
        cell_size=0.01,
        section_size=0.05,
        plane_color=(235, 235, 235),
        plane_opacity=0.6,
        position=(0.0, centre_y, -1e-4),
    )
    lock = threading.Lock()
    for sim in sims:
        make_scene(server, sim)

    @server.on_client_connect
    def _(client) -> None:
        client.camera.up_direction = (0.0, 0.0, 1.0)
        # Back off in proportion to the layout so every shape is in frame on connect.
        d = max(width, rows * spacing)
        client.camera.position = (0.0, centre_y - 0.9 * d, 0.75 * d)
        client.camera.look_at = (0.0, centre_y, 0.0)

    gui = server.gui
    gui.add_markdown(
        "Each shape's folder plays its folds with hinge **motors**, which let go afterwards. "
        "Drag a **gizmo** to pinch a plate by hand. Pink = root plate, black = hinge, "
        "orange = seam (loop closure)."
    )
    physics = gui.add_checkbox("physics", True)
    rtf = gui.add_slider("speed (x real time)", 0.05, 1.0, 0.05, 0.5)
    duration = gui.add_slider("fold stage duration (s)", 0.3, 4.0, 0.1, 1.2)
    hold = gui.add_slider("hold before release (s)", 0.0, 3.0, 0.1, 0.5)
    auto_release = gui.add_checkbox("release motors after fold", True)
    k_pinch = gui.add_slider("pinch stiffness (N/m)", 1.0, 30.0, 1.0, 8.0)
    with gui.add_folder("All shapes"):
        b_all = gui.add_button("fold each (its first fold)", color="blue")
        b_unfold_all = gui.add_button("unfold all")
        b_reset_all = gui.add_button("reset all flat")

    def on(fn):
        def cb(_):
            with lock:
                physics.value = True
                fn()

        return cb

    with gui.add_folder("Folded starts (training)", expand_by_default=True):
        gui.add_markdown(
            "`fold_pool.init_fold_starts: other`: any legal combination of a shape's NON-target folds, "
            "each part at its own angle, up to fully folded. Held by the motors unless *release* is "
            "ticked. " + ", ".join(f"{s.shape.name} {len(s.starts)}" for s in sims) + " starts."
        )
        g_range = gui.add_multi_slider("start angle range (deg)", 0.0, 180.0, 1.0, (15.0, 180.0))
        b_rand_all = gui.add_button("random start, every shape", color="green")
        g_rel = gui.add_checkbox("release a start after the hold (passive hinges)", False)
        g_cycle = gui.add_checkbox("new random start every ...", False)
        g_period = gui.add_slider("... seconds", 0.5, 6.0, 0.5, 2.0)

    def start(s, name=None):
        random_start(s, *g_range.value, name=name)
        if g_rel.value:
            s.release_at = s.t + hold.value

    def rand_all():
        for s in sims:
            start(s)

    b_rand_all.on_click(on(rand_all))

    b_all.on_click(on(lambda: [request(s, next(iter(s.shape.folds.values()))) for s in sims]))
    b_unfold_all.on_click(on(lambda: [unfold(s) for s in sims]))
    b_reset_all.on_click(on(lambda: [reset(s) for s in sims]))

    for sim in sims:
        sh = sim.shape
        with gui.add_folder(sh.name, expand_by_default=False):
            gui.add_markdown(sh.describe())
            for fold_name, stages in sh.folds.items():
                gui.add_button(fold_name).on_click(on(lambda s=sim, st=stages: request(s, st)))
            gui.add_button("unfold").on_click(on(lambda s=sim: unfold(s)))
            if sim.starts:
                g_sp = gui.add_dropdown("start pose", list(sim.starts))

                gui.add_button("show this start (random angles)").on_click(
                    on(lambda s=sim, g=g_sp: start(s, g.value)))
                gui.add_button("random start").on_click(on(lambda s=sim: start(s)))
            else:
                gui.add_markdown("*No folded start: its only fold is the target.*")
            with gui.add_folder("hinge angles (deg)", expand_by_default=False):
                gui.add_markdown(
                    "Each slider drives one hinge's motor (root held) and holds it there. "
                    "A hinge on a loop can only turn with its partners; alone it fights the seam."
                )
                for h in sh.hinges:
                    sl = gui.add_slider(f"{h.child} on {h.parent}", -180.0, 180.0, 1.0, 0.0)

                    def turn(_, s=sim, k=h.child, sl=sl):
                        with lock:
                            s.queue, s.current, s.release_at, s.pinch = [], None, None, False
                            s.set_motors(True)
                            s.cmd[k] = math.radians(sl.value)
                            physics.value = True

                    sl.on_update(turn)
            gui.add_button("reset flat").on_click(on(lambda s=sim: reset(s)))
            grasp = gui.add_dropdown(
                "pinch plate", [pl.name for pl in sh.plates], initial_value=sim.grasp[0]
            )
            gui.add_button("release pinch").on_click(on(lambda s=sim: setattr(s, "pinch", False)))

            def pick(_, s=sim, g=grasp):
                with lock:
                    xy = (
                        s.shape.grasp[1]
                        if s.shape.grasp and s.shape.grasp[0] == g.value
                        else _grasp_point(s.shape, g.value)
                    )
                    s.grasp = (g.value, np.array([xy[0], xy[1], 0.0]))
                    s.pinch = False

            grasp.on_update(pick)

        def drag(_, s=sim):
            with lock:
                s.target = np.array(s.gizmo.position)
                s.set_motors(False)
                s.queue, s.current = [], None
                s.pinch = True
                physics.value = True

        sim.gizmo.on_update(drag)

    info = gui.add_markdown("")
    frame_dt = 1.0 / 60.0
    last_info = 0.0
    last_cycle = time.time()
    print(f"viser running at http://localhost:{args.port}  ({len(sims)} shapes)", flush=True)
    while True:
        t_start = time.time()
        with lock:
            if physics.value:
                steps = max(1, round(rtf.value * frame_dt / args.dt))
                for sim in sims:
                    for _ in range(steps):
                        advance(sim, duration.value, hold.value, auto_release.value)
                        apply_pinch(sim, k_pinch.value)
                        mujoco.mj_step(sim.model, sim.data)
                        sim.t += args.dt
            with server.atomic():
                for sim in sims:
                    sync_scene(sim)
            if g_cycle.value and time.time() - last_cycle > g_period.value:
                physics.value = True
                rand_all()
                last_cycle = time.time()
            if time.time() - last_info > 0.25:
                info.content = "\n\n".join(readout(s) for s in sims)
                last_info = time.time()
        time.sleep(max(0.0, frame_dt - (time.time() - t_start)))


if __name__ == "__main__":
    main()
