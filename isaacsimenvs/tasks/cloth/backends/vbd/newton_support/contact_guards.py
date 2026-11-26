"""Cloth-robot contact guards for Newton's SolverVBD, ported from Alloy (diagnostic, 2026-11-25).

Two class-level patches of ``newton.solvers.SolverVBD``, both off unless configured, both applied
to every VBD instance (only the cloth uses VBD here). Source: Alloy (Apache-2.0,
github.com/argonrobotics-ai/alloy), ``alloy/solvers/vbd/exit_speed.py`` and
``docs/known_gaps/coupled-record-damping.md``; re-implemented against Newton 1.5.2, not copied.

1. EXIT-SPEED CAP (``cloth.exit_speed_cap`` [m/s], 0 = off). VBD reads a particle's velocity off
   the step's displacement, v = (x - x_start) / dt. A particle that STARTS a step inside a body
   record's contact distance (the finger moved onto it) is pushed out within the step and leaves at
   ~depth / dt, whatever the stiffness. After the velocity update, for each body-particle record
   (particle, edge or face) whose contact point started inside its contact distance by d0:
   the outward motion the step resolved, s = clamp(n . (x - x_start), 0, d0), must not become
   speed past v_cap. Cut = min(s / dt, n . (v - v_body)) - v_cap (> 0), shared over an edge/face
   record's corners by w_k im_k / sum_j w_j^2 im_j, never more than a corner's own outward speed.
   A particle in several records takes the largest cut (ties: lowest record). Positions are left
   as solved; only the speed the resolution created goes. Tangential velocity untouched.
   NOTE: our robot feedback is VBD's explicit contact force (``coupling_harvest_proxy_wrenches``),
   not the particles' momentum change, so the cap does not change what the robot feels THIS step;
   it removes fast particles that would hit a link (through kd) on the next.

   ``cloth.exit_speed_measure``: record each particle's largest kick speed s / dt per step (and the
   cuts) even with the cap off, for the probe. The kick is measured whether or not it is capped.

2. MASS-BASED CONTACT GAINS (``cloth.robot_contact_gains: mass``). Per record touching a dynamic
   body (in the cloth view: the robot's proxy links) of mass m, with n records on that body this
   step and cloth step dt, safety s (``cloth.contact_gain_safety``, Alloy 0.5):
       ke = min(mean(soft_contact_ke, shape ke), s^2 m / (dt^2 n)),   kd = s m / (dt n)
   so the link meets a contact of damping ratio ~s whatever its mass. Replaces the
   ``robot_contact_kd`` value on those records. The AVBD penalty ramp is clamped to the new ke.
   m is the coupler's effective proxy mass (``body_inv_mass`` in the cloth view), isotropic
   (Alloy's known gap: ~3x the mass along a hinged tip's moving axis), so the gains err stiff.
"""

from __future__ import annotations

import warp as wp

__all__ = ["body_contact_stats", "configure", "enable_body_contact_stats", "exit_stats", "install", "prepare"]

_STATE = {"installed": False, "exit_cap": 0.0, "exit_measure": False, "gains": "material", "safety": 0.5,
          "gains_reported": False, "body_stats": False}
_BIG = 1.0e30


# ----------------------------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------------------------


@wp.func
def _body_point(body: int, body_q: wp.array[wp.transform], p_local: wp.vec3) -> wp.vec3:
    if body >= 0:
        return wp.transform_point(body_q[body], p_local)
    return p_local


@wp.kernel
def _exit_pass(
    mode: int,   # 0: kick + best cut per particle, 1: pick the winning record (lowest t), 2: write its delta
    dt: float,
    v_cap: float,
    count_arr: wp.array[int],
    max_count: int,
    corners_arr: wp.array[wp.vec3i],
    bary_arr: wp.array[wp.vec3],
    shape_arr: wp.array[int],
    body_pos_arr: wp.array[wp.vec3],
    body_vel_arr: wp.array[wp.vec3],
    normal_arr: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    has_prev: int,
    x0: wp.array[wp.vec3],
    x: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    radius: wp.array[float],
    # per-particle state (the record's cut is recomputed each pass from inputs no pass changes)
    best: wp.array[float],
    winner: wp.array[int],
    dv: wp.array[wp.vec3],
    kick_max: wp.array[float],
    capped: wp.array[int],
    cut_max: wp.array[float],
):
    t = wp.tid()
    if t >= wp.min(count_arr[0], max_count):
        return
    c = corners_arr[t]
    w = wp.vec3(1.0, 0.0, 0.0)
    if c[1] >= 0:
        w = bary_arr[t]
    shape = shape_arr[t]
    body = shape_body[shape]
    n = normal_arr[t]
    margin = float(0.0)
    if shape_margin.shape[0] > 0:
        margin = shape_margin[shape]

    xp = wp.vec3(0.0)
    xp0 = wp.vec3(0.0)
    vp = wp.vec3(0.0)
    r = float(0.0)
    den = float(0.0)
    for k in range(3):
        ck = c[k]
        if ck >= 0:
            xp += w[k] * x[ck]
            xp0 += w[k] * x0[ck]
            vp += w[k] * qd[ck]
            r = wp.max(r, radius[ck])
            den += w[k] * w[k] * inv_mass[ck]

    bx = _body_point(body, body_q, body_pos_arr[t])
    d0 = -(wp.dot(n, xp0 - bx) - r - margin)
    if d0 <= 0.0 or den <= 0.0:
        return
    s = wp.clamp(wp.dot(n, xp - xp0), 0.0, d0)
    kick = s / dt
    bv = wp.vec3(0.0)
    if body >= 0:
        if has_prev != 0:
            bv = (bx - _body_point(body, body_q_prev, body_pos_arr[t])) / dt
        bv += wp.transform_vector(body_q[body], body_vel_arr[t])
    v_rel = wp.dot(n, vp - bv)
    cut = wp.min(kick, v_rel) - v_cap
    for k in range(3):
        ck = c[k]
        if ck >= 0:
            if mode == 0:
                wp.atomic_max(kick_max, ck, kick)
            ck_cut = float(0.0)
            if cut > 0.0 and w[k] > 0.0:
                ck_cut = wp.min(cut * w[k] * inv_mass[ck] / den, wp.max(wp.dot(n, qd[ck]), 0.0))
            if ck_cut > 0.0:
                if mode == 0:
                    wp.atomic_max(best, ck, ck_cut)
                elif mode == 1:
                    if ck_cut >= best[ck]:
                        wp.atomic_min(winner, ck, t)
                else:
                    if winner[ck] == t:
                        dv[ck] = -ck_cut * n
                        capped[ck] = capped[ck] + 1
                        cut_max[ck] = wp.max(cut_max[ck], ck_cut)


@wp.kernel
def _exit_apply(qd: wp.array[wp.vec3], dv: wp.array[wp.vec3]):
    i = wp.tid()
    qd[i] = qd[i] + dv[i]


@wp.kernel
def _body_record_stats(
    count_arr: wp.array[int],
    max_count: int,
    corners_arr: wp.array[wp.vec3i],
    bary_arr: wp.array[wp.vec3],
    shape_arr: wp.array[int],
    body_pos_arr: wp.array[wp.vec3],
    normal_arr: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    body_q: wp.array[wp.transform],
    x: wp.array[wp.vec3],
    radius: wp.array[float],
    n_cur: wp.array[int],
    depth_max: wp.array[float],
):
    """Per body: soft-contact records this substep, and the deepest record's penetration into the contact
    distance (particle radius + shape margin) at the END of the substep [m]."""
    t = wp.tid()
    if t >= wp.min(count_arr[0], max_count):
        return
    shape = shape_arr[t]
    b = shape_body[shape]
    if b < 0:
        return
    wp.atomic_add(n_cur, b, 1)
    c = corners_arr[t]
    w = wp.vec3(1.0, 0.0, 0.0)
    if c[1] >= 0:
        w = bary_arr[t]
    xp = wp.vec3(0.0)
    r = float(0.0)
    for k in range(3):
        ck = c[k]
        if ck >= 0:
            xp += w[k] * x[ck]
            r = wp.max(r, radius[ck])
    margin = float(0.0)
    if shape_margin.shape[0] > 0:
        margin = shape_margin[shape]
    depth = -(wp.dot(normal_arr[t], xp - _body_point(b, body_q, body_pos_arr[t])) - r - margin)
    wp.atomic_max(depth_max, b, depth)


@wp.kernel
def _body_record_roll(n_cur: wp.array[int], n_last: wp.array[int], n_max: wp.array[int]):
    i = wp.tid()
    n_last[i] = n_cur[i]
    n_max[i] = wp.max(n_max[i], n_cur[i])
    n_cur[i] = 0


@wp.kernel
def _gains_count(
    count_arr: wp.array[int],
    max_count: int,
    shape_arr: wp.array[int],
    shape_body: wp.array[int],
    robot_shape: wp.array[int],
    body_inv_mass: wp.array[float],
    n_per_body: wp.array[int],
):
    t = wp.tid()
    if t >= wp.min(count_arr[0], max_count):
        return
    shape = shape_arr[t]
    b = shape_body[shape]
    if b >= 0 and robot_shape[shape] != 0:
        if body_inv_mass[b] > 0.0:
            wp.atomic_add(n_per_body, b, 1)


@wp.kernel
def _gains_apply(
    dt: float,
    safety: float,
    soft_ke: float,
    count_arr: wp.array[int],
    max_count: int,
    shape_arr: wp.array[int],
    shape_body: wp.array[int],
    robot_shape: wp.array[int],
    shape_ke: wp.array[float],
    body_inv_mass: wp.array[float],
    n_per_body: wp.array[int],
    material_ke: wp.array[float],
    material_kd: wp.array[float],
    penalty_k: wp.array[float],
    ke_range: wp.array[float],
):
    t = wp.tid()
    if t >= wp.min(count_arr[0], max_count):
        return
    shape = shape_arr[t]
    b = shape_body[shape]
    if b < 0 or robot_shape[shape] == 0:
        return
    im = body_inv_mass[b]
    nb = n_per_body[b]
    if im <= 0.0 or nb <= 0:
        return
    m = 1.0 / im
    fn = float(nb)
    ke_avg = 0.5 * (soft_ke + shape_ke[shape])
    ke = wp.min(ke_avg, safety * safety * m / (dt * dt * fn))
    kd = safety * m / (dt * fn)
    material_ke[t] = ke
    material_kd[t] = kd
    penalty_k[t] = wp.min(penalty_k[t], ke)
    wp.atomic_min(ke_range, 0, ke)
    wp.atomic_max(ke_range, 1, ke)
    wp.atomic_min(ke_range, 2, kd)
    wp.atomic_max(ke_range, 3, kd)


# ----------------------------------------------------------------------------------------------
# patches
# ----------------------------------------------------------------------------------------------


def configure(cloth_cfg) -> None:
    """Read the guard settings from the cloth config (call before the solver steps)."""
    _STATE["exit_cap"] = float(getattr(cloth_cfg, "exit_speed_cap", 0.0) or 0.0)
    _STATE["exit_measure"] = bool(getattr(cloth_cfg, "exit_speed_measure", False))
    gains = str(getattr(cloth_cfg, "robot_contact_gains", "material") or "material")
    if gains not in ("material", "mass"):
        raise ValueError(f"cloth.robot_contact_gains must be 'material' or 'mass', got {gains!r}")
    _STATE["gains"] = gains
    _STATE["safety"] = float(getattr(cloth_cfg, "contact_gain_safety", 0.5))
    if _STATE["exit_cap"] < 0.0:
        raise ValueError("cloth.exit_speed_cap must be >= 0 (0 = off)")
    if _STATE["exit_cap"] > 0.0 or _STATE["exit_measure"] or gains == "mass":
        print(f"[contact_guards] exit_speed_cap {_STATE['exit_cap']:g} m/s (measure {_STATE['exit_measure']}), "
              f"robot_contact_gains {gains} (safety {_STATE['safety']:g})", flush=True)


def _particle_buffers(solver):
    """Per-particle guard arrays, allocated once per solver."""
    buf = getattr(solver, "_cg_buf", None)
    n = int(solver.model.particle_count)
    if buf is None or buf["n"] != n:
        dev = solver.device
        buf = {
            "n": n,
            "best": wp.zeros(n, dtype=float, device=dev),
            "winner": wp.zeros(n, dtype=int, device=dev),
            "kick_max": wp.zeros(n, dtype=float, device=dev),   # max kick speed since last read [m/s]
            "cut_max": wp.zeros(n, dtype=float, device=dev),    # max cut since last read [m/s]
            "capped": wp.zeros(n, dtype=int, device=dev),       # cuts since last read
            "dv": wp.zeros(n, dtype=wp.vec3, device=dev),       # this step's velocity cut
        }
        solver._cg_buf = buf
    return buf


def _exit_cap(solver, state_in, state_out, contacts, dt: float) -> None:
    cap = _STATE["exit_cap"]
    if (cap <= 0.0 and not _STATE["exit_measure"]) or contacts is None or contacts.soft_contact_max <= 0:
        return
    model = solver.model
    if model.particle_count == 0:
        return
    if solver.integrate_with_external_rigid_solver:
        body_q, body_q_prev = state_out.body_q, state_in.body_q
    else:
        body_q = state_in.body_q
        body_q_prev = solver.body_q_prev if model.body_count > 0 else None
    if body_q is None:
        return
    buf = _particle_buffers(solver)
    m = int(contacts.soft_contact_max)
    buf["best"].zero_()
    buf["winner"].fill_(2**31 - 1)
    buf["dv"].zero_()
    v_cap = cap if cap > 0.0 else _BIG
    args = [float(dt), float(v_cap), contacts.soft_contact_count, m, contacts.soft_contact_indices,
            contacts.soft_contact_barycentric, contacts.soft_contact_shape, contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel, contacts.soft_contact_normal, model.shape_body, model.shape_margin,
            body_q, body_q_prev if body_q_prev is not None else body_q, int(body_q_prev is not None),
            solver.particle_q_prev, state_out.particle_q, state_out.particle_qd, model.particle_inv_mass,
            model.particle_radius, buf["best"], buf["winner"], buf["dv"], buf["kick_max"], buf["capped"],
            buf["cut_max"]]
    wp.launch(_exit_pass, dim=m, device=solver.device, inputs=[0, *args])
    if cap <= 0.0:
        return
    wp.launch(_exit_pass, dim=m, device=solver.device, inputs=[1, *args])
    wp.launch(_exit_pass, dim=m, device=solver.device, inputs=[2, *args])
    wp.launch(_exit_apply, dim=buf["n"], device=solver.device, inputs=[state_out.particle_qd, buf["dv"]])


def _mass_gains(solver, contacts, dt: float) -> None:
    if _STATE["gains"] != "mass" or contacts is None or contacts.soft_contact_max <= 0:
        return
    model = solver.model
    if model.particle_count == 0 or model.body_count == 0:
        return
    if solver.body_particle_contact_penalty_k.shape[0] < contacts.soft_contact_max:
        return
    mask, nb, rng = _gain_buffers(solver)
    nb.zero_()
    m = int(contacts.soft_contact_max)
    wp.launch(_gains_count, dim=m, device=solver.device,
              inputs=[contacts.soft_contact_count, m, contacts.soft_contact_shape, model.shape_body,
                      mask, model.body_inv_mass], outputs=[nb])
    _mass_gains_apply(solver, contacts, dt, mask, nb, rng)


def _gain_buffers(solver):
    model = solver.model
    mask = getattr(solver, "_cg_robot_shape", None)
    if mask is None:
        if wp.get_device(solver.device).is_capturing:
            raise RuntimeError("robot_contact_gains=mass: buffers must be prepared before CUDA graph capture")
        # Robot links only (the patterns robot_contact_kd uses): the table is a proxy body too, with a
        # placeholder mass of ~1e7 kg, and must keep its material gains.
        import re

        import numpy as np

        labels = [str(x) for x in model.body_label]
        sb = model.shape_body.numpy()
        pat = re.compile(r"/Robot/")
        hit = np.array([1 if (b >= 0 and pat.search(labels[int(b)])) else 0 for b in sb], dtype=np.int32)
        if not hit.any():
            raise RuntimeError("robot_contact_gains=mass: no shape on a body matching '/Robot/' in the cloth view")
        mask = wp.array(hit, dtype=int, device=solver.device)
        solver._cg_robot_shape = mask
        print(f"[contact_guards] mass gains on {int(hit.sum())} robot shapes of {hit.size}", flush=True)
    nb = getattr(solver, "_cg_n_per_body", None)
    if nb is None or nb.shape[0] != model.body_count:
        nb = wp.zeros(model.body_count, dtype=int, device=solver.device)
        solver._cg_n_per_body = nb
        solver._cg_ke_range = wp.zeros(4, dtype=float, device=solver.device)
    return mask, nb, solver._cg_ke_range


def _mass_gains_apply(solver, contacts, dt, mask, nb, rng) -> None:
    model = solver.model
    m = int(contacts.soft_contact_max)
    report = not _STATE["gains_reported"] and not wp.get_device(solver.device).is_capturing
    if report:
        rng.assign(wp.array([_BIG, -_BIG, _BIG, -_BIG], dtype=float, device=solver.device))
    wp.launch(_gains_apply, dim=m, device=solver.device,
              inputs=[float(dt), float(_STATE["safety"]), float(model.soft_contact_ke), contacts.soft_contact_count, m,
                      contacts.soft_contact_shape, model.shape_body, mask, model.shape_material_ke, model.body_inv_mass, nb],
              outputs=[solver.body_particle_contact_material_ke, solver.body_particle_contact_material_kd,
                       solver.body_particle_contact_penalty_k, rng])
    if report:
        r = rng.numpy()
        if r[1] > -_BIG:   # first step with a robot-cloth contact
            import numpy as np

            im = model.body_inv_mass.numpy()
            sb = model.shape_body.numpy()
            rb = np.unique(sb[(mask.numpy() != 0) & (sb >= 0)])
            dyn = im[rb][im[rb] > 0]
            print(f"[contact_guards] mass gains, first contact step: dt {dt * 1e6:.0f} us, proxy masses "
                  f"{1.0 / dyn.max() * 1e3:.1f}-{1.0 / dyn.min() * 1e3:.1f} g over {dyn.size} bodies; per-record "
                  f"ke {r[0]:.3g}-{r[1]:.3g} N/m, kd {r[2]:.3g}-{r[3]:.3g} N s/m", flush=True)
            _STATE["gains_reported"] = True


def prepare(solver) -> None:
    """Allocate every guard buffer for ``solver`` now (host reads included), so a CUDA graph captured
    afterwards (the live viewer) records only kernel launches. Every exit-cap buffer is per particle
    (the record count can differ between the contacts objects a step receives), so nothing depends on it."""
    if solver is None or solver.model.particle_count == 0:
        return
    if _STATE["exit_cap"] > 0.0 or _STATE["exit_measure"]:
        _particle_buffers(solver)
    if _STATE["gains"] == "mass" and solver.model.body_count > 0:
        _gain_buffers(solver)


def _install_pre_capture() -> None:
    from isaaclab_newton.physics import NewtonManager

    if getattr(NewtonManager, "_contact_guards_pre_capture", False):
        return
    original = NewtonManager.__dict__["_capture_or_defer_graph"].__func__

    def _capture(cls):
        coupled = getattr(cls, "_solver", None)
        if coupled is not None and hasattr(coupled, "solver"):
            try:
                vbd = coupled.solver("cloth")
            except KeyError:
                vbd = None
            prepare(vbd)
        return original(cls)

    NewtonManager._capture_or_defer_graph = classmethod(_capture)
    NewtonManager._contact_guards_pre_capture = True


def install() -> bool:
    """Patch ``SolverVBD`` once (class level). The settings are read at each step from :func:`configure`."""
    if _STATE["installed"]:
        return False
    from newton.solvers import SolverVBD

    init_rigid = SolverVBD._initialize_rigid_bodies
    step = SolverVBD.step

    def _initialize_rigid_bodies(self, state_in, control, contacts, dt, refresh):
        init_rigid(self, state_in, control, contacts, dt, refresh)
        _mass_gains(self, contacts, dt)

    def _step(self, state_in, state_out, control, contacts, dt):
        step(self, state_in, state_out, control, contacts, dt)
        _exit_cap(self, state_in, state_out, contacts, dt)
        _body_stats(self, state_in, state_out, contacts)

    SolverVBD._initialize_rigid_bodies = _initialize_rigid_bodies
    SolverVBD.step = _step
    _install_pre_capture()
    _STATE["installed"] = True
    return True


def enable_body_contact_stats(on: bool = True) -> None:
    """Count soft-contact records per body (and the deepest one's penetration) every VBD step, for the
    probe (:func:`body_contact_stats`). Eager mode only: the buffers are allocated on first use."""
    _STATE["body_stats"] = bool(on)


def _body_stats(solver, state_in, state_out, contacts) -> None:
    if not _STATE["body_stats"] or contacts is None or contacts.soft_contact_max <= 0:
        return
    model = solver.model
    if model.particle_count == 0 or model.body_count == 0:
        return
    body_q = state_out.body_q if solver.integrate_with_external_rigid_solver else state_in.body_q
    if body_q is None:
        return
    bs = getattr(solver, "_cg_body_stats", None)
    if bs is None or bs["n_cur"].shape[0] != model.body_count:
        nb, dev = model.body_count, solver.device
        bs = {k: wp.zeros(nb, dtype=int, device=dev) for k in ("n_cur", "n_last", "n_max")}
        bs["depth_max"] = wp.full(nb, -1.0, dtype=float, device=dev)
        solver._cg_body_stats = bs
    m = int(contacts.soft_contact_max)
    wp.launch(_body_record_stats, dim=m, device=solver.device,
              inputs=[contacts.soft_contact_count, m, contacts.soft_contact_indices, contacts.soft_contact_barycentric,
                      contacts.soft_contact_shape, contacts.soft_contact_body_pos, contacts.soft_contact_normal,
                      model.shape_body, model.shape_margin, body_q, state_out.particle_q, model.particle_radius,
                      bs["n_cur"], bs["depth_max"]])
    wp.launch(_body_record_roll, dim=model.body_count, device=solver.device,
              inputs=[bs["n_cur"], bs["n_last"], bs["n_max"]])


def body_contact_stats(solver, *, reset: bool = True):
    """Per body of the cloth view: (records at the last substep, most records in any substep, deepest
    record's penetration [m], -1 = none) since the last read, as numpy; None before the first step."""
    bs = getattr(solver, "_cg_body_stats", None)
    if bs is None:
        return None
    out = (bs["n_last"].numpy().copy(), bs["n_max"].numpy().copy(), bs["depth_max"].numpy().copy())
    if reset:
        bs["n_max"].zero_(); bs["depth_max"].fill_(-1.0)
    return out


def exit_stats(solver, *, reset: bool = True):
    """(kick_max [m/s], cut_max [m/s], capped count) per particle since the last read, as numpy; None if
    the guard never ran on this solver."""
    buf = getattr(solver, "_cg_buf", None)
    if buf is None:
        return None
    out = (buf["kick_max"].numpy().copy(), buf["cut_max"].numpy().copy(), buf["capped"].numpy().copy())
    if reset:
        buf["kick_max"].zero_(); buf["cut_max"].zero_(); buf["capped"].zero_()
    return out
