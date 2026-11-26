"""Experiment (``VBD_FLEX=1``): the VBD world's sheet simulated as a MuJoCo flex shell, inside the
robot's own MJWarp solve, instead of by VBD through the proxy coupler.

The scene, the sheet's Newton particles and everything the task reads stay as they are. Three hooks:

* **compile** -- when Newton's ``SolverMuJoCo`` compiles its ``MjSpec``, a 2D flex is added with one
  body per sheet particle (three slide joints each, in the particles' order), the sheet's triangles
  as elements, and a shell material (membrane + bending) derived from the VBD material. MJWarp then
  replicates it into every world, next to that world's robot and table;
* **step** -- ``SolverCoupledProxy.step`` steps only the robot entry (MJWarp, now carrying the sheet)
  and copies every flex vertex's position and velocity into the sheet's Newton particles. VBD and the
  proxy exchange are not stepped: robot and sheet meet in one implicit contact solve, so there is no
  lagged feedback;
* **reset** -- ``write_sheet`` also writes the flex vertices' slide-joint positions in MJWarp.

Needs ``newton.use_mujoco_contacts: true`` (flex contacts exist only in MuJoCo's collision).

Material (env vars, defaults matched to vbd.yaml's sheet): ``FLEX_POISSON`` (0.25, from tri_ka / tri_ke);
``FLEX_THICK`` [m] (3.5 mm), the shell thickness that, with ``young`` set so the membrane stiffness
``young * thick`` equals ``2 tri_ke (1 + poisson)``, gives a bending stiffness ``young thick^3 /
(12 (1 - poisson^2))`` close to VBD's per-hinge ``edge_ke`` on this grid; ``FLEX_YOUNG`` overrides the
derived value. ``FLEX_DAMPING`` (1e-4 s, Rayleigh), ``FLEX_EDGE_DAMPING`` (1e-3), ``FLEX_SOLREF``
(0.005 s contact time constant), ``FLEX_SELF`` (``auto`` / ``none`` / ``narrow`` / ``bvh`` / ``sap``).
Friction: the flex has priority over the hand (``cloth_fingertip_friction`` on every hand contact) and
the table geoms priority over the flex (``cloth_table_friction`` on the table).
"""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np

__all__ = ["enabled", "FlexSheet"]

_VERT = "clothflex_v"
_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def enabled() -> bool:
    return os.environ.get("VBD_FLEX", "0") == "1"


def _env(name, default):
    v = os.environ.get(name)
    return type(default)(v) if v not in (None, "") else default


def _lumped_masses(verts: np.ndarray, tris: np.ndarray, density: float) -> np.ndarray:
    a, b, c = (verts[tris[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    m = np.zeros(len(verts))
    for k in range(3):
        np.add.at(m, tris[:, k], density * area / 3.0)
    return m


def add_sheet_flex(spec, verts_w: np.ndarray, tris: np.ndarray, *, density: float, radius: float, young: float,
                   poisson: float, thickness: float, damping: float, edge_damping: float, friction: float,
                   solref: float, selfcollide: str, contype: int, conaffinity: int, armature: float = 0.0,
                   prefix: str = _VERT, flex_name: str = "clothflex"):
    """Add the sheet to ``spec``: one body per vertex at ``verts_w`` (three slide joints) and a dim-2 flex."""
    import mujoco

    masses = _lumped_masses(verts_w, tris, density)
    names = []
    for i, (p, m) in enumerate(zip(verts_w, masses)):
        body = spec.worldbody.add_body(name=f"{prefix}{i}", pos=[float(x) for x in p])
        body.mass = float(m)
        body.inertia = [1e-9, 1e-9, 1e-9]
        body.ipos = [0.0, 0.0, 0.0]  # at the vertex: else the compiler puts it off-body, and MJWarp freezes the sheet
        body.iquat = [1.0, 0.0, 0.0, 0.0]
        body.explicitinertial = True
        for k, axis in enumerate(_AXES):
            j = body.add_joint(name=f"{prefix}{i}_{'xyz'[k]}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=list(axis))
            j.armature = float(armature)
        names.append(body.name)
    flex = spec.add_flex()
    flex.name = flex_name
    flex.dim = 2
    flex.radius = float(radius)
    flex.vertbody = names
    flex.elem = [int(x) for x in tris.reshape(-1)]
    flex.young = float(young)
    flex.poisson = float(poisson)
    flex.thickness = float(thickness)
    flex.damping = float(damping)
    flex.edgedamping = float(edge_damping)
    flex.elastic2d = 3  # both: membrane and bending
    flex.selfcollide = getattr(mujoco.mjtFlexSelf, f"mjFLEXSELF_{selfcollide.upper()}")
    flex.contype, flex.conaffinity = int(contype), int(conaffinity)
    flex.condim = 3
    flex.priority = 1
    flex.friction = [float(friction), 0.005, 0.0001]
    flex.solref = [float(solref), 1.0]
    return names


def allow_cpu_entry() -> None:
    """Isaac Lab's coupler refuses ``use_mujoco_cpu`` in a coupled entry: MuJoCo-C has one global MjData, so a
    per-world reset cannot reach it directly. With ``update_data_interval = 1`` every step re-syncs Newton's
    (per-world reset) joint state into it, which is all a reset needs; the coupler's check is bypassed by
    hiding the flag while it validates."""
    from isaaclab_contrib.coupling import coupler as cp

    for cls in vars(cp).values():
        fn = getattr(cls, "__dict__", {}).get("_validate_config") if isinstance(cls, type) else None
        if fn is None or getattr(cls, "_flex_cpu_ok", False):
            continue
        original = fn.__func__ if isinstance(fn, classmethod) else fn

        def validate(cls_or_self, *a, _orig=original, **kw):
            cfg = a[0] if a else None
            hidden = []
            for entry in getattr(cfg, "entries", None) or getattr(cls_or_self, "entries", None) or []:
                sc = getattr(entry, "solver_cfg", None)
                if getattr(sc, "use_mujoco_cpu", False):
                    sc.use_mujoco_cpu = False
                    hidden.append(sc)
            try:
                return _orig(cls_or_self, *a, **kw)
            finally:
                for sc in hidden:
                    sc.use_mujoco_cpu = True

        setattr(cls, "_validate_config", classmethod(validate) if isinstance(fn, classmethod) else validate)
        cls._flex_cpu_ok = True


class FlexSheet:
    """The hooks for one ``VbdBackend``."""

    def __init__(self, backend) -> None:
        self.b = backend
        self.qadr = self.vadr = None  # (P,) qpos / qvel address of each vertex's x slide
        self.pos0 = None  # (P, 3) the vertex bodies' rest positions in the template world
        self.pidx = None  # (N, P) global Newton particle index of env n's vertex p
        self._injected = False
        self.cpu = os.environ.get("FLEX_CPU", "0") == "1"

    # ------------------------------------------------------------------ material

    def params(self) -> dict:
        c = self.b.cfg.cloth
        poisson = _env("FLEX_POISSON", 0.25)
        thick = _env("FLEX_THICK", 0.0035)
        young = _env("FLEX_YOUNG", 2.0 * float(c.tri_ke) * (1.0 + poisson) / thick)
        return dict(density=float(c.density), radius=float(c.particle_radius()), young=young, poisson=poisson,
                    thickness=thick, damping=_env("FLEX_DAMPING", 1e-4), edge_damping=_env("FLEX_EDGE_DAMPING", 1e-3),
                    friction=float(c.cloth_fingertip_friction), solref=_env("FLEX_SOLREF", 0.005),
                    selfcollide=_env("FLEX_SELF", "auto"), armature=_env("FLEX_ARMATURE", 0.0))

    # ------------------------------------------------------------------ compile

    @contextmanager
    def injection(self):
        import mujoco

        original = mujoco.MjSpec.compile
        sheet = self

        def compile_with_sheet(spec, *a, **kw):
            if sheet._injected or not any("Robot" in (b.name or "") or "link" in (b.name or "") for b in spec.bodies):
                return original(spec, *a, **kw)
            sheet._inject(spec)
            model = original(spec, *a, **kw)
            sheet._addresses(model)
            return model

        with ExitStack() as stack:
            stack.enter_context(patch.object(mujoco.MjSpec, "compile", compile_with_sheet))
            if True:
                # SolverMuJoCo's constructor syncs into a CPU MjData (on BOTH paths: an uninitialised buffer, so
                # garbage in the flex joints) and runs mj_forward before bind() -- keep the flex joints from the start.
                from newton._src.solvers.mujoco.solver_mujoco import SolverMuJoCo

                upd = SolverMuJoCo._update_mjc_data

                def update_keeping_flex(self_, mj_data, model, state, *a, **kw):
                    q3 = getattr(sheet, "_q3", None)
                    if q3 is None or not isinstance(mj_data, mujoco.MjData) or len(mj_data.qpos) <= int(q3.max()):
                        return upd(self_, mj_data, model, state, *a, **kw)
                    q, v = mj_data.qpos[q3].copy(), mj_data.qvel[sheet._v3].copy()
                    upd(self_, mj_data, model, state, *a, **kw)
                    mj_data.qpos[q3], mj_data.qvel[sheet._v3] = q, v

                stack.enter_context(patch.object(SolverMuJoCo, "_update_mjc_data", update_keeping_flex))
            yield

    def _inject(self, spec) -> None:
        from isaacsimenvs.tasks.cloth.mdp.geometry.grid import grid_mesh

        c = self.b.cfg.cloth
        verts, tris = grid_mesh(c.size, c.resolution, c.mesh_diagonals)
        verts = np.asarray(verts, dtype=float).copy()
        tris = np.asarray(tris, dtype=int).reshape(-1, 3)
        verts[:, 2] += float(self.b.cfg.reset.table_reset_z) + float(c.start_height)
        contype = conaffinity = 0
        for g in spec.geoms:
            contype |= int(g.contype)
            conaffinity |= int(g.conaffinity)
        p = self.params()
        # The table's friction on the sheet: its geoms outrank the flex.
        n_table = 0
        for g in spec.geoms:
            body = g.parent
            if "Table" in (getattr(body, "name", "") or ""):
                g.priority = 2
                g.friction = [float(c.cloth_table_friction), 0.005, 0.0001]
                g.condim = 3
                n_table += 1
        if os.environ.get("FLEX_NOCOLLIDE") == "1":  # diagnostic: the sheet touches nothing
            contype = conaffinity = -1
        ct, ca = (max(contype, 0) if contype >= 0 else 0), (max(conaffinity, 0) if conaffinity >= 0 else 0)
        if self.cpu:
            # MuJoCo-C: Newton puts every env in ONE model, so one sheet per env, each at its env's origin.
            origins = self.b.scene.env_origins.detach().cpu().numpy().astype(float)
            self.pos0_env = np.stack([verts + o for o in origins])  # (N, P, 3)
            for k, o in enumerate(origins):
                add_sheet_flex(spec, self.pos0_env[k], tris, contype=ct, conaffinity=ca,
                               prefix=f"{_VERT}e{k}_", flex_name=f"clothflex{k}", **p)
        else:
            add_sheet_flex(spec, verts, tris, contype=ct, conaffinity=ca, **p)
        self.pos0 = verts
        self._injected = True
        print(f"[flex] sheet as a MuJoCo flex: {len(verts)} vertices, {len(tris)} triangles, {n_table} table geoms; "
              + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in p.items())
              + f"; bending D {p['young'] * p['thickness'] ** 3 / (12 * (1 - p['poisson'] ** 2)):.3g} N m, "
              f"contype {contype} conaffinity {conaffinity}", flush=True)

    def _addresses(self, model) -> None:
        import mujoco

        if self.cpu:
            N, P = self.pos0_env.shape[:2]
            q = np.zeros((N, P), np.int64); v = np.zeros((N, P), np.int64)
            for k in range(N):
                for i in range(P):
                    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{_VERT}e{k}_{i}_x")
                    q[k, i], v[k, i] = model.jnt_qposadr[j], model.jnt_dofadr[j]
            self.qadr_env, self.vadr_env = q, v
            self.qadr, self.vadr = q[0].astype(np.int32), v[0].astype(np.int32)
            self._q3 = (q[..., None] + np.arange(3)).reshape(-1)  # every flex qpos address, env-major
            self._v3 = (v[..., None] + np.arange(3)).reshape(-1)
            print(f"[flex] compiled (CPU, one model): nq {model.nq} nv {model.nv} nflex {model.nflex} flexvert {model.nflexvert}", flush=True)
            return

        P = len(self.pos0)
        q, v = np.zeros(P, np.int32), np.zeros(P, np.int32)
        for i in range(P):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{_VERT}{i}_x")
            q[i], v[i] = model.jnt_qposadr[j], model.jnt_dofadr[j]
            for k in (1, 2):  # x, y, z are consecutive
                jk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{_VERT}{i}_{'xyz'[k]}")
                assert model.jnt_qposadr[jk] == q[i] + k and model.jnt_dofadr[jk] == v[i] + k
        self.qadr, self.vadr = q, v
        self._q3 = (q[:, None] + np.arange(3)).reshape(-1)
        self._v3 = (v[:, None] + np.arange(3)).reshape(-1)
        print(f"[flex] compiled: nq {model.nq} nv {model.nv} nflex {model.nflex} flexvert {model.nflexvert}", flush=True)

    # ------------------------------------------------------------------ step

    def bind(self, model, coupled) -> None:
        """After construction (before graph capture): the index maps, on the device."""
        import warp as wp

        if self.pidx is not None or self.qadr is None:
            return
        pw = model.particle_world.numpy()
        P = len(self.pos0)
        n = int(self.b.num_envs)
        pidx = np.zeros((n, P), np.int32)
        for w in range(n):
            ids = np.nonzero(pw == w)[0]
            assert len(ids) == P, f"world {w}: {len(ids)} particles, flex has {P}"
            pidx[w] = ids
        dev = model.device
        self.pidx = wp.array(pidx, dtype=wp.int32, device=dev)
        self._qadr = wp.array(self.qadr, dtype=wp.int32, device=dev)
        self._vadr = wp.array(self.vadr, dtype=wp.int32, device=dev)
        self._pos0 = wp.array(self.pos0.astype(np.float32), dtype=wp.vec3, device=dev)
        self.rigid = next(e for e in coupled._entries.values() if hasattr(e.solver, "mj_model"))
        if self.cpu:
            self._bind_cpu(pidx)
        print(f"[flex] bound: {n} worlds x {P} vertices -> particles; robot entry {self.rigid.name!r}", flush=True)

    def _bind_cpu(self, pidx) -> None:
        """MuJoCo-C path: Newton rebuilds the WHOLE qpos / qvel (``_update_mjc_data``, from an uninitialised
        buffer) and qfrc_applied (``_apply_mjc_control``, from zeros) every step, writing only its own joints.
        Keep the flex joints' state across the first and add the armature weight after the second."""
        solver = self.rigid.solver
        assert solver.use_mujoco_cpu, "FLEX_CPU=1 but the robot entry is on MuJoCo Warp"
        sheet = self
        upd, ctl = solver._update_mjc_data, solver._apply_mjc_control

        def update_keeping_flex(mj_data, model, state, *a, **kw):
            q, v = mj_data.qpos[sheet._q3].copy(), mj_data.qvel[sheet._v3].copy()
            upd(mj_data, model, state, *a, **kw)
            mj_data.qpos[sheet._q3], mj_data.qvel[sheet._v3] = q, v

        def control_with_weight(model, state, control, mj_data):
            ctl(model, state, control, mj_data)
            if sheet._cpu_weight is not None:
                mj_data.qfrc_applied[sheet.vadr_env.reshape(-1) + 2] += sheet._cpu_weight

        solver._update_mjc_data, solver._apply_mjc_control = update_keeping_flex, control_with_weight
        a = float(self.params()["armature"])
        g = float(np.asarray(solver.mj_model.opt.gravity)[2])
        self._cpu_weight = a * g if a > 0 and os.environ.get("FLEX_ARM_GRAV", "1") == "1" else None
        self._pidx_flat = pidx.reshape(-1)
        print(f"[flex] CPU path: {len(self._q3) // 3} vertices in one MjData; armature weight {self._cpu_weight}", flush=True)

    def install_step(self) -> None:
        """Replace ``SolverCoupledProxy.step`` for the process (the env steps it outside any context)."""
        from newton._src.solvers.coupled import solver_coupled as sc
        from newton._src.solvers.coupled.solver_coupled_proxy import SolverCoupledProxy

        sheet = self
        original = SolverCoupledProxy.step

        def step(self, state_in, state_out, control, contacts, dt):
            if sheet.pidx is None:  # before bind (construction-time calls): unchanged
                return original(self, state_in, state_out, control, contacts, dt)
            self._distribute_state(state_in, dt=dt)
            sheet._fix_table_armature()
            if not sheet.cpu:
                sheet._add_armature_weight()
            self._step_entry(sheet.rigid, control, contacts, dt)
            sc._copy_state(state_in, state_out)
            self._reconcile_state(state_out)
            self._entry_output_state_valid = True
            sheet.copy_to_particles(state_out)
            if sheet.debug:
                sheet._debug_print()

        SolverCoupledProxy.step = step
        self.debug = int(os.environ.get("FLEX_DEBUG", "0"))
        self._calls = 0

    def _add_armature_weight(self) -> None:
        """``FLEX_ARMATURE`` adds inertia but not weight, so the sheet would fall (m + a) / m slower. Each
        step, before MuJoCo's, the missing weight ``a g`` goes on every vertex's z joint as an applied force
        (MuJoCo ignores a negative gravcomp). Free fall is then exact; only the inertial response is scaled."""
        a = float(self.params()["armature"])
        if a <= 0 or os.environ.get("FLEX_ARM_GRAV", "1") != "1":
            return
        import warp as wp

        if getattr(self, "_zdof_t", None) is None:
            import torch

            self._qfrc_t = wp.to_torch(self.rigid.solver.mjw_data.qfrc_applied)
            self._zdof_t = torch.as_tensor(self.vadr + 2, device=self._qfrc_t.device, dtype=torch.long)
            g = float(np.asarray(self.rigid.solver.mj_model.opt.gravity)[2])
            self._weight = a * g
            print(f"[flex] armature {a:g}: extra weight {self._weight:.4g} N on each vertex's z joint", flush=True)
        self._qfrc_t[:, self._zdof_t] = self._weight

    def _fix_table_armature(self) -> None:
        """Newton drives the kinematic table with armature 1e10 on its free joint. With the sheet's
        0.1-0.7 g vertices in the same float32 mass matrix that freezes the sheet and destabilises the
        robot, so (once, after Newton's own property writes) it is lowered to ``FLEX_TABLE_ARMATURE``
        (default 100 kg m^2: still 1e5 x anything that touches it)."""
        if getattr(self, "_table_fixed", False):
            return
        import mujoco
        import warp as wp

        if os.environ.get("FLEX_PARAMDUMP"):
            sol = self.rigid.solver
            src = sol.mj_model if self.cpu else sol.mjw_model
            def arr(name):
                a = getattr(src, name)
                a = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
                return a[0] if (not self.cpu and a.ndim > 1 and a.shape[0] == 1) else (a[0] if not self.cpu and a.ndim > 1 else a)
            out = {n: arr(n)[:27].round(5).tolist() if arr(n).ndim == 1 else arr(n)[:27, :3].round(5).tolist()
                   for n in ("dof_damping", "dof_armature", "dof_frictionloss", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "body_gravcomp")}
            import json
            json.dump(out, open(os.environ["FLEX_PARAMDUMP"], "w"))
            print(f"[flex] dumped robot params to {os.environ['FLEX_PARAMDUMP']}", flush=True)

        mj = self.rigid.solver.mj_model
        dofs = []
        for j in range(mj.njnt):
            body = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, int(mj.jnt_body_id[j]) if hasattr(mj, "jnt_body_id") else int(mj.jnt_bodyid[j])) or ""
            if "Table" in body or "table" in body:
                n = {0: 6, 1: 3, 2: 1, 3: 1}[int(mj.jnt_type[j])]
                dofs += list(range(int(mj.jnt_dofadr[j]), int(mj.jnt_dofadr[j]) + n))
        value = float(os.environ.get("FLEX_TABLE_ARMATURE", "100"))
        if self.cpu:
            before = float(mj.dof_armature[dofs].max()) if dofs else float("nan")
            if dofs and os.environ.get("FLEX_CPU_TABLE_ARMATURE", "0") == "1":  # float64 on CPU: off by default
                mj.dof_armature[dofs] = value
            self._table_fixed = True
            print(f"[flex] CPU: {len(dofs)} table dofs, armature {before:.3g} (left unless FLEX_CPU_TABLE_ARMATURE=1)", flush=True)
            return
        arm = wp.to_torch(self.rigid.solver.mjw_model.dof_armature)
        before = float(arm[..., dofs].max()) if dofs else float("nan")
        if dofs:
            arm[..., dofs] = value
        big = (arm > 1e4).nonzero()
        self._table_fixed = True
        print(f"[flex] table dofs {dofs}: armature {before:.3g} -> {value:g}; other dofs above 1e4: {big[:, -1].unique().tolist()[:10]}", flush=True)

    def _flex_body_range(self):
        import mujoco
        m = self.rigid.solver.mj_model
        b0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{_VERT}0")
        return b0, b0 + len(self.pos0)

    def _debug_print(self) -> None:
        self._calls += 1
        if self._calls > self.debug or self._calls % 6:
            return
        d = self.rigid.solver.mjw_data
        q, v = d.qpos.numpy(), d.qvel.numpy()
        fq = np.stack([q[:, self.qadr + k] for k in range(3)], -1)
        fv = np.stack([v[:, self.vadr + k] for k in range(3)], -1)
        robot = np.ones(v.shape[1], bool)
        for k in range(3):
            robot[self.vadr + k] = False
        rv = v[:, robot]
        z = fq[..., 2] + self.pos0[None, :, 2]
        if self._calls == 6:
            m = self.rigid.solver.mjw_model
            fl = np.concatenate([self.vadr + k for k in range(3)])
            b0, b1 = self._flex_body_range()
            for f in ("body_ipos", "body_inertia", "body_iquat", "body_pos"):
                a = getattr(m, f).numpy()
                print(f"[flex dbg] {f} {a.shape} flex body 0..2: {a[..., b0:b0 + 2, :].reshape(-1, a.shape[-1])[:4].round(5).tolist()}", flush=True)
            qm = self.rigid.solver.mjw_data.M.numpy()
            print(f"[flex dbg] qM shape {qm.shape}; is_sparse {m.is_sparse}; max {np.abs(qm).max():.4g}", flush=True)
            import mujoco
            mj = self.rigid.solver.mj_model
            md = mujoco.MjData(mj); md.qpos[:] = self.rigid.solver.mjw_data.qpos.numpy()[0]; mujoco.mj_forward(mj, md)
            full = np.zeros((mj.nv, mj.nv))
            for k in range(mj.nv):
                e = np.zeros(mj.nv); e[k] = 1.0; r = np.zeros(mj.nv); mujoco.mj_mulM(mj, md, r, e); full[:, k] = r
            dg = np.diag(full)
            print(f"[flex dbg] CPU qM diag flex {dg[fl].min():.4g}..{dg[fl].max():.4g} robot {np.delete(dg, fl).min():.4g}..{np.delete(dg, fl).max():.4g}; "
                  f"diag>1e3 at {np.nonzero(dg > 1e3)[0].tolist()[:12]}; armature max {mj.dof_armature.max():.4g}; offdiag flex-robot {np.abs(full[np.ix_(fl, np.delete(np.arange(mj.nv), fl))]).max():.3g}", flush=True)
            gm = m.dof_armature.numpy()[0]
            print(f"[flex dbg] GPU dof_armature max {gm.max():.4g} at {int(gm.argmax())}", flush=True)
            for f in ("dof_armature", "dof_damping", "dof_frictionloss", "body_mass", "body_gravcomp", "dof_invweight0", "jnt_stiffness"):
                a = getattr(m, f, None)
                if a is None: continue
                a = a.numpy()
                b0, b1 = self._flex_body_range()
                sl = a[..., fl] if f.startswith("dof") else (a[..., b0:b1] if f.startswith("body") else a)
                print(f"[flex dbg] {f} shape {a.shape} flex range {np.min(sl):.4g}..{np.max(sl):.4g}", flush=True)
            g = m.opt.gravity.numpy() if hasattr(m.opt.gravity, "numpy") else m.opt.gravity
            print(f"[flex dbg] opt.gravity {g} | disableflags {m.opt.disableflags} | flex body ids {self._flex_body_range()}", flush=True)
        fa = d.qfrc_applied.numpy(); fp = d.qfrc_passive.numpy(); xf = d.xfrc_applied.numpy()
        fl = np.concatenate([self.vadr + k for k in range(3)])
        b0, b1 = self._flex_body_range()
        print(f"[flex dbg {self._calls}] flex qfrc_applied {np.abs(fa[:, fl]).max(axis=1).round(5)} passive {np.abs(fp[:, fl]).max(axis=1).round(5)} "
              f"xfrc {np.abs(xf[:, b0:b1]).max(axis=(1, 2)).round(5)} actuator {np.abs(d.qfrc_actuator.numpy()[:, fl]).max(axis=1).round(5)}", flush=True)
        print(f"[flex dbg {self._calls}] robot |qvel| max {np.abs(rv).max(axis=1).round(2)} | flex |qvel| max "
              f"{np.abs(fv).max(axis=(1, 2)).round(2)} z {z.min(axis=1).round(3)}..{z.max(axis=1).round(3)} "
              f"ncon {d.nacon.numpy()[:1]} finite {np.isfinite(q).all(axis=1)}", flush=True)

    def copy_to_particles(self, state) -> None:
        import warp as wp

        if self.cpu:
            d = self.rigid.solver.mj_data
            pos = self.pos0_env.reshape(-1, 3) + d.qpos[self._q3].reshape(-1, 3)
            vel = d.qvel[self._v3].reshape(-1, 3)
            n = state.particle_q.shape[0]
            outq = np.zeros((n, 3), np.float32); outv = np.zeros((n, 3), np.float32)
            outq[self._pidx_flat] = pos; outv[self._pidx_flat] = vel
            state.particle_q.assign(outq); state.particle_qd.assign(outv)
            return

        d = self.rigid.solver.mjw_data
        wp.launch(_flex_kernel(), dim=self.pidx.shape, device=self.pidx.device,
                  inputs=[d.qpos, d.qvel, self._qadr, self._vadr, self._pos0, self.pidx],
                  outputs=[state.particle_q, state.particle_qd])

    # ------------------------------------------------------------------ reset

    def write(self, env_ids, target) -> None:
        """``target`` (n, P, 3) world positions of env_ids' sheets, at rest."""
        import torch
        import warp as wp

        if self.cpu:
            d = self.rigid.solver.mj_data
            ids = np.asarray(torch.as_tensor(env_ids).reshape(-1).cpu().numpy(), dtype=int)
            tgt = target.detach().cpu().numpy().astype(float)  # (n, P, 3)
            for row, k in enumerate(ids):
                qa = (self.qadr_env[k][:, None] + np.arange(3)).reshape(-1)
                va = (self.vadr_env[k][:, None] + np.arange(3)).reshape(-1)
                d.qpos[qa] = (tgt[row] - self.pos0_env[k]).reshape(-1)
                d.qvel[va] = 0.0
            return
        d = self.rigid.solver.mjw_data
        qpos, qvel = wp.to_torch(d.qpos), wp.to_torch(d.qvel)
        ids = torch.as_tensor(env_ids, device=qpos.device, dtype=torch.long).reshape(-1, 1, 1)
        pos0 = torch.as_tensor(self.pos0, device=qpos.device, dtype=qpos.dtype)
        k = torch.arange(3, device=qpos.device)
        qa = torch.as_tensor(self.qadr, device=qpos.device, dtype=torch.long).view(1, -1, 1) + k
        va = torch.as_tensor(self.vadr, device=qpos.device, dtype=torch.long).view(1, -1, 1) + k
        qpos[ids, qa] = (target.to(qpos.dtype) - pos0.unsqueeze(0))
        qvel[ids, va] = 0.0


def _kernel():
    import warp as wp

    @wp.kernel
    def flex_to_particles(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), qadr: wp.array(dtype=wp.int32),
                          vadr: wp.array(dtype=wp.int32), pos0: wp.array(dtype=wp.vec3), pidx: wp.array2d(dtype=wp.int32),
                          particle_q: wp.array(dtype=wp.vec3), particle_qd: wp.array(dtype=wp.vec3)):
        w, p = wp.tid()
        a, v = qadr[p], vadr[p]
        i = pidx[w, p]
        particle_q[i] = pos0[p] + wp.vec3(qpos[w, a], qpos[w, a + 1], qpos[w, a + 2])
        particle_qd[i] = wp.vec3(qvel[w, v], qvel[w, v + 1], qvel[w, v + 2])

    return flex_to_particles


_K = []


def _flex_kernel():
    if not _K:
        _K.append(_kernel())
    return _K[0]
