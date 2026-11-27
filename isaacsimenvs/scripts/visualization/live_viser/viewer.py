"""``LiveViewer``: the env, the policy and the browser, in one loop.

Threads: viser runs the GUI callbacks on its own thread. They only set fields here (``playing``,
``world``, ``reset_all``, ...); the loop, on the main thread, acts on them, so the env and the scene
are only ever touched by the loop.
"""

from __future__ import annotations

import os
import time
from collections import Counter

import numpy as np

from .pretty import TABLE_RGB
import torch

from .feature import Feature
from .fold_pool import FoldPool
from .overlays import GoalGhost, Keypoints, SurfaceGoal
from .point_cloud import PointCloud
from .policies import Players
from .scene import NewtonScene, PhysxScene

#: The episode outcomes, in the order a tie is named (the cloth task's ``termination_reasons``).
OUTCOMES = ("fold_held", "fold", "fall", "hand_far", "nonfinite", "timeout")

#: A browser's first view (camera position, look-at), env frame: the sheet at the table centre ~0.53 m
#: up, the robot at +y. The mouse orbits (drag), pans (right-drag) and zooms (scroll) from there.
START_VIEW = ((0.35, -0.45, 0.85), (0.0, 0.05, 0.53))


class EpisodeStats:
    """Episodes, held folds and the last outcome per env; outcome counts over all envs."""

    def __init__(self, n: int) -> None:
        self.episodes, self.folds, self.last = [0] * n, [0] * n, ["-"] * n
        self.outcomes: Counter = Counter()

    def record(self, reasons: dict, ended: torch.Tensor, truncated: torch.Tensor) -> None:
        for e in ended.nonzero(as_tuple=True)[0].tolist():
            fired = [k for k in OUTCOMES if k in reasons and bool(reasons[k][e])]
            out = fired[0] if fired else ("timeout" if bool(truncated[e]) else "?")
            self.episodes[e] += 1
            self.folds[e] += int(out == "fold_held")
            self.last[e] = out
            self.outcomes[out] += 1


class LiveViewer:
    def __init__(self, server, env, *, target: str, players: Players, policy: str, reset_starts: dict) -> None:
        self.server, self.env, self.inner = server, env, env.unwrapped
        inner = self.inner
        self.target, self.players, self.reset_starts = target, players, reset_starts
        self.n_env, self.device = inner.num_envs, inner.device
        self.origins = inner.scene.env_origins.detach().cpu().numpy()
        rigid = hasattr(inner.backend, "pool")  # the rigid sheet's fold-shape pool (PhysX); else VBD on Newton
        self.scene = PhysxScene(inner) if rigid else NewtonScene(inner)

        # Set from the GUI thread, acted on by the loop.
        self.playing, self.step_once, self.speed, self.deterministic = True, False, 1.0, False
        self.world = 0                 # the env the GUI shows
        self.policy_name = policy
        self.reset_all = self.reset_one = False
        self.rebuild = True
        # The loop's own.
        self.player = None
        self.drawn = 0                 # the env the scene is built for (``world`` may be ahead of it)
        self.frames: dict = {}         # body key -> its frame handle
        self.fps = 0.0
        self.stats = EpisodeStats(self.n_env)
        # Best (lowest) fold error: this episode (with its step), the last finished episode, all episodes.
        self.best_ep = [float("inf")] * self.n_env
        self.best_ep_step = [0] * self.n_env
        self.best_last = [float("nan")] * self.n_env
        self.best_all = float("inf")

        self.pool = FoldPool(self) if rigid else None
        self.pc = PointCloud(self) if inner.observations.pointcloud is not None else None
        self.features: list[Feature] = [f for f in (self.pool, Keypoints(self)) if f is not None]
        if rigid and str(inner.cfg.fold_pool.goal_task.mode) != "off":
            self.features.append(GoalGhost(self))
        if getattr(inner.task, "surface", False):
            self.features.append(SurfaceGoal(self))
        if not rigid:
            from .vbd import ClothSheet, Pierce

            self.features += [ClothSheet(self), Pierce(self)]
        if self.pc is not None:
            self.features.append(self.pc)
        self._gui()

    # ------------------------------------------------------------------ helpers
    def student(self):
        """The active player if it is a point-cloud student, else None."""
        return self.player if getattr(self.player, "is_student", False) else None

    def _maybe_snap(self) -> None:
        """``outputs/renders/snap_<port>.trigger`` present: save each client's view to snap_<port>_<k>.png (for debugging the look)."""
        from pathlib import Path

        port = self.server.get_port()
        trig = Path(f"outputs/renders/snap_{port}.trigger")
        if not trig.exists():
            return
        trig.unlink()
        import imageio.v3 as iio

        for k, c in enumerate(self.server.get_clients().values()):
            try:
                img = c.get_render(720, 1280, transport_format="png")
                iio.imwrite(f"outputs/renders/snap_{port}_{k}.png", img)
                print(f"[live] snap saved: outputs/renders/snap_{port}_{k}.png", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[live] snap failed: {exc}", flush=True)

    def object_opacity(self) -> float | None:
        """The sheet / plates are translucent under a student, so the cloud on and under them shows."""
        return self.pc.opacity() if self.pc is not None and self.student() is not None else None

    def env_label(self, w: int) -> str:
        return self.pool.env_label(w) if self.pool is not None else f"env {w}"

    # ------------------------------------------------------------------ GUI
    def _gui(self) -> None:
        gui, inner = self.server.gui, self.inner
        from .pretty import load_visuals

        self.visuals = load_visuals()
        try:
            self.server.scene.configure_environment_map("studio", environment_intensity=0.35)
            self.server.scene.enable_default_lights(True, cast_shadow=True)
        except Exception:  # noqa: BLE001 - cosmetics only
            pass
        self.g_toggle = gui.add_button("Pause")
        with gui.add_folder("Scene"):
            g_env = gui.add_dropdown("show env", [self.env_label(w) for w in range(self.n_env)])
            g_start = gui.add_dropdown("start pose", list(self.reset_starts))
            g_tol = gui.add_slider("fold tolerance [mm]", 5.0, 60.0, 0.5, float(inner.cfg.cloth.keypoint_tolerance) * 1e3)
            g_reset = gui.add_button("reset this env")
            g_reset_all = gui.add_button("reset all envs")
            self.g_collision = gui.add_checkbox("robot: collision shapes (off: visual meshes)", not self.visuals)

            @self.g_collision.on_update
            def _(_e):
                for h in getattr(self, "collision_handles", []):
                    h.visible = bool(self.g_collision.value)
            for f in self.features:
                f.scene_gui()
        with gui.add_folder("Custom start"):
            rs = inner.cfg.reset
            g_cx = gui.add_slider("sheet x offset [cm]", -15.0, 15.0, 0.5, 0.0)
            g_cy = gui.add_slider("sheet y offset [cm]", -15.0, 15.0, 0.5, 0.0)
            g_yaw = gui.add_slider("sheet heading [deg]", -180.0, 180.0, 5.0, 0.0)
            g_apply = gui.add_button("use this start (resets all envs)")
        if self.pool is None:
            self._physics_panel_vbd()
        else:
            self._physics_panel_rigid()
        with gui.add_folder("Policy"):
            self.g_pol = g_pol = gui.add_dropdown("policy", list(self.players.menu), initial_value=self.policy_name)
            g_det = gui.add_checkbox("deterministic actions", False)
            self.g_play = gui.add_checkbox("playing", True)
            g_step = gui.add_button("single step")
            g_speed = gui.add_slider("speed (x real time; 4 = as fast as possible)", 0.1, 4.0, 0.1, 1.0)
        for f in self.features:
            f.gui()
        with gui.add_folder("Status", expand_by_default=True):
            self.g_msg = gui.add_markdown("")
            self.g_hud = gui.add_markdown("")

        @g_env.on_update
        def _(_e):
            self.world = [self.env_label(w) for w in range(self.n_env)].index(g_env.value)
            for f in self.features:
                f.on_env(self.world)
            self.rebuild = True

        @g_start.on_update
        def _(_e):
            for k, v in self.reset_starts[g_start.value].items():  # read by the env at every reset
                setattr(inner.cfg.reset, k, v)

        @g_apply.on_click
        def _(_e):
            # A fixed start, no reset noise. The task decodes a reset quaternion with its components
            # reordered (backends/base.py reset_yaw): heading t is written as (sin t/2, cos t/2, 0, 0).
            import math

            t = math.radians(float(g_yaw.value))
            table_xy = getattr(rs, "table_reset_xy", (0.0, 0.0))
            for k in ("reset_position_noise_x", "reset_position_noise_y", "reset_position_noise_z", "table_reset_z_range"):
                setattr(rs, k, 0.0)
            rs.fixed_start_pose = (table_xy[0] + float(g_cx.value) / 100, table_xy[1] + float(g_cy.value) / 100,
                                   rs.table_reset_z + rs.table_object_z_offset, math.sin(t / 2), math.cos(t / 2), 0.0, 0.0)
            self.reset_all = True

        g_tol.on_update(lambda _e: setattr(inner.cfg.cloth, "keypoint_tolerance", float(g_tol.value) * 1e-3))
        g_reset.on_click(lambda _e: setattr(self, "reset_one", True))
        g_reset_all.on_click(lambda _e: setattr(self, "reset_all", True))
        g_pol.on_update(lambda _e: setattr(self, "policy_name", g_pol.value))
        g_det.on_update(lambda _e: setattr(self, "deterministic", bool(g_det.value)))
        def _sync_play(_e):
            self.playing = bool(self.g_play.value)
            self.g_toggle.label = "Pause" if self.playing else "Play"

        self.g_play.on_update(_sync_play)
        self.g_toggle.on_click(lambda _e: setattr(self.g_play, "value", not bool(self.g_play.value)))
        g_step.on_click(lambda _e: setattr(self, "step_once", True))
        g_speed.on_update(lambda _e: setattr(self, "speed", float(g_speed.value)))
        self.server.on_client_connect(self._start_view)
        self.server.scene.add_grid("/ground", width=3.0, height=3.0, cell_size=0.1, position=(0, 0, 0.0))

    def _physics_panel_rigid(self) -> None:
        """The rigid sheet's panels (restart): a custom 2vert / 2flaps design, and the plates' physics."""
        import math
        import os
        import sys

        from . import designs

        gui, inner = self.server.gui, self.inner

        _real = gui

        class _G:  # the GUI, with sliders whose range always holds their starting value
            def __getattr__(_, name):
                return getattr(_real, name)

            @staticmethod
            def add_slider(label, lo, hi, step, value, **kw):
                value = float(value)
                return _real.add_slider(label, min(lo, value), max(hi, value), step, value, **kw)

        gui = _G()
        rc, pc, fp = inner.cfg.rigid_cloth, inner.cfg.physx_chain, inner.cfg.fold_pool
        current = self.pool.names[int(self.pool.rb.pool.active[0])] if self.pool is not None else ""
        approx = [*designs.base_names(), "2vert+hinge (crease sliders)", "2flaps+hinge (crease sliders)"]
        start = current if current in approx else ("2vert+hinge (crease sliders)" if current.startswith("2vert")
                                                     else "2flaps+hinge (crease sliders)" if current.startswith("2flaps")
                                                     else approx[0])
        with gui.add_folder("Build the rigid sheet (restart)", expand_by_default=True):
            gui.add_markdown(f"running: **{current}** (1 env, this design only)")
            g_fam = gui.add_dropdown("approximation", approx, initial_value=start)
            gui.add_markdown("Coordinates in units of L = half the sheet side: the sheet is x, y in [-1, 1]; "
                             "the centre crease is x = 0 and the right half (+x) folds over.")
            g_vl = gui.add_number("2vert: left outer crease x (-1 .. 0)", -0.35, min=-0.99, max=-0.01, step=0.01)
            g_vr = gui.add_number("2vert: right outer crease x (0 .. 1)", 0.5, min=0.01, max=0.99, step=0.01)
            g_lc = gui.add_text("2flaps: left crease x1, y1, x2, y2", "-0.31, 1, -0.53, -1")
            g_rc = gui.add_text("2flaps: right crease x1, y1, x2, y2", "0, 1, 1, 0.36")
            gui.add_markdown("2flaps: each crease's two ends on its half's boundary (x = 0, the outer edge x = -1 / +1, "
                             "or y = +-1). E.g. a strip -0.3, 1, -0.5, -1; a fan from the centre corner 0, 1, 1, 0.36.")
            g_make = gui.add_button("build this sheet (design + physics below) & restart")
        # Slider ranges: the 2vert training recipe's (its DR ranges around its centre values).
        from isaacsimenvs.tasks.recipes import load_recipe

        RANGE_RECIPE = "experiments/nov25_better_rigidcloth/2vert/2vert.yaml"
        R = load_recipe(RANGE_RECIPE)["env"]
        rr, rfp, rdr = R.get("rigid_cloth", {}), R.get("fold_pool", {}), R.get("domain_randomization", {}).get("cloth", {})
        lg = math.log10
        d0, k0 = float(rr.get("density", rc.density)), float(rr.get("joint_damping", rc.joint_damping))
        m0 = float(rr.get("contact_margin", rc.contact_margin))
        ms, kds = rdr.get("mass_scale", [1, 1]), rdr.get("hinge_damping_scale", [1, 1])
        ks, yd, cmo = rdr.get("hinge_stiffness", [1e-6, 1e-3]), rdr.get("hinge_yield_deg", [0, 0]), rdr.get("contact_margin_offset", [0, 0])
        sz, thr = rfp.get("size_range", fp.size_range), rfp.get("thickness_range", [0.001, 0.001])
        ranges = {
            "size": (100 * float(sz[0]), 100 * float(sz[1])),
            "density": (d0 * float(ms[0]), d0 * float(ms[1])),
            "stiff": (lg(max(float(ks[0]), 1e-9)), lg(max(float(ks[1]), 1e-9))),
            "damp": (lg(k0 * float(kds[0])), lg(k0 * float(kds[1]))),
            "yield": (float(yd[0]), float(yd[1])),
            "margin": (1000 * max(0.0, m0 + float(cmo[0])), 1000 * (m0 + float(cmo[1]))),
        }

        def slider(label, key, step, value, wide=None):
            lo, hi = ranges[key]
            if hi <= lo:
                hi = lo + step
            if wide is None:
                return _real.add_slider(f"{label}  [{lo:.3g} .. {hi:.3g}]", lo, hi, step, min(max(float(value), lo), hi))
            wlo, whi = min(wide[0], lo), max(wide[1], hi)
            return _real.add_slider(f"{label}  (training {lo:.3g} .. {hi:.3g})", wlo, whi, step, min(max(float(value), wlo), whi))

        with gui.add_folder("Physical properties (applied by either button)"):
            gui.add_markdown(f"Ranges from **{RANGE_RECIPE}** (its DR). Fixed there (no range): plate thickness, "
                             "hinge armature, frictions; typed values below.")
            size = float(fp.size_range[0])
            th = float(fp.thickness_range[0]) if getattr(fp, "thickness_range", None) else 0.001
            g_size = slider("sheet size [cm]", "size", 0.5, size * 100)
            g_th = gui.add_number("plate thickness [mm] (2vert: fixed)", round(1000 * float(thr[0]), 2), min=0.2, max=6.0, step=0.1)
            g_dens = slider("sheet density [kg/m^2]", "density", 0.1, rc.density)
            st0 = float(inner.cfg.domain_randomization.cloth.hinge_stiffness[0]) or float(rc.joint_stiffness)
            g_ks = slider("hinge bending rigidity: log10 [N m]", "stiff", 0.1, lg(max(st0, 1e-9)) if st0 > 0 else ranges["stiff"][0],
                          wide=(-9.0, 0.0))
            g_kd = slider("hinge viscosity (damping): log10 [N m s/rad]", "damp", 0.05, lg(max(float(rc.joint_damping), 1e-9)),
                          wide=(-6.0, 0.0))
            arm = float(pc.hinge_armature) if float(pc.hinge_armature) > 0 else float(rc.joint_armature)
            g_arm = gui.add_number("hinge armature: log10 [kg m^2] (2vert: fixed)", round(lg(max(arm, 1e-9)), 2), min=-9.0, max=-2.0, step=0.1)
            g_yield = slider("hinge plasticity: yield [deg]", "yield", 0.5, rc.joint_yield_deg if float(rc.joint_yield_deg) > 0 else sum(ranges["yield"]) / 2)
            g_mu = gui.add_number("friction: plate-plate/table (2vert: fixed)", float(rc.slat_friction), min=0.0, max=2.0, step=0.01)
            g_mut = gui.add_number("friction: plate-fingertip (2vert: fixed)", float(rc.fingertip_slat_friction), min=0.0, max=2.0, step=0.01)
            g_cm = slider("rest offset (contact margin) [mm]", "margin", 0.25, float(rc.contact_margin) * 1000)
            g_com = gui.add_number("contact offset = rest + this [mm] (2vert: fixed)",
                                   round(1000 * float(R.get("physx_chain", {}).get("contact_offset_margin", pc.contact_offset_margin)), 2),
                                   min=0.1, max=20.0, step=0.1)
            g_live = gui.add_button("apply now, no restart (hinges, density, offsets, plasticity)")
            gui.add_markdown("Live: hinge stiffness / damping / armature, density, rest + contact offset, plasticity (if it "
                             "started on). Restart needed: sheet size, plate thickness, frictions, the design.")
            g_restart = gui.add_button("restart: same design, these properties")

        def f(x) -> str:
            t = f"{float(x):.6g}"
            return t if any(ch in t for ch in ".en") else t + ".0"

        def restart(select: str | None) -> None:
            new = {"env.fold_pool.size_range": f"[{f(g_size.value / 100)},{f(g_size.value / 100)}]",
                   "env.fold_pool.thickness_range": f"[{f(g_th.value / 1000)},{f(g_th.value / 1000)}]",
                   "env.rigid_cloth.density": f(g_dens.value),
                   "env.rigid_cloth.joint_stiffness": f(10 ** g_ks.value),
                   "env.rigid_cloth.joint_damping": f(10 ** g_kd.value),
                   "env.physx_chain.hinge_armature": f(10 ** g_arm.value),
                   "env.rigid_cloth.joint_yield_deg": f(g_yield.value),
                   "env.rigid_cloth.slat_friction": f(g_mu.value),
                   "env.rigid_cloth.fingertip_slat_friction": f(g_mut.value),
                   "env.rigid_cloth.contact_margin": f(g_cm.value / 1000),
                   "env.domain_randomization.cloth.mass_scale": "[1.0,1.0]",
                   "env.domain_randomization.cloth.hinge_damping_scale": "[1.0,1.0]",
                   "env.domain_randomization.cloth.hinge_stiffness": f"[{f(10 ** g_ks.value)},{f(10 ** g_ks.value)}]",
                   "env.domain_randomization.cloth.hinge_stiffness_log": "true",
                   "env.domain_randomization.cloth.hinge_from_material": "true",
                   "env.domain_randomization.cloth.hinge_yield_deg": f"[{f(g_yield.value)},{f(g_yield.value)}]",
                   "env.domain_randomization.cloth.contact_margin_offset": "[0.0,0.0]",
                   "env.physx_chain.contact_offset_margin": f(g_com.value / 1000),
                   "env.physx_chain.contact_offset_follows_rest": "true",
                   "env.fold_pool.shapes": f"[{select or current}]"}  # this design only: fast to build and step
            if not str(select or current).startswith(("2vert", "2flaps")):
                new.pop("env.domain_randomization.cloth.hinge_from_material")  # material factors: the families only
            argv = [a for a in sys.orig_argv if a.split("=", 1)[0] not in new]
            if "--num_envs" in argv:  # one env
                k = argv.index("--num_envs")
                argv[k + 1] = "1"
            else:
                argv += ["--num_envs", "1"]
            argv += [f"{k}={v}" for k, v in new.items()]
            if select:
                os.environ["LIVE_SHAPE"] = select
            self.g_msg.content = f"restarting{' on ' + select if select else ''} ... reload the page in ~3 min"
            print(f"[live] restart: {' '.join(argv)}", flush=True)
            self.server.stop()
            os.execv(sys.executable, argv)

        @g_make.on_click
        def _(_e):
            try:
                if g_fam.value in designs.base_names():
                    name = g_fam.value
                elif g_fam.value.startswith("2vert"):
                    name = designs.make_2vert_x(float(g_vl.value), float(g_vr.value))
                else:
                    nums = lambda t: tuple(float(x) for x in str(t).replace(";", ",").split(","))
                    left, right = nums(g_lc.value), nums(g_rc.value)
                    if len(left) != 4 or len(right) != 4:
                        raise ValueError("each crease needs four numbers: x1, y1, x2, y2")
                    name = designs.make_2flaps_points(left, right)
            except Exception as exc:  # noqa: BLE001 - an invalid design: say so, stay up
                self.g_msg.content = f"**design rejected:** {exc}"
                return
            restart(name)

        density0 = float(rc.density)

        def apply_live(_e) -> None:
            import torch

            b = inner.backend
            dr = getattr(b, "physics_dr", None)
            ids = torch.arange(inner.num_envs, device=inner.device)
            k_s = 10 ** float(g_ks.value)
            k_d, arm = 10 ** float(g_kd.value), 10 ** float(g_arm.value)
            scale = float(g_dens.value) / density0
            done = []
            for s_, art in enumerate(b.arts):
                keep, local = b.pool_rows(s_, ids)
                if local.numel() == 0:
                    continue
                fac = dr.hinge_factor[s_] if dr is not None else torch.ones(art.num_joints, device=inner.device)
                n_ = int(local.numel())
                art.write_joint_stiffness_to_sim((k_s * fac).expand(n_, -1).clone(), env_ids=local)
                art.write_joint_damping_to_sim((k_d * fac).expand(n_, -1).clone(), env_ids=local)
                art.write_joint_armature_to_sim(torch.full((n_, art.num_joints), arm, device=inner.device), env_ids=local)
                if dr is not None:
                    view, l_cpu = art.root_physx_view, local.to("cpu", torch.int32)
                    masses, inertias = view.get_masses(), view.get_inertias()
                    masses[l_cpu.long()] = dr.mass_base[s_][l_cpu.long()] * scale
                    inertias[l_cpu.long()] = dr.inertia_base[s_][l_cpu.long()] * scale
                    view.set_masses(masses, l_cpu)
                    view.set_inertias(inertias, l_cpu)
                    margin = float(g_cm.value) / 1000
                    dr.contact_offset_margin = float(g_com.value) / 1000
                    dr.contact_offset_follows_rest = True
                    rest = view.get_rest_offsets()
                    if dr.contact_offset_follows_rest:
                        contact = view.get_contact_offsets()
                        contact[l_cpu.long()] = torch.maximum(contact[l_cpu.long()], torch.full_like(rest[l_cpu.long()], margin + dr.contact_offset_margin))
                        view.set_contact_offsets(contact, l_cpu)
                        rest[l_cpu.long()] = margin
                        view.set_rest_offsets(rest, l_cpu)
                        contact[l_cpu.long()] = margin + dr.contact_offset_margin
                        view.set_contact_offsets(contact, l_cpu)
                    elif margin < float(view.get_contact_offsets().min()):
                        rest[l_cpu.long()] = margin
                        view.set_rest_offsets(rest, l_cpu)
                done.append(art)
            # Keep later resets from overwriting the live values (the DR re-draws only ranges that vary).
            rc.joint_stiffness, rc.joint_damping, rc.density = k_s, k_d, float(g_dens.value)
            rd_ = inner.cfg.domain_randomization.cloth
            rd_.mass_scale, rd_.hinge_damping_scale = (scale, scale), (1.0, 1.0)
            rc.joint_yield_deg = float(g_yield.value)
            rd_.hinge_stiffness, rd_.hinge_yield_deg, rd_.contact_margin_offset = (k_s, k_s), (float(g_yield.value),) * 2, (0.0, 0.0)
            pl = getattr(b, "plasticity", None)
            msg = ""
            if pl is not None and getattr(pl, "enabled", False):
                pl.yield_angle[:] = math.radians(float(g_yield.value))
            elif float(g_yield.value) > 0:
                msg = " Plasticity was off at start: use restart to turn it on."
            self.g_msg.content = (f"applied live to {len(done)} plate articulation(s): stiffness {k_s:.3g}, damping "
                                  f"{k_d:.3g}, armature {arm:.3g}, mass x{scale:.2f}, rest offset {g_cm.value:.2f} mm, contact offset +{g_com.value:.2f} mm.{msg}")
            print(f"[live] {self.g_msg.content}", flush=True)

        g_live.on_click(apply_live)
        g_restart.on_click(lambda _e: restart(None))

    def _physics_panel_vbd(self) -> None:
        """The VBD cloth's Physics (restart) panel: presets, sheet, material, frictions, solver, coupler."""
        gui, inner = self.server.gui, self.inner
        import sys
        with gui.add_folder("Physics (restart)"):
            import sys as _sys
            from isaacsimenvs import CLOTH_CONFIG_DIR

            eval_dir = CLOTH_CONFIG_DIR / "eval"
            presets = {"vbd.yaml": eval_dir / "vbd.yaml"}
            if (eval_dir / "stable.yaml").exists():
                presets["stable.yaml (tuned)"] = eval_dir / "stable.yaml"
            for f in sorted((eval_dir / "fabric").glob("*.yaml")):
                presets[f"fabric/{f.name}"] = f
            argv0 = list(_sys.orig_argv)
            cur = argv0[argv0.index("--protocol") + 1] if "--protocol" in argv0 else str(eval_dir / "vbd.yaml")
            from pathlib import Path as _Path

            cur_name = next((k for k, v in presets.items() if _Path(v).resolve() == _Path(cur).resolve()), "vbd.yaml")
            g_preset = gui.add_dropdown("physics preset (YAML)", list(presets), initial_value=cur_name)
            g_trainhand = gui.add_checkbox("training hand (sysid off)", not bool(inner.cfg.hand_sysid))
            c = inner.cfg.cloth
            g_size = gui.add_slider("cloth size [cm]", 6.0, 30.0, 1.0, float(c.size) * 100)
            g_res = gui.add_slider("cloth grid", 5, 25, 2, int(c.resolution))
            g_thick = gui.add_slider("particle thickness [mm] (2 x radius)", 4.0, 24.0, 1.0, float(c.thickness) * 1000)
            g_vbdit = gui.add_slider("VBD iterations (cloth solver)", 2, 100, 1, int(c.vbd_iterations))
            g_iters = gui.add_slider("coupler iterations", 1, 6, 1, int(c.coupler_iterations))
            g_rmode = gui.add_dropdown("relaxation mode", ("fixed", "aitken"), initial_value=str(c.proxy_relaxation_mode))
            g_relax = gui.add_slider("relaxation (1 = none)", 0.05, 1.0, 0.05, float(c.proxy_relaxation))
            import math as _m

            g_trike = gui.add_slider("stretch: log10 tri_ke (= tri_ka)", 1.0, 5.0, 0.05, round(_m.log10(float(c.tri_ke)), 2))
            g_edgeke = gui.add_slider("bending: log10 edge_ke", -6.0, -1.0, 0.05, round(_m.log10(max(float(c.edge_ke), 1e-9)), 2))
            g_edgekd = gui.add_slider("bending damping: log10 edge_kd", -6.0, -1.0, 0.05,
                                      round(_m.log10(max(float(c.edge_kd), 1e-9)), 2))
            g_dens = gui.add_slider("density [kg/m^2]", 0.05, 4.0, 0.01, float(c.density))
            g_plast = gui.add_slider("plasticity: yield curvature [1/m] (0 = springs back)", 0.0, 400.0, 5.0,
                                     float(c.plastic_yield_curvature))
            g_mu = {k: gui.add_slider(label, 0.0, 2.0, 0.01, float(getattr(c, k)))
                    for k, label in (("cloth_table_friction", "friction: cloth-table"),
                                     ("cloth_fingertip_friction", "friction: cloth-fingertips"),
                                     ("cloth_robot_friction", "friction: cloth-rest of hand (palm, phalanges)"),
                                     ("cloth_cloth_friction", "friction: cloth-cloth"))}
            g_restart = gui.add_button("restart viewer with these")
        def _f(x) -> str:
            t = f"{float(x):.6g}"
            return t if any(ch in t for ch in ".en") else t + ".0"

        @g_restart.on_click
        def _(_e):
            import os, sys
            # The preset's own values win; a panel value is passed only if it was changed here.
            panel = {"env.cloth.size": (f"{float(g_size.value) / 100:.3f}", f"{float(c.size):.3f}"),
                     "env.cloth.resolution": (str(int(g_res.value)), str(int(c.resolution))),
                     "env.cloth.thickness": (f"{float(g_thick.value) / 1000:.4f}", f"{float(c.thickness):.4f}"),
                     "env.cloth.coupler_iterations": (str(int(g_iters.value)), str(int(c.coupler_iterations))),
                     "env.cloth.vbd_iterations": (str(int(g_vbdit.value)), str(int(c.vbd_iterations))),
                     "env.cloth.proxy_relaxation_mode": (str(g_rmode.value), str(c.proxy_relaxation_mode)),
                     "env.cloth.proxy_relaxation": (f"{float(g_relax.value):.3f}", f"{float(c.proxy_relaxation):.3f}"),
                     "env.hand_sysid": ("false" if g_trainhand.value else "true", "false" if not inner.cfg.hand_sysid else "true"),
                     **{f"env.cloth.{k}": (f"{float(g.value):.3f}", f"{float(getattr(c, k)):.3f}") for k, g in g_mu.items()},
                     "env.cloth.tri_ke": (_f(10 ** float(g_trike.value)), _f(c.tri_ke)),
                     "env.cloth.tri_ka": (_f(10 ** float(g_trike.value)), _f(c.tri_ka)),
                     "env.cloth.edge_ke": (_f(10 ** float(g_edgeke.value)), _f(c.edge_ke)),
                     "env.cloth.edge_kd": (_f(10 ** float(g_edgekd.value)), _f(c.edge_kd)),
                     "env.cloth.density": (_f(g_dens.value), _f(c.density)),
                     "env.cloth.plastic_yield_curvature": (_f(g_plast.value), _f(c.plastic_yield_curvature))}
            from pathlib import Path as _P

            preset_changed = _P(presets[g_preset.value]).resolve() != _P(cur).resolve()
            if preset_changed:
                # Compare with what the NEW preset would set (its YAML, through its base chain), so the run
                # gets exactly what the panel shows.
                import yaml

                def resolved(path):
                    d = yaml.safe_load(open(path)) or {}
                    out = {}
                    if d.get("base"):
                        b = (path.parent / str(d["base"])).with_suffix(".yaml")
                        out = resolved(b) if b.exists() else {}
                    env = d.get("env") or {}
                    for k, v in (env.get("cloth") or {}).items():
                        out[f"env.cloth.{k}"] = v
                    if "hand_sysid" in env:
                        out["env.hand_sysid"] = env["hand_sysid"]
                    return out

                new_vals = resolved(presets[g_preset.value])
                fmt = {"env.cloth.size": lambda v: f"{float(v):.3f}", "env.cloth.thickness": lambda v: f"{float(v):.4f}",
                       "env.cloth.proxy_relaxation": lambda v: f"{float(v):.3f}", "env.cloth.resolution": lambda v: str(int(v)),
                       "env.cloth.coupler_iterations": lambda v: str(int(v)), "env.cloth.proxy_relaxation_mode": str,
                       "env.cloth.vbd_iterations": lambda v: str(int(v)),
                       "env.hand_sysid": lambda v: "true" if v else "false",
                       **{f"env.cloth.{k}": (lambda v: f"{float(v):.3f}") for k in g_mu},
                       **{f"env.cloth.{k}": _f for k in ("tri_ke", "tri_ka", "edge_ke", "edge_kd", "density", "plastic_yield_curvature")}}
                panel = {k: (v, fmt[k](new_vals[k]) if k in new_vals else was) for k, (v, was) in panel.items()}
            argv, skip = [], False
            for a in sys.orig_argv:
                if skip:
                    skip = False
                    continue
                if a == "--protocol":
                    skip = True
                    continue
                if preset_changed and a.startswith("env."):
                    continue  # a new preset: the old world's command-line physics go with it
                if a.split("=", 1)[0] in panel:
                    continue
                argv.append(a)
            argv = [a for a in argv if not a.startswith("env.cloth.keypoint_tolerance=")]
            argv += ["--protocol", str(presets[g_preset.value])]
            # The fold tolerance is a viewing choice, not physics: keep the Scene slider's current value.
            argv += [f"env.cloth.keypoint_tolerance={float(inner.cfg.cloth.keypoint_tolerance):.4f}"]
            def changed(v, was):  # log sliders round: within 6% is "unchanged"
                # (defined below the panel dict: float-typed config fields need a float literal)
                try:
                    a, b = float(v), float(was)
                    return abs(a - b) > 0.06 * max(abs(a), abs(b), 1e-12)
                except ValueError:
                    return v != was

            argv += [f"{k}={v}" for k, (v, was) in panel.items() if changed(v, was) or not preset_changed]
            self.g_msg.content = "restarting with the new sheet ... reload the page in ~2 min"
            print(f"[live] restart: {' '.join(argv)}", flush=True)
            self.server.stop()
            os.execv(sys.executable, argv)


    @staticmethod
    def _start_view(client) -> None:
        pos, at = START_VIEW
        client.camera.position = np.asarray(pos, dtype=float)
        client.camera.look_at = np.asarray(at, dtype=float)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    # ------------------------------------------------------------------ drawing
    def _rebuild(self) -> None:
        """Build the drawn env's bodies under /w (removing the previous env's whole)."""
        self.rebuild = False
        if hasattr(self, "_root"):
            self._root.remove()
        self._root = self.server.scene.add_frame("/w", show_axes=False)
        for f in self.features:
            f.clear()
        w = self.drawn = self.world
        shapes = self.scene.shapes(w)
        keys = sorted({s.key for s in shapes if s.key is not None}, key=str)
        names = {k: f"/w/b{i}" for i, k in enumerate(keys)}
        self.frames = {k: self.server.scene.add_frame(names[k], show_axes=False) for k in keys}
        self.server.scene.add_frame("/w/static", show_axes=False)
        opacity = self.object_opacity()
        link_of = {}  # body key -> its link name (the robot's)
        self.collision_handles = []
        for i, s in enumerate(shapes):
            moving = s.key is not None
            if s.kind == "robot" and moving:
                link_of[s.key] = s.label.rsplit("/", 1)[-1]
            h = self.server.scene.add_mesh_simple(
                f"{names[s.key] if moving else '/w/static'}/s{i}", s.verts, s.faces,
                color=TABLE_RGB if s.kind == "table" else s.rgb,
                position=s.pos if moving else s.pos - self.origins[w], wxyz=s.wxyz,
                side="double",  # the PhysX scene's triangle winding is not Newton's: never cull
                opacity=opacity if s.kind == "object" else None)
            if s.kind == "robot":
                self.collision_handles.append(h)
                h.visible = bool(self.g_collision.value) or not self.visuals
        by_link = {v: k for k, v in link_of.items()}
        for j, (link, verts, faces, rgb) in enumerate(self.visuals):
            if link in by_link:
                self.server.scene.add_mesh_simple(f"{names[by_link[link]]}/v{j}", verts, faces, color=rgb, side="double")

    def _draw(self) -> None:
        w = self.drawn
        if self.frames:
            o = self.origins[w]
            with self.server.atomic():
                for k, (pw, q) in self.scene.poses(w, list(self.frames)).items():
                    h, p = self.frames[k], np.asarray(pw, dtype=np.float64) - o
                    h.visible = not (abs(p[0]) > 1.5 or abs(p[1]) > 1.5 or p[2] > 1.6)  # pool shapes parked away
                    h.position, h.wxyz = p, tuple(float(x) for x in q)
        for f in self.features:
            f.draw(w)

    def _hud(self) -> str:
        """The status panel: one table, a row per item (the features' rows after the viewer's own)."""
        inner, w, st = self.inner, self.world, self.stats
        err, tol = float(inner.fold_error()[w]), float(inner.cfg.cloth.keypoint_tolerance)
        rows = [
            ("target", self.target),
            ("policy", self.policy_name),
            ("env", f"{w}, step {int(inner.episode_length_buf[w])} / {int(inner.max_episode_length)}"),
            ("fold error", f"**{err * 1e3:.1f} mm** / {tol * 1e3:.1f} mm" + (" **FOLDED**" if err < tol else "")),
            ("best fold, this episode", (f"**{self.best_ep[w] * 1e3:.1f} mm** at step {self.best_ep_step[w]}"
                                         if self.best_ep[w] < float("inf") else "-")),
            ("best fold, last episode / all episodes",
             f"{self.best_last[w] * 1e3:.1f} mm / " + (f"{self.best_all * 1e3:.1f} mm" if self.best_all < float("inf") else "-")
             if self.best_last[w] == self.best_last[w] else
             "- / " + (f"{self.best_all * 1e3:.1f} mm" if self.best_all < float("inf") else "-")),
            ("held folds", f"this env {st.folds[w]} / {st.episodes[w]} (last: {st.last[w]}); "
                           f"all {sum(st.folds)} / {sum(st.episodes)}"),
            ("speed", f"{self.fps:.0f} steps/s (real time {1.0 / float(inner.step_dt):.0f})"),
        ]
        for f in self.features:
            rows += f.hud(w)
        return "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)

    # ------------------------------------------------------------------ loop
    def _use_policy(self) -> None:
        try:
            player = self.players.get(self.policy_name)
        except ValueError as exc:  # a student whose cloud is not this env's: stay on the current policy
            print(f"[live] {exc}", flush=True)
            self.g_msg.content = f"**not loaded:** {exc}"
            if self.player is None:
                raise
            self.policy_name = self.g_pol.value = self.active
            return
        self.g_msg.content = ""
        self.player, self.active = player, self.policy_name

    def _action(self, obs: dict) -> torch.Tensor:
        """A student reads the env's ``policy`` vector (and ``teacher`` for its aux estimate); a keypoint
        policy the keypoint vector, which is ``teacher`` when the env builds a point cloud."""
        if self.student() is not None:
            return self.player.get_action(obs["policy"], deterministic=self.deterministic, teacher=obs.get("teacher"))
        return self.player.get_action(obs.get("teacher", obs["policy"]), deterministic=self.deterministic)

    def run(self) -> None:
        env, inner = self.env, self.inner
        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((self.n_env, inner.cfg.action_space), device=self.device))
        self._use_policy()
        dt = float(inner.step_dt)
        t_last = t_fps = t_log = time.perf_counter()
        n_steps = 0
        while True:
            if self.policy_name != self.active:
                self._use_policy()
                if self.policy_name == self.active:
                    self.reset_all = self.rebuild = True  # the sheet is translucent under a student only
            if self.rebuild:
                self._rebuild()
                self._draw()  # now, not at the next step: the view may be paused
            if self.reset_all:
                self.reset_all = False
                obs, _ = env.reset()
                self.player.reset_rnn()
                self.best_ep = [float("inf")] * self.n_env
                self._draw()
            if self.reset_one:
                self.reset_one = False
                inner.episode_length_buf[self.world] = inner.max_episode_length  # truncates at the next step

            if self.playing or self.step_once:
                self.step_once = False
                _t0 = time.perf_counter()
                _act = self._action(obs).to(self.device)
                _t1 = time.perf_counter()
                obs, _r, terminated, truncated, _x = env.step(_act)
                _t2 = time.perf_counter()
                ended = terminated.bool() | truncated.bool()
                fe = inner.fold_error().detach().cpu().tolist()
                steps_now = inner.episode_length_buf.detach().cpu().tolist()
                for e, done in enumerate(ended.cpu().tolist()):
                    if done:  # the state read now is the next episode's
                        self.best_last[e] = self.best_ep[e]
                        self.best_ep[e], self.best_ep_step[e] = float("inf"), 0
                    elif fe[e] < self.best_ep[e]:
                        self.best_ep[e], self.best_ep_step[e] = fe[e], int(steps_now[e])
                        self.best_all = min(self.best_all, fe[e])
                if bool(ended.any()):
                    self.stats.record(inner.task.termination_reasons, ended, truncated)
                    self.player.reset_rnn(ended.nonzero(as_tuple=True)[0])
                for f in self.features:
                    f.after_step(ended)
                _t3 = time.perf_counter()
                n_steps += 1
                self._draw()
                self._maybe_snap()
                _t4 = time.perf_counter()
                if os.environ.get("LIVE_PROFILE"):
                    self._prof = getattr(self, "_prof", [0.0] * 4)
                    self._prof_n = getattr(self, "_prof_n", 0) + 1
                    for k, d in enumerate((_t1 - _t0, _t2 - _t1, _t3 - _t2, _t4 - _t3)):
                        self._prof[k] += d
                    if self._prof_n % 10 == 0:
                        print("[live] profile per step (ms): policy %.0f | env.step %.0f | features %.0f | draw %.0f"
                              % tuple(1000 * x / 10 for x in self._prof), flush=True)
                        self._prof = [0.0] * 4
            else:
                time.sleep(0.03)

            now = time.perf_counter()
            if now - t_fps > 0.5:
                self.fps = n_steps / (now - t_fps)
                self.g_hud.content = self._hud()
                if now - t_log > 15.0:
                    t_log = now
                    print(f"[live] {self.fps:.0f} steps/s, outcomes {dict(self.stats.outcomes)}, "
                          f"{len(self.server.get_clients())} client(s)", flush=True)
                n_steps, t_fps = 0, now
            if self.speed < 3.95:
                wait = dt / self.speed - (time.perf_counter() - t_last)
                if wait > 0:
                    time.sleep(wait)
            t_last = time.perf_counter()
