"""``LiveViewer``: the env, the policy and the browser, in one loop.

Threads: viser runs the GUI callbacks on its own thread. They only set fields here (``playing``,
``world``, ``reset_all``, ...); the loop, on the main thread, acts on them, so the env and the scene
are only ever touched by the loop.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import torch

from .feature import Feature
from .fold_pool import FoldPool
from .overlays import GoalGhost, Keypoints
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
        self.playing, self.step_once, self.speed, self.deterministic = True, False, 1.0, True
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

        self.pool = FoldPool(self) if rigid else None
        self.pc = PointCloud(self) if inner.observations.pointcloud is not None else None
        self.features: list[Feature] = [f for f in (self.pool, Keypoints(self)) if f is not None]
        if rigid and str(inner.cfg.fold_pool.goal_task.mode) != "off":
            self.features.append(GoalGhost(self))
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

    def object_opacity(self) -> float | None:
        """The sheet / plates are translucent under a student, so the cloud on and under them shows."""
        return self.pc.opacity() if self.pc is not None and self.student() is not None else None

    def env_label(self, w: int) -> str:
        return self.pool.env_label(w) if self.pool is not None else f"env {w}"

    # ------------------------------------------------------------------ GUI
    def _gui(self) -> None:
        gui, inner = self.server.gui, self.inner
        with gui.add_folder("Scene"):
            g_env = gui.add_dropdown("show env", [self.env_label(w) for w in range(self.n_env)])
            g_start = gui.add_dropdown("start pose", list(self.reset_starts))
            g_tol = gui.add_slider("fold tolerance [mm]", 5.0, 60.0, 0.5, float(inner.cfg.cloth.keypoint_tolerance) * 1e3)
            g_reset = gui.add_button("reset this env")
            g_reset_all = gui.add_button("reset all envs")
            for f in self.features:
                f.scene_gui()
        with gui.add_folder("Policy"):
            self.g_pol = g_pol = gui.add_dropdown("policy", list(self.players.menu), initial_value=self.policy_name)
            g_det = gui.add_checkbox("deterministic actions", True)
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

        g_tol.on_update(lambda _e: setattr(inner.cfg.cloth, "keypoint_tolerance", float(g_tol.value) * 1e-3))
        g_reset.on_click(lambda _e: setattr(self, "reset_one", True))
        g_reset_all.on_click(lambda _e: setattr(self, "reset_all", True))
        g_pol.on_update(lambda _e: setattr(self, "policy_name", g_pol.value))
        g_det.on_update(lambda _e: setattr(self, "deterministic", bool(g_det.value)))
        self.g_play.on_update(lambda _e: setattr(self, "playing", bool(self.g_play.value)))
        g_step.on_click(lambda _e: setattr(self, "step_once", True))
        g_speed.on_update(lambda _e: setattr(self, "speed", float(g_speed.value)))
        self.server.on_client_connect(self._start_view)
        self.server.scene.add_grid("/ground", width=3.0, height=3.0, cell_size=0.1, position=(0, 0, 0.0))

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
        opacity = self.object_opacity()
        for i, s in enumerate(shapes):
            moving = s.key is not None
            self.server.scene.add_mesh_simple(
                f"{names[s.key] if moving else '/w/static'}/s{i}", s.verts, s.faces, color=s.rgb,
                position=s.pos if moving else s.pos - self.origins[w], wxyz=s.wxyz,
                side="double" if moving and s.kind != "robot" else "front",
                opacity=opacity if s.kind == "object" else None)

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
                self._draw()
            if self.reset_one:
                self.reset_one = False
                inner.episode_length_buf[self.world] = inner.max_episode_length  # truncates at the next step

            if self.playing or self.step_once:
                self.step_once = False
                obs, _r, terminated, truncated, _x = env.step(self._action(obs).to(self.device))
                ended = terminated.bool() | truncated.bool()
                if bool(ended.any()):
                    self.stats.record(inner.task.termination_reasons, ended, truncated)
                    self.player.reset_rnn(ended.nonzero(as_tuple=True)[0])
                for f in self.features:
                    f.after_step(ended)
                n_steps += 1
                self._draw()
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
