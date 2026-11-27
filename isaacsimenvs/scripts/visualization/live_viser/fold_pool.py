"""The rigid sheet's fold-shape pool: the shape at each env's next reset, and evenly spaced sheet sizes."""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from .feature import Feature


@contextlib.contextmanager
def even_env_sizes():
    """While the env is built: per-env sheet sizes (and slabs) spaced evenly over their ranges instead
    of drawn, so env 0 is the smallest and the last env the largest (then quantised as usual)."""
    from isaacsimenvs.tasks.cloth.backends.rigid import scene

    drawn = scene.draw_env_sizes_raw

    def even(cfg, n: int):
        p = cfg.fold_pool
        return np.linspace(*map(float, p.size_range), n), np.linspace(*map(float, p.thickness_range), n)

    scene.draw_env_sizes_raw = even
    try:
        yield
    finally:
        scene.draw_env_sizes_raw = drawn


class FoldPool(Feature):
    """Which shape each env runs from its next reset: "random" (the training draw) or one shape.

    Wraps the backend's ``draw_episode`` (its documented viewer hook): the random draw first, then the
    envs with a chosen shape are switched to it, and their folded start and goal task are re-drawn
    for that shape."""

    def __init__(self, viewer) -> None:
        super().__init__(viewer)
        from isaacsimenvs.tasks.cloth.backends.rigid.scene import draw_goal_tasks

        rb = self.rb = viewer.inner.backend
        self.names = [sh.name for sh in rb.shapes]
        self.sizes = rb.env_size_np
        carried = rb.pool.carried
        self.carried = None if carried is None or bool(carried.all()) else carried.cpu().numpy()
        self.choice = ["random"] * viewer.n_env
        import os

        if os.environ.get("LIVE_SHAPE") in self.names:  # the Design panel's new shape: every env runs it
            self.choice = [os.environ["LIVE_SHAPE"]] * viewer.n_env
        drawn = rb.draw_episode

        def draw_episode(env_ids: torch.Tensor) -> None:
            drawn(env_ids)
            forced = [(e, self.names.index(self.choice[e])) for e in env_ids.tolist() if self.choice[e] != "random"]
            forced = [(e, k) for e, k in forced if self.carried is None or self.carried[k, e]]
            if forced:
                ids = torch.tensor([e for e, _ in forced], device=viewer.device)
                shapes = torch.tensor([k for _, k in forced], device=viewer.device)
                rb.pool.select(ids, shapes)
                # The random draw drew the folded start and the goal task of another shape.
                if rb.start_tables is not None:
                    rb.draw_start_folds(ids, shapes)
                if rb.goal_tasks is not None:
                    draw_goal_tasks(rb, ids, shapes)

        rb.draw_episode = draw_episode

    def env_label(self, w: int) -> str:
        """The env dropdown's entry: env number and sheet size."""
        return f"env {w} ({self.sizes[w] * 100:.1f} cm)"

    def scene_gui(self) -> None:
        gui = self.v.server.gui
        self.g_shape = gui.add_dropdown("shape at next reset (this env)", ["random", *self.names])
        self.g_all = gui.add_button("this shape at next reset of ALL envs")

        @self.g_shape.on_update
        def _(_e):
            self.choice[self.v.world] = self.g_shape.value

        @self.g_all.on_click
        def _(_e):
            self.choice = [self.g_shape.value] * self.v.n_env

    def on_env(self, w: int) -> None:
        self.g_shape.value = self.choice[w]

    def hud(self, w: int) -> list[tuple[str, str]]:
        shape = self.names[int(self.rb.pool.active[w])]
        nxt = self.choice[w]
        return [("shape", f"**{shape}**, {self.sizes[w] * 100:.1f} cm" + (f" (next: {nxt})" if nxt != "random" else ""))]
