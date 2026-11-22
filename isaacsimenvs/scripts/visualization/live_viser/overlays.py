"""Overlays for any policy: what the fold criterion measures."""

from __future__ import annotations

import numpy as np

from .feature import Feature

GOAL_RGB = (40, 200, 90)
KP_RGB = (255, 165, 25)
GHOST_RGB = (40, 200, 90)
GHOST_POINT_RGB = (20, 150, 60)


class Keypoints(Feature):
    """The sheet's keypoints (orange) and their fold targets (green): the fold error is their distance."""

    def scene_gui(self) -> None:
        self.show = self.v.server.gui.add_checkbox("keypoints (orange) / fold targets (green)", True)

    def draw(self, w: int) -> None:
        if not self.show.value:
            self.hide("goal_kp", "kp")
            return
        o, inner = self.v.origins[w], self.v.inner
        self.points("goal_kp", inner.fold_targets_w()[w].detach().cpu().numpy() - o, GOAL_RGB, 0.007)
        self.points("kp", inner.cloth_keypoints_w()[w].detach().cpu().numpy() - o, KP_RGB, 0.005)


class GoalGhost(Feature):
    """Goal tasks (``fold_pool.goal_task``): a green ghost of the goal state the fold criterion matches
    best now (the task's own goal, or a look-alike with ``accept_lookalike``), and the task in the HUD."""

    def __init__(self, viewer) -> None:
        super().__init__(viewer)
        self._faces = None

    def scene_gui(self) -> None:
        self.show = self.v.server.gui.add_checkbox("goal tasks: matched goal state (green ghost)", True)

    def _info(self, w: int):
        geo = self.v.inner.task.geometry
        gt = geo.goal_tasks
        if gt is None or gt.task is None:
            return None
        view = self.v.inner.fold_view()
        grid, lookalike = geo.matched_goal_sheet(view)
        return {"name": gt.names[int(gt.task[w])], "des": float(view.designated_error[w]),
                "alt": float(view.lookalike_error[w]), "lookalike": bool(lookalike[w]),
                "grid": grid[w].detach().cpu().numpy()}

    def draw(self, w: int) -> None:
        info = self._info(w) if self.show.value else None
        if info is None:
            self.hide("ghost", "ghost_pts")
            return
        grid = info["grid"] - self.v.origins[w]
        if self._faces is None or self._faces[0] != len(grid):  # the sheet's square grid, two triangles a cell
            r = int(round(len(grid) ** 0.5))
            ij = np.arange(r * r).reshape(r, r)
            a, b, c, d = ij[:-1, :-1].ravel(), ij[:-1, 1:].ravel(), ij[1:, :-1].ravel(), ij[1:, 1:].ravel()
            self._faces = (len(grid), np.concatenate((np.stack((a, b, d), 1), np.stack((a, d, c), 1))))
        self.mesh("ghost", grid, self._faces[1], GHOST_RGB, opacity=0.35)
        self.points("ghost_pts", grid, GHOST_POINT_RGB, 0.004)

    def hud(self, w: int) -> list[tuple[str, str]]:
        info = self._info(w)
        if info is None:
            return []
        alt = f"{info['alt'] * 1e3:.1f} mm" if np.isfinite(info["alt"]) else "none"
        matched = "look-alike" if info["lookalike"] else "designated"
        return [("goal task", f"**{info['name']}**, matched **{matched}** "
                              f"(designated {info['des'] * 1e3:.1f} mm, look-alike {alt})")]
