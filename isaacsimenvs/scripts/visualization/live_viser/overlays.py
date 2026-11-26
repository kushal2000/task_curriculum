"""Overlays for any policy: what the fold criterion measures."""

from __future__ import annotations

import numpy as np

from .feature import Feature

GOAL_RGB = (40, 200, 90)
KP_RGB = (255, 165, 25)
GHOST_RGB = (40, 200, 90)
GHOST_POINT_RGB = (20, 150, 60)
FROZEN_GOAL_RGB = (30, 170, 210)
FROZEN_GOAL_POINT_RGB = (15, 110, 150)


def sheet_grid_faces(num_points: int) -> np.ndarray:
    """``(F, 3)`` triangles of a square, row-major particle grid of ``num_points``: two per cell."""
    side = int(round(num_points ** 0.5))
    ij = np.arange(side * side).reshape(side, side)
    a, b, c, d = ij[:-1, :-1].ravel(), ij[:-1, 1:].ravel(), ij[1:, :-1].ravel(), ij[1:, 1:].ravel()
    return np.concatenate((np.stack((a, b, d), 1), np.stack((a, d, c), 1)))


class Keypoints(Feature):
    """The sheet's keypoints (orange) and their fold targets (green): the fold error is their distance."""

    def scene_gui(self) -> None:
        self.show = self.v.server.gui.add_checkbox("keypoints (orange) / fold targets (green)", False)

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
        if self._faces is None or self._faces[0] != len(grid):
            self._faces = (len(grid), sheet_grid_faces(len(grid)))
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


class SurfaceGoal(Feature):
    """The surface criterion (``termination.success: surface``, ``mdp/surface_match.py``): the frozen goal
    sheet the criterion compares against (blue ghost, from ``surface.ref_step`` on; before that the live
    goal), and in the HUD this step's distances to it -- after the in-plane placement, layered
    (``surface.layered``, ``mdp/surface_layered.py``) and unlabelled, the criterion's against its tolerance,
    and where it is (the in-place gate's)."""

    def __init__(self, viewer) -> None:
        super().__init__(viewer)
        self._faces = None

    def scene_gui(self) -> None:
        self.show = self.v.server.gui.add_checkbox("surface criterion: frozen goal sheet (blue ghost)", True)

    def draw(self, w: int) -> None:
        if not self.show.value:
            self.hide("frozen_goal", "frozen_goal_pts")
            return
        task = self.v.inner.task
        goal_w = task.goal_reference_w(self.v.inner.fold_view())[w].detach().cpu().numpy()
        goal = goal_w - self.v.origins[w]
        if self._faces is None or self._faces[0] != len(goal):
            self._faces = (len(goal), sheet_grid_faces(len(goal)))
        self.mesh("frozen_goal", goal, self._faces[1], FROZEN_GOAL_RGB, opacity=0.35)
        self.points("frozen_goal_pts", goal, FROZEN_GOAL_POINT_RGB, 0.004)

    def hud(self, w: int) -> list[tuple[str, str]]:
        task = self.v.inner.task
        if not bool(task.goal_is_frozen[w]):
            return [("surface", f"goal freezes at step {int(task.cfg.termination.surface.ref_step)}")]
        tolerance_mm = float(task.surface_tolerance()[w]) * 1e3
        criterion_mm = float(task.surface_dist[w]) * 1e3
        aligned_mm = float(task.aligned_dist[w]) * 1e3
        in_place_mm = float(task.in_place_dist[w]) * 1e3
        verdict = "**within**" if criterion_mm < tolerance_mm else "outside"
        if task.surface_layered:
            return [("surface", f"layered **{criterion_mm:.1f} mm** ({verdict} {tolerance_mm:.1f} mm), "
                                f"unlabelled {aligned_mm:.1f} mm, in place {in_place_mm:.1f} mm")]
        return [("surface", f"aligned **{criterion_mm:.1f} mm** ({verdict} {tolerance_mm:.1f} mm), "
                            f"in place {in_place_mm:.1f} mm")]
