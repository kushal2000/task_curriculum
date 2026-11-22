"""The VBD task on Newton: the cloth sheet and fingers through it."""

from __future__ import annotations

import numpy as np
import torch

from .feature import Feature

MOVING_RGB = (242, 77, 26)       # the half that folds over
STATIONARY_RGB = (250, 199, 38)  # the half it folds onto
PIERCE_RGB = (255, 0, 0)


def _state():
    from isaaclab_newton.physics.newton_manager import NewtonManager

    return NewtonManager._state_0


class ClothSheet(Feature):
    """The sheet's triangles in two colours, moving and stationary half (translucent under a student)."""

    def __init__(self, viewer) -> None:
        super().__init__(viewer)
        from isaaclab_newton.physics.newton_manager import NewtonManager

        model = NewtonManager.get_model()
        owner = model.particle_world.numpy()
        tri = model.tri_indices.numpy()
        moving = viewer.inner.task.geometry.moving_idx.cpu().numpy()
        self.by_world = {}  # w -> (its particle indices, moving-half triangles, stationary-half triangles)
        for w in range(viewer.n_env):
            pidx = np.nonzero(owner == w)[0]
            t = tri[(owner[tri] == w).all(axis=1)] - int(pidx[0])
            in_moving = np.isin(t, moving).sum(axis=1) >= 2
            self.by_world[w] = (torch.as_tensor(pidx, device=viewer.device), t[in_moving], t[~in_moving])

    def draw(self, w: int) -> None:
        import warp as wp

        pidx, t_mv, t_st = self.by_world[w]
        pts = wp.to_torch(_state().particle_q)[pidx].cpu().numpy() - self.v.origins[w]
        for name, t, rgb in (("cloth_moving", t_mv, MOVING_RGB), ("cloth_stationary", t_st, STATIONARY_RGB)):
            if len(t):
                self.mesh(name, pts, t, rgb, opacity=self.v.object_opacity())


class Pierce(Feature):
    """Fingers through the cloth (particles are the sheet's only contact with the hand): each finger
    through it is a red line along its axis, and the HUD counts the steps it has been through."""

    def __init__(self, viewer) -> None:
        super().__init__(viewer)
        from isaaclab_newton.physics.newton_manager import NewtonManager

        from isaacsimenvs.tasks.cloth.backends.vbd.pierce import PierceDetector

        inner = viewer.inner
        self.detector = PierceDetector(NewtonManager.get_model(), viewer.n_env, viewer.device,
                                       robot_spec=getattr(inner, "robot_spec", None))
        self.run = torch.zeros(viewer.n_env, len(self.detector.names), dtype=torch.long, device=viewer.device)
        self.last = None  # (hit (N, F), axis start, axis end)

    def after_step(self, ended) -> None:
        hit, a, b = self.detector(_state())
        hit = hit & ~ended[:, None]
        self.last = (hit, a, b)
        self.run = torch.where(hit, self.run + 1, torch.zeros_like(self.run))

    def draw(self, w: int) -> None:
        if self.last is None:
            return
        hit, a, b = self.last
        segs = torch.stack((a[w], b[w]), dim=1)[hit[w]].cpu().numpy() - self.v.origins[w]
        self.lines("pierce", segs, PIERCE_RGB, 0.004)

    def hud(self, w: int) -> list[tuple[str, str]]:
        through = [f"{name.replace('left_', '')} ({k} steps)"
                   for name, k in zip(self.detector.names, self.run[w].tolist()) if k > 0]
        return [("fingers through cloth", "**" + ", ".join(through) + "** (red)" if through else "none")]
