"""``Feature``: one optional part of the viewer, with its own GUI, layers and HUD lines.

The viewer calls, on every feature it built:
    scene_gui()       add widgets to the Scene panel             (once, at start)
    gui()             add its own panels                         (once, at start)
    on_env(w)         the drawn env changed to ``w``             (GUI thread)
    after_step(ended) after every env step; ``ended``: (N,) bool  (loop)
    draw(w)           update its layers for the drawn env ``w``  (loop)
    hud(w)            -> [(label, value)] status panel rows      (loop, twice a second)
    clear()           the scene was rebuilt: its layers are gone (loop)
"""

from __future__ import annotations

import numpy as np


class Feature:
    def __init__(self, viewer) -> None:
        self.v = viewer
        self.nodes: dict[str, object] = {}

    # -- hooks
    def scene_gui(self) -> None:
        pass

    def gui(self) -> None:
        pass

    def on_env(self, w: int) -> None:
        pass

    def after_step(self, ended) -> None:
        pass

    def draw(self, w: int) -> None:
        pass

    def hud(self, w: int) -> list[tuple[str, str]]:
        return []

    def clear(self) -> None:
        # Everything is drawn under /w, which a rebuild removes whole: only the handles are left.
        self.nodes.clear()

    # -- layers (env frame, under /w)
    def points(self, name: str, xyz, rgb, size: float) -> None:
        """A point cloud; ``rgb`` one colour or one per point. No points hides the layer."""
        xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
        if len(xyz) == 0:
            self.hide(name)
            return
        rgb = np.asarray(rgb, np.uint8)
        if rgb.ndim == 1:
            rgb = np.tile(rgb, (len(xyz), 1))
        self.nodes[name] = self.v.server.scene.add_point_cloud(
            f"/w/{name}", xyz, rgb, point_size=size, point_shape="circle", precision="float32")

    def lines(self, name: str, segments, rgb, thickness: float) -> None:
        """Line segments (S, 2, 3); ``rgb`` one colour or (S, 2, 3)."""
        segments = np.asarray(segments, np.float32).reshape(-1, 2, 3)
        if len(segments) == 0:
            self.hide(name)
            return
        rgb = np.asarray(rgb, np.uint8)
        if rgb.ndim == 1:
            rgb = np.tile(rgb, (len(segments), 2, 1))
        self.nodes[name] = self.v.server.scene.add_line_segments(f"/w/{name}", segments, rgb, thickness=thickness)

    def mesh(self, name: str, verts, faces, rgb, opacity: float | None = None) -> None:
        self.nodes[name] = self.v.server.scene.add_mesh_simple(
            f"/w/{name}", np.asarray(verts, np.float32), np.asarray(faces, np.int32), color=tuple(rgb),
            side="double", opacity=opacity)

    def hide(self, *names: str) -> None:
        for name in names:
            if name in self.nodes:
                self.nodes[name].visible = False
