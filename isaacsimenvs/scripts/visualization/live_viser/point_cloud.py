"""The env's point cloud (``obs.object: pointcloud``): what a student reads, and the cameras' errors.

Built only when the env builds a cloud; it draws while the active policy is a student. The cloud is the
one the env built at the latest step, in the env frame, over the true sheet (made translucent). The
noise switches set the env's camera errors (drawn per episode, so a change resets the envs); at start
they show the env's own (all off when pinned, as the pinned evaluation).
"""

from __future__ import annotations

import numpy as np

from .feature import Feature

#: by label (``PointCloudSetup.labels``): live points, first-frame reference moving / stationary half,
#: the goal sheet (goal tasks), forward kinematics
LABEL_RGB = {"live": (245, 245, 245), "ref_moving": (255, 150, 200), "ref_stationary": (140, 195, 255),
             "goal": (255, 150, 200), "fk": (40, 40, 40)}
CAM_RGB = np.array([(0, 190, 215), (235, 60, 200)], np.uint8)   # camera A, B
NO_CAM_RGB = np.array((120, 120, 120), np.uint8)
AUX_KP_RGB, AUX_GOAL_RGB = (255, 0, 200), (0, 120, 110)
TRUE_RGB = (90, 90, 90)

#: noise switch -> (label, the camera-error fields it sets: ``domain_randomization.pointcloud``)
NOISE = {
    "cams": ("camera position +-5 cm", ("cam_pos_noise",)),
    "calib": ("calibration: each camera +-1.5 cm, +-2 deg", ("calib_xyz", "calib_rot_deg")),
    "depth": ("depth noise 1-4 mm", ("depth_noise_std_range",)),
    "drop": ("random drop 0-30%", ("drop_prob_range",)),
}


def _t(v):
    return tuple(float(x) for x in v) if isinstance(v, (tuple, list)) else float(v)


class PointCloud(Feature):
    def __init__(self, viewer) -> None:
        super().__init__(viewer)
        from isaacsimenvs.tasks.recipes import dr_block

        self.obs = viewer.inner.observations.pointcloud
        self.cloud = self.obs.cloud
        fields = [f for _label, fs in NOISE.values() for f in fs]
        #: per field: off (the DR group's default), on (the pointcloud DR block's value)
        self.off = {f: getattr(type(viewer.inner.cfg.domain_randomization.pointcloud)(), f) for f in fields}
        self.on = dr_block("pointcloud")["domain_randomization"]["pointcloud"]
        self._cam_key = None
        self._syncing = False  # the boxes are being set by the viewer, not by the user

    def opacity(self) -> float | None:
        """The true sheet's opacity under a student (None: opaque)."""
        a = float(self.g_opacity.value)
        return a if a < 0.995 else None

    def gui(self) -> None:
        gui = self.v.server.gui
        with gui.add_folder("Point cloud (student policies)"):
            self.g_colour = gui.add_dropdown("colour points by", ["label", "camera (A cyan, B magenta, none grey)"])
            self.g_show = {
                "live": gui.add_checkbox("live points: cloth, hand, arm (white)", False),
                "fk": gui.add_checkbox("forward-kinematics points: forearm, hand (black)", True),
                "ref": gui.add_checkbox("goal / first-frame cloud (pink moving, blue stationary)", False),
            }
            self.g_aux = gui.add_checkbox("aux head: keypoints (magenta), goal (teal), lines to truth", False)
            self.g_cams = gui.add_checkbox("cameras", True)
            self.g_true = gui.add_checkbox("true sheet particles (grey)", False)
            self.g_opacity = gui.add_slider("true sheet opacity", 0.05, 1.0, 0.05, 1.0)
            self.g_size = gui.add_slider("point size [mm]", 1.0, 10.0, 0.5, 4.0)
        with gui.add_folder("Point-cloud noise (a change resets all envs)"):
            setup = self.obs.setup
            self.g_noise = {k: gui.add_checkbox(label, any(_t(getattr(setup, f)) != _t(self.off[f]) for f in fs))
                            for k, (label, fs) in NOISE.items()}
            b_trained = gui.add_button("as the student was trained")
            b_clean = gui.add_button("all off")
            b_all = gui.add_button("all on")

        @self.g_opacity.on_update
        def _(_e):
            self.v.rebuild = True  # the plates carry their opacity from the build

        for box in self.g_noise.values():
            box.on_update(lambda _e: self._from_boxes())
        b_trained.on_click(lambda _e: self._as_trained())
        b_clean.on_click(lambda _e: self._set_boxes(False))
        b_all.on_click(lambda _e: self._set_boxes(True))

    def _write(self, values: dict) -> None:
        """Camera-error fields -> the env's cloud; the envs reset if anything changed."""
        setup, changed = self.obs.setup, False
        for f, v in values.items():
            v = tuple(v) if isinstance(v, list) else v
            if _t(getattr(setup, f)) != _t(v):
                setattr(setup, f, v)
                changed = True
        if changed:
            self.v.reset_all = True

    def _from_boxes(self) -> None:
        if not self._syncing:
            self._write({f: (self.on if g.value else self.off)[f]
                         for k, g in self.g_noise.items() for f in NOISE[k][1]})

    def _set_boxes(self, on: bool) -> None:
        for g in self.g_noise.values():
            g.value = on  # each box writes through _from_boxes

    def _as_trained(self) -> None:
        """The active student's own camera errors (its checkpoint's DR), box by box."""
        pl = self.v.student()
        if pl is None:
            return
        trained = pl.env["domain_randomization"]["pointcloud"]
        values = {f: trained.get(f, self.off[f]) for _label, fs in NOISE.values() for f in fs}
        self._syncing = True
        for k, (_label, fs) in NOISE.items():
            self.g_noise[k].value = any(_t(values[f]) != _t(self.off[f]) for f in fs)
        self._syncing = False
        self._write(values)

    def clear(self) -> None:
        super().clear()
        self._cam_key = None

    def draw(self, w: int) -> None:
        pl = self.v.student()
        if pl is None or pl.last is None or not hasattr(self.cloud, "last_xyz"):
            for h in self.nodes.values():
                h.visible = False
            return
        pc, o, inner = self.cloud, self.v.origins[w], self.v.inner
        xyz = pc.last_xyz[w].detach().float().cpu().numpy()  # env frame
        label = np.array(pc.labels or ("live", "goal"))[pc.last_label[w].cpu().numpy()]
        cam = pc.last_cam[w].cpu().numpy()
        ok = pc.last_mask[w].cpu().numpy() & np.isfinite(xyz).all(axis=1)
        if self.g_colour.value == "label":
            rgb = np.array([LABEL_RGB[k] for k in label], np.uint8).reshape(-1, 3)
        else:
            rgb = np.where((cam >= 0)[:, None], CAM_RGB[np.clip(cam, 0, 1)], NO_CAM_RGB)
        size = float(self.g_size.value) * 1e-3
        groups = {"live": label == "live", "fk": label == "fk",
                  "ref": np.isin(label, ("ref_moving", "ref_stationary", "goal"))}
        for name, sel in groups.items():
            sel = sel & ok & bool(self.g_show[name].value)
            self.points(f"pc_{name}", xyz[sel], rgb[sel], size)

        true = inner.cloth_points_w()[w].detach().cpu().numpy() - o if self.g_true.value else ()
        self.points("pc_true", true, TRUE_RGB, 0.003)

        self._draw_cameras(pc, w)

        if self.g_aux.value:
            # aux = [keypoints - palm, keypoints - goal] * scale; its error is in the env frame, so the
            # estimates are the truth moved by it.
            d = ((pl.last["aux_pred"][w] - pl.last["aux_true"][w]) / pl.scale).view(2, -1, 3).cpu().numpy()
            kp = inner.cloth_keypoints_w()[w].detach().cpu().numpy() - o
            goal = inner.fold_targets_w()[w].detach().cpu().numpy() - o
            kp_est, goal_est = kp + d[0], goal + d[0] - d[1]
            self.points("pc_aux_kp", kp_est, AUX_KP_RGB, 0.006)
            self.points("pc_aux_goal", goal_est, AUX_GOAL_RGB, 0.006)
            segs = np.concatenate((np.stack((kp, kp_est), 1), np.stack((goal, goal_est), 1)))
            rgb = np.concatenate((np.tile(AUX_KP_RGB, (len(kp), 2, 1)), np.tile(AUX_GOAL_RGB, (len(goal), 2, 1))))
            self.lines("pc_aux_lines", segs, rgb, 0.002)
        else:
            self.hide("pc_aux_kp", "pc_aux_goal", "pc_aux_lines")

    def _draw_cameras(self, pc, w: int) -> None:
        names = [f"pc_cam{c}" for c in range(2)] + [f"pc_cam{c}_name" for c in range(2)]
        if not self.g_cams.value:
            self.hide(*names)
            return
        pos = pc.cams[w].cpu().numpy()
        ctr = (pc.centre if pc.centre.dim() == 1 else pc.centre[w]).cpu().numpy()
        key = (w, pos.round(4).tobytes(), ctr.round(3).tobytes())
        if self._cam_key != key:  # the camera positions are drawn per episode
            self._cam_key = key
            scene = self.v.server.scene
            for c in range(2):
                self.nodes[f"pc_cam{c}"] = scene.add_camera_frustum(
                    f"/w/pc_cam{c}", fov=0.9, aspect=1.33, scale=0.08, color=tuple(int(x) for x in CAM_RGB[c]),
                    wxyz=_look_at(pos[c], ctr), position=pos[c])
                self.nodes[f"pc_cam{c}_name"] = scene.add_label(f"/w/pc_cam{c}_name", f"camera {'AB'[c]}", position=pos[c])
        for n in names:
            self.nodes[n].visible = True

    def hud(self, w: int) -> list[tuple[str, str]]:
        pl = self.v.student()
        if pl is None or pl.last is None or not hasattr(self.cloud, "last_xyz"):
            return []
        pc = self.cloud
        e = ((pl.last["aux_pred"][w] - pl.last["aux_true"][w]) / pl.scale).view(2, -1, 3).norm(dim=-1).mean(dim=-1) * 100
        shift = pc.calib_t[w].norm(dim=-1) * 1e3
        return [
            ("point cloud", f"**{int(pc.last_count[w])} / {pc.cfg.n_points}** live points, "
                            f"cloth {float(pc.last_cloth_share[w]) * 100:.0f}%, "
                            f"{float(pc.last_visible_frac[w]) * 100:.0f}% of the sheet seen "
                            f"(goal input {pc.cfg.goal_mode}, fk points {pc.cfg.fk_points})"),
            ("aux head error", f"keypoints **{float(e[0]):.1f} cm**, keypoints - goal **{float(e[1]):.1f} cm**"),
            ("cloud noise", f"calibration A {float(shift[0]):.0f} / B {float(shift[1]):.0f} mm, "
                            f"depth {float(pc.depth_std[w]) * 1e3:.1f} mm, drop {float(pc.drop_p[w]) * 100:.0f}%"),
            ("delays", f"cloud {int(pc.lag[w])} steps, proprio {int(self.v.inner.dr.proprio_lag[w])} steps"),
        ]


def _look_at(pos, target):
    """wxyz of a camera at ``pos`` (OpenCV axes: +z forward, +y down) looking at ``target``."""
    import viser.transforms as vtf

    z = (target - pos) / max(float(np.linalg.norm(target - pos)), 1e-9)
    x = np.cross(z, (0.0, 0.0, 1.0))
    x = x / max(float(np.linalg.norm(x)), 1e-9)
    return vtf.SO3.from_matrix(np.stack((x, np.cross(z, x), z), axis=1)).wxyz
