"""A point-cloud student (``distill_pc.py``) behind the player interface ``eval/live_viser.py`` drives.

The checkpoint carries its own point-cloud settings (``pc_args``) and network settings, so the
student sees the input it was trained on. ``last`` keeps what the network read and predicted at the
latest step, for drawing; the cloud itself (env frame, labels, source camera) is on ``sobs.pc``.
"""

from __future__ import annotations

import torch

#: GUI noise switch -> the ``PointCloudCfg`` fields it turns on (version C values) or off.
NOISE_GROUPS = {
    "occl": ("cam_pos_noise", "margin_range"),
    "calib": ("calib_xyz", "calib_rot_deg"),
    "depth": ("depth_noise_std_range",),
    "drop": ("drop_prob_range",),
    "tip": ("tip_drop_prob",),
}


class StudentPlayer:
    is_student = True

    def __init__(self, inner, checkpoint_path: str, device) -> None:
        from isaacsimenvs.distill.distill_pc import StudentObs
        from isaacsimenvs.distill.student import StudentNet
        from isaacsimenvs.tasks.cloth.utils.cloth_pointcloud import LABELS, PointCloudCfg, g3_noise_preset, noise_preset

        from isaacsimenvs.distill.pc_rl import load_student_any
        from isaacsimenvs.tasks.cloth.utils.scene_pointcloud import SceneCloudCfg, scene_noise_preset

        ckpt = load_student_any(checkpoint_path)             # a distilled student or an RL-trained one
        a = self.pc_args = ckpt["pc_args"]
        self.scene = a.get("cloud", "legacy") == "scene"
        if self.scene:                                       # everything in the box over the table, xyz only
            cfg = SceneCloudCfg(n_points=int(a["n_points"]), goal_mode=a["goal"], ref_step=int(a["ref_step"]),
                                fk_points=int(a["fk_points"]))
            if a["noise"]:
                scene_noise_preset(cfg)
            cfg.delay_max = int(a.get("pc_delay", 0))
            self._clean, self._full = SceneCloudCfg(), scene_noise_preset(SceneCloudCfg())
        else:
            cfg = PointCloudCfg(
                n_cloth=int(a["n_cloth"]), visibility=a["vis"], goal_mode=a["goal"],
                ref_step=int(a["ref_step"]), hand_cam=bool(a.get("hand_cam", False)),
            )
            if a["noise"]:
                noise_preset(cfg)
            if a.get("g3_noise"):
                g3_noise_preset(cfg)
            self._clean, self._full = PointCloudCfg(), noise_preset(PointCloudCfg())
        self.trained_noise = {k: bool(a["noise"]) for k in NOISE_GROUPS}
        self.n, self.device = inner.num_envs, device
        self.sobs = StudentObs(inner, cfg, proprio_delay=int(a.get("proprio_delay", 0)))
        self.net = StudentNet(
            int(self.sobs.proprio_cols.numel()), int(inner.cfg.action_space), int(self.sobs.pc.point_dim),
            int(self.sobs.aux_cols.numel()), **ckpt["net_args"],
        ).to(device).eval()
        self.net.load_state_dict(ckpt["model"], strict=not ckpt.get("from_rl", False))
        self.amp = str(device).startswith("cuda")
        self.state = self.net.initial_state(self.n, device)
        self.start = torch.ones(self.n, dtype=torch.bool, device=device)
        self.expl_coef = 0.0  # (player interface; a student has no SAPG blocks)
        self.last = None
        self.reset_rnn()
        print(f"[live] student <- {checkpoint_path} (iter {ckpt.get('iter')}, pc_args {a})", flush=True)

    # ------------------------------------------------------------------ noise switches

    def noise_state(self) -> dict[str, bool]:
        cfg = self.sobs.pc.cfg
        return {k: any(getattr(cfg, f, None) != getattr(self._clean, f, None) for f in fs) for k, fs in NOISE_GROUPS.items()}

    def set_noise(self, on: dict[str, bool]) -> bool:
        """Version C values for the groups that are on, none for the rest. True if anything changed
        (the per-episode draws are made at reset, so the caller resets the envs)."""
        cfg, changed = self.sobs.pc.cfg, False
        for k, fields in NOISE_GROUPS.items():
            src = self._full if on[k] else self._clean
            for f in fields:
                if not hasattr(cfg, f):          # the scene cloud has no capsule margin / fingertip hole
                    continue
                if getattr(cfg, f) != getattr(src, f):
                    setattr(cfg, f, getattr(src, f))
                    changed = True
        return changed

    # ------------------------------------------------------------------ player interface

    def reset_rnn(self, env_ids=None) -> None:
        """Call after the envs' reset: draws their episode's point-cloud noise and drops their state."""
        ids = torch.arange(self.n, device=self.device) if env_ids is None else env_ids
        self.sobs.reset(ids)
        self.start[ids] = True

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        so = self.sobs(obs)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            mu, aux, self.state = self.net.step(so["proprio"], so["points"], so["mask"], self.state, self.start)
        self.start = torch.zeros_like(self.start)
        self.last = {"aux_true": so["aux"], "aux_pred": aux.float()}
        return torch.nan_to_num(mu.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
