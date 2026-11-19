"""A point-cloud student (``distill_pc.py``) behind the player interface ``live_viser.py`` drives.

The viewer's env reads keypoints (it also plays keypoint policies), so the player builds the
student's input itself, the way the env does with ``obs.object: pointcloud``: the proprioceptive
columns of the keypoint vector, delayed as the student trained (``domain_randomization.proprio``),
and its own ``PointCloudObservation`` with the student's cloud and camera errors (``student_env``,
saved in the checkpoint). ``last`` keeps what the network read and predicted at the latest step, for
drawing; the cloud itself (env frame, labels, source camera) is ``cloud``.
"""

from __future__ import annotations

import copy

import torch

#: GUI noise switch -> the camera-error fields it turns on (the pointcloud DR block's values) or off.
NOISE_GROUPS = {
    "cams": ("cam_pos_noise",),
    "calib": ("calib_xyz", "calib_rot_deg"),
    "depth": ("depth_noise_std_range",),
    "drop": ("drop_prob_range",),
}


class StudentPlayer:
    is_student = True

    def __init__(self, inner, checkpoint_path: str, device) -> None:
        from isaacsimenvs.scripts.distillation.pc_rl import load_student_any
        from isaacsimenvs.scripts.distillation.student import StudentNet, apply_student_env, split_columns
        from isaacsimenvs.tasks.cloth.mdp.delay import DelayLine
        from isaacsimenvs.tasks.cloth.mdp.observations import AUX_FIELDS, OBJECT_FIELDS, obs_dim
        from isaacsimenvs.tasks.cloth.perception.observation import PointCloudObservation
        from isaacsimenvs.tasks.recipes import dr_block

        ckpt = load_student_any(checkpoint_path)             # a distilled student or an RL-trained one
        self.inner, self.n, self.device = inner, inner.num_envs, device
        cfg = copy.deepcopy(inner.cfg)                       # the student's view of this env
        apply_student_env(cfg, ckpt["env"], with_dr=True)
        self.proprio_delay_max = int(cfg.domain_randomization.proprio.delay_max)
        self.pc = PointCloudObservation(cfg, inner.robot_layout, num_envs=self.n, device=device,
                                        proprio_delay_max=self.proprio_delay_max)
        self.cloud = self.pc.cloud
        self.scale = float(self.pc.setup.scale)
        dr_pc = cfg.domain_randomization.pointcloud
        self._full = dr_block("pointcloud")["domain_randomization"]["pointcloud"]
        self._clean = {f: getattr(type(dr_pc)(), f) for fs in NOISE_GROUPS.values() for f in fs}
        self.trained_noise = {k: any(_t(getattr(dr_pc, f)) != _t(self._clean[f]) for f in fs) for k, fs in NOISE_GROUPS.items()}

        obs_list = list(inner.cfg.obs.obs_list)
        num_joints = int(inner.cfg.action_space)
        sizes = {f: obs_dim([f], num_joints) for f in obs_list}
        proprio, aux = split_columns(obs_list, sizes, OBJECT_FIELDS, AUX_FIELDS)
        self.proprio_cols = torch.tensor(proprio, dtype=torch.long, device=device)
        self.aux_cols = torch.tensor(aux, dtype=torch.long, device=device)
        self.p_delay = DelayLine(self.n, self.proprio_delay_max, len(proprio), device)
        self.p_lag = torch.zeros(self.n, dtype=torch.long, device=device)

        self.net = StudentNet(len(proprio), num_joints, self.pc.point_dim, len(aux), **ckpt["net_args"]).to(device).eval()
        self.net.load_state_dict(ckpt["model"], strict=not ckpt.get("from_rl", False))
        self.amp = str(device).startswith("cuda")
        self.state = self.net.initial_state(self.n, device)
        self.start = torch.ones(self.n, dtype=torch.bool, device=device)
        self.expl_coef = 0.0  # (player interface; a student has no SAPG blocks)
        self.last = None
        self.reset_rnn()
        print(f"[live] student <- {checkpoint_path} (iter {ckpt.get('iter')}, env {ckpt['env']})", flush=True)

    # ------------------------------------------------------------------ noise switches

    def noise_state(self) -> dict[str, bool]:
        setup = self.pc.setup
        return {k: any(_t(getattr(setup, f)) != _t(self._clean[f]) for f in fs) for k, fs in NOISE_GROUPS.items()}

    def set_noise(self, on: dict[str, bool]) -> bool:
        """The pointcloud block's values for the groups that are on, none for the rest. True if
        anything changed (the per-episode draws are made at reset, so the caller resets the envs)."""
        setup, changed = self.pc.setup, False
        for k, fields in NOISE_GROUPS.items():
            for f in fields:
                value = self._full[f] if on[k] else self._clean[f]
                value = tuple(value) if isinstance(value, list) else value
                if _t(getattr(setup, f)) != _t(value):
                    setattr(setup, f, value)
                    changed = True
        return changed

    # ------------------------------------------------------------------ player interface

    def reset_rnn(self, env_ids=None) -> None:
        """Call after the envs' reset: their next observation draws the episode's camera errors and
        proprioception lag; their network state is dropped."""
        ids = torch.arange(self.n, device=self.device) if env_ids is None else env_ids
        self.p_lag[ids] = torch.randint(0, self.proprio_delay_max + 1, (int(ids.numel()),), device=self.device)
        self.start[ids] = True

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        inner = self.inner
        state = inner.sim_state()
        view = inner.fold_view(state)
        proprio = torch.nan_to_num(obs[:, self.proprio_cols], nan=0.0, posinf=0.0, neginf=0.0)
        proprio = self.p_delay.push_and_read(proprio, self.start, self.p_lag)
        points, mask = self.pc.observe(state, view, inner.episode_length_buf, self.start, self.p_lag,
                                       inner.scene.env_origins, inner.task.geometry.goal_sheet)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            mu, aux, self.state = self.net.step(proprio, points, mask, self.state, self.start)
        self.start = torch.zeros_like(self.start)
        aux_true = torch.nan_to_num(obs[:, self.aux_cols], nan=0.0, posinf=0.0, neginf=0.0) * self.scale
        self.last = {"aux_true": aux_true, "aux_pred": aux.float()}
        return torch.nan_to_num(mu.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def _t(v):
    return tuple(float(x) for x in v) if isinstance(v, (tuple, list)) else float(v)
