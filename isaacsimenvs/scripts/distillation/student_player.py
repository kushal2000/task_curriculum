"""A point-cloud student behind the player interface (``scripts/evaluation/eval.py``, the live viewer).

The student reads the cloud the ENV builds (``obs.object: pointcloud``: ``policy`` = ``[proprio | points |
mask]``, delays and cloud DR applied by the env), as in DAgger training and point-cloud RL. The env's
cloud must be the one the student trained on (``eval.py``'s ``_check_student_cloud``).
"""

from __future__ import annotations

import torch


class EnvCloudStudentPlayer:
    is_student = True

    def __init__(self, inner, ckpt: dict, device) -> None:
        from isaacsimenvs.scripts.distillation.distill_pc import StudentInput
        from isaacsimenvs.scripts.distillation.student import StudentNet

        sin = StudentInput(inner)
        if sin.pc is None:
            raise ValueError("a point-cloud student needs env.obs.object: pointcloud")
        self.n_proprio, self.num_points, self.point_dim = sin.n_proprio, sin.num_points, sin.point_dim
        self.aux_cols, self.scale = sin.aux_cols, sin.scale
        #: The settings it was trained with (``student.student_env``): its cloud and DR.
        self.env = ckpt["env"]
        self.n, self.device = inner.num_envs, device
        self.net = StudentNet(sin.n_proprio, int(inner.cfg.action_space), sin.point_dim, int(sin.aux_cols.numel()),
                              **ckpt["net_args"]).to(device).eval()
        self.net.load_state_dict(ckpt["model"], strict=not ckpt.get("from_rl", False))
        self.amp = str(device).startswith("cuda")
        self.state = self.net.initial_state(self.n, device)
        self.start = torch.ones(self.n, dtype=torch.bool, device=device)
        self.expl_coef = 0.0  # (player interface; a student has no SAPG blocks)
        #: For viewers: the auxiliary head's estimate and its target at the latest step (``teacher`` given).
        self.last = None

    def reset_rnn(self, env_ids=None) -> None:
        """The next step of these envs starts an episode: their network state is dropped."""
        self.start[slice(None) if env_ids is None else env_ids] = True

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, deterministic: bool = True, teacher: torch.Tensor | None = None) -> torch.Tensor:
        """``obs``: the env's ``policy`` observation; ``teacher``: its ``teacher`` one (only for ``last``)."""
        n_p, m, d = self.n_proprio, self.num_points, self.point_dim
        proprio = obs[:, :n_p]
        points = obs[:, n_p:n_p + m * d].reshape(-1, m, d)
        mask = obs[:, n_p + m * d:n_p + m * d + m] > 0.5
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            mu, aux, self.state = self.net.step(proprio, points, mask, self.state, self.start)
        self.start = torch.zeros_like(self.start)
        if teacher is not None:
            aux_true = torch.nan_to_num(teacher[:, self.aux_cols], nan=0.0, posinf=0.0, neginf=0.0) * self.scale
            self.last = {"aux_true": aux_true, "aux_pred": aux.float()}
        return torch.nan_to_num(mu.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
