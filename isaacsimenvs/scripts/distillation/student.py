"""The point-cloud student and a plain-torch copy of its teacher.

The teacher is the rl_games SAPG actor (``ClothSAPG.yaml``): obs -> running mean/std (clamp 5) ->
[obs, 32-d SAPG block embedding] -> LSTM(1024) -> LayerNorm -> MLP [1024, 1024, 512, 512] (ELU) ->
mu. ``TeacherNet`` rebuilds exactly that from a checkpoint's ``model`` state dict, for one SAPG
block (default the last: coefficient 0, task reward only -- what every evaluation here uses).

``StudentNet`` is the same LSTM + MLP + mu. Its input is the teacher's proprioceptive observation
columns (same normaliser) and a PointNet feature in place of the columns that need a tracker
(``VISION_FIELDS``). ``load_teacher`` copies every teacher weight that still has a counterpart;
the SAPG embedding is a constant, so it is folded into the LSTM input bias.

Torch only -- no rl_games, no Isaac Lab -- so both are unit-tested on CPU.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["VISION_FIELDS", "split_columns", "PointNet", "TeacherNet", "StudentNet"]

#: Observation fields that come from the cloth tracker on the robot; the student does not get them.
VISION_FIELDS = ("object_rot", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales")
#: The student's auxiliary head regresses these (what the teacher read).
AUX_FIELDS = ("keypoints_rel_palm", "keypoints_rel_goal")


def split_columns(obs_list, sizes) -> tuple[list[int], list[int]]:
    """``(proprio columns, aux columns)`` of the flat observation ``obs_list`` (field -> width)."""
    proprio, aux, at = [], {}, 0
    for name in obs_list:
        cols = list(range(at, at + int(sizes[name])))
        at += int(sizes[name])
        if name in AUX_FIELDS:
            aux[name] = cols
        if name not in VISION_FIELDS:
            proprio += cols
    missing = [f for f in AUX_FIELDS if f not in aux]
    if missing:
        raise ValueError(f"obs_list has no {missing}; the auxiliary head needs them")
    return proprio, [c for f in AUX_FIELDS for c in aux[f]]


def _mlp(sizes) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ELU()]
    return nn.Sequential(*layers)


class _Core(nn.Module):
    """LSTM -> LayerNorm -> MLP -> mu, stepped one policy step at a time with per-env resets."""

    def __init__(self, in_dim: int, num_actions: int, rnn_units: int = 1024, mlp=(1024, 1024, 512, 512)) -> None:
        super().__init__()
        self.rnn_units = int(rnn_units)
        self.rnn = nn.LSTM(in_dim, rnn_units)
        self.layer_norm = nn.LayerNorm(rnn_units)
        self.actor_mlp = _mlp((rnn_units, *mlp))
        self.mu = nn.Linear(mlp[-1], num_actions)

    def initial_state(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, n, self.rnn_units, device=device)
        return z, z.clone()

    def core_step(self, x: torch.Tensor, state, start: torch.Tensor | None):
        """``x`` ``(N, D)``; ``start`` ``(N,)`` bool: this is the first step of an episode, so the
        carried state is dropped first. Returns ``(mu, lstm output after LayerNorm, state)``."""
        if start is not None:
            # where, not a multiply: a non-finite state (an env whose sim blew up) must not survive.
            drop = start.view(1, -1, 1)
            state = (torch.where(drop, torch.zeros_like(state[0]), state[0]),
                     torch.where(drop, torch.zeros_like(state[1]), state[1]))
        out, state = self.rnn(x.unsqueeze(0), (state[0].to(x.dtype), state[1].to(x.dtype)))
        h = self.layer_norm(out.squeeze(0))
        return self.mu(self.actor_mlp(h)), h, state


class TeacherNet(_Core):
    def __init__(self, model_state: dict, expl_block: int = -1) -> None:
        w_ih = model_state["a2c_network.rnn.rnn.weight_ih_l0"]
        emb = model_state["a2c_network.extra_params"]
        obs_dim = w_ih.shape[1] - emb.shape[1]
        super().__init__(w_ih.shape[1], model_state["a2c_network.mu.weight"].shape[0], w_ih.shape[0] // 4)
        self.obs_dim = int(obs_dim)
        self.register_buffer("mean", model_state["running_mean_std.running_mean"].float().clone())
        self.register_buffer("var", model_state["running_mean_std.running_var"].float().clone())
        self.register_buffer("embedding", emb[expl_block].float().clone())
        own = self.state_dict()
        for key, value in model_state.items():
            name = key.removeprefix("a2c_network.").replace("rnn.rnn.", "rnn.")
            if name in own and name not in ("mean", "var", "embedding"):
                if own[name].shape != value.shape:
                    raise ValueError(f"{key}: checkpoint {tuple(value.shape)} vs network {tuple(own[name].shape)}")
                own[name] = value.float().clone()
        self.load_state_dict(own)
        self.requires_grad_(False)

    def normalise(self, obs: torch.Tensor) -> torch.Tensor:
        return ((obs - self.mean) / torch.sqrt(self.var + 1e-5)).clamp(-5.0, 5.0)

    @torch.no_grad()
    def step(self, obs: torch.Tensor, state, start: torch.Tensor | None = None):
        """``(action, state)``: the deterministic action the evaluations use, ``clamp(mu, -1, 1)``."""
        x = torch.cat((self.normalise(obs), self.embedding.expand(obs.shape[0], -1)), dim=-1)
        mu, _, state = self.core_step(x, state, start)
        return mu.clamp(-1.0, 1.0), state


class PointNet(nn.Module):
    """Per-point MLP -> masked max-pool -> linear. An env with no valid point gets a zero feature."""

    def __init__(self, point_dim: int, widths=(64, 128, 256), out_dim: int = 64) -> None:
        super().__init__()
        self.point_mlp = _mlp((point_dim, *widths))
        self.out = nn.Sequential(nn.Linear(widths[-1], out_dim), nn.LayerNorm(out_dim))

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # An invalid slot is given a valid point of the same cloud: a duplicate does not change the
        # max-pool, and the (B, M, 3..7) input is far cheaper to mask than the (B, M, W) features.
        first = mask.to(torch.uint8).argmax(dim=1)
        fill = points[torch.arange(points.shape[0], device=points.device), first].unsqueeze(1)
        f = self.point_mlp(torch.where(mask.unsqueeze(-1), points, fill)).amax(dim=1)          # (B, W)
        f = torch.where(mask.any(dim=1, keepdim=True), f, torch.zeros_like(f))
        return self.out(f)


class StudentNet(_Core):
    def __init__(
        self, n_proprio: int, num_actions: int, point_dim: int, aux_dim: int,
        feat_dim: int = 64, use_rnn_state: bool = True,
    ) -> None:
        super().__init__(n_proprio + feat_dim, num_actions)
        self.n_proprio, self.feat_dim, self.aux_dim = int(n_proprio), int(feat_dim), int(aux_dim)
        self.use_rnn_state = bool(use_rnn_state)
        self.register_buffer("mean", torch.zeros(n_proprio))
        self.register_buffer("var", torch.ones(n_proprio))
        self.pointnet = PointNet(point_dim, out_dim=feat_dim)
        self.aux = nn.Sequential(nn.Linear(self.rnn_units, 256), nn.ELU(), nn.Linear(256, aux_dim))

    def load_teacher(self, teacher: TeacherNet, proprio_cols) -> None:
        """Copy the teacher: normaliser and LSTM input weights on the proprio columns, the SAPG
        embedding folded into the input bias, and the rest of the LSTM, LayerNorm, MLP and mu as
        they are. The PointNet feature's input weights start small."""
        cols = torch.as_tensor(proprio_cols, dtype=torch.long)
        with torch.no_grad():
            self.mean.copy_(teacher.mean[cols])
            self.var.copy_(teacher.var[cols])
            w = teacher.rnn.weight_ih_l0
            self.rnn.weight_ih_l0[:, : self.n_proprio].copy_(w[:, cols])
            self.rnn.weight_ih_l0[:, self.n_proprio:].normal_(0.0, 0.01)
            self.rnn.bias_ih_l0.copy_(teacher.rnn.bias_ih_l0 + w[:, teacher.obs_dim:] @ teacher.embedding)
            self.rnn.weight_hh_l0.copy_(teacher.rnn.weight_hh_l0)
            self.rnn.bias_hh_l0.copy_(teacher.rnn.bias_hh_l0)
            self.layer_norm.load_state_dict(teacher.layer_norm.state_dict())
            self.actor_mlp.load_state_dict(teacher.actor_mlp.state_dict())
            self.mu.load_state_dict(teacher.mu.state_dict())

    def encode(self, proprio: torch.Tensor, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p = ((proprio - self.mean) / torch.sqrt(self.var + 1e-5)).clamp(-5.0, 5.0)
        return torch.cat((p, self.pointnet(points, mask).to(p.dtype)), dim=-1)

    def step(self, proprio, points, mask, state, start: torch.Tensor | None = None):
        """One policy step: ``(mu, aux, state)``."""
        if not self.use_rnn_state:  # memory ablation: the state is dropped every step
            start = torch.ones(proprio.shape[0], dtype=torch.bool, device=proprio.device)
        mu, h, state = self.core_step(self.encode(proprio, points, mask), state, start)
        return mu, self.aux(h), state

    def forward_seq(self, proprio, points, mask, state, start):
        """Truncated BPTT over a stored segment. Inputs lead with ``(T, N)``; ``start`` ``(T, N)``
        as in ``step``; ``state`` the LSTM state before step 0. Returns ``(mu, aux)`` ``(T, N, .)``."""
        t_len, n = proprio.shape[:2]
        x = self.encode(proprio.flatten(0, 1), points.flatten(0, 1), mask.flatten(0, 1)).view(t_len, n, -1)
        mus, auxs = [], []
        for t in range(t_len):
            st = start[t] if self.use_rnn_state else torch.ones_like(start[t])
            mu, h, state = self.core_step(x[t], state, st)
            mus.append(mu)
            auxs.append(self.aux(h))
        return torch.stack(mus), torch.stack(auxs)
