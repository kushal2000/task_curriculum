"""The point-cloud policy network: proprio + a point cloud -> 27 actions. Needs torch.

Distilled students of the point-cloud fold (branch pc_round2, isaacsimenvs/distill/student.py): DAgger
from the keypoint fold policy, inputs = the robot's own fields of the observation + a labelled point
cloud. Taken from pc_deploy_handoff/pc_student_ref.py (2026-11-17), the reference the training code
was checked against; rigid2deform/checks/check_pointcloud_net.py checks this copy against its golden
files. Load it through policy_net.load_policy_net (it tells the two kinds of checkpoint apart). What
goes into the cloud and how it becomes the network's input is the task's: rigid2deform/pc_obs.py.

    proprio (P) -> (p - mean) / sqrt(var + 1e-5) -> clamp(-5, 5) ----------------------------------+
    points (M, D), mask (M) -> PointNet -> feature (64) ---------------------------------------------+
                                                                                                    v
        concat -> LSTM(1024, 1 layer) -> LayerNorm -> MLP 1024, 1024, 512, 512 (ELU) -> mu (27)
        -> nan_to_num -> clamp(-1, 1) = the action (deterministic: no sampling)
    PointNet: per point Linear(D->64), ELU, Linear(64->128), ELU, Linear(128->256), ELU
              -> max over the valid points (mask) -> Linear(256->64) -> LayerNorm(64)
              (no valid point at all: the max-pool result is 0 before the Linear)

A checkpoint is {model (state dict), pc_args (the cloud's settings: goal, n_points, fk_points, ...),
net_args, ...}. The SAPG block embedding is already folded into rnn.bias_ih_l0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _mlp(sizes) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ELU()]
    return nn.Sequential(*layers)


class PointNet(nn.Module):
    def __init__(self, point_dim: int, widths=(64, 128, 256), out_dim: int = 64) -> None:
        super().__init__()
        self.point_mlp = _mlp((point_dim, *widths))
        self.out = nn.Sequential(nn.Linear(widths[-1], out_dim), nn.LayerNorm(out_dim))

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:   # (B, M, D), (B, M)
        f = self.point_mlp(points).masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        f = torch.where(mask.any(dim=1, keepdim=True), f, torch.zeros_like(f))   # no valid point -> 0
        return self.out(f)


class PointCloudLSTM(nn.Module):
    """The network itself (training's StudentNet, inference only). Its LSTM state is the caller's."""

    def __init__(self, n_proprio: int, num_actions: int, point_dim: int, feat_dim: int = 64,
                 rnn_units: int = 1024, mlp=(1024, 1024, 512, 512)) -> None:
        super().__init__()
        self.rnn_units = rnn_units
        self.register_buffer("mean", torch.zeros(n_proprio))
        self.register_buffer("var", torch.ones(n_proprio))
        self.pointnet = PointNet(point_dim, out_dim=feat_dim)
        self.rnn = nn.LSTM(n_proprio + feat_dim, rnn_units)
        self.layer_norm = nn.LayerNorm(rnn_units)
        self.actor_mlp = _mlp((rnn_units, *mlp))
        self.mu = nn.Linear(mlp[-1], num_actions)

    def initial_state(self, batch: int = 1):
        z = torch.zeros(1, batch, self.rnn_units)
        return z, z.clone()

    @torch.no_grad()
    def act(self, proprio, feats, mask, state):
        """One policy step: unbatched (P,), (M, D), (M,) -> ((27,) float64 action in [-1, 1], new state)."""
        p = torch.as_tensor(np.asarray(proprio), dtype=torch.float32)[None]
        x = torch.as_tensor(np.asarray(feats), dtype=torch.float32)[None]
        m = torch.as_tensor(np.asarray(mask), dtype=torch.bool)[None]
        p = ((p - self.mean) / torch.sqrt(self.var + 1e-5)).clamp(-5.0, 5.0)
        h_in = torch.cat((p, self.pointnet(x, m)), dim=-1)
        out, state = self.rnn(h_in.unsqueeze(0), state)
        mu = self.mu(self.actor_mlp(self.layer_norm(out.squeeze(0))))
        a = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return a[0].numpy().astype(np.float64), state


class PointCloudNet:
    """A loaded point-cloud policy with its LSTM memory, for policy_net.load_policy_net.

    net.act((proprio, points, mask)) -> (27,) action, once per step, in order; net.reset() forgets.
    net.pc_args: the cloud the checkpoint was trained on (rigid2deform/pc_obs.py builds it);
    net.n_proprio, net.point_dim: the input sizes.
    """

    kind = "pointcloud"

    def __init__(self, ckpt: dict, num_actions: int) -> None:
        if "pc_args" not in ckpt:
            raise SystemExit("[policy_net] an rl_games point-cloud checkpoint (no pc_args, e.g. an RL-tuned one): "
                             "not supported yet; it needs converting (pc_deploy_handoff student_from_rl) "
                             "and its run's pc_args.json")
        state, self.pc_args = ckpt["model"], dict(ckpt["pc_args"])
        if not ckpt.get("net_args", {}).get("use_rnn_state", True):
            raise SystemExit("[policy_net] a no-memory ablation student: not supported")
        if self.pc_args.get("cloud") != "scene":
            raise SystemExit(f"[policy_net] only scene-cloud students are supported, got cloud={self.pc_args.get('cloud')}")
        self.n_proprio = int(state["mean"].numel())
        self.point_dim = int(state["pointnet.point_mlp.0.weight"].shape[1])
        n_out = int(state["mu.weight"].shape[0])
        if n_out != num_actions:
            raise SystemExit(f"[policy_net] the checkpoint has {n_out} actions, the robot {num_actions}")
        self.model = PointCloudLSTM(self.n_proprio, n_out, self.point_dim)
        own = self.model.state_dict()
        missing = [k for k in own if k not in state]
        if missing:
            raise SystemExit(f"[policy_net] the checkpoint lacks {missing}")
        self.model.load_state_dict({k: state[k].float() for k in own})
        self.model.eval()
        self.state = self.model.initial_state()

    def reset(self) -> None:
        self.state = self.model.initial_state()

    def act(self, obs) -> np.ndarray:
        proprio, points, mask = obs
        action, self.state = self.model.act(proprio, points, mask, self.state)
        return action

    def warm_up(self) -> None:
        """One call on an empty cloud (the first is slow), then forget it."""
        self.act((np.zeros(self.n_proprio, np.float32), np.zeros((1, self.point_dim), np.float32), np.zeros(1, bool)))
        self.reset()
