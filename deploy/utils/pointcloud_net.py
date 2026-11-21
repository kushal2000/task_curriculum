"""The point-cloud policy: proprio + a point cloud -> 27 actions. Needs torch.

Distilled students of the point-cloud fold: DAgger from the keypoint fold policy, inputs = the robot's
own fields of the observation + a labelled point cloud. The network is the training code's StudentNet
(isaacsimenvs/scripts/distillation/student.py, torch only), loaded by path through
policy_net.training_module; this file only loads a checkpoint into it and steps it one robot at a time.
rigid2deform/checks/check_pointcloud_net.py checks it against golden files the training code made. Load
it through policy_net.load_policy_net (it tells the two kinds of checkpoint apart). What goes into the
cloud and how it becomes the network's input is the task's: rigid2deform/pc_obs.py.

    proprio (P) -> (p - mean) / sqrt(var + 1e-5) -> clamp(-5, 5) -------------------+
    points (M, D), mask (M) -> PointNet -> feature (64) ------------------------------+
                                                                                     v
        concat -> LSTM(1024) -> LayerNorm -> MLP 1024, 1024, 512, 512 (ELU) -> mu (27)
        -> nan_to_num -> clamp(-1, 1) = the action (deterministic: no sampling)

A checkpoint is {model (state dict), env (the env settings it was trained with, training's
student_env: obs.pointcloud = the cloud's goal, n_points, ref_step, fk_points; the DR groups), net_args, ...};
one saved before 2026-11-19 has pc_args instead of env, converted on load by training's own
legacy_env_overrides (old goal names first / scene / shape = label / goal_fixed / goal_any). The SAPG block embedding is already folded into rnn.bias_ih_l0. The auxiliary head
(training only) may be missing from the file (the pc_deploy_handoff copies have it stripped).
"""

from __future__ import annotations

import numpy as np
import torch

from .policy_net import training_module

STUDENT_PY = "isaacsimenvs/scripts/distillation/student.py"


class PointCloudNet:
    """A loaded point-cloud policy with its LSTM memory, for policy_net.load_policy_net.

    net.act((proprio, points, mask)) -> (27,) action, once per step, in order; net.reset() forgets.
    net.cloud: the cloud the checkpoint was trained on, its env's obs.pointcloud (rigid2deform/pc_obs.py
    builds it); net.env: the whole env settings (student_env's format);
    net.n_proprio, net.point_dim: the input sizes.
    """

    kind = "pointcloud"

    def __init__(self, ckpt: dict, num_actions: int) -> None:
        if "env" not in ckpt and "pc_args" not in ckpt:
            raise SystemExit("[policy_net] an rl_games point-cloud checkpoint (no env / pc_args, e.g. an RL-tuned "
                             "one): not supported yet; it needs converting (pc_rl.load_student_any)")
        student = training_module(STUDENT_PY, "_pc_student")
        if "env" in ckpt:
            self.env = ckpt["env"]
        else:
            try:
                self.env = student.legacy_env_overrides(ckpt["pc_args"])
            except ValueError as e:   # a cloud deleted on 2026-11-19 (not the scene cloud)
                raise SystemExit(f"[policy_net] {e}") from None
        if self.env["obs"]["object"] != "pointcloud":
            raise SystemExit(f"[policy_net] the checkpoint's obs.object is {self.env['obs']['object']}, not pointcloud")
        self.cloud = dict(self.env["obs"]["pointcloud"])
        state = ckpt["model"]
        self.n_proprio = int(state["mean"].numel())
        self.point_dim = int(state["pointnet.point_mlp.0.weight"].shape[1])
        n_out = int(state["mu.weight"].shape[0])
        if n_out != num_actions:
            raise SystemExit(f"[policy_net] the checkpoint has {n_out} actions, the robot {num_actions}")
        aux_dim = int(state["aux.2.weight"].shape[0]) if "aux.2.weight" in state else 1   # 1: unused
        self.model = student.StudentNet(self.n_proprio, n_out, self.point_dim, aux_dim, **ckpt.get("net_args", {}))
        missing, unexpected = self.model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
        missing = [k for k in missing if not k.startswith("aux.")]
        if missing or unexpected:
            raise SystemExit(f"[policy_net] checkpoint vs network: missing {missing}, unexpected {unexpected}")
        self.model.eval()
        self.reset()

    def reset(self) -> None:
        self.state = self.model.initial_state(1, "cpu")

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        proprio, points, mask = obs
        p = torch.as_tensor(np.asarray(proprio), dtype=torch.float32)[None]
        x = torch.as_tensor(np.asarray(points), dtype=torch.float32)[None]
        m = torch.as_tensor(np.asarray(mask), dtype=torch.bool)[None]
        mu, _, self.state = self.model.step(p, x, m, self.state)
        a = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return a[0].numpy().astype(np.float64)

    def warm_up(self) -> None:
        """One call on an empty cloud (the first is slow), then forget it."""
        self.act((np.zeros(self.n_proprio, np.float32), np.zeros((1, self.point_dim), np.float32), np.zeros(1, bool)))
        self.reset()
