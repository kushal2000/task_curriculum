"""Reinforcement learning with the point-cloud student as the actor (rl_games SAPG).

Used by ``isaacsimenvs/scripts/training/train.py`` when the recipe sets ``obs.object: pointcloud``:
the env's ``policy`` observation is then the student's input, ``[proprio | points (M x D) | mask (M)]``
(``mdp/observations.py``, with the cloud's and the proprioception's DR); the ``critic`` observation
is the privileged state as for every run.

* ``PCA2CBuilder`` (network name ``pc_actor_critic``) -- the rl_games actor with the student's
  front end: proprio normaliser + PointNet, then the usual [.., SAPG embedding] -> LSTM ->
  LayerNorm -> MLP -> mu. ``network.pc.init_student`` starts it from a distilled student
  (the embedding's input weights start at zero, so every SAPG block is the student at first) and
  ``network.pc.init_sigma`` takes the exploration noise of a state-policy checkpoint.
* ``student_from_rl`` -- an rl_games checkpoint of that actor as a ``StudentNet`` state dict (one
  SAPG block, embedding folded into the LSTM bias), so ``distill_pc --mode eval`` runs it;
  ``write_student_env`` puts the env settings it needs next to the checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from isaacsimenvs.scripts.distillation.student import legacy_env_overrides

__all__ = ["register_network", "student_from_rl", "load_student_any", "write_student_env", "patch_gradient_step"]

FEAT_DIM = 64


def flat_dim(n_proprio: int, num_points: int, point_dim: int) -> int:
    return int(n_proprio) + int(num_points) * int(point_dim) + int(num_points)


def write_student_env(run_dir, env: dict, layout: dict, teacher: str | None) -> None:
    """Next to the checkpoints: the env settings the actor reads through (``student.student_env``),
    for ``distill_pc --mode eval`` and the viewer."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "student_env.json").write_text(json.dumps({"env": env, "layout": layout, "teacher": teacher}, indent=1))


def patch_gradient_step() -> None:
    """rl_games' SAPG returns every minibatch's flat gradient from the optimizer step and keeps a
    host copy of it (and of a zero vector the same size) for one TensorBoard scalar: a few seconds
    and ~10 GB of RAM per epoch with 128 minibatches. Here the step returns one element instead, and
    a single-rank ``--distributed`` run skips the all-reduce (its sum is the gradient itself). The
    gradients, clipping and optimizer step are unchanged; only that scalar loses its meaning."""
    from rl_games.common.a2c_common import A2CBase

    original = A2CBase.trancate_gradients_and_step
    if getattr(original, "_pc_patched", False):
        return

    def step(self):
        if self.multi_gpu and int(getattr(self, "world_size", 1)) > 1:
            return original(self)[:1]
        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return torch.ones(1, device=self.ppo_device)

    step._pc_patched = True
    A2CBase.trancate_gradients_and_step = step


def register_network() -> None:
    from rl_games.algos_torch import model_builder
    from rl_games.algos_torch.network_builder import A2CBuilder
    from rl_games.algos_torch.running_mean_std import RunningMeanStd

    from isaacsimenvs.scripts.distillation.student import PointNet

    class PCA2CBuilder(A2CBuilder):
        class Network(A2CBuilder.Network):
            def __init__(self, params, **kwargs):
                pc = dict(params["pc"])
                self_np, self_m, self_d = int(pc["n_proprio"]), int(pc["num_points"]), int(pc["point_dim"])
                feat_one = int(pc.get("feat_dim", FEAT_DIM))
                # ``n_live``: the cloud is two clouds (live | goal), each with its own PointNet
                # (student.SplitStudentNet); the encoded feature is then [f_live | f_goal].
                n_live = int(pc["n_live"]) if pc.get("split_goal") else 0
                feat = 2 * feat_one if n_live else feat_one
                full = flat_dim(self_np, self_m, self_d)
                if int(kwargs["coef_id_idx"]) != full:
                    raise ValueError(f"obs has {kwargs['coef_id_idx']} columns, the point-cloud layout needs {full}")
                kwargs = dict(kwargs)
                kwargs["coef_id_idx"] = self_np + feat            # the encoded obs: [proprio | feature | coef id]
                kwargs["input_shape"] = (self_np + feat + 1,)
                super().__init__(params, **kwargs)
                self.n_proprio, self.n_pts, self.point_dim, self.feat_dim = self_np, self_m, self_d, feat
                self.n_live = n_live
                self.pointnet = PointNet(self_d, out_dim=feat_one)
                self.goal_pointnet = PointNet(self_d, out_dim=feat_one) if n_live else None
                self.proprio_rms = RunningMeanStd((self_np,))
                self.freeze_norm = bool(pc.get("freeze_norm", False))
                if pc.get("init_student"):
                    self._init_from_student(str(pc["init_student"]))
                if pc.get("init_sigma"):
                    ckpt = torch.load(str(pc["init_sigma"]), map_location="cpu", weights_only=False)
                    ckpt = ckpt[0] if 0 in ckpt else ckpt
                    with torch.no_grad():
                        self.sigma.copy_(ckpt["model"]["a2c_network.sigma"].float())
                    print(f"[pc_rl] sigma <- {pc['init_sigma']}", flush=True)

            def _init_from_student(self, path: str) -> None:
                state = load_student_any(path)["model"]
                with torch.no_grad():
                    self.proprio_rms.running_mean.copy_(state["mean"].double())
                    self.proprio_rms.running_var.copy_(state["var"].double())
                    self.proprio_rms.count.fill_(1e8)             # a settled normaliser, not a fresh one
                    k = self.n_proprio + self.feat_dim
                    w = self.rnn.rnn.weight_ih_l0
                    w[:, :k].copy_(state["rnn.weight_ih_l0"])
                    w[:, k:].zero_()
                    self.rnn.rnn.bias_ih_l0.copy_(state["rnn.bias_ih_l0"])
                    self.rnn.rnn.weight_hh_l0.copy_(state["rnn.weight_hh_l0"])
                    self.rnn.rnn.bias_hh_l0.copy_(state["rnn.bias_hh_l0"])
                    self.layer_norm.load_state_dict({k2[len("layer_norm."):]: v for k2, v in state.items() if k2.startswith("layer_norm.")})
                    self.actor_mlp.load_state_dict({k2[len("actor_mlp."):]: v for k2, v in state.items() if k2.startswith("actor_mlp.")})
                    self.mu.load_state_dict({k2[len("mu."):]: v for k2, v in state.items() if k2.startswith("mu.")})
                    self.pointnet.load_state_dict({k2[len("pointnet."):]: v for k2, v in state.items() if k2.startswith("pointnet.")})
                    if self.goal_pointnet is not None:
                        self.goal_pointnet.load_state_dict(
                            {k2[len("goal_pointnet."):]: v for k2, v in state.items() if k2.startswith("goal_pointnet.")})
                print(f"[pc_rl] actor <- student {path}", flush=True)

            def forward(self, obs_dict):
                obs = obs_dict["obs"]
                n_p, m, d = self.n_proprio, self.n_pts, self.point_dim
                proprio = obs[:, :n_p]
                points = obs[:, n_p:n_p + m * d].reshape(-1, m, d)
                mask = obs[:, n_p + m * d:n_p + m * d + m] > 0.5
                if self.freeze_norm:
                    self.proprio_rms.eval()
                proprio = self.proprio_rms(proprio)
                # Half precision for the per-point MLP in the rollout too (training already runs
                # under autocast, and so do the distill_pc evaluations).
                with torch.autocast("cuda", dtype=torch.float16, enabled=obs.is_cuda):
                    if self.n_live:
                        k = self.n_live
                        feat = torch.cat((self.pointnet(points[:, :k], mask[:, :k]),
                                          self.goal_pointnet(points[:, k:], mask[:, k:])), dim=1)
                    else:
                        feat = self.pointnet(points, mask)
                feat = feat.to(proprio.dtype)
                encoded = dict(obs_dict)
                encoded["obs"] = torch.cat((proprio, feat, obs[:, n_p + m * d + m:]), dim=1)
                return super().forward(encoded)

        def build(self, name, **kwargs):
            return PCA2CBuilder.Network(self.params, **kwargs)

    model_builder.register_network("pc_actor_critic", lambda **kwargs: PCA2CBuilder())


def student_from_rl(model_state: dict, block: int = -1) -> dict:
    """A ``StudentNet`` state dict from an rl_games ``pc_actor_critic`` model state (the SAPG
    ``block``'s embedding folded into the LSTM input bias). The auxiliary head is not there."""
    pre = "a2c_network."
    emb = model_state[pre + "extra_params"][block].float()
    w = model_state[pre + "rnn.rnn.weight_ih_l0"].float()
    k = w.shape[1] - emb.shape[0]
    out = {
        "mean": model_state[pre + "proprio_rms.running_mean"].float(),
        "var": model_state[pre + "proprio_rms.running_var"].float(),
        "rnn.weight_ih_l0": w[:, :k].clone(),
        "rnn.bias_ih_l0": model_state[pre + "rnn.rnn.bias_ih_l0"].float() + w[:, k:] @ emb,
        "rnn.weight_hh_l0": model_state[pre + "rnn.rnn.weight_hh_l0"].float(),
        "rnn.bias_hh_l0": model_state[pre + "rnn.rnn.bias_hh_l0"].float(),
    }
    for key, value in model_state.items():
        name = key[len(pre):] if key.startswith(pre) else ""
        if name.startswith(("layer_norm.", "actor_mlp.", "mu.", "pointnet.")):
            out[name] = value.float()
    return out


def load_student_any(path: str) -> dict:
    """A student checkpoint ``{model, env, net_args, ...}`` from either a ``distill_pc`` file or an
    rl_games checkpoint of the point-cloud actor (``student_env.json`` in its run directory). ``env``:
    the env settings it was trained with (``student.student_env``'s format); students saved before
    2026-11-19 carry ``pc_args`` instead (``student.legacy_env_overrides`` translates)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "env" in ckpt:
        return ckpt
    if "pc_args" in ckpt:
        return {**ckpt, "env": legacy_env_overrides(ckpt["pc_args"])}
    ckpt = ckpt[0] if 0 in ckpt else ckpt
    if "a2c_network.pointnet.out.0.weight" not in ckpt.get("model", {}):
        raise ValueError(f"{path} is neither a distilled student nor a point-cloud RL checkpoint")
    for folder in Path(path).resolve().parents[:3]:
        for name in ("student_env.json", "pc_args.json"):
            side = folder / name
            if side.exists():
                meta = json.loads(side.read_text())
                env = meta["env"] if "env" in meta else legacy_env_overrides(meta["pc_args"])
                return {"model": student_from_rl(ckpt["model"]), "env": env, "net_args": {},
                        "iter": ckpt.get("epoch"), "teacher": meta.get("teacher"), "from_rl": True}
    raise FileNotFoundError(f"no student_env.json (or pc_args.json) next to {path}")
