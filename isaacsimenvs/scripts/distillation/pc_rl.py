"""Reinforcement learning with the point-cloud student as the actor (rl_games SAPG).

Three pieces, used by ``isaacsimenvs/scripts/training/train.py --pc_args <json>``:

* ``attach_student_obs`` -- the env's ``policy`` observation becomes the student's input, flat:
  ``[proprio | points (M x D) | mask (M)]`` (``distill_pc.StudentObs`` builds it, with the cloud
  noise and the proprioception / cloud delays). The ``critic`` observation is untouched, so the
  central value network still reads the privileged state.
* ``PCA2CBuilder`` (network name ``pc_actor_critic``) -- the rl_games actor with the student's
  front end: proprio normaliser + PointNet, then the usual [.., SAPG embedding] -> LSTM ->
  LayerNorm -> MLP -> mu. ``network.pc.init_student`` starts it from a distilled student
  (the embedding's input weights start at zero, so every SAPG block is the student at first) and
  ``network.pc.init_sigma`` takes the exploration noise of a state-policy checkpoint.
* ``student_from_rl`` -- an rl_games checkpoint of that actor as a ``StudentNet`` state dict (one
  SAPG block, embedding folded into the LSTM bias), so ``distill_pc --mode eval`` runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from isaacsimenvs.scripts.distillation.student import canonical_pc_args

__all__ = ["attach_student_obs", "register_network", "student_from_rl", "load_student_any", "write_pc_args", "patch_gradient_step"]

FEAT_DIM = 64


def flat_dim(n_proprio: int, num_points: int, point_dim: int) -> int:
    return int(n_proprio) + int(num_points) * int(point_dim) + int(num_points)


def attach_student_obs(inner, sobs) -> dict:
    """Make ``inner`` (the unwrapped env) return the student's flat input as its ``policy``
    observation. Returns the layout ``{n_proprio, num_points, point_dim}``."""
    import gymnasium as gym
    import numpy as np

    n_proprio, m, d = int(sobs.proprio_cols.numel()), int(sobs.pc.num_points), int(sobs.pc.point_dim)
    dim = flat_dim(n_proprio, m, d)
    original = inner._get_observations

    def _get_observations():
        obs = original()
        start = inner.task.episode_start(inner.episode_length_buf)
        sobs.reset(start.nonzero(as_tuple=True)[0])
        so = sobs(obs["policy"])
        obs["policy"] = torch.cat(
            (so["proprio"], so["points"].flatten(1), so["mask"].to(so["points"].dtype)), dim=1
        )
        return obs

    inner._get_observations = _get_observations
    inner.cfg.observation_space = dim
    box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
    inner.single_observation_space.spaces["policy"] = box
    inner.observation_space = gym.vector.utils.batch_space(box, inner.num_envs)   # DirectRLEnv: the policy group, batched
    return {"n_proprio": n_proprio, "num_points": m, "point_dim": d}


def write_pc_args(run_dir, pc_args: dict, layout: dict, teacher: str | None) -> None:
    """Next to the checkpoints: what ``distill_pc --mode eval`` needs to rebuild the input."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "pc_args.json").write_text(json.dumps({"pc_args": pc_args, "layout": layout, "teacher": teacher}, indent=1))


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
                feat = int(pc.get("feat_dim", FEAT_DIM))
                full = flat_dim(self_np, self_m, self_d)
                if int(kwargs["coef_id_idx"]) != full:
                    raise ValueError(f"obs has {kwargs['coef_id_idx']} columns, the point-cloud layout needs {full}")
                kwargs = dict(kwargs)
                kwargs["coef_id_idx"] = self_np + feat            # the encoded obs: [proprio | feature | coef id]
                kwargs["input_shape"] = (self_np + feat + 1,)
                super().__init__(params, **kwargs)
                self.n_proprio, self.n_pts, self.point_dim, self.feat_dim = self_np, self_m, self_d, feat
                self.pointnet = PointNet(self_d, out_dim=feat)
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
    """A student checkpoint ``{model, pc_args, net_args, ...}`` from either a ``distill_pc`` file or
    an rl_games checkpoint of the point-cloud actor (its ``pc_args.json`` is in the run directory);
    ``pc_args`` under the current names (``student.canonical_pc_args``)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "pc_args" in ckpt:
        return {**ckpt, "pc_args": canonical_pc_args(ckpt["pc_args"])}
    ckpt = ckpt[0] if 0 in ckpt else ckpt
    if "a2c_network.pointnet.out.0.weight" not in ckpt.get("model", {}):
        raise ValueError(f"{path} is neither a distilled student nor a point-cloud RL checkpoint")
    for folder in Path(path).resolve().parents[:3]:
        side = folder / "pc_args.json"
        if side.exists():
            meta = json.loads(side.read_text())
            return {"model": student_from_rl(ckpt["model"]), "pc_args": canonical_pc_args(meta["pc_args"]), "net_args": {},
                    "iter": ckpt.get("epoch"), "teacher": meta.get("teacher"), "from_rl": True}
    raise FileNotFoundError(f"no pc_args.json next to {path}")
