"""PointNet actor for the cloth point cloud, as rl_games network + model plugins.

The policy observation ends with ``cloth_points`` then ``goal_points`` (``obs.num_points = P``,
P x 3 each, metres). This network cuts those 6P values off, encodes each sample with its goal as
``[p, g, g - p]`` through a shared MLP, max-pools over the P samples (order-free), and feeds the
pooled feature in place of the raw points to the usual actor-critic trunk (LSTM, MLP, heads).
SAPG's coefficient columns, appended after the env observation, pass through untouched.

The model keeps the point slice out of rl_games' per-element input normalisation: per-slot mean/std
would rescale samples and goals independently and break ``g - p``. The points reach the encoder in
metres; everything else is normalised as before.

Config (``agent.params``)::

    model:   {name: pointnet_a2c_logstd}
    network: {name: pointnet_actor_critic, pointnet: {num_points: 64, units: [64, 128, 256]}, ...}

Importing this module registers both names; train.py and the evaluation player import it.
"""

from __future__ import annotations

import torch
from torch import nn

from rl_games.algos_torch import model_builder, models, network_builder

POINT_FEATURES = 9  # p, g, g - p


class PointNetEncoder(nn.Module):
    """``(B, P, 3), (B, P, 3) -> (B, units[-1])``: shared per-point MLP, then max over points."""

    def __init__(self, units=(64, 128, 256), in_features=POINT_FEATURES):
        super().__init__()
        layers, width = [], in_features
        for i, u in enumerate(units):
            layers.append(nn.Linear(width, u))
            if i < len(units) - 1:
                layers += [nn.LayerNorm(u), nn.ELU()]
            width = u
        self.mlp = nn.Sequential(*layers)
        self.out_dim = width

    def forward(self, points: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.cat([points, goals, goals - points], dim=-1))

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        """``(B, P, F) -> (B, units[-1])``."""
        return self.mlp(features).amax(dim=-2)


class AdapterEncoder(nn.Module):
    """Points + goals + palm -> a pretrained policy's cloth observation, in its normalised space.

    Per sample ``[p - palm, g - palm, g - p]`` (the prior's keypoints are palm-relative), pooled by
    PointNet, then an MLP head to the ``out`` cloth values."""

    def __init__(self, out: int, units=(64, 128, 256), head=(256, 256)):
        super().__init__()
        self.points = PointNetEncoder(units)
        layers, width = [], self.points.out_dim
        for u in head:
            layers += [nn.Linear(width, u), nn.ELU()]
            width = u
        layers.append(nn.Linear(width, out))
        self.head = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor, goals: torch.Tensor, palm: torch.Tensor) -> torch.Tensor:
        palm = palm.unsqueeze(1)
        return self.head(self.points.pool(torch.cat([points - palm, goals - palm, goals - points], dim=-1)))


class PointNetA2CBuilder(network_builder.A2CBuilder):
    def build(self, name, **kwargs):
        return PointNetA2CBuilder.Network(self.params, **kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            pc = params["pointnet"]
            num_points = int(pc["num_points"])
            units = tuple(int(u) for u in pc.get("units", (64, 128, 256)))
            width = 6 * num_points
            input_shape = tuple(kwargs.pop("input_shape"))
            assert len(input_shape) == 1, "the point-cloud network takes a flat observation"
            # The env observation ends at the SAPG coefficient columns (or at the end).
            env_dim = kwargs.get("coef_id_idx", input_shape[0])
            if env_dim < width:
                raise ValueError(f"observation ({env_dim}) is shorter than the point cloud ({width})")
            start = env_dim - width
            shrink = width - units[-1]
            if "coef_id_idx" in kwargs:
                kwargs["coef_id_idx"] = env_dim - shrink
            super().__init__(params, input_shape=(input_shape[0] - shrink,), **kwargs)
            self.point_slice = (start, env_dim)
            self.num_points = num_points
            self.point_encoder = PointNetEncoder(units)

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            start, end = self.point_slice
            cloud = obs[:, start:end].reshape(obs.shape[0], 2, self.num_points, 3)
            feature = self.point_encoder(cloud[:, 0], cloud[:, 1])
            obs_dict = dict(obs_dict, obs=torch.cat([obs[:, :start], feature, obs[:, end:]], dim=1))
            return super().forward(obs_dict)


class PointNetAdapterBuilder(network_builder.A2CBuilder):
    """A pretrained state policy driven by points: the adapter encoder stands in for its cloth
    inputs, everything else (trunk, heads, robot inputs) is the pretrained network unchanged.

    ``network.adapter``: ``num_points``, ``segments`` (the pretrained observation as ordered
    ``[robot|object, width]`` runs; robot runs come from this env's observation in order, object
    runs from the encoder), ``palm_slice`` (raw palm_pos in this env's observation), ``units``,
    ``head``. The encoder predicts the cloth values already normalised by the pretrained
    statistics, so they slot straight in beside the normalised robot values."""

    def build(self, name, **kwargs):
        return PointNetAdapterBuilder.Network(self.params, **kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            ad = params["adapter"]
            num_points = int(ad["num_points"])
            self.segments = [(str(kind), int(n)) for kind, n in ad["segments"]]
            if any(kind not in ("robot", "object") for kind, _ in self.segments):
                raise ValueError(f"adapter segments must be robot or object runs, got {self.segments}")
            robot = sum(n for kind, n in self.segments if kind == "robot")
            cloth = sum(n for kind, n in self.segments if kind == "object")
            input_shape = tuple(kwargs.pop("input_shape"))
            env_dim = kwargs.get("coef_id_idx", input_shape[0])
            if env_dim != robot + 6 * num_points:
                raise ValueError(f"observation {env_dim} != robot {robot} + points {6 * num_points}")
            extra = input_shape[0] - env_dim
            if "coef_id_idx" in kwargs:
                kwargs["coef_id_idx"] = robot + cloth
            super().__init__(params, input_shape=(robot + cloth + extra,), **kwargs)
            self.point_slice = (robot, env_dim)
            self.palm_slice = tuple(int(i) for i in ad["palm_slice"])
            self.num_points = num_points
            self.adapter_encoder = AdapterEncoder(cloth, tuple(ad.get("units", (64, 128, 256))),
                                                  tuple(ad.get("head", (256, 256))))

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            raw = obs_dict.get("raw_obs", obs)
            start, end = self.point_slice
            cloud = raw[:, start:end].reshape(obs.shape[0], 2, self.num_points, 3)
            cloth = self.adapter_encoder(cloud[:, 0], cloud[:, 1], raw[:, self.palm_slice[0]:self.palm_slice[1]])
            parts, r, o = [], 0, 0
            for kind, n in self.segments:
                if kind == "robot":
                    parts.append(obs[:, r:r + n])
                    r += n
                else:
                    parts.append(cloth[:, o:o + n])
                    o += n
            parts.append(obs[:, end:])  # SAPG coefficient columns
            return super().forward(dict(obs_dict, obs=torch.cat(parts, dim=1)))


class PointNetModel(models.ModelA2CContinuousLogStd):
    """``continuous_a2c_logstd`` with the point slice exempt from input normalisation, and the raw
    observation handed to the network as ``raw_obs`` (the adapter reads the raw palm position)."""

    class Network(models.ModelA2CContinuousLogStd.Network):
        def forward(self, input_dict):
            input_dict["raw_obs"] = input_dict["obs"]
            return super().forward(input_dict)

        def norm_obs(self, observation):
            normed = super().norm_obs(observation)
            start, end = self.a2c_network.point_slice
            return torch.cat([normed[:, :start], observation[:, start:end], normed[:, end:]], dim=1)


model_builder.register_network("pointnet_actor_critic", PointNetA2CBuilder)
model_builder.register_network("pointnet_adapter_actor_critic", PointNetAdapterBuilder)
model_builder.register_model("pointnet_a2c_logstd", PointNetModel)
