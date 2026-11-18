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

    def __init__(self, units=(64, 128, 256)):
        super().__init__()
        layers, width = [], POINT_FEATURES
        for i, u in enumerate(units):
            layers.append(nn.Linear(width, u))
            if i < len(units) - 1:
                layers += [nn.LayerNorm(u), nn.ELU()]
            width = u
        self.mlp = nn.Sequential(*layers)
        self.out_dim = width

    def forward(self, points: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        x = torch.cat([points, goals, goals - points], dim=-1)
        return self.mlp(x).amax(dim=-2)


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


class PointNetModel(models.ModelA2CContinuousLogStd):
    """``continuous_a2c_logstd`` with the point slice exempt from input normalisation."""

    class Network(models.ModelA2CContinuousLogStd.Network):
        def norm_obs(self, observation):
            normed = super().norm_obs(observation)
            start, end = self.a2c_network.point_slice
            return torch.cat([normed[:, :start], observation[:, start:end], normed[:, end:]], dim=1)


model_builder.register_network("pointnet_actor_critic", PointNetA2CBuilder)
model_builder.register_model("pointnet_a2c_logstd", PointNetModel)
