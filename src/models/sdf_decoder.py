"""Small MLP decoding a Hexa-Plane feature vector into a signed distance value,
with an optional geometric initialization (Atzmon & Lipman style) so the SDF
starts close to a sphere and training is stable from step 0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SDFDecoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_layers: int = 3, geometric_init: bool = True):
        super().__init__()
        dims = [feature_dim + 3] + [hidden_dim] * (num_layers - 1) + [1]  # +3 for raw xyz skip connection
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.activation = nn.Softplus(beta=100)

        if geometric_init:
            self._geometric_init(dims)

    def _geometric_init(self, dims):
        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            if is_last:
                nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dims[i]), std=1e-4)
                nn.init.constant_(layer.bias, -0.5)  # start as an inward sphere of radius ~0.5
            else:
                nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dims[i + 1]))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, feature: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feature, xyz], dim=-1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x  # (N, 1) signed distance
