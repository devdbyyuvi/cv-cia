"""Hexa-Plane feature grid: a generalization of tri-plane (K-Planes / EG3D style)
representations to six learned 2D feature planes, giving a compact,
memory-efficient, and fast-to-query volumetric feature field. Queried features
feed the SDF decoder and the geometry/material/illumination heads, enabling
feed-forward (single forward pass, no per-scene optimization) inference from
sparse multi-view inputs.

We use three "spatial" planes (XY, XZ, YZ) -- the standard tri-plane
decomposition of a 3D feature volume -- plus three "auxiliary" planes (XY', XZ',
YZ') at a coarser resolution that are conditioned on multi-view image features
and act as a lightweight cross-view fusion channel. This is what makes the grid
"hexa" (6-plane) rather than plain tri-plane: the auxiliary planes let
information from sparse input views (as few as 3) propagate into the 3D field
without a full per-point transformer/attention pass, keeping inference fast.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


PLANE_AXES = [(0, 1), (0, 2), (1, 2), (0, 1), (0, 2), (1, 2)]  # xy, xz, yz, xy', xz', yz'


class HexaPlaneField(nn.Module):
    def __init__(
        self,
        resolution: List[int],
        num_planes: int = 6,
        feature_dim: int = 32,
        scene_bound: float = 1.5,
        image_feature_dim: int = 0,
    ):
        """
        resolution: list of 6 ints, per-plane spatial resolution (spatial planes
            typically higher-res than the auxiliary fusion planes).
        feature_dim: channels per plane.
        scene_bound: query points are expected in [-scene_bound, scene_bound]^3.
        image_feature_dim: if >0, the 3 auxiliary planes are additionally
            modulated by a projected multi-view image feature (cross-view fusion).
        """
        super().__init__()
        assert num_planes == 6, "HexaPlaneField is defined as a 6-plane grid by construction."
        assert len(resolution) == 6

        self.scene_bound = scene_bound
        self.feature_dim = feature_dim
        self.image_feature_dim = image_feature_dim

        self.planes = nn.ParameterList([
            nn.Parameter(0.1 * torch.randn(1, feature_dim, resolution[i], resolution[i]))
            for i in range(6)
        ])

        if image_feature_dim > 0:
            # Projects fused multi-view image features onto the 3 auxiliary planes.
            self.aux_modulator = nn.Sequential(
                nn.Linear(image_feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim),
            )

    def _normalize(self, coords_2d: torch.Tensor) -> torch.Tensor:
        # coords in [-scene_bound, scene_bound] -> grid_sample expects [-1, 1]
        return (coords_2d / self.scene_bound).clamp(-1.0, 1.0)

    def forward(self, points: torch.Tensor, image_features: torch.Tensor = None) -> torch.Tensor:
        """
        points: (N, 3) world-space query points.
        image_features: optional (image_feature_dim,) or (N, image_feature_dim)
            pooled multi-view feature used to modulate the auxiliary planes.

        Returns: (N, feature_dim) fused hexa-plane feature per point (sum over the
            6 planes' bilinearly-interpolated features, following the standard
            tri-plane additive-fusion convention).
        """
        n = points.shape[0]
        feats = 0.0
        for i, (a, b) in enumerate(PLANE_AXES):
            coords = torch.stack([points[:, a], points[:, b]], dim=-1)  # (N, 2)
            grid = self._normalize(coords).view(1, n, 1, 2)
            sampled = F.grid_sample(
                self.planes[i], grid, mode="bilinear", padding_mode="border", align_corners=True
            )  # (1, C, N, 1)
            sampled = sampled.view(self.feature_dim, n).permute(1, 0)  # (N, C)

            if i >= 3 and self.image_feature_dim > 0 and image_features is not None:
                mod = self.aux_modulator(image_features)
                if mod.dim() == 1:
                    mod = mod.unsqueeze(0).expand(n, -1)
                sampled = sampled * torch.sigmoid(mod)

            feats = feats + sampled
        return feats

    def total_variation_loss(self) -> torch.Tensor:
        """Optional spatial-smoothness regularizer on the plane features."""
        tv = 0.0
        for p in self.planes:
            tv = tv + (p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean()
            tv = tv + (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean()
        return tv
