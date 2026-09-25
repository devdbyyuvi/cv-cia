"""Decoupled multi-head prediction: each head consumes the shared Hexa-Plane
feature (and, for geometry, the SDF decoder's intermediate signal) and predicts
one disentangled component of the scene -- geometry, material, or illumination.
Keeping these heads architecturally separate (rather than one monolithic
decoder) is what lets Phase 3's diffusion prior attach *only* to the material
head's output distribution without disturbing geometry or lighting.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class GeometryHead(nn.Module):
    """Predicts surface normal residual + local depth confidence from Hexa-Plane
    features (used to complement/verify the analytic SDF-gradient normal from
    the renderer, and to support a fast normal/depth-only rendering path)."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 predict_normals: bool = True, predict_depth: bool = True):
        super().__init__()
        self.predict_normals = predict_normals
        self.predict_depth = predict_depth
        out_dim = (3 if predict_normals else 0) + (1 if predict_depth else 0)
        self.mlp = _mlp(feature_dim, hidden_dim, out_dim, num_layers)

    def forward(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.mlp(feature)
        result = {}
        idx = 0
        if self.predict_normals:
            result["normal_residual"] = out[..., idx:idx + 3]
            idx += 3
        if self.predict_depth:
            result["depth_confidence"] = torch.sigmoid(out[..., idx:idx + 1])
        return result


class MaterialHead(nn.Module):
    """Predicts spatially-varying BRDF parameters: albedo (base color), roughness,
    and metallic, from Hexa-Plane features. This is the head regularized by the
    material diffusion prior in Phase 3."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 albedo_channels: int = 3, predict_roughness: bool = True, predict_metallic: bool = True):
        super().__init__()
        self.albedo_channels = albedo_channels
        self.predict_roughness = predict_roughness
        self.predict_metallic = predict_metallic

        self.trunk = _mlp(feature_dim, hidden_dim, hidden_dim, max(1, num_layers - 1))
        self.albedo_out = nn.Linear(hidden_dim, albedo_channels)
        self.roughness_out = nn.Linear(hidden_dim, 1) if predict_roughness else None
        self.metallic_out = nn.Linear(hidden_dim, 1) if predict_metallic else None

    def forward(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = F.relu(self.trunk(feature))
        out = {"albedo": torch.sigmoid(self.albedo_out(h))}
        if self.roughness_out is not None:
            out["roughness"] = torch.sigmoid(self.roughness_out(h)).clamp(0.04, 1.0)
        if self.metallic_out is not None:
            out["metallic"] = torch.sigmoid(self.metallic_out(h))
        out["material_latent"] = h  # exposed for the diffusion-prior adapter (Phase 3)
        return out


class IlluminationHead(nn.Module):
    """Predicts a low-frequency environment lighting estimate as a mixture of
    spherical Gaussian lobes, from a *global* (pooled across the scene / input
    views) Hexa-Plane feature -- lighting is scene-level, not per-point."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_sg_lobes: int = 24, predict_ambient: bool = True):
        super().__init__()
        self.num_sg_lobes = num_sg_lobes
        self.predict_ambient = predict_ambient

        out_dim = num_sg_lobes * (3 + 1 + 3)  # dir(3) + sharpness(1) + intensity(3) per lobe
        if predict_ambient:
            out_dim += 3
        self.mlp = _mlp(feature_dim, hidden_dim, out_dim, num_layers=2)

    def forward(self, global_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.mlp(global_feature)
        k = self.num_sg_lobes
        dirs = F.normalize(out[..., :k * 3].view(*out.shape[:-1], k, 3), dim=-1)
        sharpness = out[..., k * 3:k * 4].view(*out.shape[:-1], k, 1)
        intensity = out[..., k * 4:k * 7].view(*out.shape[:-1], k, 3)
        result = {"lobe_dirs": dirs, "lobe_sharpness": sharpness, "lobe_intensity": intensity}
        if self.predict_ambient:
            result["ambient"] = F.softplus(out[..., k * 7:k * 7 + 3])
        return result
