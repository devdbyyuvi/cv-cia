"""End-to-end feed-forward inverse-rendering network: Hexa-Plane backbone +
SDF decoder + decoupled {geometry, material, illumination} heads. A single
forward pass over a handful of input views yields the full disentangled scene
representation (geometry, SVBRDF, environment light), which the differentiable
renderer can then re-render under the original or a novel illumination.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .hexaplane import HexaPlaneField
from .sdf_decoder import SDFDecoder
from .heads import GeometryHead, MaterialHead, IlluminationHead


class ImageEncoder(nn.Module):
    """Lightweight CNN encoder that pools sparse input-view images into a global
    feature vector used to modulate the Hexa-Plane's auxiliary fusion planes."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (V, 3, H, W) input views -> (out_dim,) pooled across views."""
        feats = self.net(images).flatten(1)          # (V, 128)
        feats = feats.mean(0)                          # pool across views
        return self.proj(feats)


class InverseRenderingNetwork(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        hp = cfg["model"]["hexaplane"]
        heads = cfg["model"]["heads"]

        self.image_encoder = ImageEncoder(out_dim=128)
        self.hexaplane = HexaPlaneField(
            resolution=hp["resolution"],
            num_planes=hp["num_planes"],
            feature_dim=hp["feature_dim"],
            scene_bound=hp["scene_bound"],
            image_feature_dim=128,
        )
        self.sdf_decoder = SDFDecoder(
            feature_dim=hp["feature_dim"],
            hidden_dim=cfg["model"]["sdf_decoder"]["hidden_dim"],
            num_layers=cfg["model"]["sdf_decoder"]["num_layers"],
            geometric_init=cfg["model"]["sdf_decoder"]["geometric_init"],
        )
        self.geometry_head = GeometryHead(hp["feature_dim"], **_only(heads["geometry"], GeometryHead))
        self.material_head = MaterialHead(hp["feature_dim"], **_only(heads["material"], MaterialHead))
        self.illumination_head = IlluminationHead(128, **_only(heads["illumination"], IlluminationHead))

        self._cached_image_feature: Optional[torch.Tensor] = None

    def encode_views(self, images: torch.Tensor) -> torch.Tensor:
        """Call once per scene/object before querying the field; caches the
        pooled multi-view image feature used to condition the Hexa-Plane."""
        self._cached_image_feature = self.image_encoder(images)
        return self._cached_image_feature

    def sdf_fn(self, points: torch.Tensor) -> torch.Tensor:
        feat = self.hexaplane(points, self._cached_image_feature)
        return self.sdf_decoder(feat, points)

    def material_fn(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.hexaplane(points, self._cached_image_feature)
        return self.material_head(feat)

    def geometry_fn(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.hexaplane(points, self._cached_image_feature)
        return self.geometry_head(feat)

    def illumination_fn(self) -> Dict[str, torch.Tensor]:
        assert self._cached_image_feature is not None, "call encode_views() first"
        return self.illumination_head(self._cached_image_feature)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience full pass: encode views, then predict everything at
        `query_points` (e.g. surface points found by ray marching)."""
        self.encode_views(images)
        sdf = self.sdf_fn(query_points)
        material = self.material_fn(query_points)
        geometry = self.geometry_fn(query_points)
        illumination = self.illumination_fn()
        return {"sdf": sdf, **material, **geometry, **illumination}


def _only(d: Dict, cls) -> Dict:
    """Filter a config dict down to the kwargs a head class actually accepts."""
    import inspect
    valid = set(inspect.signature(cls.__init__).parameters) - {"self", "feature_dim"}
    return {k: v for k, v in d.items() if k in valid}
