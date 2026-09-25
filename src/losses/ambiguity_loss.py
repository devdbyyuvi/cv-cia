"""Compound loss terms that penalize physically implausible material-lighting
combinations -- the core tool for resolving the material/lighting ambiguity
(a dark albedo under bright light is pixel-identical to a bright albedo under
dim light) described in Phase 3.

Two complementary terms:

1. `energy_conservation_term`: penalizes albedo+specular reflectance exceeding
   physical energy-conservation bounds (diffuse+specular reflectance should not
   exceed incident energy), which rules out "cheating" material solutions that
   only work under one specific baked-in lighting guess.
2. `albedo_prior_consistency_term`: penalizes divergence between the predicted
   albedo and the material diffusion prior's albedo estimate (see
   src/diffusion_prior), pulling the network away from ambiguous
   lighting-baked-into-albedo solutions and toward priors learned from real
   material statistics.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


def energy_conservation_term(albedo: torch.Tensor, roughness: torch.Tensor, metallic: torch.Tensor) -> torch.Tensor:
    f0 = 0.04 * (1.0 - metallic) + albedo * metallic
    # Rough upper bound on total (diffuse+specular) directional-hemispherical
    # reflectance; anything pushing this over 1 violates energy conservation.
    approx_total_reflectance = albedo * (1.0 - metallic) + f0
    violation = (approx_total_reflectance - 1.0).clamp(min=0.0)
    return (violation ** 2).mean()


def albedo_prior_consistency_term(pred_albedo: torch.Tensor, prior_albedo: torch.Tensor) -> torch.Tensor:
    return (pred_albedo - prior_albedo).abs().mean()


def ambiguity_penalty(
    material: Dict[str, torch.Tensor],
    prior_albedo: Optional[torch.Tensor] = None,
    energy_conservation_weight: float = 0.5,
    albedo_prior_consistency_weight: float = 0.5,
) -> torch.Tensor:
    loss = energy_conservation_weight * energy_conservation_term(
        material["albedo"], material["roughness"], material["metallic"]
    )
    if prior_albedo is not None:
        loss = loss + albedo_prior_consistency_weight * albedo_prior_consistency_term(
            material["albedo"], prior_albedo
        )
    return loss
