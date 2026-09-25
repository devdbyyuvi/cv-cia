"""Loss term that regularizes the material head's predictions toward the
distribution captured by a pretrained material diffusion prior (Phase 3). This
is a distillation-style loss: the diffusion prior denoiser, evaluated at the
predicted material latent + a sampled noise level, tells us the score
(gradient direction toward higher data likelihood); we push the prediction a
small step in that direction, analogous to Score Distillation Sampling (SDS)
as used to regularize NeRF/material fields with 2D diffusion priors.
"""
from __future__ import annotations

from typing import Dict

import torch

from src.diffusion_prior.material_prior import MaterialDiffusionPrior


def diffusion_prior_loss(
    material: Dict[str, torch.Tensor],
    prior: MaterialDiffusionPrior,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    """Score-distillation-style loss pulling predicted albedo/roughness/metallic
    maps toward the pretrained material prior's learned manifold.

    Returns a scalar loss whose gradient (via straight-through score matching)
    nudges `material` predictions toward higher likelihood under the prior.
    """
    material_maps = torch.cat(
        [material["albedo"], material["roughness"], material["metallic"]], dim=-1
    )  # (..., 5): albedo(3) + roughness(1) + metallic(1)

    score = prior.predict_score(material_maps, guidance_scale=guidance_scale)  # (..., 5)

    # SDS-style loss: gradient w.r.t. material_maps is set to -score (i.e. move
    # toward the prior's denoised estimate); implemented via a detached target
    # so autograd produces exactly that gradient through `material_maps`.
    target = (material_maps - score).detach()
    loss = 0.5 * ((material_maps - target) ** 2).mean()
    return loss
