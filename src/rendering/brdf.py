"""Analytic microfacet BRDF (Cook-Torrance / GGX) used by the differentiable renderer.

All functions are fully differentiable w.r.t. albedo, roughness, metallic,
normals, and light/view directions, so gradients flow back into the geometry
and material heads during training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1, keepdim=True).clamp(min=0.0)


def _ggx_distribution(n_dot_h: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    a = roughness * roughness
    a2 = a * a
    denom = n_dot_h ** 2 * (a2 - 1.0) + 1.0
    return a2 / (torch.pi * denom ** 2 + EPS)


def _smith_geometry(n_dot_v: torch.Tensor, n_dot_l: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    k = ((roughness + 1.0) ** 2) / 8.0

    def g1(n_dot_x):
        return n_dot_x / (n_dot_x * (1.0 - k) + k + EPS)

    return g1(n_dot_v) * g1(n_dot_l)


def _fresnel_schlick(v_dot_h: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    return f0 + (1.0 - f0) * torch.pow((1.0 - v_dot_h).clamp(0.0, 1.0), 5.0)


def cook_torrance_ggx(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normal: torch.Tensor,
    view_dir: torch.Tensor,
    light_dir: torch.Tensor,
    light_radiance: torch.Tensor,
) -> torch.Tensor:
    """Evaluate outgoing radiance for a single point-ish light direction.

    Shapes: all tensors broadcastable to (..., 3) except roughness/metallic which
    are (..., 1). `light_radiance` is (..., 3) incoming radiance along `light_dir`.

    Returns:
        (..., 3) outgoing radiance contribution from this light direction.
    """
    normal = F.normalize(normal, dim=-1)
    view_dir = F.normalize(view_dir, dim=-1)
    light_dir = F.normalize(light_dir, dim=-1)
    half_dir = F.normalize(view_dir + light_dir, dim=-1)

    n_dot_v = _dot(normal, view_dir).clamp(min=EPS)
    n_dot_l = _dot(normal, light_dir).clamp(min=EPS)
    n_dot_h = _dot(normal, half_dir).clamp(min=EPS)
    v_dot_h = _dot(view_dir, half_dir).clamp(min=EPS)

    roughness = roughness.clamp(0.04, 1.0)
    f0 = 0.04 * (1.0 - metallic) + albedo * metallic

    D = _ggx_distribution(n_dot_h, roughness)
    G = _smith_geometry(n_dot_v, n_dot_l, roughness)
    F_ = _fresnel_schlick(v_dot_h, f0)

    specular = (D * G * F_) / (4.0 * n_dot_v * n_dot_l + EPS)

    k_diffuse = (1.0 - F_) * (1.0 - metallic)
    diffuse = k_diffuse * albedo / torch.pi

    brdf = diffuse + specular
    return brdf * light_radiance * n_dot_l
