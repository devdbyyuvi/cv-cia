"""Differentiable rendering module: maps (surface point, normal, BRDF params,
environment lighting) -> pixel color, and is the sole bridge through which
photometric losses back-propagate into geometry, material, and illumination heads.

This module is intentionally self-contained (pure PyTorch, sphere-traced SDF
ray marching + analytic BRDF integration) so the project has no hard
dependency on an external differentiable rasterizer. `nvdiffrast` / `redner`
can be substituted by replacing `ray_march` with a mesh rasterization pass and
keeping `shade()` unchanged, since `shade()` only consumes per-pixel
(point, normal, material) buffers.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .brdf import cook_torrance_ggx
from .sh_lighting import SphericalGaussianLighting


class DifferentiableRenderer:
    def __init__(
        self,
        num_secondary_rays: int = 0,
        background: str = "white",
        march_steps: int = 64,
        march_max_dist: float = 4.0,
        sphere_trace_iters: int = 48,
    ):
        self.num_secondary_rays = num_secondary_rays
        self.background = background
        self.march_steps = march_steps
        self.march_max_dist = march_max_dist
        self.sphere_trace_iters = sphere_trace_iters

    # ------------------------------------------------------------------ #
    # Geometry: sphere-trace the predicted SDF to find surface intersections.
    # ------------------------------------------------------------------ #
    def ray_march(
        self,
        sdf_fn: Callable[[torch.Tensor], torch.Tensor],
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Sphere-trace rays against a neural SDF.

        Args:
            sdf_fn: callable mapping (N, 3) world points -> (N, 1) signed distance.
            rays_o, rays_d: (N, 3) ray origins / normalized directions.

        Returns:
            dict with `points` (N,3), `hit_mask` (N,1) bool, `depth` (N,1).
        """
        t = torch.zeros(rays_o.shape[0], 1, device=rays_o.device)
        for _ in range(self.sphere_trace_iters):
            pts = rays_o + t * rays_d
            dist = sdf_fn(pts)
            t = t + dist.clamp(min=0.0)
            t = torch.clamp(t, max=self.march_max_dist)

        pts = rays_o + t * rays_d
        final_dist = sdf_fn(pts)
        hit_mask = (final_dist.abs() < 1e-2) & (t < self.march_max_dist)
        return {"points": pts, "hit_mask": hit_mask, "depth": t}

    @staticmethod
    def compute_normals(
        sdf_fn: Callable[[torch.Tensor], torch.Tensor], points: torch.Tensor, eps: float = 1e-3
    ) -> torch.Tensor:
        """Surface normal via finite-difference SDF gradient (differentiable)."""
        offsets = torch.eye(3, device=points.device) * eps
        grads = []
        for i in range(3):
            plus = sdf_fn(points + offsets[i])
            minus = sdf_fn(points - offsets[i])
            grads.append((plus - minus) / (2 * eps))
        normal = torch.cat(grads, dim=-1)
        return F.normalize(normal, dim=-1)

    # ------------------------------------------------------------------ #
    # Shading: BRDF + SG environment lighting -> outgoing radiance.
    # ------------------------------------------------------------------ #
    def shade(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        albedo: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        lighting: SphericalGaussianLighting,
    ) -> torch.Tensor:
        """Integrate BRDF against SG environment lighting for one batch of
        surface points. Uses the dominant SG lobe directions as discrete
        "light directions" (standard SG-relighting approximation), plus a
        closed-form diffuse irradiance term.
        """
        diffuse = lighting.integrate_diffuse(normals, albedo * (1.0 - metallic))

        specular = torch.zeros_like(diffuse)
        k = lighting.dirs.shape[-2] if lighting.dirs.dim() > 1 else lighting.dirs.shape[0]
        for i in range(k):
            light_dir = lighting.dirs[..., i, :] if lighting.dirs.dim() > 1 else lighting.dirs[i]
            light_dir = light_dir.expand_as(points)
            radiance = lighting.eval(light_dir)
            specular = specular + cook_torrance_ggx(
                albedo, roughness, metallic, normals, view_dirs, light_dir, radiance
            ) / max(k, 1)

        color = diffuse + specular
        return color.clamp(0.0, 1.0) if not color.requires_grad else color

    # ------------------------------------------------------------------ #
    # Full forward render: rays -> pixel colors, given model prediction fns.
    # ------------------------------------------------------------------ #
    def render(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        sdf_fn: Callable[[torch.Tensor], torch.Tensor],
        material_fn: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
        lighting: SphericalGaussianLighting,
    ) -> Dict[str, torch.Tensor]:
        """End-to-end differentiable render of a batch of rays.

        material_fn(points) -> {'albedo': (N,3), 'roughness': (N,1), 'metallic': (N,1)}
        """
        march = self.ray_march(sdf_fn, rays_o, rays_d)
        points, hit_mask = march["points"], march["hit_mask"]

        normals = self.compute_normals(sdf_fn, points)
        mats = material_fn(points)
        view_dirs = -rays_d

        colors = self.shade(
            points, normals, view_dirs, mats["albedo"], mats["roughness"], mats["metallic"], lighting
        )

        bg_value = 1.0 if self.background == "white" else 0.0
        bg = torch.full_like(colors, bg_value)
        colors = torch.where(hit_mask, colors, bg)

        return {
            "rgb": colors,
            "depth": march["depth"],
            "normal": normals,
            "hit_mask": hit_mask,
            "albedo": mats["albedo"],
            "roughness": mats["roughness"],
            "metallic": mats["metallic"],
        }
