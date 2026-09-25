"""Low-frequency environment lighting represented as a mixture of Spherical
Gaussian (SG) lobes: `L(w) = sum_i intensity_i * exp(sharpness_i * (dot(w, dir_i) - 1))`.

SGs give a compact, differentiable, closed-form-integrable representation of
environment illumination -- well suited to a lightweight feed-forward
illumination head (Phase 2) and to swapping in a *novel held-out* environment
at test time for the relighting validation suite (Phase 4).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class SphericalGaussianLighting:
    def __init__(self, lobe_dirs: torch.Tensor, lobe_sharpness: torch.Tensor, lobe_intensity: torch.Tensor):
        """
        lobe_dirs: (..., K, 3) unit vectors, environment head output (pre-normalized).
        lobe_sharpness: (..., K, 1) positive scalars (higher = tighter highlight).
        lobe_intensity: (..., K, 3) RGB radiance amplitude per lobe.
        """
        self.dirs = F.normalize(lobe_dirs, dim=-1)
        self.sharpness = F.softplus(lobe_sharpness) + 1e-2
        self.intensity = F.softplus(lobe_intensity)

    def eval(self, directions: torch.Tensor) -> torch.Tensor:
        """Evaluate the environment radiance along arbitrary query directions.

        directions: (..., 3)
        returns: (..., 3) incident radiance.
        """
        # Broadcast query directions against the K lobes.
        d = directions.unsqueeze(-2)                      # (..., 1, 3)
        lobe_dirs = self.dirs                              # (..., K, 3) or (K, 3)
        cos = (d * lobe_dirs).sum(-1, keepdim=True)         # (..., K, 1)
        weight = torch.exp(self.sharpness * (cos - 1.0))    # (..., K, 1)
        radiance = (weight * self.intensity).sum(-2)        # (..., 3)
        return radiance

    def integrate_diffuse(self, normal: torch.Tensor, albedo: torch.Tensor) -> torch.Tensor:
        """Closed-form-ish approximate irradiance integral of SG lighting against
        a cosine lobe at `normal`, scaled by Lambertian albedo. Uses the standard
        SG x cosine-lobe convolution approximation (sharpness attenuation)."""
        lobe_dirs = self.dirs
        cos_lobe_sharpness = 2.133  # cosine lobe approximated as an SG with this sharpness
        n = normal.unsqueeze(-2)
        cos = (n * lobe_dirs).sum(-1, keepdim=True).clamp(min=0.0)

        # Approximate SG-cosine convolution via a smooth blend between the
        # original lobe amplitude (sharp highlight regime) and a properly
        # normalized diffuse contribution (broad lobe regime).
        blended_sharpness = (self.sharpness * cos_lobe_sharpness) / (self.sharpness + cos_lobe_sharpness)
        norm_factor = blended_sharpness / (self.sharpness + 1e-6)
        irradiance = (norm_factor * self.intensity * cos).sum(-2)

        return albedo * irradiance / torch.pi

    def sample_directions(self, num_samples: int) -> torch.Tensor:
        """Importance-sample directions roughly following the dominant lobes,
        used for stochastic specular relighting when `num_secondary_rays > 0`."""
        idx = torch.randint(0, self.dirs.shape[-2], (num_samples,), device=self.dirs.device)
        base_dirs = self.dirs[..., idx, :]
        jitter = torch.randn_like(base_dirs) * 0.05
        return F.normalize(base_dirs + jitter, dim=-1)

    @staticmethod
    def from_envmap_fit(npz_path: str, device: str = "cpu") -> "SphericalGaussianLighting":
        """Load a precomputed SG fit (e.g. produced offline by fitting an .hdr
        envmap with least-squares SG decomposition) for use as a *novel held-out*
        illumination in relighting validation."""
        import numpy as np
        data = np.load(npz_path)
        return SphericalGaussianLighting(
            torch.from_numpy(data["lobe_dirs"]).float().to(device),
            torch.from_numpy(data["lobe_sharpness"]).float().to(device),
            torch.from_numpy(data["lobe_intensity"]).float().to(device),
        )
