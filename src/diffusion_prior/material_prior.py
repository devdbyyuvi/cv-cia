"""Adapter interface around a *pretrained* material diffusion prior, in the
style of MaterialFusion / IntrinsicAnything: a diffusion model trained on large
corpora of real-world material scans, used here purely as a frozen prior for
regularizing the material head's BRDF predictions (Phase 3), never fine-tuned.

This module intentionally does NOT vendor third-party model weights. Instead:

- `MaterialDiffusionPrior` defines the adapter contract (`predict_score`,
  `denoise`) that any concrete backend must implement.
- `load_pretrained_backend()` shows how a checkpoint would be loaded (left as
  an integration point -- point it at a locally downloaded MaterialFusion /
  IntrinsicAnything checkpoint).
- `NullMaterialPrior` is a no-op fallback (returns zero score) so the full
  pipeline, including `diffusion_prior_loss`, still runs end-to-end for
  optimization-only ablations when `diffusion_prior.enabled: false`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MaterialDiffusionPrior(nn.Module):
    """Abstract adapter contract for a frozen material diffusion prior."""

    def predict_score(self, material_maps: torch.Tensor, guidance_scale: float = 2.0) -> torch.Tensor:
        """Given a batch of (albedo, roughness, metallic) material samples,
        return an estimate of the denoised target (used as an SDS-style
        regularization target). Shape-preserving: (..., 5) -> (..., 5)."""
        raise NotImplementedError

    def denoise(self, noisy_material: torch.Tensor, timestep: int) -> torch.Tensor:
        """Single reverse-diffusion denoising step, exposed for direct
        material-map generation/inpainting use cases outside the SDS loss."""
        raise NotImplementedError


class NullMaterialPrior(MaterialDiffusionPrior):
    """No-op prior: zero score everywhere. Used when
    `diffusion_prior.enabled: false`, e.g. for the optimization-only ablation
    baseline in Phase 5's ablation study."""

    def predict_score(self, material_maps: torch.Tensor, guidance_scale: float = 2.0) -> torch.Tensor:
        return torch.zeros_like(material_maps)

    def denoise(self, noisy_material: torch.Tensor, timestep: int) -> torch.Tensor:
        return noisy_material


class _UNetMaterialPriorStub(MaterialDiffusionPrior):
    """Minimal placeholder small-UNet-style denoiser with the correct I/O
    contract, so the pipeline is runnable and testable before a real
    pretrained checkpoint is wired in. NOT a substitute for an actual
    pretrained prior -- replace via `load_pretrained_backend`.
    """

    def __init__(self, channels: int = 5, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, channels),
        )
        for p in self.parameters():
            p.requires_grad_(False)  # frozen prior

    @torch.no_grad()
    def predict_score(self, material_maps: torch.Tensor, guidance_scale: float = 2.0) -> torch.Tensor:
        denoised = self.net(material_maps)
        return guidance_scale * (material_maps - denoised)

    @torch.no_grad()
    def denoise(self, noisy_material: torch.Tensor, timestep: int) -> torch.Tensor:
        return self.net(noisy_material)


def load_pretrained_backend(backend: str, ckpt_path: Optional[str], device: str = "cpu") -> MaterialDiffusionPrior:
    """Integration point for a real pretrained material diffusion prior.

    Expected usage once weights are available locally:

        prior = load_pretrained_backend("materialfusion", "/path/to/materialfusion.ckpt")

    Replace the stub branch below with the actual model construction + weight
    loading code for whichever backend (MaterialFusion / IntrinsicAnything /
    other) checkpoint format you have on disk.
    """
    if backend == "none" or ckpt_path is None:
        return NullMaterialPrior()

    if not __import__("os").path.exists(ckpt_path):
        import warnings
        warnings.warn(
            f"diffusion_prior.ckpt_path='{ckpt_path}' not found; falling back to an "
            "untrained stub prior. Results with diffusion_prior.enabled=true will not "
            "reflect a real pretrained material prior until a checkpoint is supplied."
        )
        return _UNetMaterialPriorStub().to(device)

    # --- Real backend loading would go here, e.g.: ---
    # state_dict = torch.load(ckpt_path, map_location=device)
    # model = MaterialFusionUNet(...)
    # model.load_state_dict(state_dict)
    # return MaterialFusionAdapter(model).to(device)
    raise NotImplementedError(
        f"No concrete loader implemented for backend='{backend}'. "
        "Add the checkpoint-loading logic for your specific pretrained prior here."
    )


def build_material_prior(cfg: dict, device: str = "cpu") -> MaterialDiffusionPrior:
    dp_cfg = cfg["diffusion_prior"]
    if not dp_cfg.get("enabled", False):
        return NullMaterialPrior()
    return load_pretrained_backend(dp_cfg["backend"], dp_cfg.get("ckpt_path"), device=device)
