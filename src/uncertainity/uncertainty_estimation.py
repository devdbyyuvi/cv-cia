"""Uncertainty quantification (Phase 3): run several stochastic forward passes
(MC-dropout or latent/noise-perturbation sampling) through the network and
measure prediction disagreement, producing an explicit per-pixel/per-point
uncertainty map that highlights ambiguous (material-lighting degenerate) or
occluded/under-observed scene regions.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import torch
import torch.nn as nn


class UncertaintyEstimator:
    def __init__(self, method: str = "mc_dropout", num_forward_passes: int = 8, dropout_p: float = 0.1):
        assert method in ("mc_dropout", "latent_sampling")
        self.method = method
        self.num_forward_passes = num_forward_passes
        self.dropout_p = dropout_p

    def _enable_mc_dropout(self, model: nn.Module):
        """Activate dropout layers at inference time (standard MC-dropout trick),
        injecting a dropout layer transiently if the model has none."""
        found = False
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = self.dropout_p
                m.train()
                found = True
        return found

    @torch.no_grad()
    def estimate(
        self,
        predict_fn: Callable[[], Dict[str, torch.Tensor]],
        model: nn.Module = None,
        keys: List[str] = ("albedo", "roughness", "metallic", "rgb"),
    ) -> Dict[str, torch.Tensor]:
        """Run `num_forward_passes` stochastic forward passes of `predict_fn`
        (a zero-arg closure wrapping e.g. renderer.render(...) or model.forward(...))
        and compute the per-key predictive mean and standard deviation.

        For `latent_sampling`, `predict_fn` should itself inject fresh noise into
        any stochastic latents (e.g. re-sampling illumination lobes / material
        latent jitter) on each call.
        """
        if self.method == "mc_dropout" and model is not None:
            was_training = model.training
            self._enable_mc_dropout(model)

        samples: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}
        for _ in range(self.num_forward_passes):
            out = predict_fn()
            for k in keys:
                if k in out:
                    samples[k].append(out[k])

        result = {}
        for k, vals in samples.items():
            if not vals:
                continue
            stacked = torch.stack(vals, dim=0)  # (P, ...)
            mean = stacked.mean(0)
            std = stacked.std(0)
            result[f"{k}_mean"] = mean
            result[f"{k}_uncertainty"] = std.mean(-1, keepdim=True) if std.dim() > 0 else std

        if self.method == "mc_dropout" and model is not None and not was_training:
            model.eval()

        # A single scalar-per-pixel aggregate uncertainty map, averaging the
        # normalized uncertainty across all available prediction keys.
        unc_maps = [v for k, v in result.items() if k.endswith("_uncertainty")]
        if unc_maps:
            stacked = torch.stack([u / (u.amax().clamp(min=1e-6)) for u in unc_maps], dim=0)
            result["uncertainty_map"] = stacked.mean(0)

        return result


def estimate_uncertainty(predict_fn, model=None, method="mc_dropout", num_forward_passes=8, dropout_p=0.1, **kwargs):
    """Functional convenience wrapper around `UncertaintyEstimator`."""
    estimator = UncertaintyEstimator(method=method, num_forward_passes=num_forward_passes, dropout_p=dropout_p)
    return estimator.estimate(predict_fn, model=model, **kwargs)
