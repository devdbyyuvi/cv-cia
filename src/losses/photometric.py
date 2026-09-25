"""Standard photometric reconstruction losses used to train the feed-forward
backbone end-to-end (Phase 2) and for self-supervised fine-tuning on real
captures (Phase 4, where photometric consistency is the *only* supervision
signal available since GT geometry/BRDF don't exist for real photos)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def photometric_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    mask: torch.Tensor = None,
    l1_weight: float = 0.8,
    l2_weight: float = 0.2,
) -> torch.Tensor:
    """L1/L2 mix, optionally masked to the foreground object."""
    if mask is not None:
        pred_rgb = pred_rgb * mask
        gt_rgb = gt_rgb * mask
        denom = mask.sum().clamp(min=1.0) * pred_rgb.shape[-1]
    else:
        denom = pred_rgb.numel()

    l1 = F.l1_loss(pred_rgb, gt_rgb, reduction="sum") / denom
    l2 = F.mse_loss(pred_rgb, gt_rgb, reduction="sum") / denom
    return l1_weight * l1 + l2_weight * l2
