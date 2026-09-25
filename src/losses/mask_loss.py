"""Silhouette/mask supervision and the eikonal regularizer that keeps the
learned field a valid signed distance function (|grad SDF| == 1)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_loss(pred_hit_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """Binary-cross-entropy between the predicted (soft) hit probability and
    the ground-truth foreground silhouette mask."""
    pred = pred_hit_mask.float().clamp(1e-4, 1 - 1e-4)
    return F.binary_cross_entropy(pred, gt_mask.float())


def eikonal_loss(sdf_fn, points: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Penalizes deviation of the SDF gradient norm from 1, via finite
    differences (cheap, no need for autograd.grad through the whole backbone)."""
    offsets = torch.eye(3, device=points.device) * eps
    grads = []
    for i in range(3):
        plus = sdf_fn(points + offsets[i])
        minus = sdf_fn(points - offsets[i])
        grads.append((plus - minus) / (2 * eps))
    grad = torch.cat(grads, dim=-1)
    grad_norm = grad.norm(dim=-1)
    return ((grad_norm - 1.0) ** 2).mean()
