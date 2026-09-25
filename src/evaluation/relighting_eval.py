"""Phase 4: the strict relighting validation test. Unlike simple input-view
reconstruction (which a model can "cheat" by baking lighting into albedo),
this suite:

  1. Loads the recovered 3D asset (geometry + SVBRDF) from a fine-tuned checkpoint.
  2. Swaps in a *novel, held-out* environment illumination (an SG fit of an .hdr
     envmap that was never seen during training/fine-tuning).
  3. Renders the asset under that new lighting from a specified viewpoint.
  4. Compares the render quantitatively (PSNR/SSIM/LPIPS) against a *real
     photograph* of the same object actually captured under that lighting.

This is the test that actually validates geometry/material/lighting
disentanglement -- a model that merely memorized appearance will fail here
even if it reconstructs input views perfectly.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.data.real_capture_dataset import RealCaptureDataset, _read_image
from src.data.transforms import build_rays, normalize_image
from src.models.network import InverseRenderingNetwork
from src.rendering.differentiable_renderer import DifferentiableRenderer
from src.rendering.sh_lighting import SphericalGaussianLighting
from src.utils.io_utils import load_config, load_checkpoint
from src.utils.logging_utils import get_logger
from src.utils.camera import get_device

logger = get_logger(__name__)


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        return float("inf")
    return -10.0 * np.log10(mse)


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Lightweight single-scale SSIM (avoids a hard dependency on external
    SSIM implementations; swap in `skimage.metrics.structural_similarity` or
    `pytorch-msssim` for a publication-grade number)."""
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    mu_x, mu_y = pred_np.mean(), gt_np.mean()
    var_x, var_y = pred_np.var(), gt_np.var()
    cov = ((pred_np - mu_x) * (gt_np - mu_y)).mean()
    c1, c2 = (0.01 ** 2), (0.03 ** 2)
    return float(((2 * mu_x * mu_y + c1) * (2 * cov + c2)) /
                 ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)))


def _lpips_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex")
        p = pred.unsqueeze(0) * 2 - 1
        g = gt.unsqueeze(0) * 2 - 1
        return loss_fn(p, g).item()
    except ImportError:
        logger.warning("`lpips` package not installed; skipping LPIPS (pip install lpips).")
        return float("nan")


def run_relighting_eval(
    ckpt_path: str,
    capture_dir: str,
    novel_env_sg_path: str,
    gt_relit_photo_path: str,
    held_out_pose_path: str,
    cfg_path: str = "configs/default.yaml",
) -> Dict[str, float]:
    cfg = load_config(cfg_path)
    device = get_device(cfg["experiment"]["device"])

    model = InverseRenderingNetwork(cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, map_location=str(device))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = RealCaptureDataset(capture_dir)
    sample = dataset[0]
    context_views = sample["rgb"].to(device)
    model.encode_views(context_views)

    # Novel, held-out environment lighting -- NOT the lighting used at train/fine-tune time.
    novel_lighting = SphericalGaussianLighting.from_envmap_fit(novel_env_sg_path, device=str(device))

    held_out = np.load(held_out_pose_path)  # {'intrinsics': (3,3), 'c2w': (4,4), 'height', 'width'}
    rays_o, rays_d = build_rays(held_out["intrinsics"], held_out["c2w"], int(held_out["height"]), int(held_out["width"]))
    rays_o, rays_d = rays_o.reshape(-1, 3).to(device), rays_d.reshape(-1, 3).to(device)

    renderer = DifferentiableRenderer(background=cfg["rendering"]["background"])
    with torch.no_grad():
        out = renderer.render(rays_o, rays_d, model.sdf_fn, model.material_fn, novel_lighting)
    pred_img = out["rgb"].reshape(int(held_out["height"]), int(held_out["width"]), 3).permute(2, 0, 1).clamp(0, 1)

    gt_img = normalize_image(_read_image(gt_relit_photo_path)).to(device)
    if gt_img.shape[-2:] != pred_img.shape[-2:]:
        gt_img = F.interpolate(gt_img.unsqueeze(0), size=pred_img.shape[-2:], mode="bilinear").squeeze(0)

    metrics = {
        "psnr": _psnr(pred_img, gt_img),
        "ssim": _ssim(pred_img, gt_img),
        "lpips": _lpips_score(pred_img, gt_img),
    }
    logger.info(f"Relighting validation ({sample['object_id']}): "
                f"PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} LPIPS={metrics['lpips']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--capture_dir", type=str, required=True)
    parser.add_argument("--novel_env", type=str, required=True, help="Path to a held-out envmap_sg.npz fit")
    parser.add_argument("--gt_relit_photo", type=str, required=True,
                         help="Real photo of the object captured under the novel lighting")
    parser.add_argument("--held_out_pose", type=str, required=True,
                         help=".npz with intrinsics/c2w/height/width for the eval viewpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    run_relighting_eval(args.ckpt, args.capture_dir, args.novel_env, args.gt_relit_photo,
                         args.held_out_pose, args.config)


if __name__ == "__main__":
    main()
