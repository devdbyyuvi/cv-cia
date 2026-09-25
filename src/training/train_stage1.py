"""Phase 2: initial end-to-end training of the feed-forward Hexa-Plane backbone
+ decoupled heads on synthetic data, using standard photometric (L1/L2) and
mask reconstruction losses, plus eikonal regularization for a valid SDF.
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.data.synthetic_dataset import SyntheticSceneDataset
from src.models.network import InverseRenderingNetwork
from src.rendering.differentiable_renderer import DifferentiableRenderer
from src.rendering.sh_lighting import SphericalGaussianLighting
from src.losses.photometric import photometric_loss
from src.losses.mask_loss import mask_loss, eikonal_loss
from src.utils.io_utils import load_config, save_checkpoint
from src.utils.logging_utils import get_logger, MetricLogger
from src.utils.camera import get_device

logger = get_logger(__name__)


def sample_rays(sample: dict, num_rays: int, device: torch.device):
    """Flatten (V,H,W,3) ray/pixel buffers and randomly subsample `num_rays`
    for a mini-batch training step (full-image rendering every step is
    unnecessary and slow)."""
    v, h, w = sample["rgb"].shape[0], sample["rgb"].shape[2], sample["rgb"].shape[3]
    rays_o = sample["rays_o"].reshape(-1, 3)
    rays_d = sample["rays_d"].reshape(-1, 3)
    rgb = sample["rgb"].permute(0, 2, 3, 1).reshape(-1, 3)
    mask = sample["mask"].permute(0, 2, 3, 1).reshape(-1, 1)

    n_total = rays_o.shape[0]
    idx = torch.randint(0, n_total, (min(num_rays, n_total),))
    return (
        rays_o[idx].to(device), rays_d[idx].to(device),
        rgb[idx].to(device), mask[idx].to(device),
    )


def train(cfg: dict, out_dir: str):
    device = get_device(cfg["experiment"]["device"])
    torch.manual_seed(cfg["experiment"]["seed"])

    dataset = SyntheticSceneDataset(
        root=cfg["data"]["synthetic"]["root"],
        resolution=cfg["data"]["synthetic"]["resolution"],
        num_views_per_scene=cfg["data"]["synthetic"]["num_views_per_scene"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=cfg["data"]["loader"]["shuffle"],
                         num_workers=0, collate_fn=lambda b: b[0])

    model = InverseRenderingNetwork(cfg).to(device)
    renderer = DifferentiableRenderer(
        num_secondary_rays=cfg["rendering"]["num_secondary_rays"],
        background=cfg["rendering"]["background"],
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["stage1"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["training"]["stage1"]["epochs"])

    metric_logger = MetricLogger()
    num_rays_per_step = 1024

    for epoch in range(cfg["training"]["stage1"]["epochs"]):
        for sample in loader:
            rays_o, rays_d, gt_rgb, gt_mask = sample_rays(sample, num_rays_per_step, device)
            context_views = sample["rgb"].to(device)  # (V,3,H,W) input views condition the field

            model.encode_views(context_views)
            illum = model.illumination_fn()
            lighting = SphericalGaussianLighting(
                illum["lobe_dirs"], illum["lobe_sharpness"], illum["lobe_intensity"]
            )

            out = renderer.render(rays_o, rays_d, model.sdf_fn, model.material_fn, lighting)

            loss_photo = photometric_loss(
                out["rgb"], gt_rgb, gt_mask,
                l1_weight=cfg["losses"]["photometric"]["l1_weight"],
                l2_weight=cfg["losses"]["photometric"]["l2_weight"],
            )
            loss_mask = mask_loss(out["hit_mask"].float(), gt_mask) * cfg["losses"]["mask"]["weight"]
            loss_eik = eikonal_loss(model.sdf_fn, rays_o + rays_d) * cfg["losses"]["eikonal"]["weight"]

            loss = loss_photo + loss_mask + loss_eik

            opt.zero_grad()
            loss.backward()
            opt.step()

            metric_logger.update(loss=loss.item(), photo=loss_photo.item(),
                                  mask=loss_mask.item(), eikonal=loss_eik.item())

        scheduler.step()
        if epoch % 10 == 0 or epoch == cfg["training"]["stage1"]["epochs"] - 1:
            logger.info(f"[stage1][epoch {epoch}] {metric_logger.summary_str()}")
            metric_logger.reset()
            save_checkpoint({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, out_dir)

    save_checkpoint({"model": model.state_dict(), "epoch": "final", "cfg": cfg}, out_dir)
    logger.info(f"Stage 1 training complete. Checkpoints in {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir or os.path.join(cfg["experiment"]["out_dir"], "stage1")
    train(cfg, out_dir)


if __name__ == "__main__":
    main()
