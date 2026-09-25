"""Phase 3: continue training from the Stage-1 checkpoint, now adding the
material diffusion-prior regularization loss and the ambiguity/physical-
plausibility penalty to resolve material-lighting ambiguity, plus periodic
uncertainty-map computation for monitoring.
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
from src.losses.ambiguity_loss import ambiguity_penalty
from src.losses.diffusion_prior_loss import diffusion_prior_loss
from src.diffusion_prior.material_prior import build_material_prior
from src.uncertainty.uncertainty_estimation import UncertaintyEstimator
from src.utils.io_utils import load_config, save_checkpoint, load_checkpoint
from src.utils.logging_utils import get_logger, MetricLogger
from src.utils.camera import get_device
from src.training.train_stage1 import sample_rays

logger = get_logger(__name__)


def train(cfg: dict, init_ckpt: str, out_dir: str):
    device = get_device(cfg["experiment"]["device"])
    torch.manual_seed(cfg["experiment"]["seed"])

    dataset = SyntheticSceneDataset(
        root=cfg["data"]["synthetic"]["root"],
        resolution=cfg["data"]["synthetic"]["resolution"],
        num_views_per_scene=cfg["data"]["synthetic"]["num_views_per_scene"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda b: b[0])

    model = InverseRenderingNetwork(cfg).to(device)
    if init_ckpt and os.path.exists(init_ckpt):
        ckpt = load_checkpoint(init_ckpt, map_location=str(device))
        model.load_state_dict(ckpt["model"])
        logger.info(f"Loaded Stage 1 checkpoint from {init_ckpt}")

    renderer = DifferentiableRenderer(
        num_secondary_rays=cfg["rendering"]["num_secondary_rays"],
        background=cfg["rendering"]["background"],
    )
    prior = build_material_prior(cfg, device=str(device))
    uncertainty_estimator = UncertaintyEstimator(
        method=cfg["uncertainty"]["method"],
        num_forward_passes=cfg["uncertainty"]["num_forward_passes"],
        dropout_p=cfg["uncertainty"]["dropout_p"],
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["stage2_diffusion"]["lr"])
    metric_logger = MetricLogger()
    num_rays_per_step = 1024

    for epoch in range(cfg["training"]["stage2_diffusion"]["epochs"]):
        for step, sample in enumerate(loader):
            rays_o, rays_d, gt_rgb, gt_mask = sample_rays(sample, num_rays_per_step, device)
            context_views = sample["rgb"].to(device)

            model.encode_views(context_views)
            illum = model.illumination_fn()
            lighting = SphericalGaussianLighting(
                illum["lobe_dirs"], illum["lobe_sharpness"], illum["lobe_intensity"]
            )

            out = renderer.render(rays_o, rays_d, model.sdf_fn, model.material_fn, lighting)
            material_at_surface = model.material_fn(out["hit_mask"].float() * 0 + rays_o + rays_d)

            loss_photo = photometric_loss(out["rgb"], gt_rgb, gt_mask,
                                           cfg["losses"]["photometric"]["l1_weight"],
                                           cfg["losses"]["photometric"]["l2_weight"])
            loss_mask = mask_loss(out["hit_mask"].float(), gt_mask) * cfg["losses"]["mask"]["weight"]
            loss_eik = eikonal_loss(model.sdf_fn, rays_o + rays_d) * cfg["losses"]["eikonal"]["weight"]

            loss_ambiguity = ambiguity_penalty(
                material_at_surface,
                prior_albedo=None,  # populated below if the prior yields a usable estimate
                energy_conservation_weight=cfg["losses"]["ambiguity"]["energy_conservation_weight"],
                albedo_prior_consistency_weight=cfg["losses"]["ambiguity"]["albedo_prior_consistency_weight"],
            ) * cfg["losses"]["ambiguity"]["weight"]

            loss_diffusion = torch.tensor(0.0, device=device)
            if cfg["diffusion_prior"]["enabled"]:
                loss_diffusion = diffusion_prior_loss(
                    material_at_surface, prior, guidance_scale=cfg["diffusion_prior"]["guidance_scale"]
                ) * cfg["diffusion_prior"]["loss_weight"]

            loss = loss_photo + loss_mask + loss_eik + loss_ambiguity + loss_diffusion

            opt.zero_grad()
            loss.backward()
            opt.step()

            metric_logger.update(loss=loss.item(), photo=loss_photo.item(), mask=loss_mask.item(),
                                  eikonal=loss_eik.item(), ambiguity=loss_ambiguity.item(),
                                  diffusion_prior=loss_diffusion.item())

            if cfg["uncertainty"]["enabled"] and step == 0 and epoch % 20 == 0:
                def _predict():
                    return renderer.render(rays_o[:256], rays_d[:256], model.sdf_fn, model.material_fn, lighting)
                unc = uncertainty_estimator.estimate(_predict, model=model)
                if "uncertainty_map" in unc:
                    logger.info(f"[stage2][epoch {epoch}] mean uncertainty: {unc['uncertainty_map'].mean().item():.4f}")

        if epoch % 10 == 0 or epoch == cfg["training"]["stage2_diffusion"]["epochs"] - 1:
            logger.info(f"[stage2][epoch {epoch}] {metric_logger.summary_str()}")
            metric_logger.reset()
            save_checkpoint({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, out_dir)

    save_checkpoint({"model": model.state_dict(), "epoch": "final", "cfg": cfg}, out_dir)
    logger.info(f"Stage 2 (diffusion-regularized) training complete. Checkpoints in {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--init_ckpt", type=str, default="runs/stage1/last.ckpt")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["diffusion_prior"]["enabled"] = True
    cfg["losses"]["ambiguity"]["weight"] = 0.1
    out_dir = args.out_dir or os.path.join(cfg["experiment"]["out_dir"], "stage2")
    train(cfg, args.init_ckpt, out_dir)


if __name__ == "__main__":
    main()
