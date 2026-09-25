"""Phase 4: fine-tune the geometry and illumination heads (material head kept
mostly frozen / lightly regularized by the diffusion prior) on a single
curated casual smartphone capture (3-6 orbital photos), using only
self-supervised photometric consistency across the input views -- there is no
ground-truth geometry/BRDF/lighting for real captures.
"""
from __future__ import annotations

import argparse
import os

import torch

from src.data.real_capture_dataset import RealCaptureDataset
from src.models.network import InverseRenderingNetwork
from src.rendering.differentiable_renderer import DifferentiableRenderer
from src.rendering.sh_lighting import SphericalGaussianLighting
from src.losses.photometric import photometric_loss
from src.losses.mask_loss import mask_loss, eikonal_loss
from src.losses.ambiguity_loss import ambiguity_penalty
from src.diffusion_prior.material_prior import build_material_prior
from src.losses.diffusion_prior_loss import diffusion_prior_loss
from src.utils.io_utils import load_config, save_checkpoint, load_checkpoint
from src.utils.logging_utils import get_logger, MetricLogger
from src.utils.camera import get_device

logger = get_logger(__name__)


def _freeze(module: torch.nn.Module, freeze: bool):
    for p in module.parameters():
        p.requires_grad_(not freeze)


def finetune(cfg: dict, init_ckpt: str, capture_dir: str, out_dir: str):
    device = get_device(cfg["experiment"]["device"])

    dataset = RealCaptureDataset(capture_dir, target_resolution=cfg["data"]["real"].get("resolution", 512))
    if not dataset.poses is not None:  # noqa: E714 - explicit readability check below
        pass
    sample = dataset[0]
    if not sample.get("has_poses", False):
        raise RuntimeError(
            f"No COLMAP poses found for capture '{capture_dir}'. Run a structure-from-motion "
            "pass (e.g. `colmap automatic_reconstructor`) to obtain sparse/{cameras,images}.txt "
            "before fine-tuning; multi-view photometric consistency requires known poses."
        )

    model = InverseRenderingNetwork(cfg).to(device)
    ckpt = load_checkpoint(init_ckpt, map_location=str(device))
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded pretrained checkpoint from {init_ckpt} for real-capture fine-tuning.")

    # Per the phase brief: fine-tune geometry + illumination heads via
    # self-supervised photometric consistency; keep the material head's
    # weights close to the pretrained prior-regularized solution (light
    # regularization rather than a hard freeze, since materials on the real
    # object still need to adapt somewhat to its actual appearance).
    _freeze(model.hexaplane, freeze=False)
    _freeze(model.sdf_decoder, freeze=False)
    _freeze(model.geometry_head, freeze=False)
    _freeze(model.illumination_head, freeze=False)
    _freeze(model.material_head, freeze=False)  # left trainable but regularized below

    frozen_material_state = {k: v.clone() for k, v in model.material_head.state_dict().items()}

    renderer = DifferentiableRenderer(background=cfg["rendering"]["background"])
    prior = build_material_prior(cfg, device=str(device))

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["finetune_real"]["lr"],
    )
    metric_logger = MetricLogger()

    context_views = sample["rgb"].to(device)
    rays_o_all = sample["rays_o"].reshape(-1, 3)
    rays_d_all = sample["rays_d"].reshape(-1, 3)
    rgb_all = sample["rgb"].permute(0, 2, 3, 1).reshape(-1, 3)
    mask_all = sample["mask"].permute(0, 2, 3, 1).reshape(-1, 1)
    n_total = rays_o_all.shape[0]

    num_rays_per_step = 1024
    for epoch in range(cfg["training"]["finetune_real"]["epochs"]):
        idx = torch.randint(0, n_total, (min(num_rays_per_step, n_total),))
        rays_o, rays_d = rays_o_all[idx].to(device), rays_d_all[idx].to(device)
        gt_rgb, gt_mask = rgb_all[idx].to(device), mask_all[idx].to(device)

        model.encode_views(context_views)
        illum = model.illumination_fn()
        lighting = SphericalGaussianLighting(illum["lobe_dirs"], illum["lobe_sharpness"], illum["lobe_intensity"])

        out = renderer.render(rays_o, rays_d, model.sdf_fn, model.material_fn, lighting)

        loss_photo = photometric_loss(out["rgb"], gt_rgb, gt_mask,
                                       cfg["losses"]["photometric"]["l1_weight"],
                                       cfg["losses"]["photometric"]["l2_weight"])
        loss_mask = mask_loss(out["hit_mask"].float(), gt_mask) * cfg["losses"]["mask"]["weight"]
        loss_eik = eikonal_loss(model.sdf_fn, rays_o + rays_d) * cfg["losses"]["eikonal"]["weight"]

        material_at_surface = model.material_fn(rays_o + rays_d * out["depth"])
        loss_ambiguity = ambiguity_penalty(material_at_surface) * 0.1

        loss_diffusion = torch.tensor(0.0, device=device)
        if cfg["diffusion_prior"]["enabled"]:
            loss_diffusion = diffusion_prior_loss(
                material_at_surface, prior, guidance_scale=cfg["diffusion_prior"]["guidance_scale"]
            ) * cfg["diffusion_prior"]["loss_weight"]

        # Keep material head close to its Stage-2 (prior-regularized) solution,
        # since real-capture supervision alone cannot disambiguate material vs.
        # lighting -- this is the crux of the material-lighting ambiguity.
        loss_material_anchor = sum(
            (model.material_head.state_dict()[k] - v.to(device)).pow(2).mean()
            for k, v in frozen_material_state.items()
        ) * 0.01

        loss = loss_photo + loss_mask + loss_eik + loss_ambiguity + loss_diffusion + loss_material_anchor

        opt.zero_grad()
        loss.backward()
        opt.step()

        metric_logger.update(loss=loss.item(), photo=loss_photo.item(), mask=loss_mask.item())
        if epoch % 10 == 0 or epoch == cfg["training"]["finetune_real"]["epochs"] - 1:
            logger.info(f"[finetune][epoch {epoch}] {metric_logger.summary_str()}")
            metric_logger.reset()
            save_checkpoint({"model": model.state_dict(), "epoch": epoch, "cfg": cfg,
                              "object_id": sample["object_id"]}, out_dir)

    save_checkpoint({"model": model.state_dict(), "epoch": "final", "cfg": cfg,
                      "object_id": sample["object_id"]}, out_dir)
    logger.info(f"Fine-tuning complete for object '{sample['object_id']}'. Checkpoints in {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--init_ckpt", type=str, default="runs/stage2/last.ckpt")
    parser.add_argument("--capture_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir or os.path.join(cfg["experiment"]["out_dir"], "finetune")
    finetune(cfg, args.init_ckpt, args.capture_dir, out_dir)


if __name__ == "__main__":
    main()
