"""Full demo: high-quality outputs + all metrics in one run."""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from src.data.real_capture_dataset import RealCaptureDataset, _read_image
from src.data.transforms import build_rays, normalize_image
from src.models.network import InverseRenderingNetwork
from src.rendering.differentiable_renderer import DifferentiableRenderer
from src.rendering.sh_lighting import SphericalGaussianLighting
from src.uncertainty.uncertainty_estimation import UncertaintyEstimator
from src.utils.io_utils import load_config, load_checkpoint
from src.utils.camera import get_device
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def render_turntable_frame(
    model, renderer, lighting, resolution=512, azimuth=0.0, elevation=0.2, radius=2.0, device="cpu"
):
    theta, phi = azimuth, elevation
    cam_pos = radius * np.array([
        np.cos(theta) * np.cos(phi),
        np.sin(phi),
        np.sin(theta) * np.cos(phi)
    ])
    forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, forward)

    fov = np.pi / 4
    focal = resolution / (2 * np.tan(fov / 2))
    i, j = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="xy")
    x = (i - resolution / 2) / focal
    y = -(j - resolution / 2) / focal
    dirs = x[..., None] * right + y[..., None] * true_up + forward
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)

    rays_o = torch.from_numpy(np.broadcast_to(cam_pos, dirs.shape).copy()).float().reshape(-1, 3).to(device)
    rays_d = torch.from_numpy(dirs.astype(np.float32)).reshape(-1, 3).to(device)

    with torch.no_grad():
        out = renderer.render(rays_o, rays_d, model.sdf_fn, model.material_fn, lighting)
    img = out["rgb"].reshape(resolution, resolution, 3).clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        return float("inf")
    return -10.0 * np.log10(mse)


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
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


def run_full_demo(
    images_dir: str,
    ckpt_path: str,
    out_dir: str,
    relight_envs: list,
    config_path: str,
    resolution: int = 512,
    num_frames: int = 16,
    benchmark_iters: int = 50,
):
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config(config_path)
    device = get_device(cfg["experiment"]["device"])

    # Override for quality
    cfg["rendering"]["num_secondary_rays"] = 4

    model = InverseRenderingNetwork(cfg).to(device).eval()
    ckpt = load_checkpoint(ckpt_path, map_location=str(device))
    model.load_state_dict(ckpt["model"])

    dataset = RealCaptureDataset(images_dir)
    sample = dataset[0]
    context_views = sample["rgb"].to(device)
    model.encode_views(context_views)

    renderer = DifferentiableRenderer(background=cfg["rendering"]["background"])

    all_metrics = {}

    # 1) Original scene lighting turntable (high-res)
    logger.info(f"Rendering {num_frames}-frame turntable at {resolution}x{resolution}...")
    illum = model.illumination_fn()
    original_lighting = SphericalGaussianLighting(illum["lobe_dirs"], illum["lobe_sharpness"], illum["lobe_intensity"])
    frames = [
        render_turntable_frame(model, renderer, original_lighting, resolution=resolution, azimuth=a, device=device)
        for a in np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    ]
    gif_path = os.path.join(out_dir, "turntable_original_lighting_hq.gif")
    imageio.mimsave(gif_path, frames, fps=8)
    logger.info(f"Saved HQ turntable -> {gif_path}")

    # Also save individual frames
    frames_dir = os.path.join(out_dir, "turntable_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        imageio.imsave(os.path.join(frames_dir, f"frame_{idx:03d}.png"), frame)

    # 2) Relit renders under held-out environment
    for env_path in relight_envs:
        env_lighting = SphericalGaussianLighting.from_envmap_fit(env_path, device=str(device))
        img = render_turntable_frame(model, renderer, env_lighting, resolution=resolution, azimuth=np.pi/4, device=device)
        name = os.path.splitext(os.path.basename(env_path))[0]
        out_path = os.path.join(out_dir, f"relit_{name}_hq.png")
        imageio.imsave(out_path, img)
        logger.info(f"Saved HQ relit render -> {out_path}")

    # 3) Uncertainty map
    if cfg["uncertainty"]["enabled"]:
        logger.info("Computing uncertainty map...")
        estimator = UncertaintyEstimator(
            method=cfg["uncertainty"]["method"],
            num_forward_passes=cfg["uncertainty"]["num_forward_passes"],
            dropout_p=cfg["uncertainty"]["dropout_p"],
        )
        unc_resolution = 256
        cam_pos = np.array([1.5, 0.3, 1.5])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 1, 0])
        right = np.cross(forward, up); right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        i, j = np.meshgrid(np.arange(unc_resolution), np.arange(unc_resolution), indexing="xy")
        focal = unc_resolution / (2 * np.tan(np.pi / 8))
        x = (i - unc_resolution / 2) / focal
        y = -(j - unc_resolution / 2) / focal
        dirs = x[..., None] * right + y[..., None] * true_up + forward
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        rays_o = torch.from_numpy(np.broadcast_to(cam_pos, dirs.shape).copy()).float().reshape(-1, 3).to(device)
        rays_d = torch.from_numpy(dirs.astype(np.float32)).reshape(-1, 3).to(device)

        def _predict():
            return renderer.render(rays_o, rays_d, model.sdf_fn, model.material_fn, original_lighting)

        unc = estimator.estimate(_predict, model=model)
        if "uncertainty_map" in unc:
            umap = unc["uncertainty_map"].reshape(unc_resolution, unc_resolution).cpu().numpy()
            umap = (umap - umap.min()) / (umap.max() - umap.min() + 1e-8)
            unc_path = os.path.join(out_dir, "uncertainty_map_hq.png")
            imageio.imsave(unc_path, (umap * 255).astype(np.uint8))
            logger.info(f"Saved HQ uncertainty map -> {unc_path}")

    # 4) Relighting evaluation (PSNR/SSIM/LPIPS vs ground truth)
    logger.info("Running relighting evaluation...")
    gt_relit_photo = os.path.join(images_dir, "gt_relit.jpg")
    held_out_pose = os.path.join(images_dir, "pose_held_out.npz")
    novel_env = relight_envs[0] if relight_envs else "data/envmaps/held_out.npz"

    if os.path.exists(gt_relit_photo) and os.path.exists(held_out_pose) and os.path.exists(novel_env):
        novel_lighting = SphericalGaussianLighting.from_envmap_fit(novel_env, device=str(device))
        held_out = np.load(held_out_pose)
        rays_o, rays_d = build_rays(
            held_out["intrinsics"], held_out["c2w"],
            int(held_out.get("height", 256)), int(held_out.get("width", 256))
        )
        rays_o, rays_d = rays_o.reshape(-1, 3).to(device), rays_d.reshape(-1, 3).to(device)

        with torch.no_grad():
            out = renderer.render(rays_o, rays_d, model.sdf_fn, model.material_fn, novel_lighting)
        pred_img = out["rgb"].reshape(int(held_out.get("height", 256)), int(held_out.get("width", 256)), 3).permute(2, 0, 1).clamp(0, 1)

        gt_img = normalize_image(_read_image(gt_relit_photo)).to(device)
        if gt_img.shape[-2:] != pred_img.shape[-2:]:
            gt_img = F.interpolate(gt_img.unsqueeze(0), size=pred_img.shape[-2:], mode="bilinear").squeeze(0)

        relight_metrics = {
            "psnr": _psnr(pred_img, gt_img),
            "ssim": _ssim(pred_img, gt_img),
            "lpips": _lpips_score(pred_img, gt_img),
        }
        all_metrics["relighting"] = relight_metrics
        logger.info(f"Relighting metrics: PSNR={relight_metrics['psnr']:.2f} SSIM={relight_metrics['ssim']:.4f} LPIPS={relight_metrics['lpips']:.4f}")

        # Save side-by-side comparison
        pred_np = pred_img.permute(1, 2, 0).cpu().numpy()
        gt_np = gt_img.permute(1, 2, 0).cpu().numpy()
        comparison = np.hstack([gt_np, pred_np, np.abs(gt_np - pred_np)])
        imageio.imsave(os.path.join(out_dir, "relighting_comparison.png"), (comparison * 255).astype(np.uint8))
    else:
        logger.warning("Skipping relighting eval: missing gt_relit.jpg, pose_held_out.npz, or envmap")

    # 5) Speed benchmark
    logger.info("Running inference speed benchmark...")
    dummy_images = torch.rand(4, 3, resolution, resolution, device=device)
    rays_o = torch.zeros(resolution * resolution, 3, device=device)
    rays_d = torch.zeros(resolution * resolution, 3, device=device)
    rays_d[:, 2] = -1.0

    def _full_pass():
        with torch.no_grad():
            model.encode_views(dummy_images)
            illum = model.illumination_fn()
            lighting = SphericalGaussianLighting(illum["lobe_dirs"], illum["lobe_sharpness"], illum["lobe_intensity"])
            renderer.render(rays_o, rays_d, model.sdf_fn, model.material_fn, lighting)

    # Warmup
    for _ in range(5):
        _full_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(benchmark_iters):
        _full_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    per_call_ms = (elapsed / benchmark_iters) * 1000
    benchmark_metrics = {
        "device": str(device),
        "resolution": resolution,
        "num_views": 4,
        "ms_per_reconstruction": per_call_ms,
        "reconstructions_per_sec": 1000 / per_call_ms,
    }
    all_metrics["benchmark"] = benchmark_metrics
    logger.info(f"Benchmark: {per_call_ms:.1f} ms/reconstruction ({1000/per_call_ms:.2f} recon/sec)")

    # Save all metrics
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved all metrics -> {metrics_path}")

    logger.info(f"Full demo complete. Outputs in {out_dir}")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Full high-quality demo with all metrics")
    parser.add_argument("--images", type=str, required=True, help="Casual capture dir")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="demo/output")
    parser.add_argument("--relight_envs", type=str, nargs="*", default=["data/envmaps/held_out.npz"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--benchmark_iters", type=int, default=50)
    args = parser.parse_args()

    run_full_demo(
        args.images, args.ckpt, args.out, args.relight_envs, args.config,
        resolution=args.resolution, num_frames=args.num_frames, benchmark_iters=args.benchmark_iters
    )


if __name__ == "__main__":
    main()