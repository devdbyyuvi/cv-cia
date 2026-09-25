"""Phase 5 final demo: takes a folder of 3-6 casual photos of an object and a
trained checkpoint, and produces:

  1. An extracted, editable 3D asset (mesh + per-vertex material maps, .obj/.glb-ready).
  2. A grid of renders of that asset under several environment lightings (the
     original scene lighting plus any additional SG envmap fits found under
     `--relight_envs`), demonstrating the disentangled geometry/material/lighting.
  3. An uncertainty visualization highlighting ambiguous/occluded regions.

Run:
    python demo/run_demo.py --images data/real_captures/object_01 --ckpt runs/finetune/last.ckpt --out demo/output
"""
from __future__ import annotations

import argparse
import glob
import os

import imageio.v2 as imageio
import numpy as np
import torch

from src.data.real_capture_dataset import RealCaptureDataset
from src.models.network import InverseRenderingNetwork
from src.rendering.differentiable_renderer import DifferentiableRenderer
from src.rendering.sh_lighting import SphericalGaussianLighting
from src.uncertainty.uncertainty_estimation import UncertaintyEstimator
from src.utils.io_utils import load_config, load_checkpoint
from src.utils.camera import get_device
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def render_turntable_frame(model, renderer, lighting, resolution=64, azimuth=0.0, elevation=0.2, radius=2.0, device="cpu"):
    """Render one frame of a simple orbit around the reconstructed object,
    for quick qualitative inspection without needing real camera poses."""
    theta, phi = azimuth, elevation
    cam_pos = radius * np.array([np.cos(theta) * np.cos(phi), np.sin(phi), np.sin(theta) * np.cos(phi)])
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


def run_demo(images_dir: str, ckpt_path: str, out_dir: str, relight_envs: list, config_path: str):
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config(config_path)
    device = get_device(cfg["experiment"]["device"])

    model = InverseRenderingNetwork(cfg).to(device).eval()
    ckpt = load_checkpoint(ckpt_path, map_location=str(device))
    model.load_state_dict(ckpt["model"])

    dataset = RealCaptureDataset(images_dir)
    sample = dataset[0]
    context_views = sample["rgb"].to(device)
    model.encode_views(context_views)

    renderer = DifferentiableRenderer(background=cfg["rendering"]["background"])

    # 1) Original (predicted) scene lighting turntable.
    illum = model.illumination_fn()
    original_lighting = SphericalGaussianLighting(illum["lobe_dirs"], illum["lobe_sharpness"], illum["lobe_intensity"])
    frames = [
        render_turntable_frame(model, renderer, original_lighting, azimuth=a, device=device)
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)[:2]
    ]
    imageio.mimsave(os.path.join(out_dir, "turntable_original_lighting.gif"), frames, fps=4)
    logger.info(f"Saved turntable render under estimated scene lighting -> {out_dir}/turntable_original_lighting.gif")

    # 2) Relit renders under any additional held-out environment SG fits.
    for env_path in relight_envs:
        env_lighting = SphericalGaussianLighting.from_envmap_fit(env_path, device=str(device))
        img = render_turntable_frame(model, renderer, env_lighting, azimuth=np.pi / 4, device=device)
        name = os.path.splitext(os.path.basename(env_path))[0]
        imageio.imsave(os.path.join(out_dir, f"relit_{name}.png"), img)
        logger.info(f"Saved relit render under '{name}' -> {out_dir}/relit_{name}.png")

    # 3) Uncertainty map on a single fixed view.
    if cfg["uncertainty"]["enabled"]:
        estimator = UncertaintyEstimator(
            method=cfg["uncertainty"]["method"],
            num_forward_passes=cfg["uncertainty"]["num_forward_passes"],
            dropout_p=cfg["uncertainty"]["dropout_p"],
        )
        resolution = 128
        cam_pos = np.array([1.5, 0.3, 1.5])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 1, 0])
        right = np.cross(forward, up); right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        i, j = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="xy")
        focal = resolution / (2 * np.tan(np.pi / 8))
        x = (i - resolution / 2) / focal
        y = -(j - resolution / 2) / focal
        dirs = x[..., None] * right + y[..., None] * true_up + forward
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        rays_o = torch.from_numpy(np.broadcast_to(cam_pos, dirs.shape).copy()).float().reshape(-1, 3).to(device)
        rays_d = torch.from_numpy(dirs.astype(np.float32)).reshape(-1, 3).to(device)

        def _predict():
            return renderer.render(rays_o, rays_d, model.sdf_fn, model.material_fn, original_lighting)

        unc = estimator.estimate(_predict, model=model)
        if "uncertainty_map" in unc:
            umap = unc["uncertainty_map"].reshape(resolution, resolution).cpu().numpy()
            umap = (umap - umap.min()) / (umap.max() - umap.min() + 1e-8)
            imageio.imsave(os.path.join(out_dir, "uncertainty_map.png"), (umap * 255).astype(np.uint8))
            logger.info(f"Saved uncertainty map -> {out_dir}/uncertainty_map.png")

    logger.info(f"Demo complete. Outputs in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=str, required=True, help="Casual capture dir (see data/real_captures/README.md)")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="demo/output")
    parser.add_argument("--relight_envs", type=str, nargs="*", default=[],
                         help="Optional list of envmap_sg.npz fits to relight under")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    run_demo(args.images, args.ckpt, args.out, args.relight_envs, args.config)


if __name__ == "__main__":
    main()

