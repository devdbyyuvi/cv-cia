"""Phase 4: measure end-to-end inference speed (sparse photos -> full
reconstruction) on a consumer GPU, to confirm the feed-forward design meets
the "runs in seconds" usability goal -- no per-scene optimization/test-time
training, a single forward pass plus a ray-marched render.
"""
from __future__ import annotations

import argparse
import time

import torch

from src.models.network import InverseRenderingNetwork
from src.rendering.differentiable_renderer import DifferentiableRenderer
from src.rendering.sh_lighting import SphericalGaussianLighting
from src.utils.io_utils import load_config, load_checkpoint
from src.utils.logging_utils import get_logger
from src.utils.camera import get_device

logger = get_logger(__name__)


def benchmark(cfg: dict, ckpt_path: str, num_views: int = 4, resolution: int = 512):
    device = get_device(cfg["experiment"]["device"])
    model = InverseRenderingNetwork(cfg).to(device).eval()

    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path, map_location=str(device))
        model.load_state_dict(ckpt["model"])

    renderer = DifferentiableRenderer(background=cfg["rendering"]["background"])

    dummy_images = torch.rand(num_views, 3, resolution, resolution, device=device)
    rays_o = torch.zeros(resolution * resolution, 3, device=device)
    rays_d = torch.zeros(resolution * resolution, 3, device=device)
    rays_d[:, 2] = -1.0

    warmup = cfg["evaluation"]["benchmark"]["warmup_iters"]
    timed = cfg["evaluation"]["benchmark"]["timed_iters"]

    def _full_pass():
        with torch.no_grad():
            model.encode_views(dummy_images)
            illum = model.illumination_fn()
            lighting = SphericalGaussianLighting(illum["lobe_dirs"], illum["lobe_sharpness"], illum["lobe_intensity"])
            renderer.render(rays_o, rays_d, model.sdf_fn, model.material_fn, lighting)

    for _ in range(warmup):
        _full_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(timed):
        _full_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    per_call_s = elapsed / timed
    logger.info(
        f"Benchmark: {num_views} views @ {resolution}x{resolution} on {device} -> "
        f"{per_call_s * 1000:.1f} ms/reconstruction ({1.0 / per_call_s:.2f} reconstructions/sec)"
    )
    return {"device": str(device), "num_views": num_views, "resolution": resolution,
            "ms_per_reconstruction": per_call_s * 1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    cfg = load_config(args.config)
    benchmark(cfg, args.ckpt, args.num_views, args.resolution)


if __name__ == "__main__":
    main()
