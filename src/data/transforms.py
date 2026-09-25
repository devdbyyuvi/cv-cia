"""Camera and image transform utilities shared by synthetic and real dataset loaders."""
from __future__ import annotations

import numpy as np
import torch


def normalize_image(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC [0,255] -> float32 CHW [0,1] tensor."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return tensor


def build_rays(intrinsics: np.ndarray, c2w: np.ndarray, height: int, width: int):
    """Generate per-pixel ray origins/directions in world space for a pinhole camera.

    Args:
        intrinsics: (3,3) camera intrinsic matrix.
        c2w: (4,4) camera-to-world extrinsic matrix.
        height, width: image resolution.

    Returns:
        rays_o: (H, W, 3) ray origins (camera center, broadcast).
        rays_d: (H, W, 3) normalized world-space ray directions.
    """
    i, j = np.meshgrid(np.arange(width, dtype=np.float32),
                        np.arange(height, dtype=np.float32), indexing="xy")
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    dirs_cam = np.stack([(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], axis=-1)  # OpenGL convention
    rot = c2w[:3, :3]
    dirs_world = dirs_cam @ rot.T
    dirs_world = dirs_world / (np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-8)

    origin = c2w[:3, 3]
    origins_world = np.broadcast_to(origin, dirs_world.shape)

    return (
        torch.from_numpy(origins_world.copy()).float(),
        torch.from_numpy(dirs_world.copy()).float(),
    )


def normalize_scene_poses(c2ws: np.ndarray, target_radius: float = 1.5) -> np.ndarray:
    """Recenter and rescale a set of camera-to-world poses so cameras lie
    roughly on a sphere of `target_radius` around the origin. Useful for
    casual smartphone captures with no absolute scale/scene-centering.
    """
    centers = c2ws[:, :3, 3]
    scene_center = centers.mean(axis=0)
    c2ws = c2ws.copy()
    c2ws[:, :3, 3] -= scene_center

    avg_dist = np.linalg.norm(c2ws[:, :3, 3], axis=1).mean() + 1e-8
    scale = target_radius / avg_dist
    c2ws[:, :3, 3] *= scale
    return c2ws
