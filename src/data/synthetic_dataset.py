"""Synthetic scene dataset: multi-view renders paired with ground-truth geometry
(SDF/depth), spatially varying BRDF maps (albedo/roughness/metallic), and the
ground-truth environment lighting used to render each scene.

Expected on-disk layout (one folder per scene)::

    data/synthetic/<scene_id>/
        meta.json                # intrinsics, image size, num_views
        poses.npy                # (N, 4, 4) camera-to-world matrices
        rgb/000.png ... rgb/N.png
        mask/000.png ... mask/N.png
        depth/000.exr ... N.exr  # or .npy, meters
        normal/000.exr ...       # world-space normals, optional (else derived from depth)
        albedo/000.png ...
        roughness/000.png ...    # single channel
        metallic/000.png ...     # single channel
        sdf_samples.npz          # {'points': (M,3), 'sdf': (M,)} volumetric GT samples
        envmap.hdr                # ground-truth environment map used for rendering
        envmap_sg.npz             # {'lobe_dirs','lobe_sharpness','lobe_intensity'} precomputed SG fit

Any of the optional GT channels may be missing; the loader degrades gracefully
and the corresponding supervision term is simply skipped at train time (see
src/losses).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import normalize_image, build_rays, normalize_scene_poses

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _read_image(path: str) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.shape[2] == 3 else cv2.COLOR_BGRA2RGBA)
        return img
    import imageio.v2 as imageio
    return imageio.imread(path)


def _read_scalar_map(path: str) -> np.ndarray:
    img = _read_image(path)
    if img.ndim == 3:
        img = img[..., 0]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img[..., None]


class SyntheticSceneDataset(Dataset):
    """Yields one randomly-sampled sparse multi-view "episode" per scene per __getitem__:
    a handful of context views (input) plus GT supervision maps for every sampled view,
    mirroring the sparse-input / feed-forward inference setting used at test time.
    """

    def __init__(
        self,
        root: str,
        resolution: int = 256,
        num_views_per_scene: int = 8,
        load_sdf: bool = True,
        load_brdf: bool = True,
        load_envmap: bool = True,
        scene_ids: Optional[List[str]] = None,
    ):
        self.root = root
        self.resolution = resolution
        self.num_views_per_scene = num_views_per_scene
        self.load_sdf = load_sdf
        self.load_brdf = load_brdf
        self.load_envmap = load_envmap

        if scene_ids is not None:
            self.scene_ids = scene_ids
        elif os.path.isdir(root):
            self.scene_ids = sorted(
                d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
            )
        else:
            self.scene_ids = []

    def __len__(self) -> int:
        return max(len(self.scene_ids), 1)

    def _scene_dir(self, idx: int) -> str:
        return os.path.join(self.root, self.scene_ids[idx])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.scene_ids:
            return self._dummy_sample()

        scene_dir = self._scene_dir(idx)
        with open(os.path.join(scene_dir, "meta.json")) as f:
            meta = json.load(f)
        intrinsics = np.array(meta["intrinsics"], dtype=np.float32)
        poses = np.load(os.path.join(scene_dir, "poses.npy")).astype(np.float32)
        poses = normalize_scene_poses(poses)

        num_available = poses.shape[0]
        n_views = min(self.num_views_per_scene, num_available)
        view_idx = np.random.choice(num_available, size=n_views, replace=False)

        rgbs, masks, depths, normals, albedos, roughs, metals = [], [], [], [], [], [], []
        rays_o_all, rays_d_all = [], []

        for v in view_idx:
            rgb = normalize_image(_read_image(os.path.join(scene_dir, "rgb", f"{v:03d}.png")))
            rgbs.append(rgb)
            mask_path = os.path.join(scene_dir, "mask", f"{v:03d}.png")
            masks.append(
                torch.from_numpy(_read_scalar_map(mask_path)).permute(2, 0, 1)
                if os.path.exists(mask_path) else torch.ones(1, rgb.shape[1], rgb.shape[2])
            )

            h, w = rgb.shape[1], rgb.shape[2]
            ro, rd = build_rays(intrinsics, poses[v], h, w)
            rays_o_all.append(ro)
            rays_d_all.append(rd)

            if self.load_sdf:
                depth_path = os.path.join(scene_dir, "depth", f"{v:03d}.npy")
                if os.path.exists(depth_path):
                    depths.append(torch.from_numpy(np.load(depth_path)).float()[None])
                normal_path = os.path.join(scene_dir, "normal", f"{v:03d}.npy")
                if os.path.exists(normal_path):
                    normals.append(torch.from_numpy(np.load(normal_path)).float().permute(2, 0, 1))

            if self.load_brdf:
                for name, bucket in [("albedo", albedos), ("roughness", roughs), ("metallic", metals)]:
                    p = os.path.join(scene_dir, name, f"{v:03d}.png")
                    if os.path.exists(p):
                        arr = _read_image(p)
                        if name != "albedo" and arr.ndim == 3:
                            arr = arr[..., :1]
                        t = normalize_image(arr) if arr.ndim == 3 and arr.shape[-1] == 3 else \
                            torch.from_numpy(_read_scalar_map(p)).permute(2, 0, 1)
                        bucket.append(t)

        sample = {
            "scene_id": self.scene_ids[idx],
            "rgb": torch.stack(rgbs),
            "mask": torch.stack(masks),
            "rays_o": torch.stack(rays_o_all),
            "rays_d": torch.stack(rays_d_all),
            "poses": torch.from_numpy(poses[view_idx]),
            "intrinsics": torch.from_numpy(intrinsics),
        }
        if depths:
            sample["depth_gt"] = torch.stack(depths)
        if normals:
            sample["normal_gt"] = torch.stack(normals)
        if albedos:
            sample["albedo_gt"] = torch.stack(albedos)
        if roughs:
            sample["roughness_gt"] = torch.stack(roughs)
        if metals:
            sample["metallic_gt"] = torch.stack(metals)

        if self.load_sdf:
            sdf_path = os.path.join(scene_dir, "sdf_samples.npz")
            if os.path.exists(sdf_path):
                npz = np.load(sdf_path)
                sample["sdf_points"] = torch.from_numpy(npz["points"]).float()
                sample["sdf_gt"] = torch.from_numpy(npz["sdf"]).float()

        if self.load_envmap:
            sg_path = os.path.join(scene_dir, "envmap_sg.npz")
            if os.path.exists(sg_path):
                npz = np.load(sg_path)
                sample["env_sg_dirs"] = torch.from_numpy(npz["lobe_dirs"]).float()
                sample["env_sg_sharpness"] = torch.from_numpy(npz["lobe_sharpness"]).float()
                sample["env_sg_intensity"] = torch.from_numpy(npz["lobe_intensity"]).float()

        return sample

    def _dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Returned when no scenes are on disk yet, so the pipeline remains
        runnable end-to-end (e.g. for CI / smoke tests) before real data is
        downloaded. See scripts/download_data.sh."""
        n, h, w = self.num_views_per_scene, self.resolution, self.resolution
        intr = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32)
        poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
        rays_o = torch.zeros(n, h, w, 3)
        rays_d = torch.zeros(n, h, w, 3)
        rays_d[..., 2] = -1.0
        return {
            "scene_id": "dummy",
            "rgb": torch.rand(n, 3, h, w),
            "mask": torch.ones(n, 1, h, w),
            "rays_o": rays_o,
            "rays_d": rays_d,
            "poses": torch.from_numpy(poses),
            "intrinsics": torch.from_numpy(intr),
        }
