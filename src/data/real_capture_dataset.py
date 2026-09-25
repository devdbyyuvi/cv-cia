"""Loader for casual smartphone multi-view captures (3-6 orbital photos per
object), used for fine-tuning (Phase 4) and real-world qualitative eval.

Expected layout, one folder per captured object::

    data/real_captures/<object_id>/
        images/IMG_0001.jpg ... IMG_000{3..6}.jpg
        sparse/                  # COLMAP sparse reconstruction (cameras.txt, images.txt, points3D.txt)
            cameras.txt
            images.txt
            points3D.txt
        masks/IMG_0001.png ...   # optional foreground masks (else auto background removal is used)

No ground-truth geometry/BRDF/lighting is available for real captures -- only
photometric self-supervision is used (see src/training/finetune_real.py).
Poses are parsed from a COLMAP `images.txt` + `cameras.txt` pair, which is what
free tools such as COLMAP CLI or an app like Polycam/RealityScan export from a
short orbital video or photo burst.
"""
from __future__ import annotations

import os
import re
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
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    import imageio.v2 as imageio
    return imageio.imread(path)


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2],
    ])


def parse_colmap_cameras(cameras_txt: str) -> Dict[int, np.ndarray]:
    intrinsics = {}
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cam_id, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                f_, cx, cy = params[0], params[1], params[2]
                fx = fy = f_
            else:  # PINHOLE / OPENCV and friends: first four are fx,fy,cx,cy
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            intrinsics[cam_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return intrinsics


def parse_colmap_images(images_txt: str, intrinsics: Dict[int, np.ndarray]):
    poses, names, intr_list = [], [], []
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    for line in lines[::2]:  # every other line holds the pose; alternate lines hold 2D points
        parts = line.split()
        qvec = np.array(list(map(float, parts[1:5])))
        tvec = np.array(list(map(float, parts[5:8])))
        cam_id = int(parts[8])
        name = parts[9]

        R = _qvec_to_rotmat(qvec)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec
        c2w = np.linalg.inv(w2c).astype(np.float32)

        poses.append(c2w)
        names.append(name)
        intr_list.append(intrinsics[cam_id])
    return poses, names, intr_list


class RealCaptureDataset(Dataset):
    """A single object's casual capture (3-6 views). One dataset instance = one object,
    since fine-tuning (Phase 4) is per-object / few-shot rather than across a large corpus.
    """

    def __init__(self, capture_dir: str, target_resolution: Optional[int] = 512):
        self.capture_dir = capture_dir
        self.target_resolution = target_resolution
        self.images_dir = os.path.join(capture_dir, "images")
        self.masks_dir = os.path.join(capture_dir, "masks")
        sparse_dir = os.path.join(capture_dir, "sparse")

        cameras_txt = os.path.join(sparse_dir, "cameras.txt")
        images_txt = os.path.join(sparse_dir, "images.txt")

        if os.path.exists(cameras_txt) and os.path.exists(images_txt):
            intr_by_cam = parse_colmap_cameras(cameras_txt)
            self.poses, self.names, self.intrinsics_list = parse_colmap_images(images_txt, intr_by_cam)
            self.poses = normalize_scene_poses(np.stack(self.poses))
        else:
            # No COLMAP poses yet -- still usable for the diffusion-prior /
            # single-image material inference path, but multi-view photometric
            # consistency losses will be skipped until poses are supplied.
            self.names = sorted(
                n for n in os.listdir(self.images_dir)
                if n.lower().endswith((".jpg", ".jpeg", ".png"))
            ) if os.path.isdir(self.images_dir) else []
            self.poses = None
            self.intrinsics_list = None

        n = len(self.names)
        if not (3 <= n <= 6):
            import warnings
            warnings.warn(
                f"RealCaptureDataset expected 3-6 orbital photos, found {n} in {capture_dir}. "
                "Proceeding, but this pipeline is tuned for sparse casual captures."
            )

    def __len__(self) -> int:
        return 1  # whole-object episode

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgbs, masks, rays_o_all, rays_d_all = [], [], [], []
        for i, name in enumerate(self.names):
            img = _read_image(os.path.join(self.images_dir, name))
            rgb = normalize_image(img)
            rgbs.append(rgb)

            mask_path = os.path.join(self.masks_dir, os.path.splitext(name)[0] + ".png")
            if os.path.exists(mask_path):
                m = _read_image(mask_path)
                if m.ndim == 3:
                    m = m[..., 0]
                masks.append(torch.from_numpy((m > 127).astype(np.float32))[None])
            else:
                masks.append(torch.ones(1, rgb.shape[1], rgb.shape[2]))

            if self.poses is not None:
                h, w = rgb.shape[1], rgb.shape[2]
                ro, rd = build_rays(self.intrinsics_list[i], self.poses[i], h, w)
                rays_o_all.append(ro)
                rays_d_all.append(rd)

        sample = {
            "object_id": os.path.basename(self.capture_dir.rstrip("/")),
            "rgb": torch.stack(rgbs),
            "mask": torch.stack(masks),
            "has_poses": self.poses is not None,
        }
        if self.poses is not None:
            sample["rays_o"] = torch.stack(rays_o_all)
            sample["rays_d"] = torch.stack(rays_d_all)
            sample["poses"] = torch.from_numpy(np.stack(self.poses))
            sample["intrinsics"] = torch.from_numpy(np.stack(self.intrinsics_list))
        return sample
