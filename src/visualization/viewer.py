"""Phase 5: simple desktop viewer for inspecting extracted 3D geometry
(mesh, via marching cubes on the predicted SDF), per-vertex material
properties, and side-by-side relit renders, built on Open3D.
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import torch

from src.models.network import InverseRenderingNetwork
from src.utils.io_utils import load_config, load_checkpoint
from src.utils.camera import get_device
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def extract_mesh(model: InverseRenderingNetwork, resolution: int = 128, bound: float = 1.5):
    """Marching cubes over a dense grid evaluation of the predicted SDF."""
    from skimage import measure

    lin = np.linspace(-bound, bound, resolution)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

    device = next(model.parameters()).device
    sdf_vals = []
    with torch.no_grad():
        for chunk in np.array_split(pts, max(1, len(pts) // 65536)):
            t = torch.from_numpy(chunk).to(device)
            sdf_vals.append(model.sdf_fn(t).cpu().numpy())
    sdf_grid = np.concatenate(sdf_vals).reshape(resolution, resolution, resolution)

    verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0)
    verts = verts / (resolution - 1) * (2 * bound) - bound  # back to world space
    return verts, faces, normals


def colorize_by_material(model: InverseRenderingNetwork, verts: np.ndarray, channel: str = "albedo") -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        pts = torch.from_numpy(verts.astype(np.float32)).to(device)
        mats = model.material_fn(pts)
    val = mats[channel].cpu().numpy()
    if val.shape[-1] == 1:
        val = np.repeat(val, 3, axis=-1)
    return np.clip(val, 0.0, 1.0)


def launch_viewer(ckpt_path: str, cfg_path: str = "configs/default.yaml", mesh_resolution: int = 128,
                   material_channel: str = "albedo"):
    try:
        import open3d as o3d
    except ImportError:
        raise RuntimeError("open3d is required for the desktop viewer: pip install open3d")

    cfg = load_config(cfg_path)
    device = get_device(cfg["experiment"]["device"])
    model = InverseRenderingNetwork(cfg).to(device).eval()
    ckpt = load_checkpoint(ckpt_path, map_location=str(device))
    model.load_state_dict(ckpt["model"])

    logger.info("Extracting mesh via marching cubes over the predicted SDF...")
    verts, faces, normals = extract_mesh(model, resolution=mesh_resolution)
    colors = colorize_by_material(model, verts, channel=material_channel)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()

    logger.info(f"Launching Open3D viewer ({len(verts)} verts, colored by '{material_channel}')...")
    o3d.visualization.draw_geometries([mesh])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--channel", type=str, default="albedo", choices=["albedo", "roughness", "metallic"])
    args = parser.parse_args()
    launch_viewer(args.ckpt, args.config, args.resolution, args.channel)


if __name__ == "__main__":
    main()
