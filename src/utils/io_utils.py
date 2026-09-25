from __future__ import annotations

import os
from typing import Any, Dict

import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_checkpoint(state: Dict[str, Any], out_dir: str, name: str = "last.ckpt") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    return path


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
