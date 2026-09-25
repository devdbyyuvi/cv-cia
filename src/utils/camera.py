from __future__ import annotations

import torch


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    return torch.device("cpu")
