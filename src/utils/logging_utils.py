from __future__ import annotations

import logging
import sys
from collections import defaultdict
from typing import Dict


def get_logger(name: str = "neural_relighting") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class MetricLogger:
    """Tiny running-average metric tracker, printed periodically during training."""

    def __init__(self):
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def update(self, **metrics: float):
        for k, v in metrics.items():
            self._sums[k] += float(v)
            self._counts[k] += 1

    def averages(self) -> Dict[str, float]:
        return {k: self._sums[k] / max(self._counts[k], 1) for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()

    def summary_str(self) -> str:
        return " | ".join(f"{k}: {v:.4f}" for k, v in self.averages().items())
