"""Small reusable utilities."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def setup_logger(name: str = "phase1", logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on re-import
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def soft_argmax2d(heatmap: torch.Tensor) -> torch.Tensor:
    """Differentiable 2D soft-argmax. heatmap: (B,1,H,W) or (B,H,W) -> coords (B,2) as (x,y).

    Not used by the Phase 1 baselines, but lives here because Phase 2 decodes the fovea
    heatmap head with it (and the eval metric already supports fovea distance)."""
    if heatmap.dim() == 4:
        heatmap = heatmap.squeeze(1)
    b, h, w = heatmap.shape
    prob = torch.softmax(heatmap.reshape(b, -1), dim=1).reshape(b, h, w)
    ys = torch.linspace(0, h - 1, h, device=heatmap.device)
    xs = torch.linspace(0, w - 1, w, device=heatmap.device)
    y = (prob.sum(dim=2) * ys).sum(dim=1)
    x = (prob.sum(dim=1) * xs).sum(dim=1)
    return torch.stack([x, y], dim=1)
