# -*- coding: utf-8 -*-
"""
各任务 loss。

- 分类: FocalLoss 抗类别不平衡。
- 分割: BCEWithLogits + Dice (multi-label，disc/cup 各一个 sigmoid)。
- 黄斑: heatmap MSE (对高斯目标) + 坐标 L1。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 分类 Focal Loss
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """多分类 focal loss。alpha 为各类权重 (可选)。"""

    def __init__(self, gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha) if alpha is not None else None
        self._alpha = alpha

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)             # (B,C)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)  # (B,)
        pt = logpt.exp()
        focal = (1 - pt) ** self.gamma * (-logpt)
        if self._alpha is not None:
            a = self._alpha.to(logits.device)[target]
            focal = a * focal
        return focal.mean()


# --------------------------------------------------------------------------- #
# 分割 Dice + BCE
# --------------------------------------------------------------------------- #
def dice_loss(prob: torch.Tensor, target: torch.Tensor,
              eps: float = 1.0) -> torch.Tensor:
    """逐样本逐通道 soft dice loss。prob, target: (B,C,H,W)。"""
    dims = (2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return (1 - dice).mean()


class SegLoss(nn.Module):
    def __init__(self, dice_w: float = 1.0, bce_w: float = 1.0):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        d = dice_loss(torch.sigmoid(logits), target)
        return self.bce_w * bce + self.dice_w * d


# --------------------------------------------------------------------------- #
# 黄斑 heatmap + 坐标
# --------------------------------------------------------------------------- #
def gaussian_heatmap(coords: torch.Tensor, H: int, W: int,
                     sigma: float = 0.03) -> torch.Tensor:
    """由归一化坐标 (B,2)=(x,y) 生成高斯 heatmap (B,1,H,W)，峰值 1。"""
    B = coords.shape[0]
    device = coords.device
    ys = torch.linspace(0, 1, H, device=device).view(1, H, 1)
    xs = torch.linspace(0, 1, W, device=device).view(1, 1, W)
    cx = coords[:, 0].view(B, 1, 1)
    cy = coords[:, 1].view(B, 1, 1)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    g = torch.exp(-d2 / (2 * sigma * sigma))
    return g.unsqueeze(1)


class HeatmapLoss(nn.Module):
    def __init__(self, sigma: float = 0.03, coord_w: float = 1.0,
                 heat_w: float = 1.0):
        super().__init__()
        self.sigma = sigma
        self.coord_w = coord_w
        self.heat_w = heat_w

    def forward(self, pred_heat: torch.Tensor, pred_coords: torch.Tensor,
                gt_coords: torch.Tensor) -> torch.Tensor:
        B, _, H, W = pred_heat.shape
        gt_heat = gaussian_heatmap(gt_coords, H, W, self.sigma)
        heat_loss = F.mse_loss(torch.sigmoid(pred_heat), gt_heat)
        coord_loss = F.l1_loss(pred_coords, gt_coords)
        return self.heat_w * heat_loss + self.coord_w * coord_loss
