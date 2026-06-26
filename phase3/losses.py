"""
losses.py — focal loss (binary/multiclass) matching the stage-2 classifier.

Kept in its own file so stage 2 and stage 3 import the SAME implementation;
a subtly different focal loss between stages would contaminate the
"pure_e2e vs CDR-enhanced" comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multiclass focal loss. For the 2-class glaucoma head this reduces to the
    usual binary focal loss but stays general so PALM (other disease labels,
    stage 4 transfer) can reuse it.

    gamma : focusing parameter (0 -> plain CE)
    alpha : optional per-class weight tensor [C] for residual imbalance on top
            of focusing. Pass None to disable.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits [B,C], target [B] int64
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)   # [B]
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)         # [B]
        focal = (1.0 - p_t) ** self.gamma * (-logp_t)            # [B]
        if self.alpha is not None:
            at = self.alpha.gather(0, target)                    # [B]
            focal = at * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class SegCEDiceLoss(nn.Module):
    """
    Segmentation anchor loss = class-weighted CE + soft Dice over foreground.

    Why not plain CrossEntropyLoss: optic disc/cup are tiny relative to the
    fundus background. Unweighted CE on a 3-class {bg, rim, cup} map is happy to
    predict "all background" — that minimizes CE while driving Dice and the
    CDR-degenerate rate the wrong way. This was visible in the stage-3 smoke
    test: with a from-scratch seg head, a bare-CE anchor *lowered* disc Dice
    epoch over epoch. A real (already-trained) stage-2 head doesn't start from
    that failure mode, but the anchor should be robust either way, and stage 2
    itself did not train segmentation with naked CE.

    ce_class_weights : [C] weights; foreground classes up-weighted. If None,
                       inferred per-batch from inverse class frequency.
    dice_classes     : which class indices count as foreground for the Dice term
                       (default {1,2} = rim and cup).
    """

    def __init__(self, ce_class_weights: torch.Tensor | None = None,
                 dice_classes: tuple[int, ...] = (1, 2),
                 dice_weight: float = 1.0, ce_weight: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()
        self.dice_classes = tuple(dice_classes)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.eps = float(eps)
        if ce_class_weights is not None:
            self.register_buffer("ce_w", ce_class_weights)
        else:
            self.ce_w = None

    def _ce(self, logits, target):
        if self.ce_w is not None:
            w = self.ce_w
        else:
            # inverse-frequency weights from this batch, normalized to mean 1
            C = logits.size(1)
            counts = torch.bincount(target.reshape(-1), minlength=C).float()
            freq = counts / counts.sum().clamp_min(1.0)
            inv = 1.0 / (freq + self.eps)
            w = (inv / inv.mean()).to(logits.dtype)
        return F.cross_entropy(logits, target, weight=w)

    def _dice(self, logits, target):
        probs = F.softmax(logits, dim=1)                      # [B,C,H,W]
        dice_terms = []
        for c in self.dice_classes:
            p = probs[:, c]
            t = (target == c).float()
            inter = (p * t).sum(dim=(1, 2))
            denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice_terms.append((2 * inter + self.eps) / (denom + self.eps))
        dice = torch.stack(dice_terms, dim=1).mean()
        return 1.0 - dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self._ce(logits, target) + \
               self.dice_weight * self._dice(logits, target)
