# -*- coding: utf-8 -*-
"""
训练器: 指标、loss 汇总 (含任务屏蔽与分割目标路由)、两条训练路径、评测。

两条训练路径:
- 标准: AMP autocast + GradScaler + 梯度累积; 训练时 use_gt_roi=True (teacher forcing)。
- GradNorm: 强制 accum_steps=1、fp32 (不 autocast)，weighted.backward(retain_graph=True)
  得到模型梯度后，用 autograd.grad 计算 GradNorm 辅助 loss 对 w 的梯度并覆盖 w.grad，
  再 step，最后 renormalize。
"""
from __future__ import annotations

from typing import Dict

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import FocalLoss, SegLoss, HeatmapLoss
from model import roi_align_norm, paste_roi_to_full


# --------------------------------------------------------------------------- #
# 学习率调度 (之前完全缺失: warmup_epochs 是死配置, lr 全程常量)
# --------------------------------------------------------------------------- #
def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """线性 warmup + 余弦退火, 按 *优化器 step* 粒度更新 (不是 epoch 粒度)。

    steps_per_epoch = ceil(len(loader)/accum_steps); 总 step = epochs*steps_per_epoch。
    warmup 阶段 lr 从 0 线性升到 base_lr; 之后余弦退火到 min_lr。
    cfg.train.scheduler == 'none' 时返回 None (退回旧的常量 lr 行为)。
    """
    if getattr(cfg.train, "scheduler", "none") != "warmup_cosine":
        return None
    steps_per_epoch = max(1, steps_per_epoch)
    warmup_steps = int(cfg.train.warmup_epochs * steps_per_epoch)
    total_steps = max(1, cfg.train.epochs * steps_per_epoch)
    base_lr = cfg.train.lr
    min_lr = cfg.train.min_lr
    min_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        # 余弦退火: progress ∈ [0,1]
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_ratio + (1.0 - min_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """二分类 AUC (Mann-Whitney U / rank 法，带并列校正)。"""
    labels = labels.astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(scores)
    sum_pos = ranks[labels == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def dice_metric(prob: torch.Tensor, target: torch.Tensor,
                thresh: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """硬 Dice (逐样本逐通道再平均)。prob,target: (B,C,H,W)。返回每通道均值 (C,)。"""
    pred = (prob > thresh).float()
    dims = (2, 3)
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)            # (B,C)
    return dice.mean(dim=0)                              # (C,)


def macula_error(pred_xy: torch.Tensor, gt_xy: torch.Tensor,
                 disc_mask: torch.Tensor = None,
                 eps: float = 1e-12) -> torch.Tensor:
    """黄斑定位误差 (B,)。

    默认 (disc_mask=None): 归一化坐标的欧氏距离 —— 阶段二口径，行为与原来一致。

    传入 disc_mask 时: 再按视盘直径归一化 (阶段四 / REFUGE 方向)。
    视盘等效直径 = 2*sqrt(area/pi)，除以图像边长换算到与归一化坐标同一尺度后作分母，
    得到“以视盘直径为单位”的尺度无关误差；视盘为空的样本退回未归一化距离。

    注意: REFUGE 评测脚本对“视盘直径”的确切定义 (等效圆 / 垂直径 / 外接框) 可能不同，
    阶段四正式落地时请对齐目标评测实现；这里取的是等效圆直径，作为可用的近似。
    """
    d = torch.sqrt(((pred_xy - gt_xy) ** 2).sum(dim=1) + eps)        # (B,)
    if disc_mask is None:
        return d
    m = disc_mask
    if m.dim() == 4:
        m = m[:, 0]
    m = (m > 0.5).float()                                            # (B,H,W)
    H, W = m.shape[-2], m.shape[-1]
    side = float(H + W) / 2.0                                        # 方图 H==W
    area = m.flatten(1).sum(dim=1)                                   # (B,) 像素面积
    diam = (2.0 * torch.sqrt(area / math.pi + eps)) / side          # 归一化直径
    diam = torch.where(area > 0, diam.clamp(min=eps),
                       torch.ones_like(diam))                        # 空盘 → 不归一化
    return d / diam


# --------------------------------------------------------------------------- #
# 训练器
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, model, optimizer, balancer, scaler, cfg, device,
                 scheduler=None):
        self.model = model
        self.opt = optimizer
        self.balancer = balancer
        self.scaler = scaler
        self.cfg = cfg
        self.device = device
        self.scheduler = scheduler   # 可为 None (smoke_test / scheduler='none')

        alpha = None
        if cfg.cls.focal_alpha is not None:
            alpha = torch.tensor(cfg.cls.focal_alpha, dtype=torch.float32)
        self.focal = FocalLoss(gamma=cfg.cls.focal_gamma, alpha=alpha)
        self.seg_loss = SegLoss(dice_w=1.0, bce_w=1.0)
        self.heat_loss = HeatmapLoss(sigma=cfg.macula.sigma,
                                     coord_w=cfg.macula.coord_weight,
                                     heat_w=cfg.macula.heatmap_weight)
        self.is_gradnorm = (cfg.balance.method == "gradnorm")

    # ----- loss 汇总 (按 has_* 屏蔽) ----- #
    def compute_losses(self, out: Dict[str, object],
                       batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        dev = self.device
        losses: Dict[str, torch.Tensor] = {}

        # 分类
        has_cls = batch["has_cls"].to(dev)
        if has_cls.any():
            logits = out["cls_logits"][has_cls]
            tgt = batch["cls_label"].to(dev)[has_cls]
            losses["cls"] = self.focal(logits, tgt)

        # 分割
        has_seg = batch["has_seg"].to(dev)
        if has_seg.any():
            seg_target = batch["seg_target"].to(dev)       # (B,2,H,W)
            if self.model.seg_mode == "coarse2fine":
                # 粗: 把目标降到粗分辨率
                coarse_logits = out["coarse_logits"]
                ch = coarse_logits.shape[-1]
                coarse_tgt = F.interpolate(seg_target, size=(ch, ch),
                                           mode="bilinear",
                                           align_corners=False)
                coarse_tgt = (coarse_tgt > 0.5).float()
                l_coarse = self.seg_loss(coarse_logits[has_seg],
                                         coarse_tgt[has_seg])
                # 精: 用 roi_align 把目标裁到 fine 分辨率 (与预测同一 ROI)
                fine_logits = out["fine_logits"]
                fh = fine_logits.shape[-1]
                roi_boxes = out["roi_boxes"]
                fine_tgt = roi_align_norm(seg_target, roi_boxes, fh)
                fine_tgt = (fine_tgt > 0.5).float()
                l_fine = self.seg_loss(fine_logits[has_seg],
                                       fine_tgt[has_seg])
                losses["seg"] = self.cfg.seg.coarse_loss_w * l_coarse + l_fine
            else:  # highres
                seg_logits = out["seg_logits"]
                sh, sw = seg_logits.shape[-2:]
                tgt = F.interpolate(seg_target, size=(sh, sw),
                                    mode="bilinear", align_corners=False)
                tgt = (tgt > 0.5).float()
                losses["seg"] = self.seg_loss(seg_logits[has_seg],
                                              tgt[has_seg])

        # 黄斑
        has_macula = batch["has_macula"].to(dev)
        if has_macula.any():
            heat = out["macula_heatmap"][has_macula]
            coords = out["macula_coords"][has_macula]
            gt = batch["macula"].to(dev)[has_macula]
            losses["macula"] = self.heat_loss(heat, coords, gt)

        return losses

    # ----- 一个 epoch ----- #
    def train_one_epoch(self, loader, epoch: int = 0) -> Dict[str, float]:
        self.model.train()
        if self.is_gradnorm:
            return self._train_gradnorm(loader, epoch)
        return self._train_standard(loader, epoch)

    def _train_standard(self, loader, epoch: int) -> Dict[str, float]:
        cfg = self.cfg
        accum = max(1, cfg.train.accum_steps)
        use_amp = cfg.train.amp and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16

        running: Dict[str, float] = {}
        self.opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            gt_disc = batch["disc_mask"].to(self.device)

            with torch.autocast(device_type=self.device.type,
                                dtype=amp_dtype, enabled=use_amp):
                out = self.model(images, gt_disc=gt_disc, use_gt_roi=True)
                losses = self.compute_losses(out, batch)
                if len(losses) == 0:
                    continue
                total, logs = self.balancer(losses)
                total = total / accum

            self.scaler.scale(total).backward()

            if (step + 1) % accum == 0:
                if cfg.train.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(
                        self.model.trainable_parameters(), cfg.train.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                if self.scheduler is not None:   # 按优化器 step 更新 lr
                    self.scheduler.step()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            running["total"] = running.get("total", 0.0) + float(
                total.detach()) * accum

        n = max(1, len(loader))
        return {k: v / n for k, v in running.items()}

    def _train_gradnorm(self, loader, epoch: int) -> Dict[str, float]:
        """GradNorm 路径: accum=1、fp32、双反传。"""
        cfg = self.cfg
        running: Dict[str, float] = {}
        shared = self.model.gradnorm_shared_params()

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            gt_disc = batch["disc_mask"].to(self.device)

            self.opt.zero_grad(set_to_none=True)

            # fp32，不 autocast
            out = self.model(images, gt_disc=gt_disc, use_gt_roi=True)
            losses = self.compute_losses(out, batch)
            if len(losses) == 0:
                continue

            self.balancer.set_initial(losses)
            total = self.balancer.weighted_sum(losses)
            # 先回传模型梯度 (保留图，后面 GradNorm 还要再 autograd.grad)
            total.backward(retain_graph=True)

            # GradNorm 辅助 loss → 只对 w 求梯度并覆盖 w.grad
            gn = self.balancer.grad_norm_loss(losses, shared)
            w = self.balancer.w
            gw = torch.autograd.grad(gn, w, retain_graph=False)[0]
            if w.grad is None:
                w.grad = gw.detach().clone()
            else:
                w.grad.copy_(gw.detach())

            if cfg.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.trainable_parameters(), cfg.train.grad_clip)
            self.opt.step()
            self.balancer.renormalize()
            if self.scheduler is not None:   # GradNorm: accum=1, 每 iter step 一次
                self.scheduler.step()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            running["total"] = running.get("total", 0.0) + float(total.detach())

        n = max(1, len(loader))
        return {k: v / n for k, v in running.items()}

    # ----- 评测 ----- #
    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        cls_scores, cls_labels = [], []
        dice_sum = torch.zeros(self.cfg.seg.num_classes)
        dice_n = 0
        mac_errs = []
        mac_errs_ddn = []        # 阶段四口径: 按视盘直径归一化的黄斑误差

        for batch in loader:
            images = batch["image"].to(self.device)
            # 评测不喂 GT ROI，用模型自己的粗预测定位
            out = self.model(images, gt_disc=None, use_gt_roi=False)

            # 分类
            has_cls = batch["has_cls"]
            if has_cls.any():
                # 先回 CPU 再用 CPU 掩膜索引 (避免 CUDA 张量被 CPU bool mask 索引的版本兼容问题)
                prob = torch.softmax(out["cls_logits"], dim=1)[:, 1].cpu()
                cls_scores.append(prob[has_cls].numpy())
                cls_labels.append(batch["cls_label"][has_cls].cpu().numpy())

            # 分割
            has_seg = batch["has_seg"]
            if has_seg.any():
                seg_target = batch["seg_target"]
                H, W = seg_target.shape[-2:]
                if self.model.seg_mode == "coarse2fine":
                    full_logits = paste_roi_to_full(
                        out["fine_logits"], out["roi_boxes"], (H, W))
                else:
                    full_logits = F.interpolate(
                        out["seg_logits"], size=(H, W),
                        mode="bilinear", align_corners=False)
                prob = torch.sigmoid(full_logits).cpu()
                d = dice_metric(prob[has_seg], seg_target[has_seg])
                dice_sum += d
                dice_n += 1

            # 黄斑
            has_macula = batch["has_macula"]
            if has_macula.any():
                coords = out["macula_coords"].cpu()        # 先回 CPU 再索引
                gt = batch["macula"]
                mac_errs.append(macula_error(coords[has_macula],
                                             gt[has_macula]).numpy())
                # 阶段四口径: 对同时有视盘标注的样本，按视盘直径归一化
                both = has_macula & batch["has_seg"]
                if both.any():
                    disc_gt = batch["disc_mask"]           # GT 视盘 (B,1,H,W)
                    mac_errs_ddn.append(
                        macula_error(coords[both], gt[both],
                                     disc_mask=disc_gt[both]).numpy())

        metrics: Dict[str, float] = {}
        if cls_scores:
            s = np.concatenate(cls_scores)
            l = np.concatenate(cls_labels)
            metrics["auc"] = auc_score(s, l)
        if dice_n > 0:
            dmean = (dice_sum / dice_n)
            metrics["dice_disc"] = float(dmean[0])
            metrics["dice_cup"] = float(dmean[1])
            metrics["dice_mean"] = float(dmean.mean())
        if mac_errs:
            metrics["macula_err"] = float(np.concatenate(mac_errs).mean())
        if mac_errs_ddn:
            metrics["macula_err_ddn"] = float(
                np.concatenate(mac_errs_ddn).mean())
        return metrics
