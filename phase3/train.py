# -*- coding: utf-8 -*-
"""
阶段二训练入口。

示例:
  # 用 dummy 数据快速跑通 (无需任何外部文件)
  python3 train.py --preset tiny --dummy

  # 真实训练 (需 RETFound 权重 + manifest)
  python3 train.py --preset full \
      --retfound /path/RETFound_cfp_weights.pth \
      --train-manifest train.csv --val-manifest val.csv \
      --balancer uncertainty --seg-mode coarse2fine
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import Config, tiny_config
from model import MultiTaskModel
from balancing import UncertaintyWeighting, GradNorm
from dataset import (DummyMultiTaskDataset, MultiTaskManifestDataset,
                     multitask_collate, make_balanced_sampler)
from trainer import Trainer, build_scheduler


def build_balancer(cfg, device):
    m = cfg.balance.method
    if m == "uncertainty":
        return UncertaintyWeighting(
            list(cfg.task_names),
            max_weight=getattr(cfg.balance, "uw_max_weight", 0.0)).to(device)
    if m == "gradnorm":
        return GradNorm(list(cfg.task_names), alpha=cfg.balance.gradnorm_alpha).to(device)
    if m == "fixed":
        # 固定权重: 用一个无参数包装，直接加权和
        class _Fixed(torch.nn.Module):
            def __init__(self, names, weights):
                super().__init__()
                self.names = names
                self.weights = dict(zip(names, weights))
            def forward(self, losses):
                total = 0.0
                for n in self.names:
                    if n in losses:
                        total = total + self.weights[n] * losses[n]
                return total, {f"w_{n}": self.weights[n] for n in self.names}
        return _Fixed(list(cfg.task_names), cfg.balance.fixed_weights).to(device)
    raise ValueError(f"未知 balancer: {m}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["full", "tiny"], default="full")
    ap.add_argument("--dummy", action="store_true", help="用随机 dummy 数据")
    ap.add_argument("--retfound", type=str, default=None,
                    help="RETFound 预训练权重 .pth")
    ap.add_argument("--train-manifest", type=str, default=None)
    ap.add_argument("--val-manifest", type=str, default=None)
    ap.add_argument("--manifest", type=str, default=None,
                    help="包含 split 列的单张 CSV；与 --split-col 配合使用")
    ap.add_argument("--split-col", type=str, default="split",
                    help="manifest 里标记 train/val/test 的列名 (默认: split)")
    ap.add_argument("--balancer", type=str, default=None,
                    choices=["uncertainty", "gradnorm", "fixed"])
    ap.add_argument("--seg-mode", type=str, default=None,
                    choices=["coarse2fine", "highres"])
    ap.add_argument("--lora-mode", type=str, default=None,
                    choices=["lora", "adapter"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    # --- 新增: 类别不平衡 / 调度 控制 ---
    ap.add_argument("--focal-alpha", type=str, default=None,
                    help="逗号分隔的每类权重, 长度=num_classes, 例: '1.0,3.0' "
                         "(正类更高)。不传则用 config 默认 (None=不加权)。")
    ap.add_argument("--no-balanced-sampler", action="store_true",
                    help="关闭类别平衡采样 (默认开)。")
    ap.add_argument("--scheduler", type=str, default=None,
                    choices=["warmup_cosine", "none"],
                    help="学习率调度 (默认 warmup_cosine)。")
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    cfg = tiny_config() if args.preset == "tiny" else Config()

    # 命令行覆盖
    if args.retfound:
        cfg.backbone.pretrained_ckpt = args.retfound
    if args.balancer:
        cfg.balance.method = args.balancer
    if args.seg_mode:
        cfg.seg.mode = args.seg_mode
    if args.lora_mode:
        cfg.lora.mode = args.lora_mode
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.img_size is not None:
        cfg.backbone.img_size = (args.img_size, args.img_size)
    if args.focal_alpha is not None:
        cfg.cls.focal_alpha = tuple(float(x) for x in args.focal_alpha.split(","))
    if args.no_balanced_sampler:
        cfg.train.balanced_sampler = False
    if args.scheduler is not None:
        cfg.train.scheduler = args.scheduler
    if args.warmup_epochs is not None:
        cfg.train.warmup_epochs = args.warmup_epochs
    if args.lr is not None:
        cfg.train.lr = args.lr

    # GradNorm 约束: accum=1
    if cfg.balance.method == "gradnorm" and cfg.train.accum_steps != 1:
        print("[warn] GradNorm 要求 accum_steps=1，已自动设为 1。")
        cfg.train.accum_steps = 1

    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}, preset={args.preset}, "
          f"balancer={cfg.balance.method}, seg_mode={cfg.seg.mode}")

    # 支持单张含 split 列的 CSV（你的 data_all.csv 即为此格式）
    if args.manifest and not args.train_manifest:
        import pandas as pd
        df_all = pd.read_csv(args.manifest)
        col = args.split_col
        args.train_manifest = df_all[df_all[col] == "train"].reset_index(drop=True)
        if args.val_manifest is None:
            args.val_manifest = df_all[df_all[col] == "val"].reset_index(drop=True)

    # 数据
    if args.dummy or args.train_manifest is None:
        H, W = cfg.backbone.img_size
        train_ds = DummyMultiTaskDataset(n=8, img_size=(H, W),
                                         num_classes=cfg.cls.num_classes, seed=1)
        val_ds = DummyMultiTaskDataset(n=4, img_size=(H, W),
                                       num_classes=cfg.cls.num_classes, seed=2)
    else:
        train_ds = MultiTaskManifestDataset(args.train_manifest,
                                            img_size=cfg.backbone.img_size,
                                            train=True)
        _has_val = args.val_manifest is not None and (
            not hasattr(args.val_manifest, "empty") or not args.val_manifest.empty)
        val_ds = (MultiTaskManifestDataset(args.val_manifest,
                                           img_size=cfg.backbone.img_size,
                                           train=False)
                  if _has_val else None)

    # 训练采样: 类别不平衡时用 WeightedRandomSampler (与 shuffle 互斥)
    train_sampler = None
    if getattr(cfg.train, "balanced_sampler", False):
        train_sampler = make_balanced_sampler(train_ds, cfg.cls.num_classes)
        if train_sampler is None:
            print("[warn] balanced_sampler=True 但该数据集无法抽取 cls 标签, "
                  "退回普通 shuffle。")
        else:
            print(f"[info] 已启用类别平衡采样 (WeightedRandomSampler, "
                  f"n={len(train_ds)})。")
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=multitask_collate, drop_last=False)
    val_loader = (DataLoader(val_ds, batch_size=cfg.train.batch_size,
                             shuffle=False, num_workers=cfg.train.num_workers,
                             collate_fn=multitask_collate)
                  if val_ds is not None else None)

    # 模型
    model = MultiTaskModel(cfg).to(device)
    if cfg.backbone.pretrained_ckpt:
        info = model.backbone.load_pretrained(cfg.backbone.pretrained_ckpt)
        print(info)
    print("[params]", model.param_summary())

    # 平衡器 + 优化器 (模型可训练参数 + 平衡器参数)
    balancer = build_balancer(cfg, device)
    params = model.trainable_parameters() + [p for p in balancer.parameters()
                                             if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr,
                                  weight_decay=cfg.train.wd)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 学习率调度: warmup + 余弦退火 (按优化器 step 粒度)
    # steps_per_epoch = ceil(len(loader)/accum_steps)
    import math as _math
    steps_per_epoch = _math.ceil(len(train_loader) / max(1, cfg.train.accum_steps))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    if scheduler is not None:
        print(f"[info] scheduler=warmup_cosine, steps/epoch={steps_per_epoch}, "
              f"warmup={cfg.train.warmup_epochs}ep, base_lr={cfg.train.lr}, "
              f"min_lr={cfg.train.min_lr}")

    trainer = Trainer(model, optimizer, balancer, scaler, cfg, device,
                      scheduler=scheduler)

    os.makedirs(cfg.train.out_dir, exist_ok=True)
    best = -1.0
    for epoch in range(cfg.train.epochs):
        tr = trainer.train_one_epoch(train_loader, epoch)
        msg = " ".join(f"{k}={v:.4f}" for k, v in tr.items())
        print(f"[epoch {epoch}] train: {msg}")
        if val_loader is not None:
            ev = trainer.evaluate(val_loader)
            emsg = " ".join(f"{k}={v:.4f}" for k, v in ev.items())
            print(f"[epoch {epoch}] val:   {emsg}")
            score = ev.get("auc", 0.0)
            if score > best:
                best = score
                ckpt = {k: v for k, v in model.state_dict().items()}
                torch.save({"model": ckpt, "cfg": cfg.task_names},
                           os.path.join(cfg.train.out_dir, "best.pth"))

    # 末尾存一份 (只存可训练参数更省空间，这里为简单存全量)
    torch.save({"model": model.state_dict()},
               os.path.join(cfg.train.out_dir, "last.pth"))
    print("[done] saved to", cfg.train.out_dir)


if __name__ == "__main__":
    main()
