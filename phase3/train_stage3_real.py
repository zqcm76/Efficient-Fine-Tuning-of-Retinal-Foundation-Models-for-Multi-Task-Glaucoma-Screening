# -*- coding: utf-8 -*-
"""
train_stage3_real.py — 阶段三真实数据训练入口。

把阶段二的 best.pth checkpoint 加载进来，套上 CDR 融合头，
在真实 REFUGE2/PALM 等数据上做三种融合模式消融。

示例:
    # 全量消融 (三种模式逐个跑, 结果打表)
    python3 train_stage3_real.py \\
        --stage2-ckpt runs/phase2/best.pth \\
        --train-manifest data/train.csv \\
        --val-manifest   data/val.csv \\
        --mode all \\
        --epochs 15

    # 只跑 cdr_soft
    python3 train_stage3_real.py \\
        --stage2-ckpt runs/phase2/best.pth \\
        --train-manifest data/train.csv \\
        --val-manifest   data/val.csv \\
        --mode cdr_soft \\
        --epochs 15

    # 快速验证 (dummy 数据，不需要任何外部文件)
    python3 train_stage3_real.py --dummy --mode all --epochs 5
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, tiny_config
from dataset import (DummyMultiTaskDataset, MultiTaskManifestDataset,
                     multitask_collate)
from stage2_adapter import (Stage2Adapter, Stage3DataLoaderWrapper,
                             load_stage2_model)
from stage2_iface import MockStage2
from train_stage3 import Stage3Config, Stage3Trainer, _make_synthetic_loader


# --------------------------------------------------------------------------- #
# 帮助函数
# --------------------------------------------------------------------------- #
def _make_real_loader(manifest, img_size, batch_size, train, num_workers, seg_size):
    """从 manifest 构建 stage3 格式的 DataLoader。"""
    ds = MultiTaskManifestDataset(manifest, img_size=img_size, train=train)
    raw = DataLoader(ds, batch_size=batch_size, shuffle=train,
                     num_workers=num_workers, collate_fn=multitask_collate,
                     drop_last=False)
    return Stage3DataLoaderWrapper(raw, seg_size=seg_size)


def _make_dummy_loader(n, img_size, batch_size, seed, seg_size):
    """dummy 数据的 stage3 格式 DataLoader (通过 _make_synthetic_loader)。"""
    return _make_synthetic_loader(n=n, batch=batch_size,
                                  seg_size=seg_size, seed=seed)


def _print_table(rows):
    cols = ["mode", "auc", "dice_disc", "dice_cup",
            "cdr_degenerate_rate", "trainable_params", "grad_to_seg"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Stage 3: CDR fusion fine-tuning")
    ap.add_argument("--stage2-ckpt", type=str, default=None,
                    help="阶段二 best.pth 路径 (dummy 模式时不需要)")
    ap.add_argument("--train-manifest", type=str, default=None)
    ap.add_argument("--val-manifest",   type=str, default=None)
    ap.add_argument("--manifest", type=str, default=None,
                    help="含 split 列的单张 CSV")
    ap.add_argument("--split-col", type=str, default="split")
    ap.add_argument("--dummy", action="store_true",
                    help="使用 synthetic dummy 数据 (不需要任何外部文件)")
    ap.add_argument("--mode", type=str, default="all",
                    choices=["pure_e2e", "cdr_two_stage", "cdr_soft", "all"],
                    help="融合模式; 'all' 逐个跑三种并打消融表")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--seg-size", type=int, default=96,
                    help="CDR 用的分割图尺寸 (coarse_logits 会被插值到此尺寸)")
    ap.add_argument("--lr-head", type=float, default=1e-3,
                    help="分类头学习率")
    ap.add_argument("--lr-seg", type=float, default=1e-5,
                    help="cdr_soft 时分割头学习率 (小, 防止过度修改)")
    ap.add_argument("--seg-loss-weight", type=float, default=1.0)
    ap.add_argument("--focal-alpha", type=str, default=None,
                    help="focal loss 类别权重, 逗号分隔, 如 '0.25,0.75'")
    ap.add_argument("--soft-cdr-scale", type=float, default=0.1,
                    help="cls->seg 梯度缩放 (cdr_soft, 默认 0.1)")
    ap.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="./runs/phase3")
    ap.add_argument("--preset", choices=["full", "tiny"], default="full",
                    help="tiny=极小骨干, 用于快速测试")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[stage3] device={device}, mode={args.mode}, epochs={args.epochs}")

    # ---- 阶段二配置 ----
    s2cfg = tiny_config() if args.preset == "tiny" else Config()
    s2cfg.backbone.img_size = (args.img_size, args.img_size)
    img_feat_dim = s2cfg.backbone.embed_dim  # 192 (tiny) or 1024 (full ViT-L)

    # ---- 数据 ----
    if args.manifest and not args.train_manifest:
        import pandas as pd
        df_all = pd.read_csv(args.manifest)
        col = args.split_col
        args.train_manifest = df_all[df_all[col] == "train"].reset_index(drop=True)
        if args.val_manifest is None:
            args.val_manifest = df_all[df_all[col] == "val"].reset_index(drop=True)

    if args.dummy or args.train_manifest is None:
        print("[stage3] 使用 synthetic dummy 数据")
        train_loader = _make_dummy_loader(
            n=96, img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size, seed=1, seg_size=args.seg_size)
        val_loader = _make_dummy_loader(
            n=48, img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size, seed=2, seg_size=args.seg_size)
    else:
        train_loader = _make_real_loader(
            args.train_manifest, (args.img_size, args.img_size),
            args.batch_size, train=True,
            num_workers=args.num_workers, seg_size=args.seg_size)
        val_loader = _make_real_loader(
            args.val_manifest, (args.img_size, args.img_size),
            args.batch_size, train=False,
            num_workers=args.num_workers, seg_size=args.seg_size) \
            if args.val_manifest is not None else None

    # ---- Stage3Config ----
    focal_alpha = None
    if args.focal_alpha:
        focal_alpha = tuple(float(x) for x in args.focal_alpha.split(","))

    # soft-CDR warmup: 让 seg-loss anchor 先稳定约半程，再启用软 CDR 耦合
    steps_per_epoch = len(train_loader)
    warmup_steps = max(1, (steps_per_epoch * args.epochs) // 2)

    def make_s3cfg(mode):
        return Stage3Config(
            mode=mode,
            img_feat_dim=img_feat_dim,
            lr_head=args.lr_head,
            lr_seg=args.lr_seg,
            focal_alpha=focal_alpha,
            seg_loss_weight=args.seg_loss_weight,
            soft_cdr_seg_grad_scale=args.soft_cdr_scale,
            soft_cdr_warmup_steps=warmup_steps,
            amp=(not args.no_amp) and (device == "cuda"),
        )

    # ---- 加载阶段二模型 ----
    def make_stage2_model():
        """每次调用都重新加载，确保各 mode 从同一初始点出发。"""
        torch.manual_seed(args.seed)
        if args.dummy or args.stage2_ckpt is None:
            # dummy 模式: 用 MockStage2
            H = args.img_size
            seg_size_mock = H // (s2cfg.backbone.patch_size * 2)
            seg_size_mock = max(8, seg_size_mock)
            m = MockStage2(feat_dim=img_feat_dim, seg_size=args.seg_size)
            return m
        else:
            # 真实模式: 加载 checkpoint 并包装
            model_s2 = load_stage2_model(args.stage2_ckpt, s2cfg, device)
            return Stage2Adapter(model_s2, seg_out_size=args.seg_size)

    # ---- 运行 ----
    modes = (["pure_e2e", "cdr_two_stage", "cdr_soft"]
             if args.mode == "all" else [args.mode])
    rows = []

    for mode in modes:
        print(f"\n{'='*50}\n[stage3] mode = {mode}\n{'='*50}")
        torch.manual_seed(args.seed); np.random.seed(args.seed)

        model = make_stage2_model()
        cfg   = make_s3cfg(mode)
        trainer = Stage3Trainer(model, cfg, device=device)

        n_head = sum(p.numel() for p in trainer.head.parameters() if p.requires_grad)
        n_seg  = sum(p.numel() for p in trainer.model.segmentation_parameters()
                     if p.requires_grad)
        print(f"[stage3] trainable: head={n_head:,}  seg={n_seg:,}")

        trainer.fit(train_loader, val_loader, epochs=args.epochs, log_every=50)

        val = trainer.evaluate(val_loader) if val_loader else {}
        print(f"[stage3] final val: {val}")

        rows.append({
            "mode": mode,
            "auc":  round(val.get("auc", float("nan")), 4),
            "dice_disc": round(val.get("dice_disc", float("nan")), 4),
            "dice_cup":  round(val.get("dice_cup", float("nan")), 4),
            "cdr_degenerate_rate": round(val.get("cdr_degenerate_rate", 0.0), 4),
            "trainable_params": n_head + n_seg,
            "grad_to_seg": mode == "cdr_soft",
        })

        # 保存各 mode 的 checkpoint
        ckpt_path = os.path.join(args.out_dir, f"head_{mode}.pth")
        torch.save({"head": trainer.head.state_dict(),
                    "mode": mode, "val": val,
                    "img_feat_dim": img_feat_dim},
                   ckpt_path)
        print(f"[stage3] saved → {ckpt_path}")

    # ---- 打印消融表 ----
    if len(rows) > 1:
        print(f"\n{'='*50}")
        print("Stage-3 融合消融结果")
        print(f"{'='*50}")
        _print_table(rows)

    # 保存 JSON
    result_path = os.path.join(args.out_dir, "ablation_results.json")
    with open(result_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[stage3] JSON 结果 → {result_path}")


if __name__ == "__main__":
    main()
