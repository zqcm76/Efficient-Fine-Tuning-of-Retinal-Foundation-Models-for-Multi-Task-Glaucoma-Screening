"""End-to-end smoke test on synthetic data (CPU-friendly).

Runs the entire Phase 1 stack — transforms, datasets, both models, both losses, the
Trainer's fit/evaluate loop, and the metric functions — without needing any real data or
a GPU. If this prints ALL SMOKE TESTS PASSED, the plumbing is correct and you can plug in
REFUGE2 by pointing the config at your own index CSV.
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from make_synthetic_data import make_synthetic  # noqa: E402

from src.config import Config  # noqa: E402
from src.datasets import REFUGE2Dataset  # noqa: E402
from src.engine import Trainer, evaluate_classifier, evaluate_segmenter  # noqa: E402
from src.metrics import compute_cdr  # noqa: E402
from src.models import BCEClsLoss, DiceCELoss, ResNet34Classifier, UNet  # noqa: E402
from src.transforms import build_cls_transforms, build_seg_transforms  # noqa: E402
from src.utils import count_parameters, set_seed  # noqa: E402


def main() -> None:
    set_seed(0)
    device = torch.device("cpu")
    tmp = tempfile.mkdtemp()
    df = make_synthetic(tmp, n_train=16, n_val=8, size=192)
    print(f"[ok] synthetic data: {len(df)} samples")

    cfg = Config()
    cfg.train.epochs = 1
    cfg.train.batch_size = 4
    cfg.train.amp = False
    cfg.output_dir = os.path.join(tmp, "out")

    tr_df = df[df.split == "train"]
    va_df = df[df.split == "val"]
    sz = 128

    # ----- classification -----
    tl = DataLoader(REFUGE2Dataset(tr_df, build_cls_transforms(sz, True), "cls"),
                    batch_size=4, shuffle=True, num_workers=0)
    vl = DataLoader(REFUGE2Dataset(va_df, build_cls_transforms(sz, False), "cls"),
                    batch_size=4, num_workers=0)
    b = next(iter(tl))
    assert tuple(b["image"].shape[1:]) == (3, sz, sz), b["image"].shape
    clf = ResNet34Classifier(num_classes=1, pretrained=False)
    print(f"[ok] ResNet34 trainable params: {count_parameters(clf):,}")
    loss = BCEClsLoss().to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
    Trainer(clf, opt, loss, device, cfg,
            lambda bt, d: (bt["image"].to(d), bt["label"].to(d)),
            evaluate_classifier, monitor="auc", mode="max").fit(tl, vl, cfg.output_dir, "smoke_cls")
    print("[ok] classifier: 1-epoch fit + eval + checkpoint")

    # ----- segmentation -----
    tl2 = DataLoader(REFUGE2Dataset(tr_df, build_seg_transforms(sz, True), "seg"),
                     batch_size=4, shuffle=True, num_workers=0)
    vl2 = DataLoader(REFUGE2Dataset(va_df, build_seg_transforms(sz, False), "seg"),
                     batch_size=4, num_workers=0)
    b2 = next(iter(tl2))
    assert tuple(b2["mask"].shape[1:]) == (sz, sz), b2["mask"].shape
    assert b2["mask"].dtype == torch.long, b2["mask"].dtype
    seg = UNet(in_channels=3, num_classes=3, base=16)
    print(f"[ok] UNet trainable params: {count_parameters(seg):,}")
    out = seg(b2["image"])
    assert tuple(out.shape) == (4, 3, sz, sz), out.shape
    sl = DiceCELoss(num_classes=3).to(device)
    assert torch.isfinite(sl(out, b2["mask"])), "seg loss not finite"
    opt2 = torch.optim.AdamW(seg.parameters(), lr=1e-3)
    Trainer(seg, opt2, sl, device, cfg,
            lambda bt, d: (bt["image"].to(d), bt["mask"].to(d)),
            evaluate_segmenter, monitor="mean_dice", mode="max").fit(tl2, vl2, cfg.output_dir, "smoke_seg")
    print("[ok] segmenter: 1-epoch fit + eval + checkpoint")

    # ----- metric sanity -----
    cdr = compute_cdr(b2["mask"][0].numpy())
    print("[ok] CDR on a ground-truth mask:", {k: round(v, 3) for k, v in cdr.items()})

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
