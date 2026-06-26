"""Datasets.

`REFUGE2Dataset` is the only set used for TRAINING. `FundusSegDataset` is the generic
held-out / zero-shot evaluation dataset used in Phase 4 for the cross-device domains
(Drishti-GS, RIM-ONE, RIGA, PALM-disc-only, ...). None of those are ever trained on —
that is what keeps the headline result "zero-shot".
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from masks import parse_refuge_mask


def _imread_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return m


class REFUGE2Dataset(Dataset):
    """Multi-task REFUGE2 dataset.

    Required dataframe columns by task:
        task='cls' : image_path, label
        task='seg' : image_path, mask_path
    'fovea_x' / 'fovea_y' are read when present (plumbed through for Phase 2; the Phase 1
    baseline scripts do not use them).
    """

    def __init__(self, df: pd.DataFrame, transforms, task: str = "seg") -> None:
        assert task in {"cls", "seg"}, task
        self.df = df.reset_index(drop=True)
        self.t = transforms
        self.task = task

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        img = _imread_rgb(row["image_path"])
        if self.task == "cls":
            out = self.t(image=img)
            return {"image": out["image"], "label": torch.tensor(float(row["label"]))}
        # segmentation
        label = parse_refuge_mask(_read_mask(row["mask_path"]))
        out = self.t(image=img, mask=label)
        return {"image": out["image"], "mask": out["mask"].long()}


class FundusSegDataset(Dataset):
    """Generic optic disc/cup segmentation dataset for the zero-shot eval domains.

    Pass a dataset-specific `mask_parser(np.ndarray) -> label map` when the encoding
    differs from REFUGE (it usually does — each public set is different). For PALM, which
    has no optic-cup annotation, supply a disc-only parser; cup Dice / CDR are then simply
    not reported for that domain.
    """

    def __init__(self, pairs, transforms, mask_parser=parse_refuge_mask) -> None:
        self.pairs = list(pairs)  # list of (image_path, mask_path)
        self.t = transforms
        self.parse = mask_parser

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        image_path, mask_path = self.pairs[i]
        img = _imread_rgb(image_path)
        label = self.parse(_read_mask(mask_path))
        out = self.t(image=img, mask=label)
        return {"image": out["image"], "mask": out["mask"].long()}


def build_refuge_index(root, csv_out: str | None = None) -> pd.DataFrame:
    """BEST-EFFORT indexer — REFUGE2 ships in several folder layouts, so treat this as a
    starting point and adapt the path logic to YOUR download.

    Assumed layout (edit me):
        <root>/images/<name>.jpg
        <root>/masks/<name>.(bmp|png)
        <root>/glaucoma_labels.csv   columns: [image, label]
        <root>/fovea.csv             columns: [image, Fovea_X, Fovea_Y]

    Returns a dataframe with columns image_path, mask_path, label, fovea_x, fovea_y and
    optionally writes it to `csv_out`.
    """
    root = Path(root)
    img_dir, msk_dir = root / "images", root / "masks"
    labels, fovea = {}, {}

    lab_csv, fov_csv = root / "glaucoma_labels.csv", root / "fovea.csv"
    if lab_csv.exists():
        ldf = pd.read_csv(lab_csv)
        labels = dict(zip(ldf.iloc[:, 0], ldf.iloc[:, 1]))
    if fov_csv.exists():
        fdf = pd.read_csv(fov_csv)
        fovea = {r.iloc[0]: (r.iloc[1], r.iloc[2]) for _, r in fdf.iterrows()}

    rows = []
    for img_path in sorted(img_dir.glob("*")):
        if img_path.is_dir():
            continue
        mask_path = next(iter(msk_dir.glob(img_path.stem + ".*")), None)
        fx, fy = fovea.get(img_path.name, (np.nan, np.nan))
        rows.append(
            {
                "image_path": str(img_path),
                "mask_path": str(mask_path) if mask_path else "",
                "label": int(labels.get(img_path.name, 0)),
                "fovea_x": fx,
                "fovea_y": fy,
            }
        )
    df = pd.DataFrame(rows)
    if csv_out:
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_out, index=False)
    return df
