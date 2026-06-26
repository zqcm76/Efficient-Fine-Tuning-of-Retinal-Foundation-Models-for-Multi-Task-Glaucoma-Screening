"""
evaluate.py — aggregate the per-image evidence table into per-group metric
tables (group = dataset by default, or device for the stratified view).

The one subtlety is the classification operating point: AUC is threshold-free,
but sensitivity/specificity are reported at a threshold FIT ON THE IN-DOMAIN
set and then applied unchanged to every external set. That carry-over is what
turns silent calibration drift into a visible sens/spec gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import metrics as M


def fit_indomain_threshold(df: pd.DataFrame, in_domain: str = C.IN_DOMAIN_DATASET) -> float:
    """Youden-J threshold on the in-domain rows; default 0.5 if unavailable."""
    sub = df[(df["dataset"] == in_domain) & (df["has_cls"])]
    if sub.empty:
        return 0.5
    return M.youden_threshold(sub["cls_prob"].to_numpy(), sub["cls_label"].to_numpy())


def aggregate(df: pd.DataFrame, by: str = "dataset", *, cls_threshold: float = 0.5) -> pd.DataFrame:
    """One row per group with the full metric suite + sample counts.

    `by` is the grouping column ("dataset" or "device").
    `cls_threshold` is the carried-over in-domain operating point.
    """
    out: list[dict] = []
    for key, g in df.groupby(by, sort=False):
        probs = g["cls_prob"].to_numpy()
        labels = g["cls_label"].to_numpy()
        sens, spec = M.sens_spec_at_threshold(probs, labels, cls_threshold)

        loc_norm = g["loc_err_norm"].to_numpy()
        finite_loc = np.isfinite(loc_norm)
        succ = (loc_norm[finite_loc] <= C.LOC_SUCCESS_DISC_FRACTION)
        loc_success = float(succ.mean()) if finite_loc.any() else M.NAN

        row = {
            by: key,
            "n_images": int(len(g)),
            # classification
            "auc": M.classification_auc(probs, labels),
            "n_cls": int((np.isfinite(probs) & np.isin(labels, [0, 1])).sum()),
            "sens_at_thr": sens,
            "spec_at_thr": spec,
            "thr_used": cls_threshold,
            # disc/cup segmentation
            "disc_dice_mean": M.nanmean(g["disc_dice"]),
            "disc_dice_std": M.nanstd(g["disc_dice"]),
            "n_disc": int(np.isfinite(g["disc_dice"]).sum()),
            "cup_dice_mean": M.nanmean(g["cup_dice"]),
            "cup_dice_std": M.nanstd(g["cup_dice"]),
            "n_cup": int(np.isfinite(g["cup_dice"]).sum()),
            "disc_iou_mean": M.nanmean(g["disc_iou"]),
            "cup_iou_mean": M.nanmean(g["cup_iou"]),
            # CDR (clinical fusion target)
            "vcdr_mae": M.nanmean(g["vcdr_abs_err"]),
            "n_vcdr": int(np.isfinite(g["vcdr_abs_err"]).sum()),
            # fovea localization
            "loc_norm_mean": M.nanmean(g["loc_err_norm"]),
            "loc_norm_median": M.nanmedian(g["loc_err_norm"]),
            "loc_px_mean": M.nanmean(g["loc_err_px"]),
            "loc_success_rate": loc_success,
            "n_fovea": int(np.isfinite(g["loc_err_px"]).sum()),
        }
        out.append(row)
    res = pd.DataFrame(out).set_index(by)
    return res


def evaluate_per_dataset(df: pd.DataFrame, in_domain: str = C.IN_DOMAIN_DATASET) -> pd.DataFrame:
    thr = fit_indomain_threshold(df, in_domain)
    return aggregate(df, by="dataset", cls_threshold=thr)


def evaluate_per_device(df: pd.DataFrame, in_domain: str = C.IN_DOMAIN_DATASET) -> pd.DataFrame:
    thr = fit_indomain_threshold(df, in_domain)
    return aggregate(df, by="device", cls_threshold=thr)


# Columns worth showing in a compact headline table (kept short on purpose).
HEADLINE_COLS = [
    "n_images", "auc", "sens_at_thr", "spec_at_thr",
    "disc_dice_mean", "cup_dice_mean", "vcdr_mae",
    "loc_norm_mean", "loc_success_rate",
]


def headline(table: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in HEADLINE_COLS if c in table.columns]
    return table[cols].round(4)
