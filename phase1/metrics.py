"""Evaluation metrics.

This module is deliberately the *stable evaluation core* of the whole project: the same
functions are used for in-domain REFUGE2 and for the zero-shot cross-device domains
(RIGA / PALM / Drishti-GS / RIM-ONE / G1020) in later phases, so the numbers are directly
comparable across the domain-shift gradient.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from masks import cup_binary, disc_binary


# --------------------------------------------------------------------------- classification
def classification_metrics(probs, labels) -> dict:
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    out: dict = {}
    if len(np.unique(labels)) > 1:
        try:
            out["auc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            out["auc"] = float("nan")
    else:
        out["auc"] = float("nan")  # AUC is undefined when only one class is present
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    out["accuracy"] = (tp + tn) / max(len(labels), 1)
    out["sensitivity"] = tp / max(tp + fn, 1)
    out["specificity"] = tn / max(tn + fp, 1)
    return out


# --------------------------------------------------------------------------- segmentation
def dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def segmentation_metrics(pred_label: np.ndarray, gt_label: np.ndarray) -> dict:
    return {
        "dice_disc": dice_binary(disc_binary(pred_label), disc_binary(gt_label)),
        "dice_cup": dice_binary(cup_binary(pred_label), cup_binary(gt_label)),
    }


# --------------------------------------------------------------------------- CDR (clinical)
def _vh_diameter(binary: np.ndarray) -> tuple[float, float]:
    """Vertical / horizontal extent (bounding-box approximation of diameter)."""
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return 0.0, 0.0
    return float(ys.max() - ys.min() + 1), float(xs.max() - xs.min() + 1)


def compute_cdr(label: np.ndarray) -> dict:
    disc = disc_binary(label)
    cup = cup_binary(label)
    dv, dh = _vh_diameter(disc)
    cv, ch = _vh_diameter(cup)
    eps = 1e-6
    return {
        "vCDR": cv / (dv + eps),                       # vertical cup-to-disc ratio
        "hCDR": ch / (dh + eps),
        "areaCDR": float(cup.sum()) / (float(disc.sum()) + eps),
        "disc_vdiam": dv,                              # handy for fovea normalisation
    }


def cdr_error(pred_label: np.ndarray, gt_label: np.ndarray) -> dict:
    p, g = compute_cdr(pred_label), compute_cdr(gt_label)
    return {
        "vCDR_mae": abs(p["vCDR"] - g["vCDR"]),
        "hCDR_mae": abs(p["hCDR"] - g["hCDR"]),
        "areaCDR_mae": abs(p["areaCDR"] - g["areaCDR"]),
    }


# --------------------------------------------------------------------------- fovea
def fovea_distance(pred_xy, gt_xy, disc_diameter: float | None = None) -> dict:
    pred_xy = np.asarray(pred_xy, dtype=float)
    gt_xy = np.asarray(gt_xy, dtype=float)
    d = float(np.sqrt(((pred_xy - gt_xy) ** 2).sum()))
    out = {"fovea_px": d}
    if disc_diameter and disc_diameter > 0:
        out["fovea_norm"] = d / disc_diameter  # normalised by optic-disc diameter
    return out


# --------------------------------------------------------------------------- aggregation
def aggregate_segmentation(pred_labels, gt_labels) -> dict:
    """pred_labels / gt_labels: lists of HxW int label maps."""
    dd, dc, vmae = [], [], []
    for p, g in zip(pred_labels, gt_labels):
        m = segmentation_metrics(p, g)
        dd.append(m["dice_disc"])
        dc.append(m["dice_cup"])
        vmae.append(cdr_error(p, g)["vCDR_mae"])
    return {
        "dice_disc": float(np.mean(dd)) if dd else float("nan"),
        "dice_cup": float(np.mean(dc)) if dc else float("nan"),
        "mean_dice": float(np.mean(dd + dc)) if dd else float("nan"),
        "vCDR_mae": float(np.mean(vmae)) if vmae else float("nan"),
    }
