"""Fundus segmentation mask parsing.

!!! VERIFY THIS AGAINST YOUR OWN DOWNLOAD !!!
REFUGE ground-truth masks are single-channel images. The widely used encoding is:

    0   = optic cup
    128 = optic disc rim (the ring between cup boundary and disc boundary)
    255 = background

Some re-distributions invert this, or already ship {0, 1, 2}. Before training, open a
few of your masks and run `np.unique(mask)` to confirm. If the values differ, change the
three constants below — that is the only edit required.

Throughout the project the *internal* label convention is fixed:
    0 = background, 1 = disc rim, 2 = cup
so the full optic disc is (label >= 1) and the cup is (label == 2).
"""
from __future__ import annotations

import numpy as np

# raw mask pixel values (EDIT to match your data)
CUP_VALUE = 0
DISC_RIM_VALUE = 128
BG_VALUE = 255

# internal label map (do NOT change — the rest of the code depends on it)
LBL_BG, LBL_DISC, LBL_CUP = 0, 1, 2


def parse_refuge_mask(mask: np.ndarray) -> np.ndarray:
    """Raw REFUGE mask (HxW, values ~{0,128,255}) -> label map {0:bg, 1:disc, 2:cup}.

    Uses nearest-of-three assignment so mild JPEG / resize drift in the raw values does
    not corrupt the labels."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    m = mask.astype(np.int16)
    label = np.full(mask.shape, LBL_BG, dtype=np.uint8)
    label[np.abs(m - DISC_RIM_VALUE) < 64] = LBL_DISC
    label[np.abs(m - CUP_VALUE) < 64] = LBL_CUP
    return label


def disc_binary(label: np.ndarray) -> np.ndarray:
    """Full optic disc = rim + cup."""
    return (label >= LBL_DISC).astype(np.uint8)


def cup_binary(label: np.ndarray) -> np.ndarray:
    return (label == LBL_CUP).astype(np.uint8)
