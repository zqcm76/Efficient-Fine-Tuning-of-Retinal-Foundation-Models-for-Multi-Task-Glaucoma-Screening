"""Preprocessing + augmentation.

Albumentations is used because it transforms the image, the segmentation mask, and (in
Phase 2) the fovea keypoint *jointly and consistently*. Masks are resized/warped with
nearest-neighbour interpolation automatically, so the integer label values are preserved.

Normalisation uses ImageNet statistics, which matches both the ImageNet-pretrained
ResNet34 baseline and the RETFound backbone introduced in Phase 2.
"""
from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _geometric_aug() -> A.Affine:
    # mild affine jitter; CONSTANT border so padded pixels are 0 (= background in masks)
    return A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.05, 0.05),
        rotate=(-15, 15),
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5,
    )


def build_cls_transforms(size: int, train: bool) -> A.Compose:
    if train:
        aug = [
            A.Resize(height=size, width=size),
            A.HorizontalFlip(p=0.5),
            _geometric_aug(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        ]
    else:
        aug = [A.Resize(height=size, width=size)]
    aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)


def build_seg_transforms(size: int, train: bool) -> A.Compose:
    if train:
        aug = [
            A.Resize(height=size, width=size),
            A.HorizontalFlip(p=0.5),
            _geometric_aug(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ]
    else:
        aug = [A.Resize(height=size, width=size)]
    aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)
