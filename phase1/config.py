"""Configuration objects for the project. Phase 1 only needs a subset of these,
but the structure is laid out so Phase 2 (RETFound + LoRA) can extend it cleanly."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    # CSV index of REFUGE2 with columns:
    #   image_path, mask_path, label, fovea_x, fovea_y, [split]
    refuge_index: str = "data/refuge_index.csv"
    image_size: int = 512        # segmentation working resolution (multiple of 16)
    cls_image_size: int = 224    # classification working resolution
    num_seg_classes: int = 3     # 0=background, 1=optic-disc rim, 2=optic cup
    val_split: float = 0.2       # used only when the index has no 'split' column
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    pretrained: bool = True       # ImageNet init for the ResNet34 classification baseline
    unet_base_channels: int = 32  # width of the from-scratch UNet


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_accum_steps: int = 1     # raise this on the 3060 if you need a larger effective batch
    amp: bool = True              # mixed precision — helps the 12 GB card
    pos_weight: Optional[float] = None  # BCE positive-class weight (glaucoma is the rare class)
    seg_ce_weight: float = 1.0
    seg_dice_weight: float = 1.0
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "outputs"
    experiment: str = "baseline"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            output_dir=raw.get("output_dir", "outputs"),
            experiment=raw.get("experiment", "baseline"),
        )

    def to_dict(self) -> dict:
        return asdict(self)
