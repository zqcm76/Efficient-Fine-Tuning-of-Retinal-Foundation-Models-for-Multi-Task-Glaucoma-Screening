# -*- coding: utf-8 -*-
"""
阶段二 · 核心模型 配置 (Phase 2 config)

集中管理 backbone / LoRA / 三个任务头 / 多任务平衡 / 训练 的所有超参。
全部用 dataclass，方便从 train.py 里覆盖，也方便 smoke_test 用 tiny_config()。

目标运行环境: Python 3.11.15 (本容器是 3.12，但代码避免任何 3.12-only 语法)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


# --------------------------------------------------------------------------- #
# Backbone (RETFound = ViT-L/16, MAE 自监督预训练)
# --------------------------------------------------------------------------- #
@dataclass
class BackboneCfg:
    # 输入分辨率。12GB 显存默认 384；想上 512 需打开 grad_checkpointing 且减小 batch。
    img_size: Tuple[int, int] = (384, 384)
    patch_size: int = 16

    # ViT-L 结构 (与 RETFound 权重对齐，勿改)
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0

    # 从哪些 block 取中间特征给分割/黄斑头 (0-based，ViT-L 共 24 层)
    out_indices: Tuple[int, ...] = (5, 11, 17, 23)

    # 池化方式: 'token' = 用 cls token; 'mean' = 全局平均 (RETFound 微调惯例)
    pool: str = "mean"

    # 显存关键开关: 冻结主干后梯度仍要反传到早期 LoRA 层，所以激活必须留 →
    # 用 gradient checkpointing 换显存。CPU smoke test 时关掉。
    grad_checkpointing: bool = True

    # 位置编码插值后是否让 pos_embed 变成可训练 (~0.6M 参数)。
    train_pos_embed: bool = True

    # RETFound 预训练权重路径 (.pth)。None = 随机初始化 (仅供 smoke test)。
    pretrained_ckpt: Optional[str] = None

    # 构建 timm 模型时的 dropout (主干基本不用)
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1


# --------------------------------------------------------------------------- #
# LoRA / Adapter (冻结主干、只训低秩矩阵)
# --------------------------------------------------------------------------- #
@dataclass
class LoRACfg:
    mode: str = "lora"            # 'lora' | 'adapter'
    r: int = 8                    # LoRA 秩
    alpha: int = 16               # LoRA 缩放 = alpha / r
    dropout: float = 0.05
    # 注入目标: 对 attention 里名为 qkv / proj 的 nn.Linear 注入。
    # (patch_embed.proj 是 Conv2d，不会被动到；mlp.fc1/fc2 不在目标内。)
    targets: Tuple[str, ...] = ("qkv", "proj")
    adapter_bottleneck: int = 64  # mode='adapter' 时的瓶颈维度


# --------------------------------------------------------------------------- #
# 分割头 (coarse-to-fine: 先粗定位 ROI 再精修视盘/视杯)
# --------------------------------------------------------------------------- #
@dataclass
class SegCfg:
    # 'coarse2fine' = 主方案：单次 backbone 前向，在特征图上 roi_align 裁剪后精修。
    # 'highres'     = 退路：给分割分支单独挂一个更高分辨率解码器 (不裁剪)。
    mode: str = "coarse2fine"

    decoder_ch: int = 256         # 解码器主通道
    num_classes: int = 2          # disc, cup (multi-label，两个 sigmoid，非 softmax)

    coarse_out: int = 96          # 粗分割输出边长
    fine_in: int = 32             # roi_align 在特征图上裁出的边长
    fine_out: int = 128           # 精修分割输出边长 (ROI 内坐标系)
    highres_out: int = 384        # 退路解码器输出边长

    roi_margin: float = 0.3       # 由粗视盘框外扩的比例
    coarse_loss_w: float = 0.4    # 粗分割 loss 权重 (相对精修)


# --------------------------------------------------------------------------- #
# 黄斑定位头 (heatmap 回归 + soft-argmax)
# --------------------------------------------------------------------------- #
@dataclass
class MaculaCfg:
    decoder_ch: int = 128
    heatmap_out: int = 96         # heatmap 边长
    softargmax_beta: float = 10.0 # soft-argmax 温度 (越大越接近 hard argmax)
    sigma: float = 0.03           # 高斯目标标准差 (相对归一化坐标)
    coord_weight: float = 1.0     # 坐标 L1 loss 权重
    heatmap_weight: float = 1.0   # heatmap MSE loss 权重


# --------------------------------------------------------------------------- #
# 分类头 (focal loss 抗类别不平衡)
# --------------------------------------------------------------------------- #
@dataclass
class ClsCfg:
    num_classes: int = 2
    dropout: float = 0.1
    focal_gamma: float = 2.0
    # focal_alpha: None=不加权; 或长度 num_classes 的 list/tuple，正类权重更高。
    focal_alpha: Optional[Tuple[float, ...]] = None


# --------------------------------------------------------------------------- #
# 多任务 loss 平衡 (本身作为一组消融)
# --------------------------------------------------------------------------- #
@dataclass
class BalanceCfg:
    # 'uncertainty' = 不确定性加权 (默认，全程支持 AMP+累积+ckpt)
    # 'gradnorm'    = GradNorm (需 accum_steps=1 且其双反传段用 fp32)
    # 'fixed'       = 固定权重
    method: str = "uncertainty"
    fixed_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 2.5e-2
    # 不确定性加权时, log_sigma2 会被钳在 [-log(max_weight), +inf)，
    # 防止某个早早饱和的任务(如 macula, 收敛权重~143x)无限抬高自身权重、
    # 用 0.5*s 把 total 拉成深负、并淹没其它任务。max_weight=0 时不钳(旧行为)。
    uw_max_weight: float = 20.0


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
@dataclass
class TrainCfg:
    epochs: int = 50
    batch_size: int = 2
    accum_steps: int = 8          # 有效 batch = batch_size * accum_steps
    lr: float = 2e-4
    wd: float = 0.05
    warmup_epochs: int = 3        # 之前是死配置: 现已由 build_scheduler 真正生效
    # 学习率调度: 'warmup_cosine' = 线性 warmup 后余弦退火到 min_lr; 'none' = 旧行为(常量 lr)
    scheduler: str = "warmup_cosine"
    min_lr: float = 1e-6          # 余弦退火地板
    # 类别不平衡采样: True 时按 cls_label 频率给训练集做 WeightedRandomSampler
    # (只对有 cls 标注的样本生效; 无标注样本按均匀采)
    balanced_sampler: bool = True
    amp: bool = True              # 混合精度 (CPU 上自动 no-op)
    amp_dtype: str = "fp16"       # 'fp16' | 'bf16'
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    out_dir: str = "./runs/phase2"
    log_interval: int = 1


# --------------------------------------------------------------------------- #
# 顶层配置
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    lora: LoRACfg = field(default_factory=LoRACfg)
    seg: SegCfg = field(default_factory=SegCfg)
    macula: MaculaCfg = field(default_factory=MaculaCfg)
    cls: ClsCfg = field(default_factory=ClsCfg)
    balance: BalanceCfg = field(default_factory=BalanceCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    # 任务名顺序 (平衡器、loss 字典都按这个 key)
    task_names: Tuple[str, ...] = ("cls", "seg", "macula")


# --------------------------------------------------------------------------- #
# tiny 配置: CPU 冒烟测试用，几秒钟跑完，结构与真模型同构但极小
# --------------------------------------------------------------------------- #
def tiny_config() -> Config:
    cfg = Config()
    # 极小 backbone
    cfg.backbone = BackboneCfg(
        img_size=(128, 128),
        patch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=3,
        out_indices=(1, 3),
        pool="mean",
        grad_checkpointing=False,   # CPU 上默认关，单独测 ckpt 路径时再开
        train_pos_embed=True,
        pretrained_ckpt=None,
        drop_path_rate=0.0,
    )
    cfg.lora = LoRACfg(mode="lora", r=4, alpha=8, dropout=0.0,
                       targets=("qkv", "proj"), adapter_bottleneck=16)
    cfg.seg = SegCfg(mode="coarse2fine", decoder_ch=32, num_classes=2,
                     coarse_out=32, fine_in=8, fine_out=32, highres_out=64,
                     roi_margin=0.3, coarse_loss_w=0.4)
    cfg.macula = MaculaCfg(decoder_ch=32, heatmap_out=32, softargmax_beta=10.0,
                           sigma=0.05, coord_weight=1.0, heatmap_weight=1.0)
    cfg.cls = ClsCfg(num_classes=2, dropout=0.0, focal_gamma=2.0, focal_alpha=None)
    cfg.balance = BalanceCfg(method="uncertainty")
    cfg.train = TrainCfg(epochs=1, batch_size=2, accum_steps=2, lr=1e-3,
                         warmup_epochs=0, amp=False, grad_clip=1.0,
                         num_workers=0, seed=0, out_dir="./runs/tiny")
    return cfg
