# Glaucoma multi-task screening — Phase 1 (data pipeline + baselines)

Efficient fine-tuning of a retinal foundation model for glaucoma multi-task screening,
with cross-device robustness validation. **This repository is Phase 1**: the data
pipeline, the evaluation core, and two *conventional* baselines (ImageNet-pretrained
ResNet34 for classification, a from-scratch UNet for optic disc/cup segmentation). These
baselines are the comparison anchor; Phase 2 swaps the backbone for **RETFound + LoRA** and
reports the same metrics against these numbers.

## Scope

| In Phase 1 | Deferred to later phases |
|---|---|
| Data loading / preprocessing / augmentation pipeline | RETFound (ViT-L) backbone + LoRA (Phase 2) |
| Stable evaluation core (AUC, Dice, CDR, fovea distance) | Multi-task head + loss balancing (Phase 2) |
| ResNet34 classification baseline | Segmentation→CDR clinical fusion (Phase 3) |
| UNet disc/cup segmentation baseline | Zero-shot cross-device evaluation (Phase 4) |
| Synthetic-data smoke test | Report / demo (Phase 5) |

Classification and segmentation are trained as **two separate conventional models** here
(not multi-task — that begins in Phase 2). The fovea coordinates are read by the dataset
and the fovea metric is implemented, but the Phase 1 baseline scripts do not train a fovea
head (it is a one-line extension: a UNet with a single heatmap channel decoded by the
`soft_argmax2d` already in `src/utils.py`).

## Project layout

```
glaucoma-screening/
├── configs/baseline.yaml      # all hyperparameters (tuned for a 12 GB 3060)
├── src/
│   ├── config.py              # dataclass config + YAML loader
│   ├── masks.py               # REFUGE mask -> {0:bg,1:disc,2:cup}  (VERIFY encoding!)
│   ├── transforms.py          # albumentations pipelines (image+mask, ImageNet norm)
│   ├── datasets.py            # REFUGE2Dataset (train) + FundusSegDataset (zero-shot eval)
│   ├── models.py              # ResNet34Classifier, UNet, BCE / Dice-CE losses
│   ├── metrics.py             # AUC, Dice, CDR, fovea distance  (the stable eval core)
│   ├── engine.py              # generic Trainer (AMP, grad-accum, ckpt) + evaluators
│   └── utils.py               # seed, device, meters, soft-argmax
└── scripts/
    ├── make_synthetic_data.py # fake fundus data so the pipeline runs with no download
    ├── smoke_test.py          # end-to-end check on synthetic data (run this first)
    ├── train_classifier.py    # ResNet34 baseline
    └── train_segmenter.py     # UNet baseline
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Quickstart — verify the pipeline before you have any data

```bash
python scripts/smoke_test.py
```
This generates synthetic fundus images, then runs both models, both losses, the full
train/eval loop, and the metric functions on CPU. It should end with
`ALL SMOKE TESTS PASSED`.

## Data preparation

### Dataset roles (important)
- **REFUGE2 — the ONLY training set.** Train + validation (model selection) + its own
  in-domain test all come from here. Note REFUGE already contains a built-in *device*
  shift: its training split is from a Zeiss Visucam 500, while validation/test are from a
  Canon CR-2 — so even "in-domain" evaluation crosses cameras.
- **RIGA / PALM / Drishti-GS / RIM-ONE / G1020 — held-out zero-shot TEST sets.** Never
  trained on, never tuned on. They enter only in Phase 4. PALM has **no optic-cup
  annotation**, so cup-Dice / CDR are simply not reported for PALM (disc-only).

### Build the REFUGE2 index
`train_classifier.py` / `train_segmenter.py` read a CSV with columns:
```
image_path, mask_path, label, fovea_x, fovea_y, [split]
```
`src/datasets.build_refuge_index()` is a **best-effort** helper for the standard Grand
Challenge layout — open it and adapt the paths to your actual download, then point
`configs/baseline.yaml -> data.refuge_index` at the resulting CSV. If a `split` column is
present it is used directly; otherwise a stratified split is made automatically.

### ⚠️ Verify the mask encoding
`src/masks.py` assumes the common REFUGE convention `0=cup, 128=disc-rim, 255=background`.
**Open a few of your masks and run `np.unique(mask)` before training.** If the values
differ, edit the three constants at the top of `masks.py` — that is the only change needed.

## Train the baselines

```bash
python scripts/train_classifier.py --config configs/baseline.yaml   # monitors val AUC
python scripts/train_segmenter.py  --config configs/baseline.yaml   # monitors mean Dice
```
Best checkpoints and logs are written to `outputs/`. Reported metrics: classification
AUC / accuracy / sensitivity / specificity; segmentation disc-Dice, cup-Dice, and vertical
CDR error. A likely (and honest) Phase 1 result is that the ResNet34 classifier does well
while the plain UNet's **cup** Dice lags — the cup boundary is the hardest structure, which
is exactly the gap Phase 2's foundation model is meant to close.

## 3060 / 12 GB notes
Mixed precision (`amp: true`) is on by default. If you raise `image_size` or hit OOM,
increase `train.grad_accum_steps` to keep a larger *effective* batch without more memory.
On Windows, set `data.num_workers: 0` if the dataloader misbehaves.

## What was verified vs. what you must run yourself
The real datasets and a GPU are not available in the authoring environment, so the code was
verified by (1) the synthetic end-to-end smoke test above and (2) numeric unit checks of the
mask-parsing and metric math (identical masks → Dice 1.0, CDR of a 30/60-radius cup/disc →
0.5, separable scores → AUC 1.0, etc.). Convergence numbers on REFUGE2 itself are for you to
produce once your index CSV and mask encoding are confirmed.

# 青光眼多任务筛查 — 第一阶段（数据管道 + 基线模型）

针对青光眼多任务筛查的视网膜基础模型高效微调，包含跨设备鲁棒性验证。**本仓库为第一阶段**：数据管道、评估核心，以及两个*传统*基线模型（基于 ImageNet 预训练的 ResNet34 分类模型，以及从零训练的 UNet 视盘/视杯分割模型）。这些基线是比较基准；第二阶段将主干网络替换为 **RETFound + LoRA**，并对照这些数据汇报相同的指标。

## 范围

| 第一阶段包含 | 推迟到后续阶段 |
|---|---|
| 数据加载 / 预处理 / 增强管道 | RETFound（ViT-L）主干网络 + LoRA（第二阶段） |
| 稳定的评估核心（AUC、Dice、CDR、黄斑中心凹距离） | 多任务头 + 损失平衡（第二阶段） |
| ResNet34 分类基线 | 分割→CDR 临床融合（第三阶段） |
| UNet 视盘/视杯分割基线 | 零样本跨设备评估（第四阶段） |
| 合成数据冒烟测试 | 报告 / 演示（第五阶段） |

在本阶段中，分类与分割被训练为**两个独立的传统模型**（并非多任务——多任务从第二阶段才开始）。数据集会读取黄斑中心凹坐标，且相应的评估指标已实现，但第一阶段的基线脚本并不训练黄斑中心凹检测头（这只是一个一行代码量级的扩展：用单热力图通道的 UNet，配合 `src/utils.py` 中已有的 `soft_argmax2d` 进行解码即可）。

## 项目结构

```
glaucoma-screening/
├── configs/baseline.yaml      # 全部超参数（针对 12 GB 显存的 3060 调优）
├── src/
│   ├── config.py              # dataclass 配置 + YAML 加载器
│   ├── masks.py               # REFUGE 掩膜 -> {0:背景,1:视盘,2:视杯}（务必核实编码！）
│   ├── transforms.py          # albumentations 流水线（图像+掩膜，ImageNet 归一化）
│   ├── datasets.py            # REFUGE2Dataset（训练）+ FundusSegDataset（零样本评估）
│   ├── models.py               # ResNet34Classifier、UNet、BCE / Dice-CE 损失
│   ├── metrics.py              # AUC、Dice、CDR、黄斑中心凹距离（稳定的评估核心）
│   ├── engine.py                # 通用 Trainer（AMP、梯度累积、checkpoint）+ 评估器
│   └── utils.py                  # 随机种子、设备、计量器、soft-argmax
└── scripts/
    ├── make_synthetic_data.py # 生成假眼底数据，使管道无需下载即可运行
    ├── smoke_test.py          # 在合成数据上进行端到端检查（请先运行此项）
    ├── train_classifier.py    # ResNet34 基线
    └── train_segmenter.py     # UNet 基线
```

## 安装

```bash
python -m venv .venv && source .venv/bin/activate   # （Windows 下：.venv\Scripts\activate）
pip install -r requirements.txt
```

## 快速开始 —— 在拿到数据之前先验证管道

```bash
python scripts/smoke_test.py
```
该脚本会生成合成眼底图像，然后在 CPU 上运行两个模型、两种损失函数、完整的训练/评估循环以及各项评估指标函数。最终应输出
`ALL SMOKE TESTS PASSED`。

## 数据准备

### 数据集角色划分（重要）
- **REFUGE2 —— 唯一的训练集。** 训练集 + 验证集（用于模型选择）+ 其自带的域内测试集均来自该数据集。需要注意的是，REFUGE 本身就内置了一种*设备*偏移：其训练划分来自 Zeiss Visucam 500 相机，而验证/测试划分则来自 Canon CR-2 相机——因此即便是"域内"评估，也跨越了不同相机设备。
- **RIGA / PALM / Drishti-GS / RIM-ONE / G1020 —— 留存的零样本测试集。** 这些数据从不参与训练，也不参与调参。它们仅在第四阶段才会被使用。PALM **没有视杯标注**，因此对 PALM 不汇报 cup-Dice / CDR 指标（仅评估视盘）。

### 构建 REFUGE2 索引
`train_classifier.py` / `train_segmenter.py` 读取的 CSV 文件需包含以下列：
```
image_path, mask_path, label, fovea_x, fovea_y, [split]
```
`src/datasets.build_refuge_index()` 是一个针对标准 Grand Challenge 数据布局的**尽力而为（best-effort）**辅助函数——请打开该函数，根据你实际下载的数据调整路径，然后将
`configs/baseline.yaml -> data.refuge_index` 指向生成的 CSV 文件。如果存在 `split` 列，将直接使用该列；否则会自动进行分层划分。

### ⚠️ 务必核实掩膜编码
`src/masks.py` 默认采用常见的 REFUGE 约定：`0=视杯, 128=视盘边缘, 255=背景`。
**请在训练前打开几张你的掩膜，运行 `np.unique(mask)` 进行核实。** 如果数值不同，只需修改 `masks.py` 顶部的三个常量即可——这是唯一需要改动的地方。

## 训练基线模型

```bash
python scripts/train_classifier.py --config configs/baseline.yaml   # 监控验证集 AUC
python scripts/train_segmenter.py  --config configs/baseline.yaml   # 监控平均 Dice
```
最佳 checkpoint 与日志会写入 `outputs/` 目录。汇报的指标包括：分类任务的 AUC / 准确率 / 敏感度 / 特异度；分割任务的视盘 Dice、视杯 Dice，以及垂直 CDR 误差。一个可能出现（且符合预期）的第一阶段结果是：ResNet34 分类器表现良好，而普通 UNet 的**视杯** Dice 相对落后——视杯边界是最难分割的结构，这恰恰是第二阶段引入基础模型所要弥补的差距。

## 3060 / 12GB 显存使用说明
默认开启混合精度（`amp: true`）。如果你提高了 `image_size` 或遇到显存溢出（OOM），可增大 `train.grad_accum_steps`，从而在不增加显存占用的情况下保持更大的*有效*批次大小。在 Windows 系统下，如果数据加载器表现异常，请将 `data.num_workers` 设为 `0`。

## 已验证内容与需要你自行运行的内容
由于在编写本项目的环境中无法获取真实数据集与 GPU，代码的验证方式为：(1) 上述基于合成数据的端到端冒烟测试；(2) 针对掩膜解析与评估指标计算逻辑的数值单元测试（例如：相同掩膜 → Dice 为 1.0；半径 30/60 的视杯/视盘 → CDR 为 0.5；可分离的评分 → AUC 为 1.0，等等）。而在 REFUGE2 真实数据上的收敛结果，则需要你在确认好索引 CSV 与掩膜编码之后自行产出。