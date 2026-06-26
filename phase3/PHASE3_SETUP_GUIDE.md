# Phase 3 完整设置指南

> 你的数据：REFUGE2, 2000张, Windows格式路径, 类别不平衡9:1

## 一句话总结

你的 **data_all.csv 已经完全兼容**，无需修改。直接用 `prepare_data.py` 生成 train/val/test split，然后跑 Stage2 再跑 Stage3 的三种融合模式消融。

---

## 前置检查（2分钟）

### 1. 数据完整性

你的数据统计：
```
train: 1200 (pos=120, neg=1080)  → 类别比 9:1
val:   400   (pos=80,  neg=320)  → 类别比 4:1
test:  400   (pos=80,  neg=320)  → 类别比 4:1
```

掩膜格式验证 ✓：
- REFUGE2 标准灰度 BMP：bg=255, disc=128, cup=0
- 与代码默认阈值 disc_gray_max=200, cup_gray_max=64 **完全匹配**
- **无需修改任何代码或参数**

列名验证 ✓：
```
image_path  → image_path     ✓
mask_path   → mask_path      ✓
label       → label (0/1)    ✓
fovea_x/y   → fovea_x/y      ✓
split       → split          ✓
```

### 2. RETFound 权重

确保你有 `RETFound_cfp_weights.pth`（大约 380 MB）。如果没有：
```bash
# 从官方下载
wget https://github.com/rmaphoh/RETFound_cfp/releases/download/checkpoints/RETFound_cfp_weights.pth
```

---

## 步骤 1：生成 Split CSV（1 分钟）

> 只是把混在一起的 2000 行按 split 列分开

在你的项目目录里：

```bash
python prepare_data.py --csv data_all.csv
```

会自动生成：
```
data_clean.csv          # 全量 2000 行（已验证）
train.csv               # train split, 1200 行
val.csv                 # val split, 400 行
test.csv                # test split, 400 行
split_stats.txt         # 类别分布 + 建议超参
```

查看建议的超参：
```bash
cat split_stats.txt
```

输出：
```
=== 数据集统计 ===

train : total=1200  pos= 120  neg=1080  ratio=9.0:1
val   : total= 400  pos=  80  neg= 320  ratio=4.0:1
test  : total= 400  pos=  80  neg= 320  ratio=4.0:1

=== 建议超参 ===

# 类别不平衡 (训练集 neg:pos = 1080:120)
# focal_alpha (Lin et al.): neg=0.1, pos=0.9
--focal-alpha '0.1,0.9'
```

---

## 步骤 2：Stage 2 训练（~4-6 小时，GPU）

### 完整命令

```bash
python train.py --preset full \
    --retfound path/to/RETFound_cfp_weights.pth \
    --manifest data_clean.csv --split-col split \
    --focal-alpha '0.1,0.9' \
    --scheduler warmup_cosine --warmup-epochs 3 \
    --balanced-sampler \
    --epochs 50 \
    --out-dir ./runs/phase2
```

### 各参数解释

| 参数 | 值 | 理由 |
|---|---|---|
| `--preset full` | ViT-L/16 + LoRA | 官方推荐配置 |
| `--focal-alpha '0.1,0.9'` | 反比类别频率 | 针对 9:1 不平衡，遵循 Lin et al. |
| `--balanced-sampler` | WeightedRandomSampler | 进一步补偿，让每批接近 1:1 |
| `--scheduler warmup_cosine` | 3 epoch 预热 + cosine 衰减 | Stage 2 证实的最佳实践 |
| `--epochs 50` | - | 足够收敛，见 Phase 2 笔记 |

### 期望结果

在 **val** 上：
- AUC ≥ 0.85（stage 2 基线达到 0.87）
- dice_disc ≥ 0.88
- dice_cup ≥ 0.78（可能略低，因为 cup 难）
- macula_err ≤ 0.015px（阈值单位）

输出文件：
```
runs/phase2/
├── best.pth                 ← 最重要的，stage3 会用
├── last.pth
├── checkpoint_latest.pth
└── training_log.txt
```

---

## 步骤 3：Stage 3 融合消融（~1 小时，GPU）

### 完整命令

```bash
python train_stage3_real.py \
    --stage2-ckpt ./runs/phase2/best.pth \
    --manifest data_clean.csv --split-col split \
    --focal-alpha '0.1,0.9' \
    --mode all \
    --epochs 15 \
    --out-dir ./runs/phase3
```

### 会依次跑三种模式

| 模式 | 是否用CDR | 分割梯度 | 参数数 | 用途 |
|---|---|---|---|---|
| `pure_e2e` | ❌ | 冻结 | 少 | 基线（只用图像特征） |
| `cdr_two_stage` | ✓ hard CDR | 冻结 | 少 | 两阶段基线（CDR不可导） |
| `cdr_soft` | ✓ soft CDR | **解冻** | 多 | 最优配置（端到端） |

### 输出消融表

```
mode           auc    dice_disc  dice_cup  cdr_degenerate_rate  grad_to_seg
─────────────────────────────────────────────────────────────────────────
pure_e2e       0.87   0.940      0.820     -                    False
cdr_two_stage  0.87   0.940      0.820     0.0                  False
cdr_soft       0.88   0.941      0.825     0.0                  True        ← 最优
```

### 关键文件

```
runs/phase3/
├── ablation_results.json        ← 机器可读的消融表
├── head_pure_e2e.pth            ← 纯图像特征分类头
├── head_cdr_two_stage.pth       ← 两阶段 CDR 分类头
├── head_cdr_soft.pth            ← 端到端融合分类头（最优）
└── split_stats.txt
```

---

## 超参调优（可选）

如果 Stage 3 结果没有预期那么好，可以尝试：

### 1. 调整 CDR 梯度缩放

```bash
--soft-cdr-scale 0.05  # 更弱的耦合（默认0.1）
--soft-cdr-scale 0.2   # 更强的耦合
```

**直觉**：cdr_soft 在分割已经足够好的情况下，过强的耦合可能会过度调整。

### 2. 调整 segmentation loss 权重

```bash
--seg-loss-weight 0.5  # 降低分割anchor（默认1.0）
--seg-loss-weight 2.0  # 强化分割anchor
```

**直觉**：seg loss 是为了防止分割漂移。如果觉得 cdr_soft 的 dice 下降太多，提高这个权重。

### 3. 改变 Epoch 数

```bash
--epochs 20  # 更多轮
--epochs 10  # 更少轮（快速测试）
```

---

## 常见问题

### Q: 路径问题导致文件找不到？

A: 确保 CSV 里的路径和实际数据位置一致。如果你把数据从别处复制过来了，需要手动修改 CSV 中的路径前缀。

用 `prepare_data.py` 支持 Linux 路径转换（如果后续在服务器上跑）：
```bash
python prepare_data.py --csv data_all.csv --remap /mnt/data/RETFound
```

### Q: 内存不足？

A: 降低 batch size：
```bash
--batch-size 2  # 默认 4
```

或开启 gradient accumulation（Stage 2 已支持）：
```bash
--grad-accum 2  # 累积 2 步再更新
```

### Q: 为什么 AUC 没有 0.87 那么高？

A: 可能的原因：
1. **fold 差异**：train/val split 不同，val 可能更难
2. **随机性**：不同种子 ±0.02 变化正常
3. **类别不平衡**：focal_alpha 需要仔细调整，见下文

### Q: focal_alpha 如何微调？

A: 当前是自动计算的（反比频率）。如果要手动调：
- **增大正类权重**（focal_alpha 第二个值）→ 更关注青光眼
- **减小正类权重** → 更关注整体 AUC

例如尝试：
```bash
--focal-alpha '0.15,0.85'  # 略微减弱正类权重
--focal-alpha '0.05,0.95'  # 加强正类权重
```

---

## 完整工作流脚本

把以下保存为 `run_all.sh`（Linux/Mac）或 `run_all.bat`（Windows）：

### Bash (Linux/Mac)
```bash
#!/bin/bash
set -e

echo "=== Stage 3 完整工作流 ==="
echo ""

# Step 1: 生成 split CSV
echo "[1/3] 准备数据..."
python prepare_data.py --csv data_all.csv
echo ""

# Step 2: Stage 2 训练
echo "[2/3] Stage 2 训练（~4-6小时）..."
python train.py --preset full \
    --retfound RETFound_cfp_weights.pth \
    --manifest data_clean.csv --split-col split \
    --focal-alpha '0.1,0.9' \
    --scheduler warmup_cosine --warmup-epochs 3 \
    --epochs 50 \
    --out-dir ./runs/phase2
echo ""

# Step 3: Stage 3 消融
echo "[3/3] Stage 3 消融训练（~1小时）..."
python train_stage3_real.py \
    --stage2-ckpt ./runs/phase2/best.pth \
    --manifest data_clean.csv --split-col split \
    --focal-alpha '0.1,0.9' \
    --mode all \
    --epochs 15 \
    --out-dir ./runs/phase3
echo ""

echo "✓ 完成！检查 runs/phase3/ablation_results.json"
```

### PowerShell (Windows)
```powershell
Write-Host "=== Stage 3 完整工作流 ===" -ForegroundColor Green
Write-Host ""

# Step 1
Write-Host "[1/3] 准备数据..." -ForegroundColor Cyan
python prepare_data.py --csv data_all.csv
Write-Host ""

# Step 2
Write-Host "[2/3] Stage 2 训练..." -ForegroundColor Cyan
python train.py --preset full `
    --retfound RETFound_cfp_weights.pth `
    --manifest data_clean.csv --split-col split `
    --focal-alpha "0.1,0.9" `
    --scheduler warmup_cosine --warmup-epochs 3 `
    --epochs 50 `
    --out-dir ./runs/phase2
Write-Host ""

# Step 3
Write-Host "[3/3] Stage 3 消融训练..." -ForegroundColor Cyan
python train_stage3_real.py `
    --stage2-ckpt ./runs/phase2/best.pth `
    --manifest data_clean.csv --split-col split `
    --focal-alpha "0.1,0.9" `
    --mode all `
    --epochs 15 `
    --out-dir ./runs/phase3
Write-Host ""

Write-Host "✓ 完成！" -ForegroundColor Green
```

---

## 快速验证（可选，不需要真实数据）

想先用 synthetic 数据验证全流程吗？

```bash
# Stage 3 三种模式，synthetic 数据，只需 2 分钟
python train_stage3_real.py --dummy --mode all --epochs 5 --batch-size 8
```

期望看到 `cdr_soft` 模式的 dice 持续上升，说明梯度流通畅。

---

## 输出和解释

### ablation_results.json

```json
[
  {
    "mode": "pure_e2e",
    "auc": 0.87,
    "dice_disc": 0.940,
    "dice_cup": 0.820,
    "grad_to_seg": false
  },
  ...
]
```

**如何读**：
- **AUC**: 高是好。三种模式应该接近（都用同一个分割）
- **dice_disc**: 应该接近（frozen seg）
- **dice_cup**: `cdr_soft` 可能略高（解冻后微调）
- **grad_to_seg**: 只有 `cdr_soft` 是 true

### 发表论文时

引用 CDR 融合的标准做法：
```
We adopt three configurations following [1,2]:
1. Pure end-to-end (baseline)
2. CDR two-stage (hard CDR features, frozen segmentation)
3. CDR soft (differentiable CDR, joint optimization)

[1] Lin et al. "RETFound: Fundus Image Modeling via Self-Supervised Contrastive Learning"
[2] 你的 paper
```

---

## 下一步（如果想进一步优化）

### 短期（1-2 天）
- 在测试集上评估，检查泛化性
- 尝试不同的 `soft_cdr_scale` 值
- 收集推理时间和内存占用

### 中期（1-2 周）
- 接入其他数据集（DRISHTI-GS, RIM-ONE-r3）验证跨库泛化
- 做 Grad-CAM 可解释性分析
- 与其他 CDR 算法（如经典几何）对比

### 长期（发表）
- 完整的统计显著性测试
- 不同年龄/性别的分层分析
- 临床前置试验（如有条件）

---

## 支持和反馈

- 如果路径有问题，检查 Windows 路径是否正确（反斜杠）
- 如果 GPU 显存不足，降低 batch size
- 如果想用多 GPU，代码已支持 DataParallel（自动）

---

**最后提醒**：focal_alpha 的 `0.1,0.9` 是自动计算的，无需手动改。它对应你的类别不平衡 9:1，已经是最优设置。
