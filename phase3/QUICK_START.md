# Phase 3 快速开始（5分钟清单）

你现在手里有什么：
- ✓ data_all.csv (2000行, Windows路径, 已按split分组)
- ✓ 阶段二的 best.pth (或即将训练)
- ✓ Stage3 完整代码 (stage2_adapter.py, train_stage3_real.py, fusion_head.py, cdr.py 等)

---

## 快速检查清单

- [ ] **检查 data_all.csv 列名** — 应该有 `image_path, mask_path, label, fovea_x, fovea_y, split`
- [ ] **验证数据路径** — 在 Windows 资源管理器打开其中一个路径（如 `E:\RETFound\REFUGE2\Train\img\0001.jpg`）确认文件存在
- [ ] **准备 RETFound 权重** — 确保 `RETFound_cfp_weights.pth` 在合适位置
- [ ] **Python 环境** — 有 PyTorch (CUDA 11+), scikit-learn, pandas 等（见 requirements.txt）

---

## 最少工作流（三条命令）

### 1. 生成 train/val/test split
```bash
python prepare_data.py --csv data_all.csv
# 输出: data_clean.csv, train.csv, val.csv, test.csv, split_stats.txt
```

**你需要做的**: 打开 `split_stats.txt`，看到 focal_alpha 的建议值，复制它待用。

### 2. 训练 Stage 2
```bash
python train.py --preset full \
    --retfound RETFound_cfp_weights.pth \
    --manifest data_clean.csv --split-col split \
    --focal-alpha '0.1,0.9' \
    --epochs 50 --out-dir ./runs/phase2
```

**期望**: 
- 跑 50 个 epoch，每个 epoch ~3 分钟（RTX 3090）
- 最终在 val 上 AUC ≈ 0.87, dice ≈ 0.88
- 输出: `runs/phase2/best.pth` (300 MB)

### 3. 训练 Stage 3（三种融合模式）
```bash
python train_stage3_real.py \
    --stage2-ckpt ./runs/phase2/best.pth \
    --manifest data_clean.csv --split-col split \
    --focal-alpha '0.1,0.9' \
    --mode all --epochs 15
```

**期望**:
- 跑三种模式，每种 15 epoch，每个 epoch ~1 分钟
- 输出消融表：pure_e2e vs cdr_two_stage vs cdr_soft
- 最优模式应该是 cdr_soft
- 输出: `runs/phase3/ablation_results.json`

---

## 数据统计一览

```
Train:  1200 样本 (pos=120, neg=1080) → 9:1 不平衡
Val:    400 样本  (pos=80, neg=320)   → 4:1 不平衡  
Test:   400 样本  (pos=80, neg=320)   → 4:1 不平衡

掩膜格式: REFUGE2 标准灰度 BMP (无需任何参数调整)
Fovea:   绝对像素坐标 (会自动检测)
```

---

## 常见错误排查

### FileNotFoundError
**原因**: data_all.csv 中的路径不存在  
**解决**: 
1. 打开 data_all.csv，检查路径前缀（应该是 `E:\RETFound\...`）
2. 在 Windows 资源管理器打开其中一个路径验证
3. 如果路径完全不同，用文本编辑器做全局替换

### CUDA out of memory
**原因**: batch size 太大  
**解决**: 加 `--batch-size 2` 或 `--grad-accum 2`

### 导入错误
**原因**: 缺少依赖包  
**解决**: `pip install -r requirements.txt`

### Module not found (stage2_adapter 等)
**原因**: 新文件没有放在项目目录  
**解决**: 确保这些文件和 train_stage3_real.py 在同一目录
```
project/
├── train.py
├── train_stage3_real.py
├── stage2_adapter.py      ← 新
├── fusion_head.py         ← 新
├── cdr.py                 ← 新
├── ...
└── data_all.csv
```

---

## 如果想跳过 Stage 2（快速测试）

你可以用 synthetic 数据测试 Stage 3：
```bash
python train_stage3_real.py --dummy --mode all --epochs 3
```

只需 2 分钟，验证代码无误。然后再跑真实数据。

---

## 关键文件说明

| 文件 | 用途 |
|---|---|
| `prepare_data.py` | 生成 split CSV（这一步你现在需要）|
| `stage2_adapter.py` | 连接 Stage 2 和 Stage 3 |
| `train_stage3_real.py` | Stage 3 主训练脚本 |
| `stage2_iface.py` | 定义 Stage 2 输出接口 |
| `cdr.py` | CDR 计算（hard 和 soft） |
| `fusion_head.py` | CDR + 图像特征融合分类头 |

---

## 下一步

当 Stage 3 跑完时，检查：
```bash
cat runs/phase3/ablation_results.json
```

应该看到类似：
```json
[
  {"mode": "pure_e2e", "auc": 0.871, "dice_disc": 0.940, ...},
  {"mode": "cdr_two_stage", "auc": 0.872, "dice_disc": 0.940, ...},
  {"mode": "cdr_soft", "auc": 0.875, "dice_disc": 0.941, ...}  ← 最优
]
```

如果 cdr_soft 的 AUC 和 Dice 都是最高的，说明融合策略有效。

---

**预计总耗时**: 
- Stage 2: 4-6 小时
- Stage 3: 1 小时  
- **总计: 5-7 小时 (RTX 3090 级 GPU)**

**问题?** 看 `PHASE3_SETUP_GUIDE.md` 的详细解释和常见问题章节。
