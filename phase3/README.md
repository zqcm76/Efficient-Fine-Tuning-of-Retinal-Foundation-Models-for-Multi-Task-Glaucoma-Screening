# 🎯 Phase 3 完整文件清单

## 📋 文档 (读这些)

| 文件 | 用途 | 场景 |
|---|---|---|
| **QUICK_START.md** | 5分钟快速开始清单 | ⭐ 第一次读这个 |
| **PHASE3_SETUP_GUIDE.md** | 完整详细指南 | 需要解释参数/问题排查 |
| **NEXT_STEPS.md** | Phase 2 总结和 Phase 3 规划 | 了解整个项目进度 |

## 🐍 Python 脚本 (在你的项目目录使用)

### 数据准备
```
prepare_data.py              数据 CSV 分组 (生成 train/val/test split)
validate_paths.py            验证 CSV 路径是否存在 (Windows 本地)
```

### Stage 3 核心代码 (复制到项目目录)
```
phase3/
├── train_stage3_real.py     ⭐ 真实数据训练主入口
├── train_stage3.py          Stage 3 Trainer 类实现
├── stage2_adapter.py        Stage 2→3 适配器 (格式转换)
├── stage2_iface.py          定义两个阶段的接口
├── fusion_head.py           CDR + 图像特征融合分类头
├── cdr.py                   CDR 计算 (硬/软)
├── losses.py                Focal Loss 等损失函数
├── run_ablation.py          消融实验驱动 (可选, 用于 synthetic 测试)
```

### 使用流程
```bash
# 你的项目目录结构:
your_project/
├── train.py                 (Stage 2, 已有)
├── train_stage3_real.py     (复制到这里)
├── stage2_adapter.py        (复制到这里)
├── fusion_head.py           (复制到这里)
├── cdr.py                   (复制到这里)
├── stage2_iface.py          (复制到这里)
├── losses.py                (可选, 或从 phase3 目录导入)
├── train_stage3.py          (可选, 或从 phase3 目录导入)
├── prepare_data.py          (复制到这里)
├── validate_paths.py        (复制到这里)
├── data_all.csv             (你已有的)
├── RETFound_cfp_weights.pth (需要下载)
├── runs/
│   └── phase2/best.pth      (Stage 2 输出)
└── ... (其他阶段二代码)
```

---

## 🚀 快速开始（复制粘贴）

### 第 0 步：验证环境
```bash
# 检查数据完整性
python validate_paths.py --csv data_all.csv
```

### 第 1 步：准备数据
```bash
python prepare_data.py --csv data_all.csv
# 输出: data_clean.csv, train.csv, val.csv, test.csv, split_stats.txt
```

查看建议超参：
```bash
cat split_stats.txt
# 你会看到 --focal-alpha '0.1,0.9' 的建议
```

### 第 2 步：Stage 2 训练 (如果还没有 runs/phase2/best.pth)
```bash
python train.py --preset full \
    --retfound RETFound_cfp_weights.pth \
    --manifest data_clean.csv --split-col split \
    --focal-alpha '0.1,0.9' \
    --epochs 50 --out-dir ./runs/phase2
```

### 第 3 步：Stage 3 消融
```bash
python train_stage3_real.py \
    --stage2-ckpt ./runs/phase2/best.pth \
    --manifest data_clean.csv --split-col split \
    --focal-alpha '0.1,0.9' \
    --mode all --epochs 15 --out-dir ./runs/phase3
```

### 查看结果
```bash
cat runs/phase3/ablation_results.json
```

---

## 📊 你的数据概览

**data_all.csv 统计：**
```
总行数: 2000
Split 分布:
  - train: 1200 (pos=120, neg=1080, ratio=9:1)
  - val:   400  (pos=80,  neg=320,  ratio=4:1)
  - test:  400  (pos=80,  neg=320,  ratio=4:1)

列名: image_path, mask_path, label, fovea_x, fovea_y, split
Mask 格式: REFUGE2 灰度 BMP (无需参数调整 ✓)
Fovea: 绝对像素坐标 (自动检测 ✓)
路径: Windows 格式 (保留原样 ✓)
```

**完全兼容，零修改需求！**

---

## ⚙️ 参数说明

### 必须的
- `--manifest` : CSV 文件路径
- `--split-col` : CSV 中的 split 列名 (这里是 "split")
- `--focal-alpha` : 从 split_stats.txt 复制 (这里是 '0.1,0.9')

### Stage 3 特有
- `--mode` : 融合模式
  - `all` : 依次跑三种模式 (pure_e2e, cdr_two_stage, cdr_soft)
  - `pure_e2e` : 仅图像特征 (基线)
  - `cdr_two_stage` : CDR 不可导 (中等基线)
  - `cdr_soft` : CDR 可导 (最优, 推荐)
- `--epochs` : 建议 15 (足够融合头收敛)
- `--soft-cdr-scale` : CDR 梯度权重 (默认 0.1, 可调)

### 其他 (可选)
- `--batch-size 4` : 默认值，显存不足改成 2
- `--lr-head 1e-3` : 分类头学习率 (默认值很好)
- `--lr-seg 1e-5` : 分割头学习率，仅 cdr_soft 用 (很小，防止过度调整)
- `--no-amp` : 关闭混合精度 (如 GPU 不支持)

---

## 📈 预期结果

### Stage 2 输出 (best.pth)
在 **val** 上应该看到：
```
AUC = 0.85-0.88
dice_disc = 0.88-0.94
dice_cup = 0.78-0.85
macula_err = 0.01-0.02
```

### Stage 3 消融表
```
mode           auc    dice_disc  dice_cup  cdr_degenerate
pure_e2e       0.87   0.940      0.820     -
cdr_two_stage  0.87   0.940      0.820     0.0%
cdr_soft       0.88   0.941      0.825     0.0%    ← 最优
```

如果 `cdr_soft` 在 AUC 和 Dice 上都最高，说明融合策略有效 ✓

---

## 🆘 问题排查

### 问：找不到文件？
答：运行 `validate_paths.py` 检查路径。如果提示缺失，检查：
1. data_all.csv 的路径前缀是否正确 (E:\RETFound\...)
2. 数据是否完整下载

### 问：显存不足？
答：
```bash
--batch-size 2          # 降低批大小
--no-amp                # 关闭混合精度
--grad-accum 2          # (Stage 2 only) 梯度累积
```

### 问：为什么 AUC 没有 0.88 那么高？
答：可能原因：
- train/val split 数据分布不同
- 随机性变化 ±0.02 正常
- focal_alpha 需要微调

尝试调整 focal_alpha：
```bash
--focal-alpha '0.15,0.85'  # 减弱正类权重
--focal-alpha '0.05,0.95'  # 增强正类权重
```

### 问：Stage 3 的 cdr_soft 模式 dice 下降了？
答：正常现象。分割被微调来优化分类，dice 可能略低。
如果下降太多，提高分割 anchor：
```bash
--seg-loss-weight 2.0  # 默认 1.0
```

### 问：导入错误 (ModuleNotFoundError)?
答：确保所有 .py 文件都在同一目录。不要放在子文件夹里。

---

## 📚 进一步阅读

### 如果想自己改代码
- `train_stage3_real.py` : 主函数，可改超参逻辑
- `stage2_adapter.py` : 格式转换，需要对 Stage 2 有理解
- `cdr.py` : CDR 计算细节，可调 tau_col/tau_occ (软CDR温度)
- `fusion_head.py` : 分类头架构，可改 hidden size 等

### 论文参考
- CDR 软可导计算: [RETFound paper](https://arxiv.org/abs/2304.14417)
- Focal Loss: [Lin et al. 2017](https://arxiv.org/abs/1708.02002)
- ViT + LoRA: 见 Stage 2 文档

---

## 🎓 学习路线

**从零开始:**
1. 读 QUICK_START.md (5min)
2. 运行第 0-1 步 (prepare_data) (2min)
3. 查看 split_stats.txt (1min)
4. 读 PHASE3_SETUP_GUIDE.md 的参数解释章节 (10min)
5. 运行 Stage 2 (4-6小时，自动跑)
6. 运行 Stage 3 (1小时，自动跑)
7. 看结果

**如果遇到问题:**
- 先查 PHASE3_SETUP_GUIDE.md 的"常见问题"
- 用 validate_paths.py 验证数据
- 尝试 --dummy 模式快速测试 (2min)

---

## 📝 最后检查清单

- [ ] 已读 QUICK_START.md
- [ ] validate_paths.py 验证通过 (所有文件存在)
- [ ] prepare_data.py 生成了 split CSV
- [ ] split_stats.txt 显示正确的 focal_alpha
- [ ] Stage 2 best.pth 已准备好
- [ ] 所有 phase3/*.py 文件复制到项目目录
- [ ] 准备好 GPU (或打好心理准备等 CPU 跑😅)
- [ ] 随时可以开始 Stage 3！

---

**现在就可以开始 Phase 3 了！祝你顺利！** 🚀
