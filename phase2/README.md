# 阶段二 · 核心模型 (Phase 2 — LoRA-RETFound 多任务)

视网膜眼底多任务模型: **青光眼分类 + 视盘/视杯分割 + 黄斑定位**。
骨干用 RETFound (ViT-L/16, MAE 自监督)，冻结主干、只训 LoRA/Adapter 低秩矩阵，
靠小 batch + 混合精度 + 梯度累积压进 RTX 3060 的 12GB。

> 这是 5 阶段路线里的**阶段二**。阶段一 (数据+ResNet34/UNet 基线) 的产物——统一预处理
> pipeline、REFUGE 口径评测脚本——是本阶段数据接入点 (见 `dataset.py` 的 TODO)。

---

## 1. 架构

```
                       输入眼底图 (默认 384x384)
                              │
                ┌─────────────▼──────────────┐
                │  RETFound ViT-L/16 (冻结)   │   ← LoRA/Adapter 注入到每个
                │  + LoRA/Adapter (可训练)    │     attention 的 qkv / proj
                │  + 可训练插值位置编码        │
                └─────────────┬──────────────┘
            多层级特征 levels (out_indices=5,11,17,23)
              + 全局 pooled 特征 (mean pool)
                              │
        ┌──────────────┬──────┴───────┬───────────────┐
        ▼              ▼              ▼               ▼
   分类头(Focal)   粗分割解码器     精修分割解码器     黄斑头(heatmap
   pooled→logits  levels→粗logits  ROI特征→精logits   +soft-argmax)
                       │   └── roi_align ───┘
                       └────────(coarse-to-fine)────────
```

三个头:
- **分类头**: LayerNorm + Dropout + Linear，**Focal Loss** 抗类别不平衡。
- **分割头 (coarse-to-fine)**: 先用多层级特征出**粗** disc/cup，再由粗视盘前景框在
  **特征图**上 `roi_align` 裁 ROI，**精修**解码器在 ROI 内坐标系出细粒度 disc/cup。
  disc/cup 用 **两个独立 sigmoid (multi-label)**，因为 cup ⊂ disc 是嵌套关系，不是互斥。
- **黄斑定位头**: heatmap 回归 + **soft-argmax** 出连续坐标，loss = heatmap MSE + 坐标 L1。

多任务 loss 平衡支持三种 (本身作为一组消融)：`uncertainty` (默认) / `gradnorm` / `fixed`。

---

## 2. 分割分辨率"退路"——已在本阶段锁定

路线里要求把"分割分辨率退路"在阶段二定下来。结论:

| 方案 | 状态 | 说明 |
|---|---|---|
| **特征空间 coarse-to-fine** | **主方案** (`seg.mode='coarse2fine'`) | 单次骨干前向，在融合特征图上 `roi_align` 裁 ROI 再精修。显存友好，视盘区域获得高"有效分辨率"。 |
| **独立高分辨率解码器** | **退路** (`seg.mode='highres'`) | 给分割分支单挂一个多次上采样的解码器，不裁剪、全图精修。结构更直，显存更吃紧。 |
| 图像重裁剪+重编码 | 仅文档 (未实现) | 把 ROI 在**原图**上裁出再过一次骨干。能恢复高频细节，但要两次前向，成本高，列为未来工作。 |

**诚实说明 (重要)**: 主方案的 `roi_align` 是在**已下采样的特征图**上裁剪，
它放大的是特征、不是像素——**无法恢复 patch 化时已经丢掉的高频细节**。
它的价值在于把解码器的容量与上采样预算集中到视盘/视杯所在的小区域，
从而在**不增加骨干分辨率**的前提下提升该区域的分割精细度；但天花板受限于
骨干 patch grid。若 Dice (尤其 cup 边界) 在主方案下到顶，再切 `highres`，
最终对照"图像重裁剪+重编码"作为加分项。这条权衡会进阶段四的评测叙事。

---

## 3. 怎么压进 12GB

- **冻结主干仍要反传到早期 LoRA 层**，所以中间激活不能省 → 默认开
  **gradient checkpointing** (`backbone.grad_checkpointing=True`) 用算力换显存。
- **默认分辨率 384 而非 512**：ViT 显存随 token 数平方增长，384 (24×24=576 token)
  是 12GB 上挂三头 + 精修解码器较稳的点。上 512 请同时减 `batch_size`、保持 checkpointing。
- **小 batch + 梯度累积**: 默认 `batch_size=2, accum_steps=8` → 有效 batch 16。
- **混合精度**: 默认 fp16 (`amp_dtype='fp16'`)；数值不稳可切 `bf16`。
- coarse-to-fine 让视盘区域以小代价获得高有效分辨率，避免为分割把整图分辨率拉满。

> AMP/GradScaler 在 CPU 上自动 no-op (`enabled=False`)，方便本地调试。

---

## 4. 位置编码插值

RETFound 在 224 上预训练 (14×14 grid)，要吃 384/512 必须把 `pos_embed` 的 patch 部分
**双三次插值**到新 grid，cls token 那一行单独保留 (`backbone.py: interpolate_pos_embed`)。
插值在两处发生：(a) 按 `img_size` 构建模型时；(b) `load_pretrained` 加载 RETFound 权重时
(把 ckpt 里 224 的 pos_embed 插到当前 grid)。插值后 `pos_embed` 设为**可训练**
(`train_pos_embed=True`, ~0.6M 参数)，让它适配新分辨率。

---

## 5. 运行

```bash
# 安装
pip install -r requirements.txt

# 0) CPU 冒烟测试 (几秒，验证冻结/梯度/GradNorm/位置编码插值/评测全链路)
python3 smoke_test.py

# 1) dummy 数据快速跑通 (无需任何外部文件)
python3 train.py --preset tiny --dummy

# 2) 真实训练 (需 RETFound 权重 + manifest csv)
python3 train.py --preset full \
    --retfound /path/RETFound_cfp_weights.pth \
    --train-manifest data/train.csv \
    --val-manifest   data/val.csv \
    --balancer uncertainty \
    --seg-mode coarse2fine

# 多任务平衡消融: 换 --balancer gradnorm (会自动把 accum_steps 设为 1)
# 分割退路消融:   换 --seg-mode highres
# 微调方式消融:   换 --lora-mode adapter
```

`manifest` (csv) 建议字段：`image_path, cls_label, disc_path, cup_path, macula_xy, dataset`
(任一标签列留空即视为该任务无监督，loss 自动按样本屏蔽——对应不同数据集标签不齐的现实；
`dataset` 列留作阶段四按相机/设备维度拆指标)。

---

## 6. 多任务平衡: GradNorm vs 不确定性加权

- **uncertainty (默认)**: 学一组 log σ²，全程兼容 AMP + 梯度累积 + checkpointing。
- **gradnorm**: 让各任务在共享层 (取最后一个 transformer block 的可训练参数) 的梯度范数对齐。
  **约束**: 需 `accum_steps=1` (train.py 会自动设置)；其权重更新走一次额外的 w 梯度计算。
  实现上利用 "Gᵢ = wᵢ·‖∇Lᵢ‖ 对标量 wᵢ 线性" 这一点，用**一阶**梯度算 ‖∇Lᵢ‖ 后把 Gᵢ
  显式写成 wᵢ·‖∇Lᵢ‖，从而**避开二阶导** (roi_align / fused-attention 都不支持 double-backward)，
  数学上与标准 GradNorm 等价。

两者作为一组消融对比 (路线要求)。

---

## 7. 文件

| 文件 | 作用 |
|---|---|
| `config.py` | 所有 dataclass 配置 + `tiny_config()` (CPU 冒烟用) |
| `lora.py` | LoRA / Adapter 实现 + `inject_tuning` 注入 + 冻结/计数工具 |
| `backbone.py` | RETFound ViT-L 封装、位置编码插值、多层级特征、预训练加载 |
| `heads.py` | 解码器积木 + 分类/粗分割/精修/高分辨率/黄斑头 + soft-argmax |
| `model.py` | `MultiTaskModel` 装配 + ROI 框/roi_align/paste-back |
| `losses.py` | Focal / Dice+BCE / heatmap+坐标 loss |
| `balancing.py` | UncertaintyWeighting / GradNorm |
| `dataset.py` | 样本 schema、dummy 数据集、collate、真实 manifest 数据集骨架 |
| `trainer.py` | 指标、loss 汇总 (任务屏蔽)、标准/GradNorm 两条训练路径、评测 |
| `train.py` | 命令行入口 |
| `smoke_test.py` | CPU 全链路自检 |

---

## 8. 已知简化 / 未来工作 (不挤进核心交付)

- 黄斑误差当前用归一化坐标欧氏距离；阶段四要改成**按视盘直径归一化** (REFUGE 口径)，
  见 `trainer.py: macula_error` 的 TODO。
- 真实数据增广 (图像/掩膜/坐标同步变换) 接阶段一 pipeline，`dataset.py` 已留接口。
- 加分项 (Grad-CAM、ONNX 导出、conformal prediction、可微 cup⊂disc 拓扑约束) 按未来工作处理。
