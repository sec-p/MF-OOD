# Dual Adapter + LoCoOp OOD Regularization 使用指南

## 概述

实现了 Dual Adapter 架构，让 CLS token 和 patch token 分别通过独立的 adapter 进行处理，并分别计算对应的 OOD 正则化损失。

## 核心改进

### 1. DualAdapterFuser 类

新增了 `DualAdapterFuser` 类，可以同时处理 CLS token 和 patch token：

```python
class DualAdapterFuser(BaseFuser):
    """
    Dual Adapter that processes both CLS token and patch tokens separately.
    Returns adapted CLS and adapted patches for separate loss computation.
    """
```

**特点**：
- 独立的 CLS adapter 和 patch adapter
- 分别的 LayerNorm 层
- 零初始化输出层，保证初始阶段不破坏原特征

### 2. 分离的 OOD 损失函数

新增了三个损失函数：

#### `compute_locoop_cls_loss`
专门用于 CLS token 的 OOD 正则化损失：
```python
def compute_locoop_cls_loss(cls_feats, text_feats, labels, top_k=200, logit_scale=100.0)
```

#### `compute_locoop_ood_loss`
专门用于 patch tokens 的 OOD 正则化损失（原有）：
```python
def compute_locoop_ood_loss(local_feats, text_feats, labels, top_k=200, logit_scale=100.0)
```

#### `compute_locoop_dual_loss`
组合 CLS 和 patch 的 OOD 损失：
```python
def compute_locoop_dual_loss(cls_feats, patch_feats, text_feats, labels, 
                           top_k=200, logit_scale=100.0,
                           lambda_cls=0.5, lambda_patch=0.5)
```

### 3. 自动检测 Fuser 类型

在 `ModularCustomCLIP.forward` 中自动检测 fuser 类型：

```python
if isinstance(self.fuser, DualAdapterFuser):
    # 使用 dual adapter，分别计算 CLS 和 patch 损失
    adapted_cls, adapted_patches = self.fuser(selected_feats, global_feat=image_features)
    
    # 使用 dual loss
    ood_loss = compute_locoop_dual_loss(
        adapted_cls_for_loss,
        adapted_patches_for_loss,
        text_feats,
        labels,
        top_k=top_k,
        logit_scale=scale_val,
        lambda_cls=lambda_cls,
        lambda_patch=lambda_patch
    )
else:
    # 使用标准 fuser，只计算 patch 损失
    ood_loss = compute_locoop_ood_loss(
        adapted_patches_for_loss,
        text_feats,
        labels,
        top_k=top_k,
        logit_scale=scale_val
    )
```

## 使用方法

### 基本使用（Dual Adapter 模式）

```bash
python src/train_eval.py \
    --method GL_MCM_FA \
    --selector_type identity \
    --fuser_type dual_adapter \
    --use_locoop_ood \
    --lambda_locoop_ood 0.1 \
    --locoop_topk 200 \
    --lambda_locoop_cls 0.5 \
    --lambda_locoop_patch 0.5 \
    --epochs 50 \
    --lr 0.005 \
    --batch_size 64 \
    --shots 16 \
    --seed 42
```

### 参数说明

#### Fuser 类型

- `--fuser_type dual_adapter`: 使用 DualAdapterFuser（推荐）
- `--fuser_type simple_adapter`: 使用 SimpleAdapterFuser（原有）
- `--fuser_type query_attn`: 使用 QueryGuidedAttentionFuser
- `--fuser_type self_attn`: 使用 SelfAttentionFuser
- `--fuser_type cross_attn`: 使用 CrossAttentionFuser
- `--fuser_type mean`: 使用 MeanPoolFuser

#### LoCoOp 参数

- `--use_locoop_ood`: 启用 LoCoOp OOD 正则化
- `--lambda_locoop_ood`: 总的 LoCoOp 损失权重（默认 0.1）
- `--locoop_topk`: Top-k 值（默认 200）
- `--lambda_locoop_cls`: CLS token OOD 损失权重（默认 0.5，仅在 dual_adapter 模式下有效）
- `--lambda_locoop_patch`: Patch token OOD 损失权重（默认 0.5，仅在 dual_adapter 模式下有效）

### 不使用 Selector

由于你主要使用 adapter，可以将 selector 设置为 identity：

```bash
--selector_type identity
```

这样 selector 不会对 patches 进行筛选，所有 patches 都会传递给 fuser。

## 架构对比

### 原有架构

```
Image -> CLIP Encoder -> [CLS, Patches]
                          |
                          v
                    Selector (可选)
                          |
                          v
                    Fuser -> Final Features
                          |
                          v
                    Classification
```

### 新架构（Dual Adapter）

```
Image -> CLIP Encoder -> [CLS, Patches]
                          |
                          +-------------------+
                          |                   |
                          v                   v
                    Selector (可选)      Identity (CLS)
                          |                   |
                          v                   v
                    Patches               CLS
                          |                   |
                          +-------------------+
                          |
                          v
                    DualAdapterFuser
                          |
                          +-------------------+
                          |                   |
                          v                   v
                    Adapted CLS       Adapted Patches
                          |                   |
                          +-------------------+
                          |
                          v
                    Classification + OOD Loss (CLS + Patches)
```

## 损失计算流程

### Dual Adapter 模式

1. **CLS Token 处理**：
   - CLS token 通过 cls_adapter
   - 计算 CLS token 与所有类别的相似度
   - 识别 OOD samples（top-k 中不包含真实标签）
   - 计算 OOD samples 的熵

2. **Patch Tokens 处理**：
   - Patch tokens 通过 patch_adapter
   - 计算每个 patch 与所有类别的相似度
   - 识别 OOD patches（top-k 中不包含真实标签）
   - 计算 OOD patches 的熵

3. **损失组合**：
   ```python
   total_loss = lambda_cls * cls_loss + lambda_patch * patch_loss
   ```

### 标准 Fuser 模式

1. 只计算 patch tokens 的 OOD 损失
2. 不使用 CLS token 进行 OOD 正则化

## 参数调优建议

### 1. Adapter Ratio

```bash
--adapter_ratio 0.5  # 默认值
```

建议范围：0.25 - 1.0
- 较小的值（0.25）：参数少，计算快，但表达能力有限
- 较大的值（1.0）：参数多，表达能力强，但可能过拟合

### 2. LoCoOp 权重

```bash
--lambda_locoop_ood 0.1      # 总权重
--lambda_locoop_cls 0.5       # CLS 权重
--lambda_locoop_patch 0.5     # Patch 权重
```

建议：
- 如果 ID 分类准确率下降太多，减小 `--lambda_locoop_ood`
- 如果 OOD 检测性能不够，增大 `--lambda_locoop_ood`
- 可以调整 `--lambda_locoop_cls` 和 `--lambda_locoop_patch` 的比例

### 3. Top-K 值

```bash
--locoop_topk 200  # 默认值
```

建议范围：50 - 500
- 较小的值（50）：识别更多 OOD patches，正则化更强
- 较大的值（500）：识别更少的 OOD patches，正则化更弱

## 实验建议

### 消融实验

1. **Baseline**：
   ```bash
   --fuser_type simple_adapter --selector_type identity
   ```

2. **Dual Adapter only**：
   ```bash
   --fuser_type dual_adapter --selector_type identity
   ```

3. **Dual Adapter + LoCoOp**：
   ```bash
   --fuser_type dual_adapter --selector_type identity --use_locoop_ood
   ```

4. **Different CLS/Patch weights**：
   ```bash
   --lambda_locoop_cls 0.7 --lambda_locoop_patch 0.3
   --lambda_locoop_cls 0.3 --lambda_locoop_patch 0.7
   ```

### 与其他损失函数组合

```bash
python src/train_eval.py \
    --fuser_type dual_adapter \
    --selector_type identity \
    --use_locoop_ood \
    --lambda_locoop_ood 0.1 \
    --lambda_locoop_cls 0.5 \
    --lambda_locoop_patch 0.5 \
    --lambda_mixup 0.1 \
    --lambda_intra_class 0.1 \
    --lambda_llm_negatives 0.05 \
    # ... 其他参数
```

## 预期效果

### 训练效果

1. **ID 分类准确率**：
   - Dual Adapter 可能略微提升（因为 CLS 和 patches 分别优化）
   - LoCoOp 正则化可能略微降低（因为增加了约束）

2. **OOD 检测性能**：
   - CLS OOD 损失：提升全局特征的 OOD 检测能力
   - Patch OOD 损失：提升局部特征的 OOD 检测能力
   - 组合效果：预期 AUROC 和 FPR95 指标提升

### 计算开销

- **Dual Adapter**：比 Simple Adapter 略慢（需要处理 CLS 和 patches）
- **LoCoOp 损失**：需要计算所有 patches 的相似度，可能增加训练时间

## 注意事项

1. **内存使用**：
   - Dual Adapter 需要存储 CLS 和 patches 的适配后特征
   - LoCoOp 损失需要存储所有 patches 的 logits
   - 注意 batch size 的设置

2. **训练稳定性**：
   - 所有损失函数在 FP32 下计算，数值稳定
   - 使用梯度裁剪防止梯度爆炸

3. **超参数敏感性**：
   - `lambda_locoop_cls` 和 `lambda_locoop_patch` 的比例可能影响性能
   - 建议进行网格搜索找到最佳配置

## 故障排查

### 损失始终为 0

如果 LoCoOp 损失始终为 0：
- 检查 `--locoop_topk` 值是否太大
- 检查模型是否已经过拟合
- 减小 `--locoop_topk` 值

### 训练不稳定

如果训练出现 NaN：
- 检查学习率是否过大
- 检查梯度裁剪设置
- 减小 `--lambda_locoop_ood` 权重

### 内存不足

如果出现 OOM 错误：
- 减小 batch size
- 减小 `--num_select` 值
- 使用 `--selector_type identity` 减少中间特征存储

## 代码示例

### 完整训练命令

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_eval.py \
    --method GL_MCM_FA \
    --selector_type identity \
    --fuser_type dual_adapter \
    --use_locoop_ood \
    --lambda_locoop_ood 0.1 \
    --locoop_topk 200 \
    --lambda_locoop_cls 0.5 \
    --lambda_locoop_patch 0.5 \
    --adapter_ratio 0.5 \
    --epochs 50 \
    --lr 0.005 \
    --batch_size 64 \
    --shots 16 \
    --seed 42 \
    --backbone ViT-B/16
```

### 参数网格搜索

```bash
for lambda_cls in 0.3 0.5 0.7; do
    for lambda_patch in 0.3 0.5 0.7; do
        for topk in 100 200 300; do
            python src/train_eval.py \
                --fuser_type dual_adapter \
                --selector_type identity \
                --use_locoop_ood \
                --lambda_locoop_ood 0.1 \
                --lambda_locoop_cls $lambda_cls \
                --lambda_locoop_patch $lambda_patch \
                --locoop_topk $topk \
                --seed $seed
        done
    done
done
```

## 总结

Dual Adapter + LoCoOp OOD Regularization 提供了一个灵活的框架，可以：

1. 分别优化 CLS token 和 patch tokens
2. 分别计算 CLS 和 patches 的 OOD 损失
3. 通过超参数控制 CLS 和 patches 损失的权重
4. 与现有的损失函数无缝集成

这个架构特别适合你的需求：不使用 selector，让 CLS 和 patches 都通过 adapter 进行处理，并分别计算对应的损失。
