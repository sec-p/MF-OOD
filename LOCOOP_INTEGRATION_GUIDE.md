# LoCoOp OOD Regularization Integration Guide

## 概述

成功将 LoCoOp 的文本相似度筛选 OOD patch token 的方法集成到 MF-OOD 项目中。该方法通过识别和正则化 OOD patch tokens 来增强 OOD 检测性能。

## 核心原理

### LoCoOp 方法简介

LoCoOp (Local regularized Context Optimization) 通过以下机制进行 OOD 检测：

1. **局部特征提取**：使用 CLIP 的 Vision Transformer 提取图像的 patch tokens（局部特征）
2. **文本相似度计算**：计算每个 patch token 与所有类别文本描述的余弦相似度
3. **OOD patch 识别**：如果一个 patch 的 top-k 预测中不包含真实标签，则认为是 OOD patch
4. **正则化训练**：最大化 OOD patch 的预测熵，让模型学习将 OOD 特征与 ID 类别分离

### 集成到 MF-OOD

在 MF-OOD 项目中，我们将 LoCoOp 方法作为一个额外的正则化损失函数集成到现有的训练框架中。

## 代码修改

### 1. 新增损失函数 (`src/model_modular.py`)

添加了 `compute_locoop_ood_loss` 函数：

```python
@autocast(enabled=False)
def compute_locoop_ood_loss(local_feats: torch.Tensor,
                           text_feats: torch.Tensor,
                           labels: torch.Tensor,
                           top_k: int = 200,
                           logit_scale: float = 100.0) -> torch.Tensor:
    """
    LoCoOp-style OOD regularization loss.
    识别 OOD patches 并最大化它们的熵。
    """
```

**功能说明**：
- 计算所有 patch tokens 与所有类别文本的相似度
- 识别 top-k 预测中不包含真实标签的 OOD patches
- 计算这些 OOD patches 的熵并返回负熵作为损失

### 2. 模型集成 (`src/model_modular.py`)

在 `ModularCustomCLIP.forward` 中添加了 OOD 正则化损失：

```python
# F. LoCoOp-style OOD regularization
if self.cfg.get('use_locoop_ood', False) and labels is not None:
    scale_val = logit_scale.item()
    top_k = self.cfg.get('locoop_topk', 200)
    
    ood_loss = compute_locoop_ood_loss(
        local_features,  # 使用所有局部特征
        text_feats,
        labels,
        top_k=top_k,
        logit_scale=scale_val
    )
    aux_losses['locoop_ood'] = self.cfg.get('lambda_locoop_ood', 0.1) * ood_loss
```

### 3. 训练脚本配置 (`src/train_eval.py`)

添加了相关超参数：

**TrainEvalOrchestrator 初始化参数**：
```python
# LoCoOp OOD regularization parameters
use_locoop_ood: bool = False,          # 是否使用 LoCoOp OOD 正则化
lambda_locoop_ood: float = 0.1,       # LoCoOp 损失权重
locoop_topk: int = 200,                # Top-k 值
```

**命令行参数**：
```python
parser.add_argument('--use_locoop_ood', action='store_true',
                    help='Use LoCoOp-style OOD regularization loss')
parser.add_argument('--lambda_locoop_ood', type=float, default=0.1,
                    help='Weight for LoCoOp OOD regularization loss (default: 0.1)')
parser.add_argument('--locoop_topk', type=int, default=200,
                    help='Top-k value for LoCoOp OOD patch selection (default: 200)')
```

## 使用方法

### 基本使用

启用 LoCoOp OOD 正则化进行训练：

```bash
python src/train_eval.py \
    --method GL_MCM_FA \
    --selector_type slot \
    --fuser_type query_attn \
    --use_locoop_ood \
    --lambda_locoop_ood 0.1 \
    --locoop_topk 200 \
    --epochs 50 \
    --lr 0.005 \
    --batch_size 64 \
    --shots 16 \
    --seed 42
```

### 参数调优建议

1. **`--lambda_locoop_ood`** (默认: 0.1)
   - 控制 LoCoOp 损失的权重
   - 建议范围: 0.01 - 0.5
   - 值越大，对 OOD patches 的正则化越强

2. **`--locoop_topk`** (默认: 200)
   - 用于识别 OOD patches 的 top-k 值
   - 建议范围: 50 - 500
   - 值越大，识别的 OOD patches 越少（更严格）

### 与现有损失函数的组合

LoCoOp OOD 正则化可以与现有的损失函数组合使用：

```bash
python src/train_eval.py \
    --use_locoop_ood \
    --lambda_locoop_ood 0.1 \
    --lambda_mixup 0.1 \
    --lambda_intra_class 0.1 \
    --lambda_llm_negatives 0.05 \
    # ... 其他参数
```

## 技术细节

### 损失函数工作流程

1. **特征归一化**：将局部特征和文本特征进行 L2 归一化
2. **相似度计算**：使用 einsum 计算所有 patches 与所有类别的余弦相似度
3. **Top-K 筛选**：对每个 patch 找出 top-k 个最相似的类别
4. **OOD 识别**：检查真实标签是否在 top-k 中
5. **熵计算**：对识别出的 OOD patches 计算预测熵
6. **损失返回**：返回负熵（最大化熵 = 最小化负熵）

### 数值稳定性

- 使用 `@autocast(enabled=False)` 强制在 FP32 下计算
- 添加小的 epsilon (1e-8) 防止 log(0)
- 使用稳定的 softmax 实现

### 内存优化

- 使用 `torch.einsum` 进行高效的批量矩阵运算
- 避免不必要的张量复制
- 使用原地操作减少内存分配

## 预期效果

### 训练效果

1. **ID 分类准确率**：可能略有下降（因为增加了正则化约束）
2. **OOD 检测性能**：预期 AUROC 和 FPR95 指标提升
3. **训练稳定性**：损失函数在 FP32 下计算，数值稳定

### 与 MF-OOD 的协同

- **特征选择器** (Selector)：选择重要的 patches
- **特征融合器** (Fuser)：融合全局和局部特征
- **LoCoOp 正则化**：利用所有 patches 识别 OOD

三者协同工作：
1. Selector 选择重要 patches
2. Fuser 融合特征
3. LoCoOp 利用所有 patches（包括被筛选掉的）进行 OOD 正则化

## 实验建议

### 消融实验

1. **基线**：不使用 LoCoOp 正则化
2. **仅 LoCoOp**：只使用 LoCoOp 正则化
3. **组合**：LoCoOp + 其他损失函数

### 超参数搜索

建议进行网格搜索：

```bash
for lambda in 0.01 0.05 0.1 0.2 0.5; do
    for topk in 50 100 200 300; do
        python src/train_eval.py \
            --use_locoop_ood \
            --lambda_locoop_ood $lambda \
            --locoop_topk $topk \
            --seed $seed
    done
done
```

## 注意事项

1. **计算开销**：LoCoOp 损失需要计算所有 patches 的相似度，可能增加训练时间
2. **内存使用**：需要存储所有 patches 的 logits，注意 batch size 的设置
3. **类别数量**：对于类别数很多的任务，可能需要调整 top_k 值

## 故障排查

### 损失为 0

如果 LoCoOp 损失始终为 0，可能原因：
- `locoop_topk` 值太大，所有 patches 都被识别为 ID
- 模型已经过拟合，所有 patches 的预测都包含真实标签

**解决方案**：
- 减小 `locoop_topk` 值
- 增加 `lambda_locoop_ood` 权重

### 训练不稳定

如果训练出现 NaN 或不稳定：
- 检查是否正确使用了 FP32 计算
- 减小学习率
- 检查梯度裁剪设置

## 参考文献

- LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning (NeurIPS 2023)
- 论文链接: https://arxiv.org/abs/2306.01293

## 联系与支持

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
