# 训练速度优化建议

## 当前优化状态

### 已有的优化

1. **DataLoader 配置** ([`utils/train_eval_util.py`](file:///home/yeliu/huhuhu3/MF-OOD/utils/train_eval_util.py#L39-L149))
   - `num_workers: 8` - 多进程数据加载
   - `pin_memory: True` - 使用 pin_memory 加速 CPU-GPU 传输
   - `persistent_workers: True` - 持久化 worker 进程
   - `prefetch_factor: 2` - 预取数据

2. **训练循环** ([`src/train_eval.py`](file:///home/yeliu/huhuhu3/MF-OOD/src/train_eval.py#L375-L453))
   - 使用了 AMP (Automatic Mixed Precision)
   - 使用了 GradScaler
   - 使用了梯度裁剪

3. **模型优化**
   - 去掉了 selector，减少了计算量
   - 使用了 shared adapter，参数量减少

## 可以进一步优化的地方

### 1. 增加 DataLoader 的 num_workers

**当前配置**：
```python
kwargs = {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True}
```

**建议**：
```python
# 根据你的 CPU 核心数调整
# 一般设置为 CPU 核心数的 2-4 倍
kwargs = {'num_workers': 16, 'pin_memory': True, 'persistent_workers': True}
# 或者
kwargs = {'num_workers': 32, 'pin_memory': True, 'persistent_workers': True}
```

**如何确定最佳 num_workers**：
```bash
# 查看 CPU 核心数
nproc

# 或者
lscpu | grep "^CPU(s):"

# 建议：num_workers = CPU 核心数 × 2
```

**修改位置**：
- [`utils/train_eval_util.py`](file:///home/yeliu/huhuhu3/MF-OOD/utils/train_eval_util.py#L39) - val_loader
- [`utils/train_eval_util.py`](file:///home/yeliu/huhuhu3/MF-OOD/utils/train_eval_util.py#L97) - OOD loader
- [`utils/train_eval_util.py`](file:///home/yeliu/huhuhu3/MF-OOD/utils/train_eval_util.py#L124) - train_loader

### 2. 增加 batch_size

**当前配置**：
```bash
DEFAULT_BATCH_SIZE=1024  # 在 common_params.sh 中
```

**建议**：
```bash
# 根据你的 GPU 显存调整
# 如果显存足够，可以增加到 2048 或 4096
DEFAULT_BATCH_SIZE=2048
```

**注意**：
- 需要确保 GPU 显存足够
- 可以先尝试 2048，如果 OOM 就减小
- 使用 `nvidia-smi` 监控显存使用

### 3. 使用 torch.compile (PyTorch 2.0+)

**当前**：没有使用 torch.compile

**建议**：
```python
# 在 train_eval.py 中添加
import torch

# 在模型加载后添加
self.model = torch.compile(self.model, mode='max-autotune')
```

**效果**：
- 可以提升 20-30% 的训练速度
- 需要较长的 warmup 时间（第一次运行会慢）

**修改位置**：
[`src/train_eval.py`](file:///home/yeliu/huhuhu3/MF-OOD/src/train_eval.py#L342-L343)

```python
# Build modular model
self.model = build_modular_model(cfg, self.classnames, clip_model, class_negatives=self.class_negatives)
self.model = self.model.to(self.device)

# 添加 torch.compile
if torch.__version__ >= '2.0':
    self.model = torch.compile(self.model, mode='max-autotune')
    self.logger.debug('  ✓ Model compiled with torch.compile')
```

### 4. 优化 LoCoOp 损失计算

**当前**：每次前向传播都计算所有 patches 的相似度

**建议**：
- 使用 `torch.einsum` 优化矩阵乘法
- 使用 `torch.topk` 的 `sorted=False` 参数（如果不需要排序）

**修改位置**：
[`src/model_modular.py`](file:///home/yeliu/huhuhu3/MF-OOD/src/model_modular.py#L760-L817)

```python
# 优化前
patch_logits = logit_scale * torch.einsum('bnd,cd->bnc', local_feats_norm, text_feats_norm)
pred_topk = torch.topk(patch_logits_flat, k=top_k, dim=1)[1]

# 优化后（如果不需要排序）
pred_topk = torch.topk(patch_logits_flat, k=top_k, dim=1, largest=True, sorted=False)[1]
```

### 5. 使用更高效的优化器

**当前**：
```python
self.optimizer = torch.optim.SGD(
    trainable_params,
    lr=self.lr,
    momentum=0.9,
    weight_decay=1e-5
)
```

**建议**：
```python
# 使用 AdamW（通常收敛更快）
self.optimizer = torch.optim.AdamW(
    trainable_params,
    lr=self.lr,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)
```

**或者使用 Adam**：
```python
self.optimizer = torch.optim.Adam(
    trainable_params,
    lr=self.lr,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)
```

**修改位置**：
[`src/train_eval.py`](file:///home/yeliu/huhuhu3/MF-OOD/src/train_eval.py#L351-L356)

### 6. 减少评估频率

**当前**：每个 epoch 都进行评估

**建议**：
```python
# 每 N 个 epoch 评估一次
EVAL_EVERY_N_EPOCHS = 5

# 在 train_with_eval 中修改
if (epoch + 1) % EVAL_EVERY_N_EPOCHS == 0:
    # 进行评估
```

**修改位置**：
[`src/train_eval.py`](file:///home/yeliu/huhuhu3/MF-OOD/src/train_eval.py#L614-L615)

### 7. 使用梯度累积

**当前**：每个 batch 都更新参数

**建议**：
```python
# 使用梯度累积，模拟更大的 batch size
ACCUMULATION_STEPS = 4

# 在 train_epoch 中修改
for batch_idx, (images, labels) in enumerate(pbar):
    with autocast():
        output_dict = self.model(images, labels=labels)
        loss = ...
    
    loss = loss / ACCUMULATION_STEPS  # 归一化
    self.scaler.scale(loss).backward()
    
    if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
```

**修改位置**：
[`src/train_eval.py`](file:///home/yeliu/huhuhu3/MF-OOD/src/train_eval.py#L385-L420)

### 8. 使用更快的数据增强库

**当前**：使用 torchvision 的 transforms

**建议**：
```python
# 使用 Kornia 或 Albumentations（更快的 GPU 加速）
import kornia.augmentation as K

# 或者
from albumentations import Compose, RandomResizedCrop, Normalize
```

**修改位置**：
[`utils/train_eval_util.py`](file:///home/yeliu/huhuhu3/MF-OOD/utils/train_eval_util.py#L115-L123)

### 9. 使用分布式训练（多 GPU）

**当前**：单 GPU 训练

**建议**：
```bash
# 使用 torchrun 或 torch.distributed
torchrun --nproc_per_node=4 src/train_eval.py \
    --method GL_MCM_FA \
    --fuser_type shared_adapter \
    --use_locoop_ood \
    ...
```

**修改**：
需要修改训练脚本以支持分布式训练

### 10. 优化日志输出

**当前**：每个 batch 都更新进度条

**建议**：
```python
# 减少进度条更新频率
UPDATE_FREQ = 10

if batch_idx % UPDATE_FREQ == 0:
    pbar.set_postfix(loss_info)
```

**修改位置**：
[`src/train_eval.py`](file:///home/yeliu/huhuhu3/MF-OOD/src/train_eval.py#L442)

## 优先级建议

### 高优先级（立即实施）

1. **增加 num_workers**：简单，效果明显
2. **增加 batch_size**：如果显存足够
3. **使用 torch.compile**：PyTorch 2.0+，效果显著

### 中优先级（测试后实施）

4. **使用 AdamW**：可能提升收敛速度
5. **减少评估频率**：节省评估时间
6. **使用梯度累积**：模拟更大的 batch size

### 低优先级（需要较大改动）

7. **使用更快的增强库**：需要修改数据加载
8. **使用分布式训练**：需要较大改动
9. **优化 LoCoOp 损失**：需要仔细测试

## 快速实施示例

### 1. 增加 num_workers

修改 [`utils/train_eval_util.py`](file:///home/yeliu/huhuhu3/MF-OOD/utils/train_eval_util.py#L39)：
```python
kwargs = {'num_workers': 16, 'pin_memory': True, 'persistent_workers': True}
```

### 2. 使用 torch.compile

修改 [`src/train_eval.py`](file:///home/yeliu/huhuhu3/MF-OOD/src/train_eval.py#L342-L343)：
```python
self.model = build_modular_model(cfg, self.classnames, clip_model, class_negatives=self.class_negatives)
self.model = self.model.to(self.device)

if torch.__version__ >= '2.0':
    self.model = torch.compile(self.model, mode='max-autotune')
    self.logger.debug('  ✓ Model compiled with torch.compile')
```

### 3. 使用 AdamW

修改 [`src/train_eval.py`](file:///home/yeliu/huhuhu3/MF-OOD/src/train_eval.py#L351-L356)：
```python
self.optimizer = torch.optim.AdamW(
    trainable_params,
    lr=self.lr,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)
```

## 预期效果

- **增加 num_workers (8→16)**：提升 10-20%
- **增加 batch_size (1024→2048)**：提升 10-15%（如果显存足够）
- **使用 torch.compile**：提升 20-30%
- **使用 AdamW**：可能提升 5-10% 的收敛速度

**总计**：预期提升 30-50% 的训练速度

## 注意事项

1. **torch.compile**：
   - 第一次运行会慢（需要编译）
   - 某些操作可能不支持
   - 需要测试是否兼容

2. **AdamW**：
   - 可能需要调整学习率
   - 收敛曲线可能与 SGD 不同

3. **num_workers**：
   - 不要设置过大（可能导致 CPU 过载）
   - 一般不超过 CPU 核心数的 4 倍

4. **batch_size**：
   - 需要确保 GPU 显存足够
   - 可能需要调整学习率

## 监控工具

使用以下工具监控训练速度：

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控 CPU 使用
htop

# 监控内存使用
free -h

# 监控磁盘 I/O
iostat -x 1
```

## 总结

最简单且效果最明显的优化：

1. **增加 num_workers**：8 → 16 或 32
2. **使用 torch.compile**：PyTorch 2.0+
3. **使用 AdamW**：替换 SGD

这三个优化可以快速实施，预期提升 30-50% 的训练速度。
