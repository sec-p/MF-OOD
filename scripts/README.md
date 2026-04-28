# Shared Adapter + LoCoOp 训练脚本使用指南

## 脚本列表

### 1. `train_shared_adapter_locoop.sh`
**用途**：单次训练，快速测试 Shared Adapter + LoCoOp

**使用场景**：
- 快速验证 Shared Adapter + LoCoOp 是否工作
- 测试特定参数组合
- 调试和开发

**运行方式**：
```bash
bash scripts/train_shared_adapter_locoop.sh
```

**参数配置**：
在脚本中直接修改以下参数：
```bash
# LoCoOp parameters
USE_LOCOOP_OOD=true
LAMBDA_LOCOOP_OOD=0.1
LOCOOP_TOPK=200
LAMBDA_LOCOOP_CLS=0.5
LAMBDA_LOCOOP_PATCH=0.5
```

### 2. `search_shared_adapter_locoop.sh`
**用途**：完整的超参数网格搜索

**使用场景**：
- 系统性地搜索最佳参数组合
- 找到最优的 LoCoOp 配置
- 论文实验

**运行方式**：
```bash
bash scripts/search_shared_adapter_locoop.sh
```

**搜索空间**：
- 学习率：`${LEARNING_RATES[@]}`
- Batch size：`${BATCH_SIZES[@]}`
- 随机种子：`${SEEDS[@]}`
- LoCoOp 权重：`${LAMBDA_LOCOOP_OOD[@]}`
- Top-K 值：`${LOCOOP_TOPK[@]}`
- CLS 权重：`${LAMBDA_LOCOOP_CLS[@]}`
- Patch 权重：`${LAMBDA_LOCOOP_PATCH[@]}`
- Adapter ratio：`${ADAPTER_RATIO[@]}`

**结果保存**：
- 日志目录：`$PROJECT_ROOT/logs/search_GL_MCM_FA_identity_shared_adapter_locoop/`
- 日志文件：`grid_search.log`

### 3. `search_locoop_params.sh`
**用途**：专注于 LoCoOp 参数的搜索

**使用场景**：
- 快速找到最佳的 LoCoOp 参数配置
- 对比不同 CLS/Patch 权重比例
- 对比不同 Top-K 值

**运行方式**：
```bash
bash scripts/search_locoop_params.sh
```

**搜索空间**：
- Top-K：`100 200 300`
- CLS 权重：`0.3 0.5 0.7`
- Patch 权重：`0.3 0.5 0.7`
- 随机种子：`${SEEDS[@]}`

**固定参数**：
- 学习率：`0.005`
- Batch size：`128`
- Epochs：`30`
- Adapter ratio：`0.5`

**结果保存**：
- 日志目录：`$PROJECT_ROOT/logs/locoop_search_GL_MCM_FA_shared_adapter/`
- 日志文件：`search_topk_*.log`

## 参数说明

### 通用参数（在 `common_params.sh` 中配置）

```bash
# 默认训练参数
DEFAULT_EPOCHS=30          # 训练轮数
DEFAULT_BATCH_SIZE=1024      # 批次大小
DEFAULT_LR=0.001             # 学习率
DEFAULT_SEED=1               # 随机种子
DEFAULT_BACKBONE="ViT-B/16"  # 骨干网络
DEFAULT_ROOT_PATH="/amax/yeliu/data"  # 数据集路径
DEFAULT_SHOTS=16            # Few-shot 样本数
DEFAULT_CLASS_NEGATIVES_PATH="/root/huhuhu2/MF-OOD/negatives_non_photographic.json"  # 负样本路径
```

### LoCoOp 参数

```bash
# LoCoOp OOD regularization parameters
LAMBDA_LOCOOP_OOD=(0.1)     # 总的 LoCoOp 损失权重
LOCOOP_TOPK=(200)               # Top-K 值
LAMBDA_LOCOOP_CLS=(0.5)        # CLS token OOD 损失权重
LAMBDA_LOCOOP_PATCH=(0.5)      # Patch token OOD 损失权重
```

### 其他损失函数参数

```bash
# Loss function coefficients grids
LAMBDA_LLM_NEGATIVES=(5)       # LLM 负样本损失权重
LAMBDA_MIXUP=(0)               # Mixup 损失权重
MARGIN_VALUES=(0.05)            # Margin 值
LAMBDA_INTRA_CLASS=(0)        # 类内一致性损失权重
INTRA_CLASS_TEMP=(0)           # 类内一致性温度
```

### Selector 参数

```bash
# Selector parameters grid
NUM_SELECT_VALUES=(64)         # 选择的 token 数量
SELECTOR_TEMPERATURE=(1.0)     # Selector 温度
PATCHES_PER_SLOT_ATTN=(4)      # 每个 slot 的 patch 数量
```

### 维度参数

```bash
# Dimension parameters
MLP_HIDDEN_RATIO=(1)           # MLP selector 隐藏层比例
SLOT_FFN_RATIO=(4.0)          # Slot selector FFN 比例
FUSER_FFN_RATIO=(4)            # Fuser FFN 比例
ADAPTER_RATIO=(0.125)          # Adapter 比例（重要！）
```

### OOD 评分参数

```bash
# OOD score parameters
SCORE_TYPE="GL-MCM"            # OOD 评分方法
TEMPERATURE=1.0                # 温度参数
LAMBDA_LOCAL=0.5              # 局部特征权重
```

## 使用示例

### 示例 1：快速测试

```bash
# 修改 train_shared_adapter_locoop.sh 中的参数
# 然后运行
bash scripts/train_shared_adapter_locoop.sh
```

### 示例 2：完整网格搜索

```bash
# 运行完整的超参数搜索
bash scripts/search_shared_adapter_locoop.sh

# 监控进度
tail -f logs/search_GL_MCM_FA_identity_shared_adapter_locoop/grid_search.log
```

### 示例 3：专注于 LoCoOp 参数搜索

```bash
# 快速找到最佳 LoCoOp 参数
bash scripts/search_locoop_params.sh

# 查看结果
ls -lh logs/locoop_search_GL_MCM_FA_shared_adapter/
```

### 示例 4：自定义参数运行

```bash
# 直接使用 train_eval.py，自定义所有参数
python3 src/train_eval.py \
    --method GL_MCM_FA \
    --selector_type identity \
    --fuser_type shared_adapter \
    --use_locoop_ood \
    --lambda_locoop_ood 0.1 \
    --locoop_topk 200 \
    --lambda_locoop_cls 0.5 \
    --lambda_locoop_patch 0.5 \
    --adapter_ratio 0.5 \
    --epochs 30 \
    --lr 0.005 \
    --batch_size 128 \
    --seed 1 \
    --backbone ViT-B/16 \
    --root_path /amax/yeliu/data \
    --shots 16 \
    --score_type GL-MCM \
    --temperature 1.0 \
    --lambda_local 0.5
```

## 结果分析

### 日志文件格式

训练日志包含：
- 训练进度（每个 epoch）
- ID 分类准确率
- OOD 检测指标（AUROC, FPR95）
- 各种损失值（CE, LoCoOp, Mixup 等）

### 提取最佳结果

```bash
# 查找所有训练中的最佳 AUROC
grep "Avg OOD AUROC" logs/search_GL_MCM_FA_identity_shared_adapter_locoop/grid_search.log | sort -t -k3 -n3

# 查找最佳 FPR95
grep "Avg OOD FPR95" logs/search_GL_MCM_FA_identity_shared_adapter_locoop/grid_search.log | sort -n -k3 -n3
```

### 可视化结果

```bash
# 提取结果到 CSV
grep "Avg OOD" logs/search_GL_MCM_FA_identity_shared_adapter_locoop/grid_search.log | \
    awk '{print $2","$4","$6","$8","$10}' > results.csv

# 使用 Python 可视化
python3 << EOF
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv', 
                  names=['AUROC', 'FPR95', 'Config', 'Time', 'Status'])

# 绘制 AUROC 分布
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(df['AUROC'], bins=20)
plt.title('AUROC Distribution')
plt.xlabel('AUROC')
plt.ylabel('Count')

# 绘制 FPR95 分布
plt.subplot(1, 2, 2)
plt.hist(df['FPR95'], bins=20)
plt.title('FPR95 Distribution')
plt.xlabel('FPR95')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('results_distribution.png')
plt.show()
EOF
```

## 常见问题

### Q1: 如何调整搜索范围？

修改 `common_params.sh` 中的数组：
```bash
# 例如：扩大学习率搜索范围
LEARNING_RATES=(0.001 0.005 0.01)

# 例如：扩大 Top-K 搜索范围
LOCOOP_TOPK=(50 100 200 300 500)
```

### Q2: 如何减少搜索时间？

1. **减少搜索维度**：
```bash
# 只搜索关键参数
LAMBDA_LOCOOP_OOD=(0.1)  # 固定为单一值
LOCOOP_TOPK=(200)           # 固定为单一值
```

2. **减少组合数**：
```bash
# 减少每个参数的选项
SEEDS=(1)  # 只使用一个种子
```

3. **使用 focused search**：
```bash
# 使用 search_locoop_params.sh 而不是完整的 search_shared_adapter_locoop.sh
```

### Q3: 如何并行运行多个实验？

使用 GNU parallel：
```bash
# 安装 parallel
sudo apt-get install parallel

# 并行运行多个种子
parallel -j 4 bash scripts/train_shared_adapter_locoop.sh ::: 1 2 3 4
```

### Q4: 如何监控训练进度？

```bash
# 实时查看日志
tail -f logs/GL_MCM_FA_identity_shared_adapter_locoop/train.log

# 查看最近的错误
grep "Failed" logs/search_GL_MCM_FA_identity_shared_adapter_locoop/grid_search.log | tail -20
```

### Q5: 如何恢复训练？

```bash
# 训练脚本会自动保存 checkpoint
# checkpoint 位置：logs/*/checkpoints/epoch_*.pt

# 使用 checkpoint 继续训练（需要修改 train_eval.py 支持 resume）
# 或者直接使用最佳 checkpoint 进行评估
```

## 最佳实践

### 1. 参数搜索策略

**阶段 1：粗搜索**
- 使用较大的参数范围
- 较少的 epoch（10-20）
- 较少的种子（1-2）

**阶段 2：精细搜索**
- 在最佳参数附近进行精细搜索
- 完整的 epoch（30-50）
- 多个种子（3-5）

**阶段 3：最终验证**
- 使用最佳参数进行完整训练
- 多个随机种子
- 完整的评估

### 2. Shared Adapter 参数建议

```bash
# 推荐的 adapter ratio
ADAPTER_RATIO=(0.125 0.25 0.5)

# 推荐的 LoCoOp 权重配置
# 配置 1：平衡 CLS 和 Patch
LAMBDA_LOCOOP_CLS=0.5
LAMBDA_LOCOOP_PATCH=0.5

# 配置 2：更重视 CLS
LAMBDA_LOCOOP_CLS=0.7
LAMBDA_LOCOOP_PATCH=0.3

# 配置 3：更重视 Patch
LAMBDA_LOCOOP_CLS=0.3
LAMBDA_LOCOOP_PATCH=0.7
```

### 3. 资源管理

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控内存使用
watch -n 1 free -h

# 监控磁盘使用
watch -n 1 df -h
```

### 4. 日志管理

```bash
# 定期清理旧日志
find logs/ -name "*.log" -mtime +30d -delete

# 压缩日志节省空间
find logs/ -name "*.log" -mtime +7d -exec gzip {} \;
```

## 总结

这三个脚本提供了完整的训练和超参数搜索流程：

1. **train_shared_adapter_locoop.sh**：快速测试和开发
2. **search_shared_adapter_locoop.sh**：完整的超参数搜索
3. **search_locoop_params.sh**：专注于 LoCoOp 参数的搜索

所有脚本都：
- 支持自动日志记录
- 支持错误处理
- 支持进度跟踪
- 支持结果汇总

根据你的需求选择合适的脚本开始实验吧！
