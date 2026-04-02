# SAE离线训练模块

本模块提供从预采集激活值进行SAE离线训练的功能，相比在线训练更快且更灵活。

## 模块结构

```
offline_training/
├── activation_collector.py  # 激活值采集器
├── train_offline.py         # 离线训练脚本
├── test_offline.py          # 离线测试脚本
├── __init__.py
└── README.md               # 本文档
```

## 使用流程

### 1. 采集激活值

首先使用 `activation_collector.py` 从DiT模型中采集激活值并保存到磁盘。

```bash
python -m wan.sae.offline_training.activation_collector \
    --checkpoint_dir ./Wan2.1-T2V-1.3B \
    --prompt_dir ./nsfw_prompts \
    --output_dir offline_data/run1 \
    --hook_mode block_out \
    --hook_layers "15,29" \
    --sampling_steps 30 \
    --batch_prompts 4 \
    --max_prompts 1000 \
    --storage_format numpy
```

**参数说明：**

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `checkpoint_dir` | Wan2.1模型权重目录 | `./Wan2.1-T2V-1.3B` |
| `prompt_dir` | 提示词目录 | 包含.txt文件的目录 |
| `output_dir` | 激活值输出目录 | `offline_data/run1` |
| `hook_layers` | 要采集的层 | `"15,29"`（中层+深层） |
| `sampling_steps` | 每个prompt的扩散步数 | 30-50 |
| `storage_format` | 存储格式 | `numpy`（推荐） |

**输出结构：**

```
offline_data/run1/
├── collection_config.json    # 采集配置
├── manifest.jsonl           # 数据索引（每行一条记录）
├── metadata.json            # 整体元信息
└── data/
    ├── batch_000/
    │   ├── record_000.json           # prompt和元信息
    │   ├── record_000_block_out_layer15.npy   # 层15激活值
    │   └── record_000_block_out_layer29.npy   # 层29激活值
    ├── batch_001/
    └── ...
```

**存储格式说明：**

- `json`: 全部存为JSON（体积大，兼容性最好）
- `numpy`: 大数组存为.npy/.npz文件（推荐，体积小10-100倍）
- `hybrid`: 小数组JSON，大数组numpy（折中）

### 2. 离线训练SAE

使用 `train_offline.py` 从采集的激活值训练SAE。

```bash
python -m wan.sae.offline_training.train_offline \
    --data_dir offline_data/run1 \
    --run_dir sae_runs/offline_exp1 \
    --layers "15,29" \
    --epochs 10 \
    --batch_size 4096 \
    --lr 1e-3 \
    --d_hidden 6144
```

**参数说明：**

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `data_dir` | 激活值数据目录 | `offline_data/run1` |
| `run_dir` | SAE训练输出目录 | `sae_runs/offline_exp1` |
| `layers` | 要训练的层 | `"15,29"` |
| `epochs` | 训练轮数 | 5-20 |
| `batch_size` | 每批token数 | 2048-8192 |
| `lr` | 学习率 | 1e-3 ~ 1e-4 |
| `validation_split` | 验证集比例 | 0.05 |

**特点：**

1. **快速训练**：无需运行DiT，直接在预存激活值上训练
2. **灵活调参**：可快速调整学习率、batch_size等参数重新训练
3. **多轮训练**：支持对同一数据多次遍历（epoch）
4. **验证监控**：自动划分验证集监控过拟合

**输出结构：**

```
sae_runs/offline_exp1/
├── train_config.json           # 训练配置
├── training_history.json       # 训练历史
└── block_out.layer15/
│   ├── sae_config.json        # SAE配置
│   ├── sae_latest.pt          # 最新权重
│   ├── sae_best.pt            # 最佳验证权重
│   └── sae_step1000.pt        # 历史版本
└── block_out.layer29/
    └── ...
```

### 3. 离线测试SAE

使用 `test_offline.py` 测试训练好的SAE性能。

```bash
python -m wan.sae.offline_training.test_offline \
    --data_dir offline_data/run1 \
    --run_dir sae_runs/offline_exp1 \
    --hook_layers "15,29" \
    --output_path test_results.pt
```

**输出结果：**

```python
{
    "block_out.layer15": {
        "results": [...],  # 每个prompt的分析结果
        "statistics": {
            "num_prompts": 500,
            "avg_sparsity": 0.0234,
            "avg_recon_mse": 0.0012,
            "feature_frequency": [...],  # 特征激活频率
            "most_common_features": [123, 456, ...],  # 最常激活的特征
        }
    }
}
```

## 在线训练 vs 离线训练对比

| 特性 | 在线训练 | 离线训练 |
|------|----------|----------|
| **速度** | 慢（需要运行DiT） | 快（直接训练SAE） |
| **显存占用** | 高（DiT + SAE） | 低（仅SAE） |
| **灵活性** | 低（难以调整采样参数） | 高（固定激活值可反复实验） |
| **数据分布** | 实时采样，分布更准确 | 依赖预采集数据 |
| **适用场景** | 最终训练、完整验证 | 快速实验、参数搜索 |

## 高级用法

### 批量采集多个配置

```bash
# 采集不同层的激活值
for layers in "0,5" "15,20" "25,29"; do
    python -m wan.sae.offline_training.activation_collector \
        --hook_layers "$layers" \
        --output_dir "offline_data/layers_${layers//,/}"
done
```

### 增量训练

```bash
# 在已有checkpoint基础上继续训练
python -m wan.sae.offline_training.train_offline \
    --data_dir offline_data/run1 \
    --run_dir sae_runs/offline_exp1 \
    --resume \
    --epochs 20
```

### 多GPU训练（数据并行）

目前单GPU训练已足够快（几分钟到几十分钟），如需多GPU可修改代码使用PyTorch DDP。

## 常见问题

**Q: 采集的激活值占用多少磁盘空间？**

A: 使用numpy格式约每1000条prompt每层的激活值占用500MB-1GB（取决于sampling_steps和token数）。

**Q: 离线训练的SAE质量与在线训练相比如何？**

A: 如果采集数据覆盖了足够多样的提示词和时间步，离线训练的质量与在线训练相当。

**Q: 可以混合多个采集数据集吗？**

A: 可以，只需将多个manifest.jsonl合并，或修改代码支持多数据目录。

**Q: 如何控制保存的文件大小？**

A: 调整`max_prompts_per_file`参数（默认100），较小的值会产生更多小文件。

## 配置文件示例

完整的训练配置文件（JSON）：

```json
{
    "data_dir": "offline_data/run1",
    "run_dir": "sae_runs/offline_exp1",
    "epochs": 10,
    "batch_size": 4096,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "lr_scheduler": "cosine",
    "validation_split": 0.05,
    "gradient_accumulation_steps": 1,
    "sae": {
        "d_model": 1536,
        "d_hidden": 6144,
        "activation": "relu",
        "sparsity": "topk",
        "top_k": 64
    },
    "seed": 42
}
```

使用配置文件：

```bash
python -m wan.sae.offline_training.train_offline --config train_config.json
```
