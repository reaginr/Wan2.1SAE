# SAE Loss 持久化与可视化功能总结

## 功能概述

本次更新为 SAE 训练系统添加了完整的 **loss 持久化** 和 **可视化分析** 功能。

---

## 1. 训练脚本增强 (`wan/sae_train_t2v_1_3b.py`)

### 新增日志参数

```python
log_params = {
    "log_to_file": True,          # 自动保存日志到文件
    "loss_log_interval": 1,        # 每步记录详细 metrics
}
```

### 新增记录的 Metrics

每步现在记录以下详细指标：

| 指标 | 说明 | 用途 |
|------|------|------|
| `loss` | 总损失 | 观察整体收敛 |
| `recon_mse` | 重建 MSE | 评估重建质量 |
| `l2_norm` | 参数 L2 范数 | 监控权重衰减 |
| `sparsity` | 稀疏度 | 验证稀疏约束效果 |
| `num_activations` | 平均激活数 | 检查 topk 实际值 |

### 输出文件结构

```
sae_runs/exp__20250324/
├── logs/
│   ├── training.log              # 控制台输出（完整日志）
│   ├── loss_history.jsonl        # 结构化数据（推荐用于分析）
│   └── loss_history.csv          # 扁平化数据（pandas 友好）
├── block_out.layer15/
│   └── ...
├── loss_multi.png                # 自动生成的图表（可选）
└── train_state.json
```

### 日志输出示例

```
[50/500] batch=4 step_time=595.70s elapsed=8.3h ETA=74.5h cur_mem=8.25GB peak_mem=19.69GB
  block_out.layer15: loss=0.4234 mse=0.3891 spar=0.010 acts=63.8
```

---

## 2. 可视化工具包 (`wan/sae_vis/`)

### 文件结构

```
wan/sae_vis/
├── __init__.py              # 包初始化，导出主要函数
├── plot_loss.py             # 主可视化脚本（参数分离结构）
├── quick_plot.py            # 快速启动脚本
├── example_config.json      # 配置文件模板
└── example_usage.py         # 编程使用示例
```

### 核心功能

#### plot_loss.py - 主可视化脚本

**参数分离结构**：
- `data_params` - 数据路径配置
- `plot_params` - 图表类型配置
- `style_params` - 视觉样式配置
- `output_params` - 输出配置
- `compare_params` - 多实验对比配置
- `advanced_params` - 高级配置

**支持的图表类型**：

| 类型 | 命令 | 用途 |
|------|------|------|
| Loss 曲线 | `--plot_type loss` | 基础收敛分析 |
| 多指标组合 | `--plot_type multi` | 全面训练分析 |
| 多实验对比 | `--plot_type compare` | 超参数调优 |

**样式特性**：
- 平滑处理（移动平均）
- 对数/线性 Y 轴
- 自定义 DPI（150/300/600）
- 多格式输出（PNG/PDF/SVG）

#### quick_plot.py - 快速启动

```bash
# 自动查找最新实验
python wan/sae_vis/quick_plot.py

# 指定实验
python wan/sae_vis/quick_plot.py exp__20250324

# 论文质量
python wan/sae_vis/quick_plot.py --paper

# 对比实验
python wan/sae_vis/quick_plot.py --compare exp1,exp2,exp3
```

---

## 3. 使用指南

### 3.1 基础用法

**训练后自动生成图表**：

```bash
# 训练完成后，使用快速脚本
python wan/sae_vis/quick_plot.py exp__20250324
```

**输出**: `sae_runs/exp__20250324/loss_multi.png`

### 3.2 自定义图表

```bash
python wan/sae_vis/plot_loss.py \
  --loss_file sae_runs/exp__20250324/logs/loss_history.jsonl \
  --plot_type multi \
  --dpi 300 \
  --smoothing_window 20 \
  --output_file figures/my_analysis.png
```

### 3.3 多实验对比

```bash
python wan/sae_vis/plot_loss.py \
  --plot_type compare \
  --compare_dirs "exp_topk32,exp_topk64,exp_topk128" \
  --title "TopK 参数对比" \
  --dpi 300
```

### 3.4 编程使用

```python
from wan.sae_vis import load_loss_data, plot_loss_curve

# 加载数据
df = load_loss_data("sae_runs/exp__20250324/logs/loss_history.jsonl")

# 查看统计
print(f"总步数: {df['step'].max()}")
print(f"最终 loss: {df['loss'].iloc[-1]:.4f}")

# 自定义绘图
params = {
    "figure_size": (12, 6),
    "dpi": 300,
    "smoothing_window": 10,
}
fig = plot_loss_curve(df, params)
fig.savefig("my_plot.png")
```

---

## 4. 参数配置

### 修改默认参数

编辑 `wan/sae_vis/plot_loss.py` 顶部的参数字典：

```python
style_params = {
    "figure_size": (14, 8),      # 改这里
    "dpi": 300,                   # 改这里
    "smoothing_window": 10,       # 改这里
}
```

### 使用配置文件

```json
{
  "data_params": {
    "loss_file": "path/to/loss.jsonl"
  },
  "style_params": {
    "dpi": 600
  }
}
```

```bash
python wan/sae_vis/plot_loss.py --config my_config.json
```

---

## 5. 典型分析场景

### 场景 1: 检查收敛性

```bash
python wan/sae_vis/quick_plot.py --log
```

观察 loss 是否趋于平稳，对数坐标更易看出指数衰减。

### 场景 2: 验证稀疏性

查看 `sparsity` 和 `num_activations`：
- topk=64 时 `num_activations` 应接近 64
- `sparsity` 应在 1% 左右（6144 维时约 0.01）

### 场景 3: 调参对比

```bash
# 对比不同学习率
python wan/sae_vis/quick_plot.py --compare exp_lr1e-3,exp_lr5e-4
```

### 场景 4: 生成论文图表

```bash
python wan/sae_vis/plot_loss.py \
  --dpi 600 \
  --smoothing_window 0 \
  --output_format pdf
```

---

## 6. 数据格式参考

### JSONL 格式

```json
{
  "step": 50,
  "timestamp": 1711388715.654,
  "elapsed": 29835.5,
  "step_time": 595.7,
  "metrics": {
    "block_out.layer15": {
      "loss": 0.423456,
      "recon_mse": 0.389123,
      "l2_norm": 15234.56,
      "sparsity": 0.0104,
      "num_activations": 63.8
    }
  }
}
```

### CSV 格式

```csv
step,timestamp,sae_key,loss,recon_mse,l2_norm,sparsity,num_activations
50,1711388715.654,block_out.layer15,0.423456,0.389123,15234.56,0.0104,63.8
```

---

## 7. 文档索引

| 文档 | 内容 |
|------|------|
| `SAE_VISUALIZATION.md` | 完整可视化指南 |
| `CHECKPOINT.md` | Checkpoint 管理指南 |
| `wan/sae_vis/example_config.json` | 配置模板 |
| `wan/sae_vis/example_usage.py` | 编程示例 |

---

## 8. 后续扩展建议

如需进一步扩展可视化功能：

1. **添加新的指标**: 在 `sae_train_t2v_1_3b.py` 中计算并记录
2. **添加新的图表类型**: 在 `plot_loss.py` 中添加绘图函数
3. **添加交互式图表**: 使用 plotly 替代 matplotlib
4. **实时监控**: 读取 JSONL 文件并实时更新图表

---

## 总结

现在你可以：

1. ✅ **持久化**: 训练时自动记录详细 metrics
2. ✅ **可视化**: 一键生成专业 loss 曲线
3. ✅ **对比**: 轻松对比不同实验结果
4. ✅ **分析**: 编程方式深入分析数据
5. ✅ **发表**: 输出论文质量高清图表
