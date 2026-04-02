# SAE 训练可视化指南

本文档介绍如何使用可视化工具分析 SAE 训练过程，包括 loss 曲线、稀疏性分析和多实验对比。

---

## 1. 文件结构

```
wan/sae_vis/
├── __init__.py              # 包初始化
├── plot_loss.py             # 主可视化脚本（参数分离结构）
├── quick_plot.py            # 快速启动脚本
├── example_config.json      # 配置文件示例
└── README.md                # 本文件
```

训练产生的持久化数据：

```
sae_runs/exp__20250324/
├── logs/
│   ├── training.log         # 完整训练日志
│   ├── loss_history.jsonl   # 详细 metrics（JSONL 格式）
│   └── loss_history.csv     # 扁平化 CSV（方便 pandas）
├── loss_multi.png           # 自动生成的图表
└── ...
```

---

## 2. 持久化数据结构

### 2.1 JSONL 格式（`loss_history.jsonl`）

每行一个 JSON 对象，包含该 step 的所有 metrics：

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

字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `step` | int | 训练步数 |
| `timestamp` | float | Unix 时间戳 |
| `elapsed` | float | 已训练时间（秒）|
| `step_time` | float | 单步耗时（秒）|
| `metrics.{sae_key}.loss` | float | 总损失 |
| `metrics.{sae_key}.recon_mse` | float | 重建 MSE |
| `metrics.{sae_key}.l2_norm` | float | 参数 L2 范数 |
| `metrics.{sae_key}.sparsity` | float | 稀疏度（0-1）|
| `metrics.{sae_key}.num_activations` | float | 平均激活数 |

### 2.2 CSV 格式（`loss_history.csv`）

扁平化格式，每行一个 SAE 的一个 step：

```csv
step,timestamp,sae_key,loss,recon_mse,l2_norm,sparsity,num_activations
50,1711388715.654,block_out.layer15,0.423456,0.389123,15234.56,0.0104,63.8
```

---

## 3. 快速使用

### 3.1 自动查找最新实验并绘图

```bash
python wan/sae_vis/quick_plot.py
```

自动查找 `sae_runs/` 下最新的实验目录，绘制多指标图表。

### 3.2 指定实验目录

```bash
python wan/sae_vis/quick_plot.py exp__20250324
```

### 3.3 指定具体文件

```bash
python wan/sae_vis/quick_plot.py --file sae_runs/exp__20250324/logs/loss_history.jsonl
```

### 3.4 论文质量输出

```bash
python wan/sae_vis/quick_plot.py exp__20250324 --paper
```

输出 300 DPI 高清图片，无平滑处理。

---

## 4. 高级用法

### 4.1 使用主脚本 `plot_loss.py`

```bash
# 基础 loss 曲线
python wan/sae_vis/plot_loss.py --plot_type loss

# 多指标组合图（loss + recon_mse + sparsity）
python wan/sae_vis/plot_loss.py --plot_type multi

# 对数 Y 轴
python wan/sae_vis/plot_loss.py --log_scale_y

# 无平滑（显示原始数据）
python wan/sae_vis/plot_loss.py --smoothing_window 0

# 指定输出文件
python wan/sae_vis/plot_loss.py --output_file figures/my_loss.png --dpi 600

# 显示图表窗口（本地开发）
python wan/sae_vis/plot_loss.py --show_plot
```

### 4.2 多实验对比

```bash
# 对比不同 top_k 配置
python wan/sae_vis/plot_loss.py \
  --plot_type compare \
  --compare_dirs "exp_topk32,exp_topk64,exp_topk128" \
  --title "TopK Comparison"

# 或使用 quick_plot
python wan/sae_vis/quick_plot.py --compare exp_topk32,exp_topk64,exp_topk128
```

### 4.3 使用配置文件

```bash
python wan/sae_vis/plot_loss.py --config my_config.json
```

配置文件示例见 `wan/sae_vis/example_config.json`。

---

## 5. 参数详解

### 5.1 数据参数（`data_params`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `loss_file` | `"sae_runs/..."` | 主 loss 文件路径 |
| `compare_dirs` | `""` | 对比实验目录（逗号分隔）|
| `run_dir_base` | `"sae_runs"` | 实验根目录 |

### 5.2 图表参数（`plot_params`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `plot_type` | `"multi"` | `loss`/`multi`/`compare` |
| `metrics` | `["loss", "recon_mse", "sparsity"]` | 要绘制的指标 |
| `sae_keys` | `[]` | 指定 SAE 层（空=全部）|

### 5.3 样式参数（`style_params`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `figure_size` | `[14, 8]` | 图表尺寸（英寸）|
| `dpi` | `300` | 输出分辨率 |
| `line_width` | `1.5` | 线宽 |
| `marker_size` | `0` | 标记大小（0=无标记）|
| `alpha` | `0.8` | 透明度 |
| `grid` | `True` | 显示网格 |
| `log_scale_y` | `False` | 对数 Y 轴 |
| `smoothing_window` | `10` | 平滑窗口（0=不平滑）|

### 5.4 输出参数（`output_params`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_file` | `""` | 输出路径（空=自动）|
| `output_format` | `"png"` | `png`/`pdf`/`svg`/`all` |
| `show_plot` | `False` | 显示图表窗口 |
| `title` | `""` | 图表标题 |
| `xlabel` | `"Training Step"` | X 轴标签 |
| `ylabel` | `"Metric Value"` | Y 轴标签 |

---

## 6. 编程使用

### 6.1 加载数据

```python
from wan.sae_vis import load_loss_data
import pandas as pd

# 自动检测格式
df = load_loss_data("sae_runs/exp__20250324/logs/loss_history.jsonl")

# 查看数据
print(df.head())
print(df.columns)

# 筛选特定 SAE
df_layer15 = df[df["sae_key"] == "block_out.layer15"]

# 查看统计
print(f"总步数: {df['step'].max()}")
print(f"最终 loss: {df_layer15['loss'].iloc[-1]:.4f}")
print(f"最小 loss: {df_layer15['loss'].min():.4f}")
```

### 6.2 自定义绘图

```python
from wan.sae_vis import plot_loss_curve, plot_multi_metrics
import matplotlib.pyplot as plt

# 自定义参数
params = {
    "figure_size": (12, 6),
    "dpi": 300,
    "line_width": 2.0,
    "smoothing_window": 20,
    "log_scale_y": True,
    "grid": True,
    "title": "My Custom Plot",
}

# 绘制
fig = plot_loss_curve(df, params)
plt.savefig("custom_plot.png", dpi=300, bbox_inches="tight")
```

### 6.3 平滑处理

```python
from wan.sae_vis import smooth_data
import numpy as np

# 原始数据
loss_values = df_layer15["loss"].values

# 平滑
smoothed = smooth_data(loss_values, window=20)

# 绘制对比
plt.plot(loss_values, alpha=0.3, label="Raw")
plt.plot(smoothed, linewidth=2, label="Smoothed")
plt.legend()
```

---

## 7. 典型分析场景

### 7.1 检查收敛性

```bash
python wan/sae_vis/quick_plot.py exp__20250324 --log
```

观察 loss 是否趋于平稳，对数坐标更易看出指数衰减。

### 7.2 验证稀疏性

```bash
python wan/sae_vis/plot_loss.py --plot_type multi
```

查看 `sparsity` 和 `num_activations` 是否符合预期（如 topk=64 时激活数应接近 64）。

### 7.3 调参对比

```bash
# 对比不同学习率
python wan/sae_vis/quick_plot.py --compare exp_lr1e-3,exp_lr5e-4,exp_lr1e-4

# 对比不同层
python wan/sae_vis/quick_plot.py --compare exp_layer15,exp_layer29
```

### 7.4 生成论文图表

```bash
python wan/sae_vis/plot_loss.py \
  --dpi 600 \
  --smoothing_window 0 \
  --output_format pdf \
  --output_file figures/fig1_loss_curve.pdf
```

---

## 8. 故障排除

### Q1: 找不到 loss 文件

```bash
# 检查文件是否存在
ls sae_runs/exp__20250324/logs/

# 如果只有 CSV，指定格式
python wan/sae_vis/plot_loss.py --file sae_runs/.../loss_history.csv
```

### Q2: 图表中文显示为方框

```python
# 在脚本开头添加
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

### Q3: 步数范围不对

编辑 `plot_loss.py` 中的 `advanced_params["step_range"]` 参数。

### Q4: 多实验对比线条太乱

```bash
# 使用平滑
python wan/sae_vis/quick_plot.py --compare exp1,exp2,exp3 --paper
```

---

## 9. 扩展开发

如需添加新的可视化类型：

1. 在 `plot_loss.py` 中添加新的绘图函数
2. 在 `plot_params["plot_type"]` 中添加新选项
3. 在 `main()` 函数中添加路由逻辑

示例：

```python
def plot_sparsity_distribution(df: pd.DataFrame, params: Dict) -> Figure:
    """绘制稀疏度分布直方图"""
    fig, ax = plt.subplots(figsize=params["figure_size"])
    # ... 绘图逻辑 ...
    return fig
```

---

## 10. 最佳实践

1. **训练时**：设置 `loss_log_interval=1` 记录每步数据
2. **监控时**：使用 `quick_plot.py` 快速查看
3. **分析时**：使用 `smoothing_window=10` 平滑波动
4. **发表时**：使用 `--paper` 参数输出高清矢量图
5. **对比时**：保持相同坐标轴范围便于比较
