# t-SNE 可视化测试

使用模拟数据测试 t-SNE 可视化效果，验证正负样本的聚类分离度。

## 文件结构

```
simulate/test-tSNE/
├── generate_mock_data.py    # 生成模拟数据
├── test_tsne.py             # 运行t-SNE测试
└── README.md                # 本文件
```

## 生成的模拟数据类型

| 数据集 | 描述 | 预期 Silhouette | 预期可视化效果 |
|--------|------|-----------------|----------------|
| `well_separated` | 正负样本明显分离 | 0.7-0.9 | 两个清晰的聚类团 |
| `sparse_separated` | 稀疏特征，明显分离 | 0.6-0.8 | 稀疏但分离的聚类 |
| `partially_separated` | 部分重叠 | 0.3-0.5 | 两个团但有重叠区域 |
| `overlapping` | 高度重叠 | 0.0-0.2 | 混合在一起无明显边界 |

## 使用方法

### 一键测试（推荐）

```bash
cd simulate/test-tSNE
python test_tsne.py
```

这会：
1. 自动生成4种模拟数据
2. 对每个数据集运行t-SNE
3. 计算聚类指标（Silhouette Score等）
4. 生成可视化图像
5. 输出对比报告

### 分步执行

**步骤1：生成模拟数据**

```bash
python generate_mock_data.py \
    --output_root "simulate/test-tSNE/mock_activations" \
    --n_samples 200 \
    --d_hidden 6144 \
    --n_timesteps 30
```

**步骤2：对单个数据集运行t-SNE**

```bash
python wan/sae/interpretability/visualize_tsne.py \
    --activation_root "simulate/test-tSNE/mock_activations" \
    --category "well_separated" \
    --layer_key "sae_layer15" \
    --output_dir "simulate/test-tSNE/results" \
    --perplexity 30 \
    --n_iter 1000
```

## 输出文件

```
simulate/test-tSNE/
├── mock_activations/           # 模拟数据（阶段一格式）
│   ├── sae_layer15/
│   │   ├── well_separated/
│   │   ├── partially_separated/
│   │   ├── overlapping/
│   │   └── sparse_separated/
│   └── extraction_config.json
├── results/                    # t-SNE可视化结果
│   ├── well_separated_sae_layer15_tsne.png
│   ├── well_separated_sae_layer15_metrics.json
│   ├── well_separated_sae_layer15_tsne.npz
│   ├── partially_separated_...
│   ├── overlapping_...
│   └── sparse_separated_...
└── comparison_report.json      # 对比报告
```

## 结果解读

### Silhouette Score 解释

| 值范围 | 含义 | 可视化表现 |
|--------|------|------------|
| 0.7 - 1.0 | 良好分离 | 两个清晰分开的团 |
| 0.5 - 0.7 | 较好分离 | 分离但边界可能有些模糊 |
| 0.25 - 0.5 | 部分分离 | 两个团但有明显重叠 |
| 0.0 - 0.25 | 弱分离 | 大量重叠，边界不清 |
| < 0.0 | 错误聚类 | 样本被分到错误的类 |

### 可视化图像说明

生成的 PNG 包含两个子图：

1. **左图：t-SNE散点图**
   - 蓝色点：负样本
   - 红色点：正样本
   - X标记：各类的中心

2. **右图：距离分布直方图**
   - 显示样本到各自类中心的距离分布
   - 绿色虚线：决策边界

## 验证真实数据

使用真实采集的数据：

```bash
python wan/sae/interpretability/visualize_tsne.py \
    --activation_root "activations" \
    --category "violence" \
    --layer_key "sae_layer15" \
    --output_dir "visualizations"
```

如果 Silhouette Score > 0.5，说明正负样本在SAE特征空间中有良好分离，概念提取有效。
