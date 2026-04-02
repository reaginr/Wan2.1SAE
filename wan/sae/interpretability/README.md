# SAE可解释性分析模块

本模块提供通过对比正负提示词提取概念向量的功能，用于理解SAE中学到的特征含义。

## 核心思想

概念提取的基本假设：
1. 正例提示词（包含目标概念）会在某些特征上产生强激活
2. 负例提示词（不包含目标概念）在这些特征上激活较弱
3. 两者的差异即为该概念的"方向向量"

## 模块结构

```
interpretability/
├── concept_extractor.py    # 概念向量提取器
├── __init__.py
└── README.md              # 本文档
```

## 使用方法

### 1. 准备正负提示词

创建两个文本文件，每行一条提示词：

**concepts/violence_positive.txt**（包含暴力概念的提示词）：
```
a person punching another person
fighters in a boxing ring
war scene with explosions
soldiers fighting in battlefield
a knife attack in dark alley
...
```

**concepts/violence_negative.txt**（不包含暴力概念的提示词）：
```
a peaceful garden with flowers
people having picnic in park
children playing on playground
calm ocean waves at sunset
a cozy living room interior
...
```

### 2. 采集激活值

首先需要为正负提示词分别采集激活值（参见offline_training/README.md）：

```bash
# 采集正例激活值
python -m wan.sae.offline_training.activation_collector \
    --prompt_dir concepts/violence_positive \
    --output_dir offline_data/violence_positive

# 采集负例激活值
python -m wan.sae.offline_training.activation_collector \
    --prompt_dir concepts/violence_negative \
    --output_dir offline_data/violence_negative
```

### 3. 提取概念向量

```bash
python -m wan.sae.interpretability.concept_extractor \
    --run_dir sae_runs/exp1 \
    --positive_file concepts/violence_positive.txt \
    --negative_file concepts/violence_negative.txt \
    --concept_name violence \
    --hook_mode block_out \
    --hook_layers "15,29" \
    --method mean_diff \
    --output_dir concept_vectors
```

**参数说明：**

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `run_dir` | SAE训练输出目录 | `sae_runs/exp1` |
| `positive_file` | 正例提示词文件 | `concepts/xxx_positive.txt` |
| `negative_file` | 负例提示词文件 | `concepts/xxx_negative.txt` |
| `concept_name` | 概念名称 | 如`violence`、`nsfw`等 |
| `hook_layers` | 要分析的层 | `"15,29"` |
| `method` | 提取方法 | `mean_diff`或`contrastive` |
| `output_dir` | 输出目录 | `concept_vectors` |

## 提取方法

### 1. 平均差分法 (mean_diff)

最简单的提取方法：

```
concept_vector = mean(positive_activations) - mean(negative_activations)
```

**优点：**
- 计算简单，无需优化
- 结果直观可解释

**适用场景：**
- 概念与激活值有明确线性关系
- 正负样本分布相对分离

### 2. 对比学习方法 (contrastive)

通过优化使得：
- 正例在概念方向上的投影尽可能大
- 负例在概念方向上的投影尽可能小

```python
# 优化目标
loss = -log(sigmoid(proj_positive / temp)) - log(sigmoid(-proj_negative / temp))
```

**优点：**
- 可以找到更具判别性的方向
- 适合边界模糊的概念

**缺点：**
- 需要迭代优化
- 结果可能不如mean_diff稳定

## 输出格式

### JSON格式 (.json)

包含完整的元信息和统计：

```json
{
  "concept_name": "violence",
  "concept_vector": [0.01, -0.005, 0.023, ...],
  "layer_key": "block_out.layer15",
  "run_dir": "sae_runs/exp1",
  "extraction_method": "mean_diff",
  "positive_prompts_count": 100,
  "negative_prompts_count": 100,
  "positive_prompts_sample": [
    "a person punching another person",
    "fighters in a boxing ring",
    ...
  ],
  "negative_prompts_sample": [
    "a peaceful garden with flowers",
    "people having picnic in park",
    ...
  ],
  "statistics": {
    "pos_mean_activation": [...],
    "neg_mean_activation": [...],
    "activation_difference": [...],
    "pos_std": [...],
    "neg_std": [...],
    "top_k_indices": [123, 456, 789, ...],
    "top_k_values": [0.15, 0.12, 0.10, ...],
    "top_k_features": [
      {"index": 123, "value": 0.15},
      {"index": 456, "value": 0.12},
      ...
    ],
    "selectivity": {
      "selectivity_scores": [...],
      "high_selectivity_indices": [123, 456, ...],
      "high_selectivity_count": 15,
      "pos_activation_freq": [...],
      "neg_activation_freq": [...]
    }
  },
  "metadata": {
    "sae_config": {...},
    "extraction_config": {...},
    "stats_config": {...}
  }
}
```

### NumPy格式 (.npy)

纯向量数据，方便快速加载：

```python
import numpy as np
vector = np.load("concept_vectors/violence_block_out.layer15.npy")
# vector.shape: (d_hidden,)
```

## 统计指标解读

### 1. Top-K特征

概念向量中绝对值最大的K个特征索引，表示对该概念最敏感的特征。

```json
"top_k_features": [
  {"index": 123, "value": 0.15},  // 特征123正向贡献最大
  {"index": 456, "value": -0.12}, // 特征456负向贡献最大
  ...
]
```

### 2. 特征选择度 (Selectivity)

衡量特征对概念的区分能力：

```
selectivity = P(feature active | positive) - P(feature active | negative)
```

- 高选择度（>0.7）：特征几乎只在正例中激活
- 低选择度（<0.3）：特征在正负例中都有激活

### 3. 激活频率

```json
"pos_activation_freq": [...],  // 各特征在正例中的激活频率
"neg_activation_freq": [...]   // 各特征在负例中的激活频率
```

## 概念向量验证

提取概念向量后，建议进行验证：

### 1. 方向一致性验证

检查正例在概念方向上的投影是否普遍大于负例：

```python
# 伪代码
for prompt in positive_prompts:
    activation = get_sae_activation(prompt)
    projection = dot(activation, concept_vector)
    assert projection > threshold
```

### 2. 干预验证

使用概念向量进行干预生成（参见steering模块），检查是否产生预期效果。

## 多个概念的提取

可以同时提取多个相关概念：

```bash
for concept in violence nudity gore weapons; do
    python -m wan.sae.interpretability.concept_extractor \
        --positive_file concepts/${concept}_positive.txt \
        --negative_file concepts/${concept}_negative.txt \
        --concept_name $concept \
        --hook_layers "15,29"
done
```

## 概念向量的组合

多个概念向量可以线性组合：

```python
# 组合概念向量
combined = 0.5 * violence_vector + 0.3 * gore_vector - 0.2 * peaceful_vector

# 归一化
combined = combined / np.linalg.norm(combined)
```

## 常见问题

**Q: 需要多少正负样本？**

A: 建议各50-200条。样本越多统计越稳定，但采集时间也越长。

**Q: 如何选择提取的层？**

A: 通常中层（15左右）最具可解释性，深层（29左右）更抽象。可以同时提取多层后比较。

**Q: mean_diff和contrastive哪个更好？**

A: 建议先用mean_diff，如果效果不佳再尝试contrastive。mean_diff更稳定，contrastive可能找到更好的方向。

**Q: 概念向量的范数应该是多少？**

A: 默认归一化后范数为1。不归一化时，范数反映概念的"强度"。

## 迭代改进指南

1. **检查样本质量**
   - 确保正负样本确实区分了目标概念
   - 去除模糊样本

2. **调整提取参数**
   - 尝试不同的min_activation_threshold
   - 对比不同提取方法的结果

3. **多层对比**
   - 比较不同层的概念向量
   - 选择最具判别性的层

4. **验证干预效果**
   - 使用steering模块测试概念向量
   - 根据生成结果反馈调整

## 相关论文

- [Understanding Neural Networks through Representation Erasure](https://arxiv.org/abs/1612.08220)
- [Discovering Latent Concepts Learned in BERT](https://arxiv.org/abs/1902.07296)
- [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
