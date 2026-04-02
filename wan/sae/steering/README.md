# SAE干预生成模块

本模块通过SAE概念向量干预视频生成过程，实现对生成内容的精细控制。

## 核心思想

干预生成的基本流程：
1. 在DiT生成过程中hook特定层的输出
2. 将特征通过SAE编码到隐空间
3. 沿概念向量方向调整激活值
4. 通过SAE解码回特征空间
5. 继续正常生成流程

## 模块结构

```
steering/
├── steering_generator.py    # 干预生成器
├── __init__.py
└── README.md               # 本文档
```

## 使用方法

### 1. 准备概念向量

首先使用interpretability模块提取概念向量：

```bash
python -m wan.sae.interpretability.concept_extractor \
    --concept_name violence \
    --positive_file concepts/violence_positive.txt \
    --negative_file concepts/violence_negative.txt \
    --hook_layers "15,29"
```

概念向量将保存到 `concept_vectors/` 目录：
```
concept_vectors/
├── violence_block_out.layer15.json
├── violence_block_out.layer15.npy
├── violence_block_out.layer29.json
└── violence_block_out.layer29.npy
```

### 2. 创建干预配置

创建JSON配置文件：

```json
{
  "prompt": "Two people arguing in the street",
  "size_w": 832,
  "size_h": 480,
  "frame_num": 81,
  "sampling_steps": 30,
  "seed": 42,
  "interventions": [
    {
      "concept_name": "violence",
      "layer_key": "block_out.layer15",
      "strength": -0.5,
      "method": "additive",
      "timestep_range": [0, 30]
    }
  ]
}
```

### 3. 运行干预生成

```bash
python -m wan.sae.steering.steering_generator \
    --config steering_config.json \
    --checkpoint_dir ./Wan2.1-T2V-1.3B \
    --run_dir sae_runs/exp1 \
    --concept_dir concept_vectors \
    --output_dir steering_outputs
```

## 干预配置详解

### 干预参数

| 参数 | 说明 | 可选值 | 建议值 |
|------|------|--------|--------|
| `concept_name` | 概念名称 | 任意字符串 | `violence`, `nsfw`, `gore`等 |
| `layer_key` | 目标层 | `"block_out.layer{N}"` | `"block_out.layer15"` |
| `strength` | 干预强度 | -1.0 ~ 1.0 | 0.3 ~ 0.8 |
| `method` | 干预方法 | `"additive"`, `"multiplicative"`, `"projection"`, `"clamp"` | `"additive"` |
| `timestep_range` | 干预时间步范围 | `[start, end]` | `[0, 30]` |
| `feature_mask` | 只干预特定特征 | 特征索引列表 | `null`（全部） |

### 干预方法

#### 1. 加法干预 (additive)

```python
z_new = z + strength * concept_vector
```

**特点：**
- 最直接的方法
- 正值增强概念，负值抑制概念
- 适合大多数场景

**建议strength：** 0.3 ~ 0.8（增强），-0.5 ~ -0.8（抑制）

#### 2. 乘法干预 (multiplicative)

```python
z_new = z * (1 + strength * concept_vector)
```

**特点：**
- 对原有激活值进行缩放
- 适合调整概念的"强度"而非简单开关

**建议strength：** 0.1 ~ 0.3

#### 3. 投影干预 (projection)

```python
projection = dot(z, concept_vector)
z_new = z + strength * projection * concept_vector
```

**特点：**
- 只调整沿概念向量方向的分量
- 保持正交方向不变
- 更"精确"的干预

**建议strength：** 0.5 ~ 1.0

#### 4. 钳制干预 (clamp)

```python
projection = dot(z, concept_vector)
clamped = clamp(projection, -abs(strength), abs(strength))
z_new = z - projection * concept_vector + clamped * concept_vector
```

**特点：**
- 限制概念方向的激活值范围
- 适合"安全过滤"场景

**建议strength：** 0.1 ~ 0.3

### 时间步范围

干预可以只在特定时间步进行：

```json
{
  "timestep_range": [0, 15]  // 只在早期干预（影响构图）
}
```

```json
{
  "timestep_range": [15, 30]  // 只在后期干预（影响细节）
}
```

**不同时间步的影响：**
- 早期（0-10）：影响整体构图和场景设置
- 中期（10-20）：影响主体内容和动作
- 后期（20-30）：影响细节和纹理

## 高级配置

### 多概念干预

可以同时应用多个概念向量：

```json
{
  "interventions": [
    {
      "concept_name": "violence",
      "layer_key": "block_out.layer15",
      "strength": -0.6,
      "method": "additive",
      "timestep_range": [0, 30]
    },
    {
      "concept_name": "gore",
      "layer_key": "block_out.layer29",
      "strength": -0.4,
      "method": "projection",
      "timestep_range": [10, 30]
    },
    {
      "concept_name": "peaceful",
      "layer_key": "block_out.layer15",
      "strength": 0.3,
      "method": "additive",
      "timestep_range": [0, 20]
    }
  ]
}
```

### 动态强度调整

干预强度可以随时间动态衰减：

```json
{
  "dynamic_strength": true,
  "strength_decay": "linear"  // "linear" | "exponential" | "cosine"
}
```

**衰减函数：**
- `linear`: strength * (1 - progress)
- `exponential`: strength * exp(-3 * progress)
- `cosine`: strength * 0.5 * (1 + cos(π * progress))

### 特征掩码

只干预特定特征：

```json
{
  "feature_mask": [123, 456, 789],
  "strength": 0.5
}
```

这在已知哪些特征对概念最敏感时很有用。

## 输出结构

```
steering_outputs/
├── steered_20250401_120000.mp4      # 生成的视频
├── session_20250401_120000.json     # 会话配置（持久化）
└── ...
```

### 会话配置文件

```json
{
  "prompt": "Two people arguing in the street",
  "video_path": "steering_outputs/steered_20250401_120000.mp4",
  "interventions": [
    {
      "concept_name": "violence",
      "layer_key": "block_out.layer15",
      "strength": -0.5,
      "method": "additive",
      "timestep_range": [0, 30]
    }
  ],
  "size_w": 832,
  "size_h": 480,
  "frame_num": 81,
  "sampling_steps": 30,
  "seed": 42,
  "run_dir": "sae_runs/exp1",
  "checkpoint_dir": "./Wan2.1-T2V-1.3B",
  "concept_vectors_dir": "concept_vectors",
  "timestamp": "2025-04-01T12:00:00",
  "duration_seconds": 45.2,
  "metadata": {
    "global_strength_scale": 1.0,
    "dynamic_strength": false
  }
}
```

## 使用建议

### 1. 渐进式调整

不要一开始就用很大的strength：

```
尝试顺序：0.1 -> 0.3 -> 0.5 -> 0.8
```

### 2. 分层干预

不同层控制不同抽象层次：

- 浅层（0-10）：低级视觉特征
- 中层（10-20）：物体和动作
- 深层（20-29）：语义概念

### 3. A/B对比

同时生成有干预和无干预的版本进行对比：

```bash
# 无干预
python -m wan.sae.steering.steering_generator --config config_no_intervention.json

# 有干预
python -m wan.sae.steering.steering_generator --config config_with_intervention.json
```

### 4. 组合实验

尝试不同层和概念的组合：

```bash
for layer in 10 15 20 25 29; do
    for strength in 0.3 0.5 0.8; do
        # 修改配置文件并运行
        ...
    done
done
```

## 常见问题

**Q: 干预后视频质量下降怎么办？**

A: 可能原因和解决方法：
- strength过大 → 减小到0.3以下
- 干预层太深 → 尝试浅层
- 干预时间步太长 → 限制timestep_range

**Q: 如何判断干预是否生效？**

A: 建议方法：
1. 对比有/无干预的生成结果
2. 使用相同的seed确保可比性
3. 收集多个样本进行人工评估

**Q: 可以同时抑制和增强不同概念吗？**

A: 可以，如示例中的多概念干预配置。

**Q: 干预对生成速度的影响？**

A: 每干预一层增加约10-20%的生成时间（额外的SAE编解码）。

**Q: 为什么干预某些层没有效果？**

A: 可能原因：
- 该层没有学到目标概念
- 概念向量提取不准确
- 干预强度不够
- 概念在该层不是线性可分的

## 安全考虑

⚠️ **重要提示：**

1. **内容安全**
   - 干预生成可能产生意外内容
   - 建议配合其他安全措施使用
   - 不应用于生成违法内容

2. **概念向量质量**
   - 确保概念向量从足够多样的样本提取
   - 定期验证干预效果
   - 注意概念向量的偏见问题

3. **干预边界**
   - 理解干预有局限性
   - 不是所有概念都能被精确控制
   - 干预可能带来副作用

## 迭代优化流程

```
1. 提取概念向量
   ↓
2. 小规模干预测试（5-10条prompt）
   ↓
3. 评估干预效果
   ↓
4. 调整干预参数（层、strength、时间步）
   ↓
5. 重复2-4直到满意
   ↓
6. 大规模验证（50+条prompt）
   ↓
7. 部署使用
```

## 相关模块

- `interpretability`: 概念向量提取
- `offline_training`: SAE训练和激活值采集
- `enhanced_configs`: 增强配置系统
