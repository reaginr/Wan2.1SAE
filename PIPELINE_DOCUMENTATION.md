# SAE概念提取完整流程详解

本文档详细说明从激活值采集到概念向量提取的完整流程。

---

## 一、阶段一：激活值采集（GPU必需）

### 1.1 运行命令

```bash
python -m wan.sae.interpretability.concept_extractor_stage1 \
    --model_path "./Wan2.1-T2V-1.3B" \
    --sae_run_dir "sae_runs/exp1" \
    --pos_prompts "final_cleaned/pos_prompt_1.txt" \
    --neg_prompts "final_cleaned/neg_prompt_1.txt" \
    --category "sex" \
    --output_root "activations" \
    --sae_layers "15" \
    --sampling_steps 30
```

### 1.2 内部处理流程

```
对于每一对提示词 (pos_prompt, neg_prompt):

    Step 1: 文本编码
        pos_context = T5编码(pos_prompt)   # [1, L_text, 4096]
        neg_context = T5编码(neg_prompt)   # [1, L_text, 4096]

    Step 2: 初始化噪声
        latent = randn([16, 21, 60, 104])   # VAE latent空间

    Step 3: 扩散采样循环 (30 timesteps)
        for t in [29, 28, ..., 0]:

            # 3.1 注册Hook捕获DiT输出
            handles = register_dit_hooks(model, layers=[15])

            # 3.2 DiT前向传播
            with torch.no_grad():
                output = model(latent, t=t, context=pos_context)

            # 3.3 获取Hook捕获的激活
            dit_hidden = hook_capture["block_out.layer15"]  # [1, 32760, 1536]

            # 3.4 SAE编码
            z, _, _ = sae.encode(dit_hidden)  # [32760, 6144]

            # 3.5 实时池化（关键！不保存原始数据）
            pool["sum"] += z.sum(axis=0)       # 累加所有token
            pool["sum_sq"] += (z**2).sum(axis=0)
            pool["max"] = max(pool["max"], z.max(axis=0))
            pool["min"] = min(pool["min"], z.min(axis=0))
            pool["count"] += 32760

            # 3.6 更新latent（Euler方法）
            latent = latent - output * dt

            # 3.7 删除中间变量（内存管理）
            del z, dit_hidden, output

    Step 4: 生成统计特征
        # 从池化结果计算7个统计量
        mean = pool["sum"] / pool["count"]           # [6144]
        std = sqrt(pool["sum_sq"]/count - mean^2)    # [6144]
        max_val = pool["max"]                        # [6144]
        min_val = pool["min"]                        # [6144]
        median = mean  # 近似
        p95 = mean + 1.645 * std
        p05 = mean - 1.645 * std

        # 合并为 [7, 6144]
        stats = stack([mean, std, max_val, min_val, median, p95, p05])

    Step 5: 保存到磁盘
        # 正样本
        save(activations/sae_layer15/sex/pos/activations.npy, stats[newaxis, ...])
        # [1, 7, 6144] = 168KB

        # 负样本（同样流程）
        save(activations/sae_layer15/sex/neg/activations.npy, stats[newaxis, ...])
```

### 1.3 输出文件结构

```
activations/                          # --output_root 指定的目录
├── extraction_config.json            # 全局配置（阶段二会读取）
│   {
│       "category": "sex",
│       "sae_run_dir": "sae_runs/exp1",
│       "hook_mode": "block_out",
│       "sae_layers": [15],
│       "timesteps": [29, 28, ..., 0],
│       "num_timesteps": 30,
│       "pool_activations": true,              # 关键标志
│       "activation_format": {
│           "shape": "[N, 7, d_hidden]",
│           "stats": ["mean", "std", "max", "min", "median", "p95", "p05"],
│           "description": "实时池化统计特征"
│       },
│       "seq_len": 32760,
│       "d_hidden": 6144
│   }
│
└── sae_layer15/                      # 每个SAE层一个目录
    └── sex/                          # 每个概念一个目录
        ├── pos/                      # 正样本
        │   ├── activations.npy       # [N_pos, 7, 6144] 主数据文件
        │   │                         # N_pos = 正样本对数
        │   │                         # 7 = 统计量数量
        │   │                         # 6144 = SAE d_hidden
        │   │
        │   ├── metadata.json         # 元信息列表
        │   │   [
        │   │       {
        │   │           "idx": 0,
        │   │           "pair_idx": 0,
        │   │           "prompt": "A naked person...",
        │   │           "category": "sex",
        │   │           "polarity": "pos"
        │   │       },
        │   │       ...
        │   │   ]
        │   │
        │   └── checkpoint.json       # 断点信息（支持断点续传）
        │       {
        │           "completed_pair_indices": [0, 1, 2, ...],
        │           "total_pairs": 200
        │       }
        │
        └── neg/                      # 负样本（结构相同）
            ├── activations.npy       # [N_neg, 7, 6144]
            ├── metadata.json
            └── checkpoint.json
```

### 1.4 数据格式详解

#### activations.npy 文件

```python
import numpy as np

# 加载数据
data = np.load("activations/sae_layer15/sex/pos/activations.npy")

print(data.shape)  # (100, 7, 6144)
#           │   │    └── SAE隐空间维度
#           │   └────── 7个统计量
#           └────────── 样本数 (100对提示词)

# 7个统计量的索引
STAT_MEAN    = 0  # 均值 - 主要使用这个
STAT_STD     = 1  # 标准差
STAT_MAX     = 2  # 最大值
STAT_MIN     = 3  # 最小值
STAT_MEDIAN  = 4  # 中位数（近似）
STAT_P95     = 5  # 95%分位数
STAT_P05     = 6  # 5%分位数

# 提取所有样本的均值特征
sample_means = data[:, STAT_MEAN, :]  # (100, 6144)

# 计算类别均值（阶段二的核心计算）
category_mean = sample_means.mean(axis=0)  # (6144,)
```

#### 内存占用对比

| 格式 | 形状 | 100样本内存 | 说明 |
|------|------|------------|------|
| **旧格式** | [100, 30, 32760, 6144] | **23 TB** | 不可能存下 |
| **实时池化** | [100, 7, 6144] | **16.8 MB** | ✅ 实际使用 |

---

## 二、阶段二：概念向量提取（CPU即可）

### 2.1 运行命令

```bash
python -m wan.sae.interpretability.concept_extractor_stage2 \
    --activation_root "activations" \
    --category "sex" \
    --layer_key "sae_layer15" \
    --output_dir "concept_vectors"
```

### 2.2 内部处理流程

```
Step 1: 读取阶段一配置
    config = load("activations/extraction_config.json")

    # 自动检测数据格式
    if config["pool_activations"]:
        logger.info("检测到实时池化格式 [N, 7, 6144]")
        logger.info("将使用第0维(mean)作为样本特征")

Step 2: 流式加载正样本
    pos_running_mean = RunningMean()  # Welford算法

    for batch in load_batches("activations/sae_layer15/sex/pos/activations.npy"):
        # batch: [32, 7, 6144] (batch_size=32)

        # 提取mean统计量（第0维）
        batch_means = batch[:, 0, :]  # [32, 6144]

        # 增量更新均值
        pos_running_mean.update(batch_means)

    pos_mean = pos_running_mean.get_mean()  # [6144]

Step 3: 流式加载负样本
    neg_running_mean = RunningMean()

    for batch in load_batches("activations/sae_layer15/sex/neg/activations.npy"):
        batch_means = batch[:, 0, :]  # [32, 6144]
        neg_running_mean.update(batch_means)

    neg_mean = neg_running_mean.get_mean()  # [6144]

Step 4: 计算概念向量
    concept_vector = pos_mean - neg_mean  # [6144]

    # 阈值过滤（去除噪声）
    concept_vector[abs(concept_vector) < 0.01] = 0

    # 归一化
    concept_vector = concept_vector / norm(concept_vector)

Step 5: 提取Top-K特征
    # 找出与概念最相关的SAE特征
    top_indices = argsort(abs(concept_vector))[-50:][::-1]

    top_k_features = [
        {"index": 445, "value": 0.234},
        {"index": 1203, "value": 0.198},
        ...
    ]

Step 6: 保存结果
    save("concept_vectors/sex_sae_layer15.npy", concept_vector)
    save("concept_vectors/sex_sae_layer15.json", {
        "concept_vector_shape": [6144],
        "norm": 1.0,
        "active_features": 523,
        "sparsity": 0.915,
        "top_k_features": [...],
        "source": {
            "stat_used": "mean",
            "activation_format": {...}
        }
    })
```

### 2.3 核心算法：RunningMean（Welford算法）

```python
class RunningMean:
    """
    增量计算均值，O(1)内存复杂度

    不需要保存所有样本，只维护：
    - count: 已处理样本数
    - mean: 当前均值
    """
    def __init__(self):
        self.count = 0
        self.mean = None

    def update(self, new_values):
        """
        new_values: [B, D] 批次数据
        """
        batch_count = new_values.shape[0]
        batch_mean = new_values.mean(axis=0)  # [D]

        if self.count == 0:
            self.mean = batch_mean
        else:
            # Welford更新公式
            delta = batch_mean - self.mean
            self.mean += delta * batch_count / (self.count + batch_count)

        self.count += batch_count

    def get_mean(self):
        return self.mean  # [D]
```

### 2.4 输出文件

```
concept_vectors/
├── sex_sae_layer15.npy          # [6144] 概念向量（NumPy数组）
│                                  # 可直接加载用于阶段三干预
│
└── sex_sae_layer15.json         # 元信息
    {
        "concept_name": "sex",
        "category": "sex",
        "layer_key": "sae_layer15",
        "layer_type": "sae",
        "layer_idx": 15,
        "method": "mean_diff",
        "vector_shape": [6144],
        "norm": 1.0,
        "active_features": 523,           # 非零特征数
        "sparsity": 0.9148,               # 91.48%稀疏度

        "top_k_features": [               # 最重要的50个特征
            {"index": 445, "value": 0.2341},
            {"index": 1203, "value": 0.1983},
            {"index": 2891, "value": -0.1856},  # 负值表示抑制
            ...
        ],

        "statistics": {
            "pos_count": 100,
            "neg_count": 100,
            "pos_mean_score": 0.4523,       # 正样本平均概念分数
            "neg_mean_score": -0.3841,      # 负样本平均概念分数
        },

        "source": {
            "activation_root": "activations",
            "num_pos": 100,
            "num_neg": 100,
            "stat_used": "mean",            # 明确标注使用mean
            "activation_format": {
                "shape": "[N, 7, d_hidden]",
                "stats": ["mean", "std", "max", "min", "median", "p95", "p05"]
            }
        },

        "extraction_time": "2025-04-20T10:30:00"
    }
```

---

## 三、阶段一→阶段二数据流转

### 3.1 可视化流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         阶段一 (GPU)                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  提示词对    │───→│ DiT+SAE编码  │───→│ 实时池化: [7, 6144]  │  │
│  │ (pos, neg)  │    │ 30×32760次  │    │ mean, std, max...   │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                │                 │
│                                                ▼                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  保存: activations/sae_layer15/sex/{pos,neg}/activations.npy │  │
│  │  格式: [N, 7, 6144]                                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ 加载
┌─────────────────────────────────────────────────────────────────┐
│                         阶段二 (CPU)                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  读取: activations.npy                                    │  │
│  │  提取: [:, 0, :] → [N, 6144] (mean统计量)                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  RunningMean流式处理                                      │  │
│  │  pos_mean = mean(all pos samples)  # [6144]              │  │
│  │  neg_mean = mean(all neg samples)  # [6144]              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  concept_vector = pos_mean - neg_mean  # [6144]          │  │
│  │  normalize → threshold → top_k                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  保存: concept_vectors/sex_sae_layer15.{npy,json}        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 关键数据形状变化

| 阶段 | 形状 | 含义 | 内存 |
|------|------|------|------|
| DiT输出 | [32760, 1536] | 单时间步所有token | 201 MB |
| SAE编码 | [32760, 6144] | 单时间步SAE激活 | 805 MB |
| 池化后 | [7, 6144] | 单样本统计特征 | 168 KB |
| 保存 | [1, 7, 6144] | 单样本（带batch维） | 168 KB |
| 批量 | [N, 7, 6144] | N个样本 | N×168 KB |
| 阶段二输入 | [N, 6144] | 取mean统计量 | N×24 KB |
| 阶段二输出 | [6144] | 概念向量 | 24 KB |

---

## 四、阶段三：实时干预（概念向量使用）

### 4.1 加载概念向量

```python
import numpy as np

# 加载阶段二生成的概念向量
concept_vector = np.load("concept_vectors/sex_sae_layer15.npy")  # [6144]

# 查看元信息
import json
with open("concept_vectors/sex_sae_layer15.json") as f:
    metadata = json.load(f)

print(f"概念: {metadata['concept_name']}")
print(f"活跃特征: {metadata['active_features']}/{metadata['vector_shape'][0]}")
print(f"Top特征: {metadata['top_k_features'][:3]}")
```

### 4.2 实时干预代码

```python
def intervene_dit_hidden(dit_hidden, sae, concept_vector, strength=0.5, threshold=0.1):
    """
    在DiT生成过程中进行概念干预

    Args:
        dit_hidden: [1, 32760, 1536] DiT隐藏状态
        sae: 训练好的SAE模型
        concept_vector: [6144] 概念向量（阶段二输出）
        strength: 干预强度
        threshold: 风险阈值

    Returns:
        corrected_hidden: 干预后的DiT隐藏状态
    """
    with torch.no_grad():
        # 1. SAE编码
        z, _, _ = sae.encode(dit_hidden.reshape(-1, 1536))  # [32760, 6144]

        # 2. 计算风险分数（与概念向量的相似度）
        risk_score = (z * concept_vector).sum(dim=-1).mean()

        # 3. 判断是否干预
        if risk_score > threshold:
            # 4. 沿概念向量反方向修正
            z_corrected = z - strength * concept_vector

            # 5. SAE解码回DiT空间
            dit_corrected = sae.decode(z_corrected)

            return dit_corrected.reshape(dit_hidden.shape)
        else:
            return dit_hidden  # 无需干预
```

---

## 五、完整命令总结

### 阶段一（GPU，约10小时/200对）

```bash
# 提取sex类别
python -m wan.sae.interpretability.concept_extractor_stage1 \
    --model_path "./Wan2.1-T2V-1.3B" \
    --sae_run_dir "sae_runs/exp1" \
    --pos_prompts "final_cleaned/pos_prompt_1.txt" \
    --neg_prompts "final_cleaned/neg_prompt_1.txt" \
    --category "sex" \
    --output_root "activations" \
    --sae_layers "15" \
    --sampling_steps 30

# 断点续传（如果中断）
python -m wan.sae.interpretability.concept_extractor_stage1 ... --resume
```

### 阶段二（CPU，约1分钟）

```bash
python -m wan.sae.interpretability.concept_extractor_stage2 \
    --activation_root "activations" \
    --category "sex" \
    --layer_key "sae_layer15" \
    --output_dir "concept_vectors" \
    --normalize \
    --min_threshold 0.01
```

### 阶段三（干预实验）

```python
# 在generate.py或自定义脚本中加载概念向量
concept_vector = np.load("concept_vectors/sex_sae_layer15.npy")

# 在DiT生成循环中调用intervene_dit_hidden()
```

---

## 六、常见问题

### Q1: 阶段一保存的7个统计量都有什么用？

| 统计量 | 阶段二使用 | 其他用途 |
|--------|-----------|---------|
| mean (0) | ✅ 主要使用 | 概念向量计算 |
| std (1) | ❌ | 可用于特征稳定性分析 |
| max (2) | ❌ | 异常值检测 |
| min (3) | ❌ | 异常值检测 |
| median (4) | ❌ | 分布对称性分析 |
| p95 (5) | ❌ | 动态范围分析 |
| p05 (6) | ❌ | 动态范围分析 |

### Q2: 为什么阶段二只用mean？

概念向量的定义是 **正负样本的分布差异**：

```
concept_vector = E[pos_samples] - E[neg_samples]
```

mean就是这个期望值的最优估计。

### Q3: 如果我想用其他统计量？

可以修改阶段二代码：

```python
# 使用std代替mean
batch_stds = batch[:, 1, :]  # 第1维是std

# 或使用max
batch_maxs = batch[:, 2, :]  # 第2维是max
```

但不推荐，因为mean最具代表性。

### Q4: 阶段一和阶段二可以分开跑吗？

**可以！** 这是设计目标：
- 阶段一必须在GPU上跑（需要DiT模型）
- 阶段二可以在任何CPU机器上跑（只需NumPy）

甚至可以：
1. 在服务器A（有GPU）跑阶段一
2. 把`activations/`目录复制到本地笔记本
3. 在笔记本上跑阶段二
4. 阶段三再回到服务器跑干预实验
