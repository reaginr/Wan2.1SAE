# SAE系统扩展模块概述

本文档概述了为满足新需求而开发的SAE系统扩展模块。

## 背景与动机

原有SAE系统存在以下限制：
1. **恢复训练不灵活**：无法通过配置文件直接加载ckpt并修改参数
2. **不支持离线训练**：必须在线运行DiT才能训练SAE，耗时且占用显存
3. **缺乏可解释性工具**：无法系统性地提取和分析概念向量
4. **缺少干预生成**：无法利用SAE概念向量干预视频生成

新的扩展解决了这些问题，同时保持与现有ckpt格式的兼容性。

---

## 修改1：增强恢复训练 (sae_train_t2v_1_3b.py)

### 新增功能
- 支持从指定目录恢复训练 (`--resume_from`)
- 支持新增层训练 (`--additional_layers`)
- 支持冻结已有层 (`--frozen_layers`)
- 支持重置优化器和步数 (`--reset_optimizer`, `--reset_step_count`)

### 使用示例

**1. 从其他目录恢复并修改参数**
```bash
python wan/sae_train_t2v_1_3b.py \
    --resume_from sae_runs/exp_layer15 \
    --run_dir sae_runs/exp_layer15_29 \
    --hook_layers "15,29" \
    --additional_layers "29" \
    --batch_prompts 8
```

**2. 冻结已有层，只训练新层**
```bash
python wan/sae_train_t2v_1_3b.py \
    --resume_from sae_runs/exp_layer15 \
    --hook_layers "15,29" \
    --additional_layers "29" \
    --frozen_layers "block_out.layer15" \
    --reset_optimizer
```

**3. 从其他实验加载特定层**
```bash
python wan/sae_train_t2v_1_3b.py \
    --run_dir sae_runs/exp_combined \
    --hook_layers "15,29" \
    --resume_from sae_runs/exp_style  # 层15从这里加载
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--resume_from` | 源实验目录 | `sae_runs/exp1` |
| `--additional_layers` | 新增层索引 | `"20,25"` |
| `--frozen_layers` | 冻结层key | `"block_out.layer15"` |
| `--reset_optimizer` | 重置优化器 | 标志参数 |
| `--reset_step_count` | 重置步数 | 标志参数 |

---

## 修改2：多层源测试 (sae_test_t2v_1_3b.py)

### 新增功能
- 支持从不同run_dir加载不同层的SAE

### 使用示例

**1. 命令行指定层源**
```bash
python wan/sae_test_t2v_1_3b.py \
    --run_dir sae_runs/exp_default \
    --hook_layers "15,29" \
    --layer_sources "15:sae_runs/exp_A,29:sae_runs/exp_B"
```

**2. 使用JSON配置文件**
```bash
python wan/sae_test_t2v_1_3b.py \
    --hook_layers "15,29" \
    --layer_sources layer_sources.json
```

`layer_sources.json`:
```json
{
  "block_out.layer15": "sae_runs/exp_style",
  "block_out.layer29": "sae_runs/exp_content"
}
```

---

## 模块2：离线训练 (offline_training/)

### 功能
- 在线采集激活值并保存到磁盘
- 从激活值文件离线训练SAE（无需运行DiT）
- 离线测试SAE性能

### 文件结构
```
offline_training/
├── activation_collector.py  # 激活值采集
├── train_offline.py         # 离线训练
├── test_offline.py          # 离线测试
└── README.md
```

### 使用流程

**Step 1: 采集激活值**
```bash
python -m wan.sae.offline_training.activation_collector \
    --checkpoint_dir ./Wan2.1-T2V-1.3B \
    --prompt_dir ./prompts \
    --output_dir offline_data/run1 \
    --hook_layers "15,29"
```

**Step 2: 离线训练**
```bash
python -m wan.sae.offline_training.train_offline \
    --data_dir offline_data/run1 \
    --run_dir sae_runs/offline_exp1 \
    --epochs 10 \
    --batch_size 4096
```

**Step 3: 离线测试**
```bash
python -m wan.sae.offline_training.test_offline \
    --data_dir offline_data/run1 \
    --run_dir sae_runs/offline_exp1
```

### 关键特性
- **快速训练**：无需运行DiT，直接训练SAE
- **灵活调参**：可快速调整学习率、batch_size等重新训练
- **存储优化**：支持numpy格式，体积小10-100倍
- **验证监控**：自动划分验证集

### 详细文档
参见 [offline_training/README.md](offline_training/README.md)

---

## 模块3：可解释性分析 (interpretability/)

### 功能
- 通过正负提示词对比提取概念向量
- 计算特征选择度和统计信息
- 保存概念向量用于后续干预

### 文件结构
```
interpretability/
├── concept_extractor.py     # 概念向量提取器
└── README.md
```

### 核心思想
```
concept_vector = mean(positive_activations) - mean(negative_activations)
```

正例（包含概念）和负例（不包含概念）在SAE激活上的差异即为概念方向。

### 使用方法

**1. 准备提示词**
```
concepts/violence_positive.txt   # 包含暴力概念的提示词
concepts/violence_negative.txt   # 不包含暴力概念的提示词
```

**2. 提取概念向量**
```bash
python -m wan.sae.interpretability.concept_extractor \
    --positive_file concepts/violence_positive.txt \
    --negative_file concepts/violence_negative.txt \
    --concept_name violence \
    --hook_layers "15,29"
```

**3. 输出格式**
```
concept_vectors/
├── violence_block_out.layer15.json  # 元信息和统计
├── violence_block_out.layer15.npy   # 向量数据
├── violence_block_out.layer29.json
└── violence_block_out.layer29.npy
```

### 提取方法
- **mean_diff**：平均差分法，简单直观
- **contrastive**：对比学习，更具判别性

### 统计指标
- Top-K最激活特征
- 特征选择度（selectivity）
- 正负例激活频率

### 详细文档
参见 [interpretability/README.md](interpretability/README.md)

---

## 模块4：干预生成 (steering/)

### 功能
- 加载已训练的SAE和概念向量
- 在视频生成过程中干预特定层
- 支持多个概念向量的组合干预
- 持久化干预配置和结果

### 文件结构
```
steering/
├── steering_generator.py    # 干预生成器
└── README.md
```

### 核心思想
```
# 在DiT生成过程中
1. Hook层输出特征
2. SAE编码: z = SAE.encode(feature)
3. 干预: z' = z + strength * concept_vector
4. SAE解码: feature' = SAE.decode(z')
5. 继续生成
```

### 使用方法

**1. 创建干预配置**
```json
{
  "prompt": "Two people arguing",
  "interventions": [
    {
      "concept_name": "violence",
      "layer_key": "block_out.layer15",
      "strength": -0.5,        // 负值抑制概念
      "method": "additive",
      "timestep_range": [0, 30]
    }
  ]
}
```

**2. 运行干预生成**
```bash
python -m wan.sae.steering.steering_generator \
    --config steering_config.json \
    --output_dir steering_outputs
```

**3. 输出**
```
steering_outputs/
├── steered_20250401_120000.mp4      # 生成的视频
└── session_20250401_120000.json     # 干预配置（持久化）
```

### 干预方法
- **additive**：加法干预 `z + strength * concept`
- **multiplicative**：乘法干预 `z * (1 + strength * concept)`
- **projection**：投影干预
- **clamp**：钳制干预

### 高级特性
- 多概念组合干预
- 动态强度衰减
- 时间步范围限制
- 特征掩码

### 详细文档
参见 [steering/README.md](steering/README.md)

---

## 模块间关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        SAE系统整体架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │ 在线训练 (原有) │───┐                                        │
│  └─────────────────┘   │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            增强配置系统 (enhanced_configs)               │    │
│  │  - 灵活恢复训练                                        │    │
│  │  - 参数修改                                            │    │
│  │  - 多层管理                                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                        │                                        │
│           ┌────────────┼────────────┐                          │
│           ▼            ▼            ▼                          │
│  ┌──────────────┐ ┌──────────┐ ┌──────────────┐               │
│  │ 离线训练     │ │ 在线训练 │ │ 测试/分析    │               │
│  │ offline_     │ │ (原有)   │ │ (原有)       │               │
│  │ training/    │ │          │ │              │               │
│  └──────────────┘ └──────────┘ └──────────────┘               │
│           │                                     │               │
│           ▼                                     ▼               │
│  ┌───────────────────┐            ┌──────────────────┐         │
│  │ 可解释性分析      │            │ SAE Checkpoint   │         │
│  │ interpretability/ │            │ (sae_latest.pt)  │         │
│  │                   │            └────────┬─────────┘         │
│  │ - 概念提取        │                     │                   │
│  │ - 向量保存        │                     ▼                   │
│  └─────────┬─────────┘            ┌──────────────────┐         │
│            │                       │ 概念向量         │         │
│            ▼                       │ (xxx.npy)        │         │
│     ┌────────────┐                 └────────┬─────────┘         │
│     │ 概念向量   │                          │                   │
│     │ (.npy)     │◄─────────────────────────┘                   │
│     └─────┬──────┘                                              │
│           │                                                     │
│           ▼                                                     │
│  ┌───────────────────┐                                          │
│  │ 干预生成          │                                          │
│  │ steering/         │                                          │
│  │                   │                                          │
│  │ - 干预DiT生成     │─────► 生成视频 + 配置持久化              │
│  │ - 多概念组合      │                                          │
│  └───────────────────┘                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 完整工作流示例

### 场景：检测和抑制暴力内容

**Step 1: 采集激活值（离线训练）**
```bash
# 采集训练数据
python -m wan.sae.offline_training.activation_collector \
    --prompt_dir ./nsfw_prompts \
    --output_dir offline_data/run1 \
    --hook_layers "15,29"
```

**Step 2: 离线训练SAE**
```bash
python -m wan.sae.offline_training.train_offline \
    --data_dir offline_data/run1 \
    --run_dir sae_runs/violence_exp \
    --epochs 10
```

**Step 3: 提取暴力概念向量**
```bash
# 准备正负提示词
# concepts/violence_positive.txt
# concepts/violence_negative.txt

python -m wan.sae.interpretability.concept_extractor \
    --run_dir sae_runs/violence_exp \
    --positive_file concepts/violence_positive.txt \
    --negative_file concepts/violence_negative.txt \
    --concept_name violence \
    --hook_layers "15,29"
```

**Step 4: 干预生成（抑制暴力）**
```bash
# 创建干预配置 steering_config.json
{
  "prompt": "Two people fighting",
  "interventions": [
    {
      "concept_name": "violence",
      "layer_key": "block_out.layer15",
      "strength": -0.6,        // 负值抑制
      "method": "additive"
    }
  ]
}

python -m wan.sae.steering.steering_generator \
    --config steering_config.json \
    --run_dir sae_runs/violence_exp \
    --concept_dir concept_vectors
```

**Step 5: 评估和迭代**
- 对比有/无干预的生成结果
- 调整干预强度和层
- 重复Step 4-5直到满意

---

## 快速参考

| 任务 | 命令 | 文档 |
|------|------|------|
| 增强配置恢复 | `python wan/sae_train_t2v_1_3b.py --config resume.json` | [enhanced_configs.md](enhanced_configs.md) |
| 采集激活值 | `python -m wan.sae.offline_training.activation_collector` | [offline_training/README.md](offline_training/README.md) |
| 离线训练 | `python -m wan.sae.offline_training.train_offline` | [offline_training/README.md](offline_training/README.md) |
| 离线测试 | `python -m wan.sae.offline_training.test_offline` | [offline_training/README.md](offline_training/README.md) |
| 提取概念 | `python -m wan.sae.interpretability.concept_extractor` | [interpretability/README.md](interpretability/README.md) |
| 干预生成 | `python -m wan.sae.steering.steering_generator` | [steering/README.md](steering/README.md) |

---

## 文件列表

新创建的文件：

```
wan/sae/
├── enhanced_configs.py              # 增强配置系统
├── enhanced_configs.md              # 配置系统文档
├── offline_training/
│   ├── __init__.py
│   ├── activation_collector.py      # 激活值采集
│   ├── train_offline.py             # 离线训练
│   ├── test_offline.py              # 离线测试
│   └── README.md
├── interpretability/
│   ├── __init__.py
│   ├── concept_extractor.py         # 概念向量提取
│   └── README.md
├── steering/
│   ├── __init__.py
│   ├── steering_generator.py        # 干预生成
│   └── README.md
└── NEW_MODULES_OVERVIEW.md          # 本文档
```

---

## 后续开发建议

1. **激活值可视化**
   - 添加 t-SNE/UMAP 降维可视化
   - 特征激活热力图

2. **概念向量数据库**
   - 管理多个概念向量
   - 支持向量相似度搜索

3. **自动化评估**
   - 干预效果的自动评估
   - A/B测试框架

4. **多GPU支持**
   - 离线训练的多GPU并行
   - 干预生成的批量处理

5. **Web界面**
   - 交互式概念提取
   - 可视化干预配置
