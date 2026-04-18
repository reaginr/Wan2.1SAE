# SAE可解释性分析模块 - 详细使用指南

## 目录
1. [快速开始](#快速开始)
2. [正负提示词设计](#正负提示词设计)
3. [两阶段工作流程](#两阶段工作流程)
4. [启动命令详解](#启动命令详解)
5. [中断处理与恢复](#中断处理与恢复)
6. [日志系统](#日志系统)
7. [概念向量的学术理解](#概念向量的学术理解)
8. [存储开销分析](#存储开销分析)
9. [常见问题FAQ](#常见问题faq)

---

## 快速开始

### 最小可运行示例

```bash
# 步骤1: 准备正负提示词（各至少20条）
mkdir -p concepts

cat > concepts/nsfw_positive.txt << 'EOF'
a nude woman standing in front of camera
explicit sexual content scene
naked body photography
adult content video
...
EOF

cat > concepts/nsfw_negative.txt << 'EOF'
a woman wearing formal business suit
family gathering photo
landscape mountain scenery
city street photography
...
EOF

# 步骤2: 两阶段工作流 - 阶段1: 采集激活值
python -m wan.sae.offline_training.activation_collector \
    --checkpoint_dir "./Wan2.1-T2V-1.3B" \
    --prompt_file concepts/nsfw_positive.txt \
    --output_path "activations/nsfw_positive.pt" \
    --hook_layers "15,29" \
    --batch_prompts 4

python -m wan.sae.offline_training.activation_collector \
    --checkpoint_dir "./Wan2.1-T2V-1.3B" \
    --prompt_file concepts/nsfw_negative.txt \
    --output_path "activations/nsfw_negative.pt" \
    --hook_layers "15,29" \
    --batch_prompts 4

# 步骤3: 两阶段工作流 - 阶段2: 提取概念向量
python -m wan.sae.interpretability.concept_extractor \
    --run_dir sae_runs/exp1 \
    --positive_activations activations/nsfw_positive.pt \
    --negative_activations activations/nsfw_negative.pt \
    --positive_prompts concepts/nsfw_positive.txt \
    --negative_prompts concepts/nsfw_negative.txt \
    --concept_name nsfw \
    --hook_layers "15,29" \
    --method mean_diff \
    --output_dir concept_vectors
```

---

## 正负提示词设计

### 为什么需要负向提示词？

**核心原理**: 概念向量 = 正例平均激活 - 负例平均激活

单纯使用正例只能得到"这些提示词激活了什么特征"，无法区分：
- 特定于概念的激活（只在正例出现）
- 通用激活（正负例都有）

负例的作用是**减去通用背景激活**，突出概念特异性。

### NSFW概念示例

**正向提示词（NSFW）**: 包含色情、裸露、性暗示内容
```
a nude woman in bedroom
explicit sexual scene
naked body photography
...
```

**负向提示词（非NSFW）**: 同类但不包含NSFW内容
```
a woman in business suit
family portrait photo
fashion model in dress
...
```

**关键原则**: 负例应与正例在**其他维度相似**（都有人、场景、摄影风格），仅在**目标概念维度不同**（是否NSFW）。

### 数量对应要求

| 场景 | 正例数量 | 负例数量 | 说明 |
|-----|---------|---------|-----|
| 最小可行 | 20 | 20 | 能运行，但统计不稳定 |
| 推荐 | 50-100 | 50-100 | 数量不必严格相等 |
| 高质量 | 200+ | 200+ | 概念边界更清晰 |

**重要**: 正负例**不需要一一对应**。不需要 "A+" 对应 "A-"，只需要：
- 正例集合整体体现目标概念
- 负例集合整体排除目标概念
- 两者在其他维度分布相似

### 负例设计策略

1. **随机采样**（最简单）
   - 从训练数据中随机采样
   - 假设随机样本大概率不包含目标概念

2. **同类排斥**（推荐）
   - 正例: "nude woman"
   - 负例: "woman in dress"（同类但有衣物）

3. **反义对照**
   - 正例: "violent scene"
   - 负例: "peaceful scene"

---

## 两阶段工作流程

### 阶段1: 激活值采集（前向传播）

**目的**: 获取提示词在SAE中的激活值

**特点**:
- 需要GPU（DiT前向传播）
- 耗时较长（每个提示词约10-30秒）
- 存储开销大（见下文分析）
- 可中断恢复

**输出**: `.pt` 文件，包含 `[N, L, C]` 的原始激活值

```bash
python -m wan.sae.offline_training.activation_collector \
    --checkpoint_dir "./Wan2.1-T2V-1.3B" \
    --prompt_file concepts/nsfw_positive.txt \
    --output_path "activations/nsfw_positive.pt" \
    --hook_mode block_out \
    --hook_layers "15,29" \
    --size_w 512 \
    --size_h 288 \
    --frame_num 17 \
    --batch_prompts 4 \
    --device_id 0
```

### 阶段2: 概念向量提取

**目的**: 从激活值计算概念向量

**特点**:
- 纯CPU计算（矩阵运算）
- 非常快（秒级）
- 存储开销小（仅保存向量）
- 可快速迭代不同提取方法

**输入**: 阶段1的 `.pt` 文件 + 提示词文件

**输出**:
- `concept_vectors/nsfw_block_out.layer15.json` - 元信息
- `concept_vectors/nsfw_block_out.layer15.npy` - 概念向量

```bash
python -m wan.sae.interpretability.concept_extractor \
    --run_dir sae_runs/exp1 \
    --positive_activations activations/nsfw_positive.pt \
    --negative_activations activations/nsfw_negative.pt \
    --positive_prompts concepts/nsfw_positive.txt \
    --negative_prompts concepts/nsfw_negative.txt \
    --concept_name nsfw \
    --hook_layers "15,29" \
    --method mean_diff \
    --normalize True \
    --min_activation_threshold 0.01 \
    --output_dir concept_vectors
```

### 为什么要分两阶段？

| 优势 | 说明 |
|-----|------|
| **存储灵活** | 可以删除原始激活值，只保留概念向量 |
| **快速迭代** | 换提取方法/参数无需重新前向传播 |
| **多概念复用** | 同一批激活值可提取多个不同概念 |
| **分布式采集** | 激活值可在多台机器上并行采集 |

---

## 启动命令详解

### 完整参数列表

```bash
python -m wan.sae.interpretability.concept_extractor \
    # === 输入配置 ===
    --positive_activations PATH      # 正例激活值文件 (.pt)
    --negative_activations PATH      # 负例激活值文件 (.pt)
    --positive_prompts PATH          # 正例提示词文件 (.txt)
    --negative_prompts PATH          # 负例提示词文件 (.txt)

    # === SAE配置 ===
    --run_dir DIR                    # SAE训练目录
    --hook_mode MODE                 # hook模式 (默认: block_out)
    --hook_layers "15,29"            # 要分析的层

    # === 提取方法 ===
    --method METHOD                  # mean_diff | contrastive
    --use_abs BOOL                   # 是否取绝对值 (默认: False)
    --normalize BOOL                 # 是否归一化 (默认: True)
    --min_activation_threshold FLOAT # 最小激活阈值 (默认: 0.01)
    --contrastive_temp FLOAT         # 对比学习温度 (默认: 0.1)

    # === 统计配置 ===
    --top_k_features INT             # 保存top-k特征 (默认: 50)
    --compute_selectivity BOOL       # 计算选择度 (默认: True)
    --selectivity_threshold FLOAT    # 选择度阈值 (默认: 0.7)

    # === 输出配置 ===
    --output_dir DIR                 # 输出目录 (默认: concept_vectors)
    --concept_name NAME              # 概念名称

    # === 系统配置 ===
    --device_id INT                  # GPU设备ID (默认: 0)
    --seed INT                       # 随机种子 (默认: 0)
```

### 典型使用场景

**场景1: 快速测试**
```bash
# 最小配置，仅分析单层
python -m wan.sae.interpretability.concept_extractor \
    --positive_activations act/pos.pt \
    --negative_activations act/neg.pt \
    --positive_prompts pos.txt \
    --negative_prompts neg.txt \
    --concept_name test \
    --hook_layers "15" \
    --output_dir ./test_output
```

**场景2: 对比学习提取**
```bash
# 使用对比学习方法，适合边界模糊的概念
python -m wan.sae.interpretability.concept_extractor \
    --positive_activations act/pos.pt \
    --negative_activations act/neg.pt \
    --positive_prompts pos.txt \
    --negative_prompts neg.txt \
    --concept_name violence \
    --hook_layers "15,29" \
    --method contrastive \
    --contrastive_temp 0.05 \
    --output_dir concept_vectors
```

**场景3: 批量处理多个概念**
```bash
# 使用相同激活值提取多个概念
for concept in nsfw violence gore weapons; do
    python -m wan.sae.interpretability.concept_extractor \
        --positive_activations "activations/${concept}_positive.pt" \
        --negative_activations "activations/common_negative.pt" \
        --positive_prompts "concepts/${concept}_positive.txt" \
        --negative_prompts "concepts/common_negative.txt" \
        --concept_name $concept \
        --hook_layers "15,29" \
        --output_dir concept_vectors &
done
wait
```

---

## 中断处理与恢复

### 阶段1采集时的中断

**问题**: 激活值采集耗时很长（100条提示词约30-60分钟），可能中断。

**当前实现**: 采集模块支持断点续传（需查看具体实现）。

**手动恢复策略**:
```bash
# 1. 检查已完成的提示词
python -c "
import torch
data = torch.load('activations/nsfw_positive.pt')
print(f'已完成: {len(data)} 条')
"

# 2. 从断点继续（需修改提示词文件，删除已完成的部分）
# 或使用支持断点续传的采集脚本
```

**建议**: 对于长任务，使用 `screen` 或 `tmux` 保持会话：
```bash
screen -S collect_activations
python -m wan.sae.offline_training.activation_collector ...
# Ctrl+A, D  detach
# screen -r collect_activations  恢复
```

### 阶段2提取时的中断

概念提取很快（秒级），通常不需要中断恢复。如果中断，直接重新运行即可。

---

## 日志系统

### 日志保存位置

```
sae_runs/exp1/                          # SAE训练目录
└── logs/
    └── analysis/                       # 可解释性分析日志
        ├── run.log                     # 主日志（文本）
        ├── metrics.jsonl               # 指标日志（结构化）
        ├── events.jsonl                # 事件日志
        ├── results.jsonl               # 结果日志
        ├── summary.json                # 总结报告
        └── config.json                 # 配置记录
```

### 日志文件格式

**run.log** (文本，人类可读)
```
2026-04-14 20:30:15 - sae_analysis - INFO - 提取概念 'nsfw' 从层 block_out.layer15
2026-04-14 20:30:15 - sae_analysis - INFO - 正例: 100, 负例: 100
2026-04-14 20:30:15 - sae_analysis - DEBUG - [LAYER] 正例激活[0] shape: (81, 2560, 1536)
2026-04-14 20:30:16 - sae_analysis - INFO - 已加载SAE: d_model=1536, d_hidden=6144
2026-04-14 20:30:16 - sae_analysis - DEBUG - [EXTRACT] mean_diff 开始: 正例=100, 负例=100
2026-04-14 20:30:18 - sae_analysis - DEBUG - [EXTRACT] mean_diff 完成: 输出shape=(6144,)
2026-04-14 20:30:18 - sae_analysis - INFO - 概念向量提取完成！
2026-04-14 20:30:18 - sae_analysis - INFO -   向量范数: 1.0000
2026-04-14 20:30:18 - sae_analysis - INFO -   Top-5特征: [3421, 1892, 4501, 1023, 5567]
2026-04-14 20:30:18 - sae_analysis - INFO -   Top-5值: [0.1523, 0.1345, 0.1289, 0.1156, 0.1098]
```

**metrics.jsonl** (JSON行，结构化)
```json
{"timestamp": 1713108615.123, "datetime": "2026-04-14T20:30:15.123456", "step": 0, "layer_key": "block_out.layer15", "loss": 0.0, "sparsity": 0.0156}
{"timestamp": 1713108618.456, "datetime": "2026-04-14T20:30:18.456789", "step": 0, "layer_key": "block_out.layer15", "extraction_time": 2.333, "vector_norm": 1.0}
```

**results.jsonl** (逐条结果)
```json
{"timestamp": 1713108618.456, "result_id": "nsfw_layer15", "concept_name": "nsfw", "vector_norm": 1.0, "top_k_indices": [3421, 1892, 4501], "top_k_values": [0.1523, 0.1345, 0.1289]}
```

**summary.json** (总结)
```json
{
  "total_prompts": 200,
  "positive_count": 100,
  "negative_count": 100,
  "extraction_method": "mean_diff",
  "vector_norm": 1.0,
  "top_k_features": 50,
  "high_selectivity_count": 23
}
```

### 启用详细调试日志

```bash
# 方法1: 设置环境变量
export SAE_LOG_LEVEL=DEBUG
python -m wan.sae.interpretability.concept_extractor ...

# 方法2: 修改代码中的logger级别（在脚本开头）
import logging
logging.getLogger().setLevel(logging.DEBUG)

# 方法3: 使用 verbose 参数（如果脚本支持）
python -m wan.sae.interpretability.concept_extractor --verbose ...
```

### 查看实时日志

```bash
# 终端1: 运行任务
python -m wan.sae.interpretability.concept_extractor ...

# 终端2: 实时查看日志
tail -f sae_runs/exp1/logs/analysis/run.log

# 或使用 grep 过滤关键信息
tail -f sae_runs/exp1/logs/analysis/run.log | grep "EXTRACT\|LAYER\|ERROR"
```

---

## 概念向量的学术理解

### 什么是概念向量？

概念向量是在SAE隐空间中的一个方向向量，沿着这个方向移动会增强/减弱目标概念在生成内容中的表现。

**数学定义**:
```
concept_vector = E[z | positive] - E[z | negative]
```

其中 z 是SAE编码后的稀疏表示（维度 = d_hidden）。

### mean_diff 方法详解

```python
# 1. 对每个正例提示词获取激活
pos_activations = []
for prompt in positive_prompts:
    # DiT forward 获取隐藏状态 h
    h = dit_forward(prompt)  # [L, C]
    # SAE 编码
    z = sae.encode(h)  # [L, d_hidden]
    # 平均池化
    z_mean = z.mean(dim=0)  # [d_hidden]
    pos_activations.append(z_mean)

# 2. 计算正例平均
pos_mean = np.mean(pos_activations, axis=0)  # [d_hidden]

# 3. 同理计算负例平均
neg_mean = np.mean(neg_activations, axis=0)  # [d_hidden]

# 4. 差分 = 概念向量
concept_vector = pos_mean - neg_mean  # [d_hidden]

# 5. 阈值过滤（去除噪声）
concept_vector[abs(concept_vector) < threshold] = 0

# 6. 归一化
concept_vector = concept_vector / np.linalg.norm(concept_vector)
```

### 干预原理

概念向量可以用于**干预生成**：

```python
# 正常生成
z = sae.encode(h)

# 概念增强生成（沿着概念向量方向移动）
z_enhanced = z + strength * concept_vector

# 概念抑制生成（反方向移动）
z_suppressed = z - strength * concept_vector
```

`strength` 是干预强度，越大效果越明显。

### 特征选择度 (Selectivity)

**定义**:
```
selectivity_i = P(z_i > 0 | positive) - P(z_i > 0 | negative)
```

- 选择度 ≈ 1.0: 特征几乎只在正例激活（概念特异性特征）
- 选择度 ≈ 0.0: 特征在正负例都激活（通用特征）
- 选择度 ≈ -1.0: 特征几乎只在负例激活（反向特征）

**用途**: 识别哪些特征真正编码了目标概念。

---

## 存储开销分析

### 阶段1: 激活值存储

**计算公式**:
```
存储大小 = N × L × C × 4 bytes

N = 提示词数量
L = token数量 ≈ (H/16) × (W/16) × (F/4)  # 取决于尺寸
C = 隐藏维度 = 1536 (1.3B模型)
4 = float32字节数
```

**示例**: 100条提示词，832×480，81帧
```
L ≈ (832/16) × (480/16) × (81/4) ≈ 52 × 30 × 20 ≈ 31,200 tokens
存储 = 100 × 31,200 × 1536 × 4 bytes
     = 19,200,000,000 bytes
     ≈ 17.9 GB
```

**不同配置的存储开销**:

| 提示词数 | 尺寸 | 帧数 | Token数 | 存储大小 |
|---------|------|------|--------|---------|
| 100 | 512×288 | 17 | ~3,000 | ~1.8 GB |
| 100 | 832×480 | 81 | ~31,000 | ~17.9 GB |
| 500 | 512×288 | 17 | ~3,000 | ~9.0 GB |

**优化策略**:
1. 减小尺寸: `--size_w 512 --size_h 288`
2. 减少帧数: `--frame_num 17`
3. 仅保留关键层激活（而非所有层）
4. 使用 float16: `--dtype float16`

### 阶段2: 概念向量存储

**计算公式**:
```
存储大小 = d_hidden × 4 bytes

# 1.3B模型
= 6144 × 4 = 24,576 bytes ≈ 24 KB
```

**极其微小**，可以忽略不计。

### 对比总结

| 阶段 | 100条提示词存储 | 可删除？ | 用途 |
|-----|---------------|---------|-----|
| 原始视频 | ~50 GB | ✓ | 无需保存 |
| 激活值 (.pt) | ~18 GB | ✓ | 提取概念后可删 |
| 概念向量 (.npy) | ~24 KB | ✗ | 干预生成必需 |
| 元信息 (.json) | ~50 KB | ✗ | 分析和记录 |

**推荐策略**:
- 激活值采集后提取概念向量
- 验证概念向量有效后删除激活值
- 保留概念向量用于后续干预生成

---

## 常见问题FAQ

### Q1: 正负例数量必须相等吗？

**不需要**。两者数量可以不等，但建议都在50-200范围内。

数量不平衡时的处理:
```python
# mean_diff 自动处理不等数量
pos_mean = np.mean(pos_activations, axis=0)  # 不管多少条
neg_mean = np.mean(neg_activations, axis=0)  # 不管多少条
```

### Q2: 可以只用正例吗？

**技术上可以，但不推荐**。

仅用正例:
```python
#  naive approach (不推荐)
concept_vector = mean(positive_activations)
```

问题:
- 包含大量通用激活（如"画面"、"颜色"）
- 无法区分概念特异性特征

替代方案（无负例时）:
- 使用随机样本作为负例
- 使用数据集的总体统计作为baseline

### Q3: 如何验证概念向量的有效性？

**方法1: 方向一致性验证**
```python
# 正例在概念方向上投影应为正
for act in positive_activations:
    proj = np.dot(act, concept_vector)
    assert proj > 0, "正例投影应为正"

# 负例在概念方向上投影应为负或较小
for act in negative_activations:
    proj = np.dot(act, concept_vector)
    assert proj < threshold, "负例投影应较小"
```

**方法2: 干预生成验证**
```bash
# 使用steering模块测试
python -m wan.sae.steering.generate \
    --concept_vector concept_vectors/nsfw.npy \
    --strength 0.5 \
    --prompt "a woman in red dress"
# 观察生成结果是否更NSFW
```

**方法3: 可视化Top特征**
```python
# 查看概念向量中最激活的特征
# 在训练集中找到激活这些特征的示例图像/视频
# 人工检查是否确实与概念相关
```

### Q4: 提取的概念向量可以跨层使用吗？

**不可以**。每个层的概念向量是特定于该层的：
- `block_out.layer15` 提取的向量只能用于 layer15
- 不同层的SAE有不同权重，隐空间不共享

如果要干预多层，需要为每层分别提取或使用不同的干预策略。

### Q5: 如何处理模糊样本？

有些提示词可能难以明确分类（如"艺术人体摄影"对NSFW概念）：

**策略1: 严格筛选**
- 只保留明确属于正例或负例的样本
- 丢弃模糊样本

**策略2: 软标签**
- 给样本打0-1的分数（如"可能是NSFW的程度"）
- 使用加权平均（当前实现不支持，需修改）

**策略3: 多轮迭代**
- 第一轮提取概念向量
- 用概念向量给所有样本打分
- 移除得分在中等区间（如0.3-0.7）的模糊样本
- 重新提取

### Q6: 为什么提取的概念向量范数不为1？

如果设置了 `--normalize True`，范数应该为1。

不归一化的情况:
```python
# 不归一化时，范数反映概念"强度"
norm = np.linalg.norm(concept_vector)
# 范数越大，正负例差异越大，概念越"清晰"
```

### Q7: 可以提取复合概念吗？

**线性组合法**:
```python
# 组合多个概念
nsfw_vector = np.load("nsfw.npy")
violence_vector = np.load("violence.npy")
gore_vector = np.load("gore.npy")

# 加权组合
combined = 0.5 * nsfw_vector + 0.3 * violence_vector + 0.2 * gore_vector
combined = combined / np.linalg.norm(combined)  # 重新归一化
```

**注意**: 这种方法假设概念是正交的，实际上可能有重叠。

---

## 相关资源

- [SAE训练指南](../SAE_CHECKPOINT_GUIDE.md)
- [离线训练模块](../offline_training/README.md)
- [干预生成模块](../steering/README.md)
- [SAE论文合集](https://docs.google.com/document/d/1s5QLJ2k7dx4N9UZsQcNn5fXPIb3_v5i_s8bNpZiKLhQ/edit?tab=t.0)

---

## 附录: 完整两阶段脚本模板

```bash
#!/bin/bash
# extract_concept.sh - 完整概念提取流程

set -e  # 出错时停止

# 配置
CONCEPT_NAME="nsfw"
POSITIVE_PROMPTS="concepts/nsfw_positive.txt"
NEGATIVE_PROMPTS="concepts/nsfw_negative.txt"
CHECKPOINT_DIR="./Wan2.1-T2V-1.3B"
SAE_RUN_DIR="sae_runs/exp1"
OUTPUT_DIR="concept_vectors"
HOOK_LAYERS="15,29"

# 参数
SIZE_W=512
SIZE_H=288
FRAME_NUM=17
BATCH_PROMPTS=4

echo "=== 阶段1: 采集正例激活值 ==="
python -m wan.sae.offline_training.activation_collector \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --prompt_file "$POSITIVE_PROMPTS" \
    --output_path "activations/${CONCEPT_NAME}_positive.pt" \
    --hook_layers "$HOOK_LAYERS" \
    --size_w $SIZE_W \
    --size_h $SIZE_H \
    --frame_num $FRAME_NUM \
    --batch_prompts $BATCH_PROMPTS

echo "=== 阶段1: 采集负例激活值 ==="
python -m wan.sae.offline_training.activation_collector \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --prompt_file "$NEGATIVE_PROMPTS" \
    --output_path "activations/${CONCEPT_NAME}_negative.pt" \
    --hook_layers "$HOOK_LAYERS" \
    --size_w $SIZE_W \
    --size_h $SIZE_H \
    --frame_num $FRAME_NUM \
    --batch_prompts $BATCH_PROMPTS

echo "=== 阶段2: 提取概念向量 ==="
python -m wan.sae.interpretability.concept_extractor \
    --run_dir "$SAE_RUN_DIR" \
    --positive_activations "activations/${CONCEPT_NAME}_positive.pt" \
    --negative_activations "activations/${CONCEPT_NAME}_negative.pt" \
    --positive_prompts "$POSITIVE_PROMPTS" \
    --negative_prompts "$NEGATIVE_PROMPTS" \
    --concept_name "$CONCEPT_NAME" \
    --hook_layers "$HOOK_LAYERS" \
    --method mean_diff \
    --output_dir "$OUTPUT_DIR"

echo "=== 完成 ==="
echo "概念向量保存在: ${OUTPUT_DIR}/${CONCEPT_NAME}_block_out.layer*.npy"
echo "查看日志: ${SAE_RUN_DIR}/logs/analysis/run.log"
```

运行:
```bash
chmod +x extract_concept.sh
./extract_concept.sh 2>&1 | tee extraction.log
```
