# SAE概念提取器架构说明

## 概述

SAE概念提取采用**两阶段离线分离式设计**，将激活值采集与概念向量提取完全解耦。

## 文件结构

```
wan/sae/interpretability/
├── activation_io.py               # 统一的激活值文件I/O接口
├── concept_extractor_stage1.py    # 阶段一：激活值采集（GPU必需）
├── concept_extractor_stage2.py    # 阶段二：概念向量提取（CPU即可）
├── concept_extractor_offline.py   # [已弃用] 单文件版，保留向后兼容
├── visualize_activations.py       # 可视化工具
└── __init__.py                    # 模块导出接口
```

## 架构对比

### 旧版（单文件）
```
┌─────────────────────────────────────────────────────┐
│          concept_extractor_offline.py               │
│  ┌───────────────────────────────────────────────┐  │
│  │           阶段一: 采集激活值                   │  │
│  │  - 加载WanT2V模型                              │  │
│  │  - 加载SAE模型                                 │  │
│  │  - 运行DiT前向传播                             │  │
│  │  - Hook收集DiT状态                             │  │
│  │  - SAE编码                                     │  │
│  │  - 保存.npz文件                                │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │           阶段二: 提取概念向量                 │  │
│  │  - 流式加载.npz文件                            │  │
│  │  - mean_diff计算概念向量                       │  │
│  │  - 保存结果                                    │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 新版（分离式）
```
阶段一服务器（GPU）                    阶段二工作站（CPU）
┌──────────────────────────────┐      ┌──────────────────────────────┐
│ concept_extractor_stage1.py  │      │ concept_extractor_stage2.py  │
│                              │      │                              │
│ - 加载WanT2V模型              │      │ - 无需模型加载                │
│ - 加载SAE模型                 │      │ - ActivationIO流式读取       │
│ - 前向传播 + Hook            │      │ - 纯NumPy向量运算            │
│ - DiT/SAE状态编码            │─────▶│ - mean_diff提取概念向量      │
│ - ActivationIO保存           │      │ - 保存结果                    │
└──────────────────────────────┘      └──────────────────────────────┘
         │                                      │
         │    传输目录结构（sae_layer/...）       │
         └──────────────────────────────────────┘
```

## 设计优势

### 1. 完全解耦
- **阶段一**依赖PyTorch、CUDA、Wan2.1模型，必须在GPU服务器运行
- **阶段二**仅需NumPy，可在任何CPU工作站运行，无需模型
- 两阶段通过标准.npz文件交互，接口清晰

### 2. 内存效率
- 阶段二使用`numpy.memmap`内存映射，无需加载整个文件到内存
- `RunningMean`增量计算，O(1)内存复杂度处理任意大数据集
- 支持流式批处理，内存使用可控

### 3. 灵活性
- 阶段一采集一次，可被多个阶段二任务复用（不同概念、不同层）
- 阶段二可离线反复实验，无需重新运行昂贵的模型推理
- 支持分布式采集（多台GPU服务器采集，集中到一台CPU服务器分析）

## 数据流

```
正向提示词 ──┐
             │    ┌─────────────┐    ┌──────────────────┐    ┌────────────────┐
负向提示词 ──┼───▶│ 阶段一采集  │───▶│ activations/     │───▶│ 阶段二提取     │
             │    │ (GPU必需)   │    │   sae_layer15/   │    │ (CPU即可)      │
时间步配置 ──┤    │             │    │   └── violence/  │    │                │
层配置    ───┤    │ - WanT2V    │    │       ├── pos/   │    │ - 流式加载     │
SAE路径   ───┘    │ - SAE编码   │    │       └── neg/   │    │ - mean_diff    │
                  └─────────────┘    └──────────────────┘    │ - 阈值过滤     │
                                                             │ - 归一化       │
                                                             └───────┬────────┘
                                                                     │
                                                             ┌───────▼────────┐
                                                             │ concept_vector │
                                                             │   .npy/.json   │
                                                             └────────────────┘
```

## 使用示例

### 阶段一：采集（GPU服务器）

阶段一使用**配对处理**，同时处理正负提示词，保证样本对齐：

```bash
python wan/sae/interpretability/concept_extractor_stage1.py \
    --model_path "./Wan2.1-T2V-1.3B" \
    --sae_run_dir "sae_runs/exp1" \
    --pos_prompts "final_cleaned/pos_prompt_3.txt" \
    --neg_prompts "final_cleaned/neg_prompt_3.txt" \
    --category "violence" \
    --output_root "activations" \
    --sae_layers "15,29" \
    --save_dit_layers "15" \
    --sampling_steps 30 \
    --use_cfg \
    --guide_scale 5.0
```

生成的目录结构：
```
activations/
├── sae_layer15/
│   └── violence/
│       ├── pos/
│       │   ├── activations.npy      # [N, T, L, d_hidden]
│       │   ├── metadata.json
│       │   └── checkpoint.json
│       └── neg/
│           ├── activations.npy
│           └── metadata.json
├── dit_layer15/                     # 可选的DiT状态
└── extraction_config.json           # 全局配置
```

### 阶段二：提取（CPU工作站）

```bash
python wan/sae/interpretability/concept_extractor_stage2.py \
    --activation_root "activations" \
    --category "violence" \
    --layer_key "sae_layer15" \
    --output_dir "concept_vectors" \
    --method "mean_diff" \
    --normalize \
    --min_threshold 0.01
```

输出文件：
- `concept_vectors/violence_sae_layer15.npy` - 概念向量 [d_hidden]
- `concept_vectors/violence_sae_layer15.json` - 元信息和统计

## 文件格式

### 阶段一输出（分层目录结构）

```
{root}/
├── {layer_type}_layer{idx}/           # 如 sae_layer15, dit_layer15
│   └── {category}/                    # 如 violence
│       ├── pos/
│       │   ├── activations.npy        # [N, T, L, D] 激活值数组
│       │   ├── metadata.json          # 样本元信息列表
│       │   └── checkpoint.json        # 增量采集断点
│       └── neg/
│           ├── activations.npy
│           └── metadata.json
└── extraction_config.json             # 全局配置
```

**activations.npy**: [N, T, L, D] 数组
- N: 样本数量
- T: 时间步数量
- L: token数量（空间位置）
- D: 特征维度（SAE d_hidden 或 DiT dim）

**metadata.json**:
```json
[
  {"idx": 0, "pair_idx": 0, "prompt": "...", "category": "violence", "polarity": "pos"},
  {"idx": 1, "pair_idx": 1, "prompt": "...", "category": "violence", "polarity": "pos"}
]
```

### 阶段二输出

**概念向量文件** (.npy): [d_hidden] NumPy数组

**元信息文件** (.json):
```json
{
    "concept_name": "violence",
    "category": "violence",
    "layer_key": "sae_layer15",
    "layer_type": "sae",
    "layer_idx": 15,
    "method": "mean_diff",
    "vector_shape": [6144],
    "norm": 1.0,
    "top_k_features": [{"index": 0, "value": 0.5}, ...],
    "statistics": {
        "pos_count": 200,
        "neg_count": 200,
        "active_features": 150,
        "total_features": 6144,
        "sparsity": 0.9756,
        "norm_before_normalize": 15.234,
        "norm_after_normalize": 1.0
    },
    "parameters": {
        "normalize": true,
        "min_threshold": 0.01,
        "batch_size": 32
    }
}
```

## 接口调用

### Python API

```python
# 阶段一：采集
from wan.sae.interpretability import PairedActivationCollector

collector = PairedActivationCollector(
    model_path="./Wan2.1-T2V-1.3B",
    sae_run_dir="sae_runs/exp1",
    hook_mode="block_out",
    sae_layers=[15, 29],
    save_dit_layers=[15],
    device="cuda:0",
)

# 配对处理正负提示词
collector.collect_paired(
    pos_prompts=["positive prompt 1", "positive prompt 2"],
    neg_prompts=["negative prompt 1", "negative prompt 2"],
    category="violence",
    output_root="activations",
    num_timesteps=30,
)


# 阶段二：提取
from wan.sae.interpretability import ConceptExtractor, ActivationIO

# 使用 ActivationIO 读取保存的激活值
io = ActivationIO("activations")

# 创建提取器
extractor = ConceptExtractor(
    io=io,
    category="violence",
    layer_type="sae",
    layer_idx=15,
    method="mean_diff",
    normalize=True,
    min_threshold=0.01,
)

# 提取概念向量
concept_vector, statistics = extractor.extract()

# 获取Top-K特征
top_k = extractor.get_top_k_features(concept_vector, k=50)
```

## 注意事项

1. **阶段一和阶段二必须在相同层上操作**
   - 阶段一保存的层必须与阶段二指定的`layer_key`匹配
   - 阶段一使用 `--sae_layers "15,29"` 指定要保存的层
   - 阶段二使用 `--layer_key "sae_layer15"` 指定要提取的层

2. **存储结构变化**
   - 新版使用分层目录结构而非单个 `.npz` 文件
   - 便于增量采集和增量处理
   - 支持内存映射流式读取，处理大文件不OOM

3. **时间步配置**
   - 阶段一采集完整扩散轨迹（所有时间步）
   - 概念向量在时间步和token维度上取平均
   - 默认30步，与训练时一致

4. **内存控制**
   - 阶段二使用`batch_size`参数控制内存使用
   - 对于大文件，调整`batch_size`以适应可用内存
   - 使用`mmap=True`进行内存映射读取

5. **增量采集**
   - 阶段一支持 `--resume` 断点续传
   - 断点信息保存在每个极性目录的 `checkpoint.json` 中

## 迁移指南

从旧版`.npz`格式迁移：

| 旧版 | 新版 |
|------|------|
| 单个 `.npz` 文件 | 分层目录结构 `{layer}/{category}/{polarity}/` |
| `JointActivationCollector` | `PairedActivationCollector` |
| `StreamingActivationLoader` | `ActivationIO` + `ConceptExtractor` |
| `--prompt_file`（单文件） | `--pos_prompts` + `--neg_prompts`（配对文件） |
| `--output_path`（文件路径） | `--output_root`（根目录） |
| `--concept_name`（阶段二） | `--category`（统一命名） |

主要优势：
- 配对处理保证样本对齐
- 分层存储便于管理多个概念和层
- 内存映射支持大文件流式处理
- 增量采集避免重复计算
