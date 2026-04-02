# SAE Checkpoint 参数配置与使用指南

本指南详细介绍 SAE 训练和测试脚本中的 checkpoint 参数配置，包括恢复训练、跨实验加载、多层源配置等高级用法。

> **版本 2.0 更新**: Checkpoint 格式已升级，配置内置到 `.pt` 文件中，不再依赖 `.json` 文件。本文档同时涵盖新旧格式的兼容处理。

---

## 参数命名规范

| 参数名称 | 用途 | 示例值 |
|---------|------|--------|
| `model_path` | Wan 2.1 DiT 基础模型权重目录 | `./Wan2.1-T2V-1.3B` |
| `run_dir` | SAE 训练输出目录 | `sae_runs/exp_20250324` |
| `sae_checkpoint` | SAE checkpoint 路径或目录（可选） | `sae_runs/exp1` 或具体 `.pt` 文件 |

**注意**: `--checkpoint_dir` 参数已弃用，请使用 `--model_path`。

## 恢复训练配置自动加载机制

### 架构配置优先级

当恢复训练时，SAE 架构参数（d_model, d_hidden, top_k 等）按以下优先级加载：

1. **从 checkpoint 的 `sae_config.json` 加载**（如果存在）- **推荐**
2. 从代码中的 `model_params` 字典加载（后备）

这意味着：
- ✅ **可以安全修改代码中的参数**，恢复时会自动使用 checkpoint 保存的原始配置
- ✅ **跨实验迁移**时，会自动加载源实验的架构配置
- ✅ **防止配置不匹配**导致的加载错误

### 需要哪些文件？

| 文件 | 用途 | 是否必须 | 丢失时的行为 |
|------|------|---------|-------------|
| `sae_latest.pt` | SAE 权重（state_dict） | **是** | 当作新层随机初始化 |
| `sae_config.json` | SAE 架构配置（d_hidden, top_k 等） | **强烈推荐** | 使用代码中的 `model_params`，**可能因维度不匹配而报错** |
| `train_state.json` | 训练步数、hook 配置等 | 否 | 从 step=0 开始计数 |

### ⚠️ 重要：文件丢失时的详细行为

#### 情况1：`sae_config.json` 丢失，但 `.pt` 存在

```
block_out.layer15/
├── sae_latest.pt          ✅ 存在
└── sae_config.json        ❌ 丢失
```

**代码行为**：
1. 检测到 `.pt` 文件存在 → 尝试恢复权重
2. 检测不到 `.json` → 使用代码中的 `model_params`
3. **结果取决于配置是否匹配**：

| 代码配置 vs 原配置 | 结果 |
|------------------|------|
| 相同（d_hidden=6144） | ✅ 正常恢复训练 |
| 不同（d_hidden=12288） | ❌ **报错**：`RuntimeError: size mismatch` |

**错误示例**：
```
RuntimeError: Error(s) in loading state_dict for SparseAutoEncoder:
    size mismatch for W_enc: copying a param with shape torch.Size([1536, 6144])
    from checkpoint, the shape in current model is torch.Size([1536, 12288]).
```

#### 情况2：`.pt` 文件丢失（无论 `.json` 是否存在）

```
block_out.layer15/
├── sae_latest.pt          ❌ 丢失
└── sae_config.json        ✅ 存在（但无用）
```

**代码行为**：
1. 检测不到 `.pt` 文件
2. 当作**新层**处理，随机初始化
3. 日志输出：`初始化新 SAE: block_out.layer15`

⚠️ **注意**：即使 `sae_config.json` 存在，也不会使用其配置（因为没有权重可加载）

#### 情况3：两个文件都丢失

```
block_out.layer15/
├── sae_latest.pt          ❌ 丢失
└── sae_config.json        ❌ 丢失
```

**代码行为**：当作全新层，使用代码配置初始化。

### 配置不匹配警告

如果从 `sae_config.json` 加载的配置与代码中的 `model_params` 不同，会输出提示：

```
从 checkpoint 恢复 SAE 配置: d_model=1536, d_hidden=6144, top_k=64
```

**如果没有此提示**，说明使用了代码配置（可能因为 `sae_config.json` 不存在）。

### 安全恢复检查清单

恢复训练前，确认以下文件存在：

```bash
# 检查每个要恢复的层
for layer in 15 29; do
    dir="sae_runs/exp_name/block_out.layer${layer}"
    echo "检查层 $layer:"
    ls -la "$dir/sae_latest.pt" 2>/dev/null && echo "  ✅ .pt 存在" || echo "  ❌ .pt 丢失"
    ls -la "$dir/sae_config.json" 2>/dev/null && echo "  ✅ .json 存在" || echo "  ❌ .json 丢失"
done
```

**如果 `.json` 丢失但记得原配置**：
- 手动修改代码中的 `model_params` 匹配原配置，或
- 从备份恢复 `sae_config.json`

---

## 架构配置与恢复训练的交互机制

### 核心问题：每层使用哪个配置？

代码逻辑：**每层独立决定自己的配置**

```python
# 伪代码逻辑
for each layer:
    if layer has checkpoint config (sae_config.json):
        use checkpoint's config  # 恢复原有架构
    else:
        use code's model_params   # 新层使用当前配置
```

### 场景示例详解

#### 场景：恢复层15 + 新增层29，代码配置与checkpoint不一致

**配置设置：**

```python
# model_params（代码中的当前配置）
model_params = {
    "d_model": 1536,
    "d_hidden": 12288,    # ⚠️ 改为 12288（与原 6144 不同）
    "top_k": 128,         # ⚠️ 改为 128（与原 64 不同）
}

# resume_params
resume_params = {
    "enabled": True,
    "sae_checkpoint": "sae_runs/exp_20250324",  # 包含层15的已训练模型
    "additional_layers": [29],                   # 新增层 29
}

# hook_params
hook_params = {
    "hook_mode": "block_out",
    "hook_layers": "15,29",  # 同时训练两层
}
```

**源目录结构：**

```
sae_runs/exp_20250324/
├── block_out.layer15/
│   ├── sae_config.json      # 保存 d_hidden=6144, top_k=64
│   ├── sae_latest.pt        # 使用哪个pt？→ 总是用 latest
│   └── sae_step500.pt       # 历史版本，不会被自动使用
└── train_state.json
```

**实际行为：**

| 层 | 是否新增 | 配置来源 | d_hidden | top_k | 权重初始化 |
|---|---------|---------|---------|-------|-----------|
| 15 | ❌ 否 | `sae_config.json` | **6144** (来自checkpoint) | **64** (来自checkpoint) | 从 `sae_latest.pt` 加载 |
| 29 | ✅ 是 | `model_params` (代码) | **12288** (新配置) | **128** (新配置) | 随机初始化 |

**日志输出：**

```
从 checkpoint 恢复 SAE 配置: d_model=1536, d_hidden=6144, top_k=64
从 sae_runs/exp_20250324/block_out.layer15/sae_latest.pt 恢复 SAE: block_out.layer15
初始化新 SAE: block_out.layer29
SAE 初始化完成: 总共=2, 新增=1, 冻结=0, 可训练=2
```

### 关键结论

1. **使用哪个 pt 文件？**
   - 总是使用 `sae_latest.pt`
   - 历史版本（如 `sae_step500.pt`）需要手动重命名或指定

2. **配置不一致会怎样？**
   - **不会报错**，每层使用各自的配置
   - 已有层（如15）：使用 checkpoint 保存的原始配置（6144）
   - 新增层（如29）：使用代码当前配置（12288）

3. **这在什么情况下有用？**
   - ✅ **实验不同架构**：保持层15为基线，测试层29的大维度效果
   - ✅ **渐进式扩展**：小维度快速迭代，大维度精细化
   - ⚠️ **注意**：层15和层29的 SAE 输出维度不同，不能直接比较特征

### 进阶：手动指定特定步数的 checkpoint

如果想从 `sae_step500.pt` 恢复（而非 latest）：

```bash
# 方法1：复制到 latest
cp sae_runs/exp_20250324/block_out.layer15/sae_step500.pt \
   sae_runs/exp_20250324/block_out.layer15/sae_latest.pt

# 方法2：修改代码（不推荐）
# 需要修改 SAERunLocator.latest_ckpt_path() 逻辑
```

---

## Checkpoint 格式版本说明

### 新版本 (v2.0) - 推荐

**特点**: 配置内置到 `.pt` 文件中，永不分离

```python
# v2.0 .pt 文件内容
{
    "state_dict": {...},           # SAE 权重
    "step": int,                   # 训练步数
    "sae_config": {...},           # SAE 架构配置（内置，必需）
    "hook_info": {...},            # hook 信息
    "timestamp": float,            # 保存时间戳
    "version": "2.0",              # 格式版本
}
```

**优势**:
- ✅ 配置和权重永不分离，避免丢失 `.json` 导致的恢复失败
- ✅ 单个文件即可完整恢复
- ✅ 自动向后兼容旧格式

### 旧版本 (v1.0)

**特点**: 配置和权重分开存储

```
block_out.layer15/
├── sae_config.json      # 配置
├── sae_latest.pt        # 仅权重
└── sae_step500.pt       # 仅权重
```

**问题**:
- ⚠️ `.json` 丢失后无法确定模型架构，可能导致恢复失败
- ⚠️ 需要同时维护两个文件

---

## 从旧版本迁移

### 自动迁移（推荐）

迁移单个 checkpoint:
```bash
python wan/sae/migrate_checkpoints.py \
  --pt sae_runs/exp/block_out.layer15/sae_latest.pt
```

迁移整个实验目录:
```bash
python wan/sae/migrate_checkpoints.py --dir sae_runs/exp
```

预览将要迁移的文件:
```bash
python wan/sae/migrate_checkpoints.py --dir sae_runs/exp --dry-run
```

### 手动迁移

如果自动迁移失败，可以手动复制配置:
```bash
# 1. 备份原文件
cp sae_latest.pt sae_latest.pt.backup

# 2. 使用 Python 手动迁移
python -c "
import torch
from wan.sae.checkpoint_io import CheckpointMigrator
CheckpointMigrator.migrate_file(
    'sae_runs/exp/block_out.layer15/sae_latest.pt',
    backup=False
)
"
```

### 向后兼容说明

**新代码可以读取旧格式**（自动从 `.json` 回退加载），但会输出警告:
```
从旧格式 .json 恢复 SAE: block_out.layer15 (d_hidden=6144, top_k=64) [建议迁移]
```

**建议**: 尽快迁移旧 checkpoint 到新格式。

---

## 一、基础训练示例（从头开始）

### 配置方式（文件内）

```python
# wan/sae_train_t2v_1_3b.py

path_params = {
    "model_path": "./Wan2.1-T2V-1.3B",      # DiT 模型路径
    "prompt_dir": "./nsfw_prompts",         # 提示词目录
    "run_dir": "sae_runs/exp_baseline",     # 实验输出目录
}

resume_params = {
    "enabled": False,                        # 不启用恢复训练
    "sae_checkpoint": "",                   # 无 checkpoint
    "additional_layers": [],                # 无新增层
    "frozen_layers": [],                    # 无冻结层
    "reset_optimizer": False,
    "reset_step_count": False,
}

hook_params = {
    "hook_mode": "block_out",
    "hook_layers": "15,29",                 # 训练层 15 和 29
}
```

### 运行命令

```bash
python wan/sae_train_t2v_1_3b.py
```

或命令行覆盖：

```bash
python wan/sae_train_t2v_1_3b.py \
  --model_path "./Wan2.1-T2V-1.3B" \
  --run_dir "sae_runs/exp_baseline" \
  --hook_layers "15,29"
```

---

## 二、恢复训练示例（继续同一实验）

### 场景

训练在中断后需要继续，保持相同的层配置和优化器状态。

### 配置方式（文件内）

```python
# wan/sae_train_t2v_1_3b.py

path_params = {
    "model_path": "./Wan2.1-T2V-1.3B",
    "prompt_dir": "./nsfw_prompts",
    "run_dir": "sae_runs/exp_baseline",      # 同一实验目录
}

resume_params = {
    "enabled": True,                         # 启用恢复
    "sae_checkpoint": "",                   # 空表示从 run_dir 自动检测
    "additional_layers": [],
    "frozen_layers": [],
    "reset_optimizer": False,               # 保持优化器状态
    "reset_step_count": False,              # 继续计数
}
```

### 运行命令

```bash
python wan/sae_train_t2v_1_3b.py --resume
```

或完整参数：

```bash
python wan/sae_train_t2v_1_3b.py \
  --resume \
  --run_dir "sae_runs/exp_baseline"
```

### 恢复行为

1. 自动查找 `sae_runs/exp_baseline/train_state.json`
2. 读取上次保存的 step 数
3. 加载各层的 `sae_latest.pt`
4. 恢复优化器状态（如果未指定 `--reset_optimizer`）
5. 从断点继续训练

---

## 三、扩展层训练示例（已有层15，新增层29）

### 场景

已经训练好了层 15，现在想同时训练层 15 和 29，其中层 15 从 checkpoint 恢复，层 29 全新初始化。

### 配置方式（文件内）

```python
# wan/sae_train_t2v_1_3b.py

path_params = {
    "model_path": "./Wan2.1-T2V-1.3B",
    "prompt_dir": "./nsfw_prompts",
    "run_dir": "sae_runs/exp_extended",      # 新的实验目录
}

resume_params = {
    "enabled": True,
    "sae_checkpoint": "sae_runs/exp_baseline",  # 源实验目录
    "additional_layers": [29],              # 层 29 是新增加的
    "frozen_layers": [],                    # 不冻结任何层
    "reset_optimizer": True,                # 新增层需要新优化器
    "reset_step_count": False,              # 继续计数（可选 True 重置）
}

hook_params = {
    "hook_mode": "block_out",
    "hook_layers": "15,29",                 # 现在训练两层
}
```

### 运行命令

```bash
python wan/sae_train_t2v_1_3b.py \
  --resume \
  --run_dir "sae_runs/exp_extended" \
  --hook_layers "15,29" \
  --sae_checkpoint "sae_runs/exp_baseline" \
  --additional_layers "29" \
  --reset_optimizer
```

### 行为说明

| 层 | 来源 | 初始化方式 | 训练状态 |
|---|------|-----------|---------|
| 15 | `exp_baseline/block_out.layer15/` | 从 checkpoint 加载 | 可训练 |
| 29 | 无（新层） | 随机初始化 | 可训练 |

---

## 四、冻结层训练示例（冻结层15，只训练层29）

### 场景

层 15 已经训练得很好，想固定它只训练新增的层 29，防止已学习特征被破坏。

### 配置方式（文件内）

```python
# wan/sae_train_t2v_1_3b.py

resume_params = {
    "enabled": True,
    "sae_checkpoint": "sae_runs/exp_baseline",
    "additional_layers": [29],
    "frozen_layers": ["block_out.layer15"],  # 冻结层 15
    "reset_optimizer": True,
    "reset_step_count": False,
}

hook_params = {
    "hook_mode": "block_out",
    "hook_layers": "15,29",
}
```

### 运行命令

```bash
python wan/sae_train_t2v_1_3b.py \
  --resume \
  --sae_checkpoint "sae_runs/exp_baseline" \
  --additional_layers "29" \
  --frozen_layers "block_out.layer15" \
  --reset_optimizer
```

### 行为说明

| 层 | 初始化 | 是否训练 | 优化器 |
|---|--------|---------|--------|
| 15 | 从 checkpoint 加载 | **否**（冻结） | 无（不参与优化） |
| 29 | 随机初始化 | **是** | AdamW |

---

## 五、跨实验加载示例（从不同实验加载不同层）

### 场景

从实验 A 加载层 15，从实验 B 加载层 29，合并到新实验 C 进行测试或进一步训练。

### 5.1 测试时跨实验加载

```python
# wan/sae_test_t2v_1_3b.py

checkpoint_params = {
    "sae_checkpoint": "",                   # 不使用统一源
    "layer_sources": {
        "block_out.layer15": "sae_runs/exp_base",
        "block_out.layer29": "sae_runs/exp_finetune",
    },
    "allow_partial_load": False,
    "strict_loading": True,
}
```

### 命令行方式

```bash
python wan/sae_test_t2v_1_3b.py \
  --model_path "./Wan2.1-T2V-1.3B" \
  --prompt_dir "./test_prompts" \
  --layer_sources "15:sae_runs/exp_base,29:sae_runs/exp_finetune" \
  --hook_layers "15,29"
```

### 5.2 训练时跨实验加载

```python
# wan/sae_train_t2v_1_3b.py - 需通过命令行指定

resume_params = {
    "enabled": True,
    "sae_checkpoint": "",                   # 空，使用 run_dir 作为基础
    "additional_layers": [],
    "frozen_layers": [],
}
```

```bash
python wan/sae_train_t2v_1_3b.py \
  --resume \
  --run_dir "sae_runs/exp_combined" \
  --hook_layers "15,29" \
  --sae_checkpoint "sae_runs/exp_base" \
  --resume_from_layer29 "sae_runs/exp_finetune"
```

**注意**: 训练时跨实验加载需要通过自定义代码或多次指定 `--sae_checkpoint` 实现。

---

## 六、测试时多层源配置

### 6.1 从单一源加载（默认）

```bash
python wan/sae_test_t2v_1_3b.py \
  --model_path "./Wan2.1-T2V-1.3B" \
  --run_dir "sae_runs/exp_baseline" \
  --hook_layers "15,29"
```

### 6.2 从 JSON 文件加载多层源

创建 `layer_sources.json`：

```json
{
    "block_out.layer15": "sae_runs/exp_v1",
    "block_out.layer29": "sae_runs/exp_v2",
    "self_attn.layer20": "sae_runs/exp_attn"
}
```

运行：

```bash
python wan/sae_test_t2v_1_3b.py \
  --model_path "./Wan2.1-T2V-1.3B" \
  --layer_sources "layer_sources.json"
```

### 6.3 容错模式（允许部分加载失败）

```bash
python wan/sae_test_t2v_1_3b.py \
  --model_path "./Wan2.1-T2V-1.3B" \
  --run_dir "sae_runs/exp_partial" \
  --hook_layers "15,29,30" \
  --allow_partial_load
```

如果层 30 的 checkpoint 不存在，将跳过该层继续测试其他层。

---

## 七、参数优先级说明

### 7.1 训练脚本参数优先级

优先级从高到低：

1. 命令行参数（如 `--sae_checkpoint`）
2. 外部 JSON 配置文件（`--config`）
3. 代码中的 `resume_params` 字典
4. 默认值

### 7.2 测试脚本参数优先级

1. 命令行 `--layer_sources` 和 `--sae_checkpoint`
2. 代码中的 `checkpoint_params` 字典
3. `--run_dir` 作为默认源

---

## 八、Checkpoint 文件结构

### 训练输出结构

```
sae_runs/exp_name/
├── train_state.json                      # 全局训练状态
├── logs/
│   ├── training.log                      # 训练日志
│   ├── loss_history.jsonl                # 详细 loss 记录
│   └── loss_history.csv                  # CSV 格式 loss
├── block_out.layer15/
│   ├── sae_config.json                   # SAE 架构配置
│   ├── sae_latest.pt                     # 最新权重（软链接）
│   └── sae_step500.pt                    # 历史版本
└── block_out.layer29/
    ├── sae_config.json
    ├── sae_latest.pt
    └── sae_step500.pt
```

### train_state.json 内容

```json
{
    "step": 500,
    "max_steps": 2000,
    "hook_mode": "block_out",
    "hook_layers": [15, 29],
    "sae_config": {
        "d_model": 1536,
        "d_hidden": 6144,
        "sparsity": "topk",
        "top_k": 64
    },
    "sampling_steps": 30,
    "seed": 0
}
```

---

## 九、常见问题 FAQ

### Q1: 如何只测试特定层的 SAE？

```bash
python wan/sae_test_t2v_1_3b.py \
  --run_dir "sae_runs/exp_name" \
  --hook_layers "15"                      # 只测试层 15
```

### Q2: 恢复训练时如何改变学习率？

```python
# 修改 training_params["lr"] 后，重置优化器
resume_params = {
    "enabled": True,
    "reset_optimizer": True,                # 必须重置以使用新学习率
}
```

### Q3: 可以从训练到一半的 checkpoint 开始吗？

可以。`sae_latest.pt` 始终指向最新的 checkpoint，无论训练到哪个 step。

### Q4: 不同 hook_mode 的 checkpoint 能混用吗？

**不能**。SAE 配置必须与训练时的 hook_mode 匹配，否则特征维度可能不一致。

### Q5: 如何查看某个 checkpoint 的详细信息？

```python
import torch

ckpt = torch.load("sae_runs/exp/block_out.layer15/sae_latest.pt")
print(f"训练步数: {ckpt['step']}")
print(f"权重键: {ckpt['state_dict'].keys()}")
```

---

## 十、高级用法示例

### 示例：迁移学习（从层15迁移到层20）

```python
# wan/sae_train_t2v_1_3b.py

resume_params = {
    "enabled": True,
    "sae_checkpoint": "sae_runs/exp_layer15",  # 源实验
    "additional_layers": [20],                   # 新层 20
    "frozen_layers": [],                         # 不冻结，微调所有层
    "reset_optimizer": True,
    "reset_step_count": True,                    # 新任务，重置计数
}

hook_params = {
    "hook_mode": "block_out",
    "hook_layers": "15,20",                      # 同时训练两层
}
```

### 示例：渐进式训练（逐层解冻）

第一阶段：训练层 15
```python
hook_params = {"hook_layers": "15"}
resume_params = {"enabled": False}
```

第二阶段：添加层 20，冻结层 15
```python
hook_params = {"hook_layers": "15,20"}
resume_params = {
    "enabled": True,
    "sae_checkpoint": "sae_runs/exp_stage1",
    "additional_layers": [20],
    "frozen_layers": ["block_out.layer15"],
}
```

第三阶段：解冻所有层联合训练
```python
resume_params = {
    "enabled": True,
    "sae_checkpoint": "sae_runs/exp_stage2",
    "frozen_layers": [],                         # 解冻
}
```

---

## 十一、向后兼容性说明

- `--checkpoint_dir` 参数仍可使用，但会显示弃用警告
- 现有的 `train_state.json` 格式保持不变
- 现有的 `sae_latest.pt` 加载逻辑保持不变
- 旧版本训练的 checkpoint 可以直接在新版本脚本中加载

---

## 参考链接

- [SAE 训练脚本](../sae_train_t2v_1_3b.py)
- [SAE 测试脚本](../sae_test_t2v_1_3b.py)
- [SAE 模块文档](./NEW_MODULES_OVERVIEW.md)

---

## 统一读写接口更新记录

### 已更新的模块

| 模块路径 | 修改内容 | 状态 |
|---------|---------|------|
| `wan/sae/checkpoint_io.py` | 新增统一 IO 类 `SAECheckpointIO` | 新增 |
| `wan/sae/migrate_checkpoints.py` | 新增迁移脚本 | 新增 |
| `wan/sae_train_t2v_1_3b.py` | 使用 `SAECheckpointIO` 保存/加载 | 已更新 |
| `wan/sae_test_t2v_1_3b.py` | 使用 `SAECheckpointIO.load()` 加载 | 已更新 |
| `wan/sae/offline_training/train_offline.py` | 保存和恢复逻辑更新 | 已更新 |
| `wan/sae/offline_training/test_offline.py` | 加载逻辑更新 | 已更新 |
| `wan/sae/interpretability/concept_extractor.py` | 加载逻辑更新 | 已更新 |
| `wan/sae/steering/steering_generator.py` | 加载逻辑更新 | 已更新 |

### 向后兼容性

所有模块均保持向后兼容，可以读取：
- **新格式 (v2.0)**: 配置内置在 `.pt` 文件中
- **旧格式 (v1.0)**: 配置在 `.json` 文件中，权重在 `.pt` 文件中

从旧格式加载时会输出警告：
```
从旧格式 .json 加载配置 [建议迁移]
```

### 迁移旧 Checkpoint 的方法

```bash
# 单个文件迁移
python -m wan.sae.migrate_checkpoints --pt sae_runs/exp/block_out.layer15/sae_latest.pt

# 整个目录迁移
python -m wan.sae.migrate_checkpoints --dir sae_runs/exp

# 预览将要迁移的文件
python -m wan.sae.migrate_checkpoints --dir sae_runs/exp --dry-run
```

### 统一读写接口使用示例

**保存 Checkpoint**:
```python
from wan.sae.checkpoint_io import SAECheckpointIO, SAERunLocator

loc = SAERunLocator(run_dir="sae_runs/exp", hook_mode="block_out", layer_idx=15)
io = SAECheckpointIO(
    sae=sae,
    step=500,
    hook_mode="block_out",
    layer_idx=15,
    extra_info={"best_val_loss": 0.01},
)
io.save(loc)  # 同时保存 .pt（内置配置）和 .json（便于查看）
```

**加载 Checkpoint**:
```python
from wan.sae.checkpoint_io import SAECheckpointIO, SAERunLocator

loc = SAERunLocator(run_dir="sae_runs/exp", hook_mode="block_out", layer_idx=15)
io = SAECheckpointIO.load(loc, device="cuda:0")

sae = io.sae          # 加载好的 SAE 模型
step = io.step        # 训练步数
config = io.sae_config  # SAE 配置
```
