# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在本仓库中工作的指导。

## 概述

这是阿里巴巴开源的视频生成模型 **Wan2.1**。它包含原始的 Wan2.1 推理代码，以及仓库所有者添加的自定义 SAE（稀疏自编码器）可解释性研究代码。

- **原始功能**: Wan2.1 视频生成（T2V、I2V、FLF2V、VACE 任务）
- **自定义扩展**: 在 Wan 1.3B T2V 模型隐藏状态上进行 SAE 训练和分析

## SAE 代码结构特点

所有 SAE 相关脚本采用**代码与参数分离**的设计：
- 参数定义在文件顶部的 `{xxx}_params` 字典中
- 每个参数都有详细的中文注释，说明学术意义、实际用法和建议值
- 命令行参数可以覆盖代码中的默认值
- 支持通过 `--config` 加载 JSON 配置文件

## 常用命令

### 环境设置与安装

```bash
# 安装依赖（需要 torch >= 2.4.0）
pip install -r requirements.txt

# 下载模型
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
```

### 代码格式化

```bash
# 格式化代码（isort + yapf）
make format
```

### 视频生成（原始 Wan2.1）

```bash
# T2V 1.3B 单 GPU
python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B \
  --prompt "两只拟人化的猫咪穿着舒适的拳击装备..."

# T2V 1.3B 显存卸载模式（适用于低显存）
python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B \
  --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 \
  --prompt "你的提示词"

# 多 GPU 推理（FSDP + xDiT USP）
torchrun --nproc_per_node=8 generate.py --task t2v-14B --size 1280*720 \
  --ckpt_dir ./Wan2.1-T2V-14B --dit_fsdp --t5_fsdp --ulysses_size 8
```

### SAE 训练与测试

#### 方式一：在线训练（推荐，完整扩散轨迹）

**修改参数**: 直接编辑 `wan/sae_train_t2v_1_3b.py` 顶部的参数字典

```python
# 关键参数配置区域（在文件顶部）
path_params = {
    "checkpoint_dir": "./Wan2.1-T2V-1.3B",
    "prompt_dir": "./nsfw_prompts",
    "run_dir": "sae_runs/exp_blockout_20250319",
}

model_params = {
    "d_model": 1536,      # DiT 维度，1.3B 模型固定
    "d_hidden": 6144,     # SAE 扩展维度，建议 4x
    "sparsity": "topk",   # "topk" 或 "l1"
    "top_k": 64,          # topk 稀疏度，约 1%
}

training_params = {
    "steps": 2000,        # 训练步数
    "batch_prompts": 4,   # 每步批大小
    "sampling_steps": 30, # 扩散步数（覆盖噪声范围）
    "lr": 1e-3,           # 学习率
}

hook_params = {
    "hook_mode": "block_out",  # "self_attn" | "cross_attn" | "block_out"
    "hook_layers": "15,29",    # 要分析的层
}
```

**运行训练**:
```bash
# 使用代码中的默认参数
python wan/sae_train_t2v_1_3b.py

# 命令行覆盖关键参数
python wan/sae_train_t2v_1_3b.py \
  --checkpoint_dir "./Wan2.1-T2V-1.3B" \
  --prompt_dir "./nsfw_prompts" \
  --run_dir "sae_runs/exp1" \
  --hook_layers "0,15,29" \
  --steps 3000

# 从 JSON 配置文件加载
python wan/sae_train_t2v_1_3b.py --config my_config.json

# 导出默认配置模板
python wan/sae_train_t2v_1_3b.py --dump_default_config template.json

# 恢复训练
python wan/sae_train_t2v_1_3b.py --resume --run_dir sae_runs/exp1
```

**训练日志输出**（带 ETA 预估）:
```
[100/2000] batch=4 keys=['block_out.layer15', 'block_out.layer29']
step_time=2.35s elapsed=3m52s ETA=1h14m
```

#### 方式二：离线训练（先采集后训练）

**步骤 1: 采集隐藏状态**

编辑 `wan/sae_collect.py` 顶部的参数，然后运行：
```bash
python wan/sae_collect.py --checkpoint_dir "./Wan2.1-T2V-1.3B"
```

**步骤 2: 离线训练 SAE**

编辑 `train_sae.py` 顶部的参数，然后运行：
```bash
python train_sae.py \
  --features_path features_layer29.pt \
  --d_model 1536 \
  --d_hidden 6144 \
  --epochs 5 \
  --batch_size 4096
```

#### SAE 测试/推理

编辑 `wan/sae_test_t2v_1_3b.py` 顶部的参数，然后运行：
```bash
python wan/sae_test_t2v_1_3b.py \
  --checkpoint_dir "./Wan2.1-T2V-1.3B" \
  --prompt_dir "./test_prompts" \
  --run_dir "sae_runs/exp1" \
  --output_path "sae_test_out.pt"
```

### SAE Checkpoint 查找与使用指南

#### Checkpoint 文件结构

每个训练实验的 checkpoint 按以下结构组织：

```
sae_runs/exp1/                              # run_dir: 实验根目录
├── train_state.json                        # 训练状态（步数、配置等）
├── block_out.layer15/                      # 特定 hook 层目录
│   ├── sae_config.json                     # SAE 架构配置
│   ├── sae_latest.pt                       # 最新权重（软链接/复制）
│   └── sae_step1000.pt                     # 历史版本
└── block_out.layer29/
    └── ...
```

**重要**: Checkpoint 仅包含 SAE 参数（轻量级，约 100MB），不包含 DiT/T5/VAE 等基础模型权重。

#### 查找已训练的 SAE

**方式 1: 列出实验目录**
```bash
# 查看所有实验
ls sae_runs/

# 查看特定实验的层配置
ls sae_runs/exp_20250324/
# 输出: block_out.layer15/  block_out.layer29/  train_state.json

# 查看某层的 checkpoint
ls sae_runs/exp_20250324/block_out.layer15/
# 输出: sae_config.json  sae_latest.pt  sae_step500.pt  sae_step1000.pt
```

**方式 2: 查看训练状态**
```bash
# 查看实验配置和当前进度
cat sae_runs/exp_20250324/train_state.json
```

**方式 3: 编程方式查找**
```python
from wan.sae.sae_run_naming import SAERunLocator, list_available_layers

# 列出某实验的所有可用层
layers = list_available_layers("sae_runs/exp_20250324")
print(layers)  # [("block_out", 15), ("block_out", 29)]

# 获取特定层的 checkpoint 路径
loc = SAERunLocator(run_dir="sae_runs/exp_20250324", hook_mode="block_out", layer_idx=15)
print(loc.latest_ckpt_path())   # sae_runs/exp_20250324/block_out.layer15/sae_latest.pt
print(loc.config_path())        # sae_runs/exp_20250324/block_out.layer15/sae_config.json
```

#### 使用 SAE Checkpoint

**方式 1: 使用测试脚本（推荐）**

编辑 `wan/sae_test_t2v_1_3b.py` 顶部参数：
```python
path_params = {
    "checkpoint_dir": "./Wan2.1-T2V-1.3B",     # DiT 基础模型路径
    "prompt_dir": "./test_prompts",            # 测试提示词目录
    "run_dir": "sae_runs/exp_20250324",        # SAE 训练输出目录
    "output_path": "test_results.pt",
}

hook_params = {
    "hook_mode": "block_out",                  # 必须与训练时一致
    "hook_layers": "15,29",                    # 要测试的层（逗号分隔）
}
```

运行测试：
```bash
python wan/sae_test_t2v_1_3b.py
```

**方式 2: 命令行参数覆盖**
```bash
# 测试单个层
python wan/sae_test_t2v_1_3b.py \
  --run_dir "sae_runs/exp_20250324" \
  --hook_mode "block_out" \
  --hook_layers "15" \
  --prompt_dir "./test_prompts" \
  --checkpoint_dir "./Wan2.1-T2V-1.3B" \
  --output_path "layer15_results.pt"

# 测试多个层
python wan/sae_test_t2v_1_3b.py \
  --run_dir "sae_runs/exp_20250324" \
  --hook_layers "15,29" \
  --batch_prompts 8
```

**方式 3: 加载 checkpoint 进行自定义分析**
```python
import torch
from wan.sae.sae_run_naming import SAERunLocator, load_json
from wan.modules.sae_new import SAEConfig, SparseAutoEncoder

# 配置
run_dir = "sae_runs/exp_20250324"
hook_mode = "block_out"
layer_idx = 15
device = "cuda:0"

# 定位并加载 SAE
loc = SAERunLocator(run_dir=run_dir, hook_mode=hook_mode, layer_idx=layer_idx)

# 加载配置
cfg_dict = load_json(loc.config_path())["sae"]
sae_cfg = SAEConfig(**cfg_dict)

# 加载权重
sae = SparseAutoEncoder(sae_cfg).to(device)
ckpt = torch.load(loc.latest_ckpt_path(), map_location=device)
sae.load_state_dict(ckpt["state_dict"])
sae.eval()

# 现在可以使用 sae.encode(x) 进行编码分析
# z, info = sae.encode(hidden_states)  # z: 稀疏表示
```

#### 在训练时指定保存配置

编辑 `wan/sae_train_t2v_1_3b.py` 配置：

```python
path_params = {
    "checkpoint_dir": "./Wan2.1-T2V-1.3B",
    "prompt_dir": "./nsfw_prompts",
    "run_dir": "sae_runs/exp_20250324",        # 实验目录名
}

training_params = {
    "save_every": 200,                         # 每 200 步保存 checkpoint
}

hook_params = {
    "hook_mode": "block_out",                  # hook 模式
    "hook_layers": "15,29",                    # 同时训练哪些层
}
```

#### Checkpoint 内容说明

| 文件 | 大小 | 内容 |
|------|------|------|
| `sae_latest.pt` | ~100MB | SAE 权重 (W_enc, W_dec, b_enc, b_dec) + 训练步数 |
| `sae_step{step}.pt` | ~100MB | 历史版本（可选保留） |
| `sae_config.json` | ~1KB | SAE 架构参数 (d_model, d_hidden, top_k 等) |
| `train_state.json` | ~1KB | 全局训练状态 |

**注意**:
- 恢复训练时使用 `sae_latest.pt`
- 测试分析时使用 `sae_latest.pt` 或特定的 `sae_step{step}.pt`
- `sae_config.json` 必须与权重配套使用，用于重建 SAE 架构

## 架构概述

### Core Wan2.1 Components

Wan2.1 模型采用 **Flow Matching + 扩散 Transformer (DiT)** 架构：

- **`wan/text2video.py`**: `WanT2V` 类 - 文本生成视频主流水线
- **`wan/image2video.py`**: `WanI2V` 类 - 图像生成视频流水线
- **`wan/first_last_frame2video.py`**: `WanFLF2V` 类 - 首尾帧生成视频
- **`wan/vace.py`**: `WanVace` 类 - 视频编辑和风格迁移

关键模块：
- **`wan/modules/model.py`**: `WanModel` - DiT Transformer（核心生成模型）
- **`wan/modules/vae.py`**: `WanVAE` - 用于视频编解码的 3D 因果 VAE
- **`wan/modules/t5.py`**: 使用 UMT5-XXL 的文本编码器
- **`wan/configs/`**: 模型配置（1.3B、14B 变体）

### DiT 架构细节

`wan/modules/model.py` 中的 `WanModel` 包含：
- 30 个 Transformer 块（1.3B 模型）或 40 个块（14B 模型）
- 每个块包含：`WanSelfAttention`、`WanCrossAttention` 和 FFN
- 使用 RoPE（旋转位置编码）进行时空位置编码
- 使用 Flash Attention 进行高效注意力计算

模型维度：
| 模型 | dim | 注意力头数 | 层数 | ffn_dim |
|------|-----|-----------|------|---------|
| 1.3B | 1536| 12        | 30   | 8960    |
| 14B  | 5120| 40        | 40   | 13824   |

### SAE 架构

**SAE 训练方式对比：**

| 方式 | 脚本 | 优点 | 缺点 |
|------|------|------|------|
| 在线训练 | `sae_train_t2v_1_3b.py` | 覆盖完整扩散轨迹，特征分布更真实 | 训练慢，需持续运行 DiT |
| 离线训练 | `train_sae.py` | 训练快，可多轮调参 | 只覆盖单时间步 |

**关键文件：**
- **`wan/modules/sae_new.py`**: `SparseAutoEncoder` 类，支持 topk/L1 稀疏
- **`wan/sae/hooking.py`**: Hook 系统，用于提取 DiT 隐藏状态
- **`wan/sae/configs.py`**: 配置数据类
- **`wan/sae/prompt_io.py`**: 提示词加载和清洗
- **`wan/sae/sae_run_naming.py`**: 检查点管理

**Hook 模式（学术意义）：**
- `self_attn`: 捕获空间-时间自相关模式（物体形状、运动）
- `cross_attn`: 捕获文本-视觉对齐特征（概念、属性）
- `block_out`: 完整的残差表征（推荐）

**检查点组织：**
```
sae_runs/exp1/
├── train_state.json              # 全局训练状态
├── block_out.layer15/
│   ├── sae_config.json           # SAE 配置
│   ├── sae_latest.pt             # 最新权重
│   └── sae_step1000.pt           # 历史版本
└── block_out.layer29/
    └── ...
```

## 关键参数指南

### SAE 结构参数

| 参数 | 学术意义 | 默认值 | 建议值 |
|------|---------|--------|--------|
| `d_hidden` | 扩展维度，决定特征容量 | 6144 | 4x~8x d_model |
| `sparsity` | 稀疏化策略 | "topk" | "topk"（可解释）/ "l1"（重建好） |
| `top_k` | 每样本非零特征数 | 64 | d_hidden 的 1%~10% |

### 训练参数

| 参数 | 学术意义 | 默认值 | 建议值 |
|------|---------|--------|--------|
| `steps` | 训练步数 | 2000 | 2000~5000 |
| `batch_prompts` | 每步批大小 | 4 | 4~8（显存限制） |
| `sampling_steps` | 扩散步数 | 30 | 30~50（覆盖噪声范围） |
| `use_cfg` | 使用 CFG | False | False（快）/ True（分布准确） |

### Hook 参数

| 参数 | 学术意义 | 默认值 | 建议值 |
|------|---------|--------|--------|
| `hook_mode` | 捕获哪种激活 | "block_out" | "block_out"（全面） |
| `hook_layers` | 分析的层 | "15" | "15,29"（中层+深层） |

## 关键配置文件

- **`pyproject.toml`**: 包元数据、依赖、工具配置
- **`requirements.txt`**: 核心依赖
- **`.style.yapf`**: YAPF 代码格式化规则
- **`wan/configs/wan_t2v_1_3B.py`**: 1.3B 模型配置
- **`wan/configs/wan_t2v_14B.py`**: 14B 模型配置

## 重要提示

- **参数修改**: 直接编辑脚本顶部的 `{xxx}_params` 字典，每个参数都有详细注释
- **内存管理**: 使用 `--offload_model True` 和 `--t5_cpu` 用于低显存 GPU
- **多 GPU**: SAE 训练目前仅支持单 GPU
- **恢复训练**: 支持 `--resume` 从上次中断处继续
- **日志输出**: 训练时自动显示剩余时间（ETA）估计

## 文档

- `README-EN.md`: 原始 Wan2.1 文档（英文）
- `README.md`: 中文版本文档
- `sae_readme.md`: 详细的 SAE 系统文档（中文）
- 各 SAE 脚本内的参数注释: 最详细的参数说明
