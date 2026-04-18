# 阶段一：激活值采集 - 使用说明

## 功能

1. **配对提取**：逐对处理正负提示词，保证紧凑
2. **复用现有接口**：SAECheckpointIO（兼容新旧格式）、WanT2V、register_dit_hooks
3. **分层存储**：按 `sae_layer{idx}/{category}/{polarity}/` 分层
4. **增量采集**：支持断点续传
5. **完整扩散轨迹**：采集所有时间步（如30步）

## 存储结构

```
activations/
├── sae_layer15/
│   └── violence/
│       ├── pos/
│       │   ├── activations.npy      # [N, T, L, d_hidden]
│       │   ├── metadata.json        # [{idx, prompt, pair_idx}, ...]
│       │   └── checkpoint.json      # 增量断点
│       └── neg/
│           ├── activations.npy
│           └── metadata.json
├── dit_layer15/                     # DiT状态（可选）
│   └── violence/
│       ├── pos/
│       └── neg/
└── extraction_config.json           # 全局配置（SAE路径、时间步等）
```

## 使用示例

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
    --hook_mode "block_out" \
    --sampling_steps 30 \
    --use_cfg false
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `./Wan2.1-T2V-1.3B` | DiT模型路径 |
| `--sae_run_dir` | `sae_runs/exp1` | SAE训练输出目录 |
| `--pos_prompts` | - | 正样本提示词文件 |
| `--neg_prompts` | - | 负样本提示词文件 |
| `--category` | `violence` | 概念类别名称 |
| `--sae_layers` | `15,29` | 要采集的SAE层 |
| `--save_dit_layers` | `` | 要保存的DiT层（空=不保存） |
| `--hook_mode` | `block_out` | Hook位置 |
| `--sampling_steps` | `30` | 时间步数 |
| `--use_cfg` | `false` | 是否使用CFG |
| `--resume` | `false` | 增量采集模式 |

## 增量采集

如果采集中断，使用 `--resume` 从断点继续：

```bash
python wan/sae/interpretability/concept_extractor_stage1.py \
    --resume \
    --pos_prompts "final_cleaned/pos_prompt_3.txt" \
    --neg_prompts "final_cleaned/neg_prompt_3.txt" \
    --category "violence" \
    ...其他参数
```

## 元信息格式

`metadata.json`（每个极性一个）：
```json
[
  {"idx": 0, "pair_idx": 0, "prompt": "...", "category": "violence", "polarity": "pos"},
  {"idx": 1, "pair_idx": 1, "prompt": "...", "category": "violence", "polarity": "pos"}
]
```

## 全局配置

`extraction_config.json`：
```json
{
  "category": "violence",
  "sae_checkpoints": {"layer15": ".../sae_latest.pt", ...},
  "hook_mode": "block_out",
  "timesteps": [999, 966, ..., 0],
  "num_timesteps": 30,
  "use_cfg": false
}
```

## ActivationIO 接口

`activation_io.py` 提供了统一的文件I/O接口，供阶段一和阶段二使用：

```python
from wan.sae.interpretability import ActivationIO

# 初始化IO管理器
io = ActivationIO("activations")

# 保存激活值
io.save_activations(
    layer_type="sae",
    layer_idx=15,
    category="violence",
    polarity="pos",
    activations=array,  # [N, T, L, D]
    append=True
)

# 加载激活值（支持内存映射）
acts = io.load_activations(
    layer_type="sae",
    layer_idx=15,
    category="violence",
    polarity="pos",
    mmap=True  # 使用内存映射，大文件不OOM
)

# 流式分批读取
for batch in io.iter_activation_batches(
    layer_type="sae", layer_idx=15,
    category="violence", polarity="pos",
    batch_size=32
):
    # 处理每批 [B, T, L, D]
    process(batch)

# 获取样本数量（不加载数据）
num_pos = io.get_num_samples("sae", 15, "violence", "pos")

# 保存/加载元信息
io.save_metadata(layer_type, layer_idx, category, polarity, metadata_list)
metadata = io.load_metadata(layer_type, layer_idx, category, polarity)

# 断点管理
io.save_checkpoint(layer_type, layer_idx, category, polarity, checkpoint)
checkpoint = io.load_checkpoint(layer_type, layer_idx, category, polarity)

# 列出可用层
layers = io.list_available_layers("violence")
# 返回: [("sae", 15), ("sae", 29), ("dit", 15)]

# 打印存储摘要
io.print_summary("violence")
```
