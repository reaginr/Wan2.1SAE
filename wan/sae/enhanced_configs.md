# 增强配置系统

本文档介绍增强的SAE训练配置系统，支持灵活的恢复训练、参数修改和多层管理。

## 背景

原有配置系统的限制：
1. 恢复训练时不能修改参数（如batch_size）
2. 不支持加载已有层并新增其他层
3. 不支持从配置文件直接切换实验模式

增强配置系统解决了这些问题。

## 核心功能

### 1. 配置文件恢复训练

可以直接通过配置文件恢复训练：

```bash
python wan/sae_train_t2v_1_3b.py --config resume_config.json
```

配置文件示例 (`resume_config.json`)：

```json
{
  "checkpoint_dir": "./Wan2.1-T2V-1.3B",
  "prompt_dir": "./nsfw_prompts",

  "sae": {
    "d_model": 1536,
    "d_hidden": 6144,
    "activation": "relu",
    "sparsity": "topk",
    "top_k": 64
  },

  "training": {
    "steps": 3000,
    "batch_prompts": 8,
    "lr": 0.0005
  },

  "hook": {
    "hook_mode": "block_out",
    "hook_layers": [15, 29]
  },

  "resume": {
    "enabled": true,
    "source_run_dir": "sae_runs/exp_layer15_only",
    "additional_layers": [
      {"layer_idx": 29}
    ],
    "frozen_layers": [],
    "reset_optimizer": false,
    "reset_step_count": false
  },

  "ckpt": {
    "run_dir": "sae_runs/exp_layer15_29",
    "save_every": 200
  }
}
```

### 2. 修改训练参数

恢复训练时可以修改参数：

```json
{
  "resume": {
    "enabled": true,
    "source_run_dir": "sae_runs/old_exp"
  },

  "training": {
    "batch_prompts": 8,      // 从4增加到8
    "lr": 0.0005,            // 从0.001调整到0.0005
    "lr_scheduler": "cosine" // 新增学习率调度
  }
}
```

### 3. 加载已有层并新增层

场景：之前只训练了层15，现在想同时训练层15和层29

```json
{
  "hook": {
    "hook_layers": [15, 29]  // 最终要训练的层
  },

  "resume": {
    "enabled": true,
    "source_run_dir": "sae_runs/exp_layer15_only",

    "additional_layers": [
      {"layer_idx": 29}  // 新增层29，使用当前配置初始化
    ],

    "frozen_layers": [],  // 不冻结任何层（都继续训练）
    "load_weights_only": false  // 恢复训练状态
  }
}
```

### 4. 从多个源加载不同层

场景：层15从实验A加载，层29从实验B加载

```json
{
  "hook": {
    "hook_layers": [15, 29]
  },

  "resume": {
    "enabled": true,
    "additional_layers": [
      {
        "layer_idx": 15,
        "source_run_dir": "sae_runs/exp_A"
      },
      {
        "layer_idx": 29,
        "source_run_dir": "sae_runs/exp_B"
      }
    ]
  }
}
```

### 5. 冻结已有层，只训练新层

场景：层15已经训练好，只想训练层29

```json
{
  "hook": {
    "hook_layers": [15, 29]
  },

  "resume": {
    "enabled": true,
    "source_run_dir": "sae_runs/exp_layer15_only",

    "additional_layers": [
      {"layer_idx": 29}
    ],

    "frozen_layers": ["block_out.layer15"],  // 冻结层15
    "reset_optimizer": true  // 重置优化器（因为层15被冻结）
  }
}
```

## 配置参数详解

### ResumeConfig 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `enabled` | bool | 是否启用恢复模式 |
| `source_run_dir` | str | 源实验目录 |
| `load_weights_only` | bool | 仅加载权重，不恢复训练状态 |
| `additional_layers` | list | 要新增的层配置列表 |
| `frozen_layers` | list | 要冻结的层key列表 |
| `reset_optimizer` | bool | 是否重置优化器状态 |
| `reset_step_count` | bool | 是否重置步数计数器 |

### additional_layers 格式

```json
{
  "additional_layers": [
    {
      "layer_idx": 29,                    // 层索引（必填）
      "hook_mode": "block_out",           // hook模式（可选，默认使用当前配置）
      "source_run_dir": "sae_runs/exp1"   // 源目录（可选，不填则初始化新SAE）
    }
  ]
}
```

### 层加载行为

| 场景 | source_run_dir | 行为 |
|------|----------------|------|
| 恢复并继续训练 | 与ckpt.run_dir相同 | 加载权重和训练状态，从断点继续 |
| 迁移学习 | 不同目录 | 加载权重，可选重置优化器和步数 |
| 新增层 | 不填 | 初始化新SAE，使用当前配置 |
| 从其他实验加载 | 指定其他目录 | 从指定目录加载该层权重 |

## 测试配置增强

测试脚本同样支持多层源配置：

```json
{
  "checkpoint_dir": "./Wan2.1-T2V-1.3B",
  "prompt_dir": "./test_prompts",

  "hook": {
    "hook_mode": "block_out",
    "hook_layers": [15, 29]
  },

  "layer_sources": {
    "block_out.layer15": "sae_runs/exp_A",
    "block_out.layer29": "sae_runs/exp_B"
  },

  "output_path": "test_results.pt"
}
```

如果不指定`layer_sources`，则默认使用`default_run_dir`或`run_dir`。

## 命令行使用

### 生成示例配置

```bash
python -c "from wan.sae.enhanced_configs import save_sample_resume_config; save_sample_resume_config('sample_resume.json')"
```

### 使用增强配置训练

```bash
python wan/sae_train_t2v_1_3b.py --config resume_config.json
```

### 命令行覆盖配置参数

```bash
python wan/sae_train_t2v_1_3b.py \
    --config resume_config.json \
    --batch_prompts 8 \
    --steps 5000
```

## 配置文件完整示例

### 场景1：在新服务器上继续训练

```json
{
  "description": "在新服务器上继续训练，增大batch_size",

  "checkpoint_dir": "./Wan2.1-T2V-1.3B",
  "prompt_dir": "./nsfw_prompts",

  "training": {
    "steps": 3000,
    "batch_prompts": 8,       // 之前是4，现在改为8
    "sampling_steps": 30,
    "lr": 0.001,
    "lr_scheduler": "cosine", // 新增余弦调度
    "warmup_steps": 500
  },

  "sae": {
    "d_model": 1536,
    "d_hidden": 6144,
    "sparsity": "topk",
    "top_k": 64
  },

  "hook": {
    "hook_mode": "block_out",
    "hook_layers": [15]
  },

  "resume": {
    "enabled": true,
    "source_run_dir": "sae_runs/exp_old_server",
    "frozen_layers": [],
    "reset_optimizer": false  // 继承优化器状态
  },

  "ckpt": {
    "run_dir": "sae_runs/exp_new_server",
    "save_every": 200
  }
}
```

### 场景2：扩展现有实验到更多层

```json
{
  "description": "在层15基础上增加层29的训练",

  "checkpoint_dir": "./Wan2.1-T2V-1.3B",
  "prompt_dir": "./nsfw_prompts",

  "training": {
    "steps": 2000,
    "batch_prompts": 4,
    "lr": 0.001
  },

  "sae": {
    "d_model": 1536,
    "d_hidden": 6144,
    "sparsity": "topk",
    "top_k": 64
  },

  "hook": {
    "hook_mode": "block_out",
    "hook_layers": [15, 29]    // 最终训练两层
  },

  "resume": {
    "enabled": true,
    "source_run_dir": "sae_runs/exp_layer15",

    "additional_layers": [
      {"layer_idx": 29}        // 层29是新层
    ],

    "frozen_layers": [],       // 层15继续训练（不冻结）
    "load_weights_only": false,
    "reset_optimizer": false
  },

  "ckpt": {
    "run_dir": "sae_runs/exp_layer15_29",
    "save_every": 200
  }
}
```

### 场景3：只训练新层，冻结已有层

```json
{
  "description": "冻结层15，只训练层29",

  "training": {
    "steps": 2000,
    "batch_prompts": 4,
    "lr": 0.001
  },

  "hook": {
    "hook_mode": "block_out",
    "hook_layers": [15, 29]
  },

  "resume": {
    "enabled": true,
    "source_run_dir": "sae_runs/exp_layer15",

    "additional_layers": [
      {"layer_idx": 29}
    ],

    "frozen_layers": ["block_out.layer15"],  // 冻结层15
    "reset_optimizer": true                    // 重置优化器（因为参数变了）
  },

  "ckpt": {
    "run_dir": "sae_runs/exp_layer29_only",
    "save_every": 200
  }
}
```

### 场景4：合并多个实验

```json
{
  "description": "从两个不同实验分别加载层15和层29",

  "training": {
    "steps": 1000,  // 少量微调
    "batch_prompts": 4,
    "lr": 0.0005    // 较低学习率微调
  },

  "hook": {
    "hook_mode": "block_out",
    "hook_layers": [15, 29]
  },

  "resume": {
    "enabled": true,

    "additional_layers": [
      {
        "layer_idx": 15,
        "source_run_dir": "sae_runs/exp_style"    // 从风格实验加载
      },
      {
        "layer_idx": 29,
        "source_run_dir": "sae_runs/exp_content"  // 从内容实验加载
      }
    ],

    "frozen_layers": [],
    "load_weights_only": true,  // 只加载权重，重新训练
    "reset_optimizer": true,
    "reset_step_count": true
  },

  "ckpt": {
    "run_dir": "sae_runs/exp_combined",
    "save_every": 100
  }
}
```

## 训练状态格式

增强的训练状态保存在 `enhanced_train_state.json`：

```json
{
  "global_step": 1500,
  "max_steps": 3000,
  "layer_states": {
    "block_out.layer15": {
      "key": "block_out.layer15",
      "hook_mode": "block_out",
      "layer_idx": 15,
      "step": 1500,
      "loaded_from_ckpt": true,
      "source_ckpt_path": "sae_runs/exp_old/block_out.layer15/sae_latest.pt",
      "is_frozen": false
    },
    "block_out.layer29": {
      "key": "block_out.layer29",
      "hook_mode": "block_out",
      "layer_idx": 29,
      "step": 500,
      "loaded_from_ckpt": false,
      "source_ckpt_path": null,
      "is_frozen": false
    }
  },
  "seed": 42,
  "config_snapshot": {...},
  "history": {
    "start_time": "2025-04-01T10:00:00",
    "last_save_time": "2025-04-01T12:00:00"
  }
}
```

## 兼容性

增强配置系统向后兼容旧版配置：
- 如果检测到旧版 `train_state.json`，自动转换
- 命令行参数 `--resume` 仍然有效
- 原有脚本无需修改即可使用

## 注意事项

1. **路径解析**
   - 配置文件中支持相对路径（相对于项目根目录）
   - 支持 `~` 展开为用户目录

2. **参数验证**
   - 系统会验证source_run_dir是否存在
   - 会检查SAE配置是否兼容
   - 不匹配时会给出警告或错误

3. **显存管理**
   - 新增的层会占用更多显存
   - 注意调整batch_prompts

4. **训练时间**
   - 每增加一层，训练时间大约线性增加
   - 新层初始化后需要一定步数收敛
