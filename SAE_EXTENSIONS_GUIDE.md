# SAE系统扩展使用指南

## 修改总结

本次扩展在保持与现有ckpt格式完全兼容的前提下，增加了以下功能：

### 1. 增强训练恢复 (sae_train_t2v_1_3b.py)

新增命令行参数：
```bash
--resume_from DIR          # 从指定目录恢复
--additional_layers "15,29" # 新增层
--frozen_layers "key1,key2" # 冻结层
--reset_optimizer           # 重置优化器
--reset_step_count          # 重置步数
```

**示例场景：**

**场景A：扩展已有实验到更多层**
```bash
# 之前只训练了层15，现在想同时训练层15和29
python wan/sae_train_t2v_1_3b.py \
    --resume_from sae_runs/exp_layer15_only \
    --run_dir sae_runs/exp_layer15_29 \
    --hook_layers "15,29" \
    --additional_layers "29"
```

**场景B：冻结已有层，只训练新层**
```bash
python wan/sae_train_t2v_1_3b.py \
    --resume_from sae_runs/exp_layer15 \
    --run_dir sae_runs/exp_layer29_only \
    --hook_layers "15,29" \
    --additional_layers "29" \
    --frozen_layers "block_out.layer15" \
    --reset_optimizer
```

**场景C：从其他实验加载特定层**
```bash
# 层15从exp_A加载，层29从exp_B加载
python wan/sae_train_t2v_1_3b.py \
    --run_dir sae_runs/exp_combined \
    --hook_layers "15,29" \
    --resume_from sae_runs/exp_A  # 默认从exp_A加载
# 注意：层29会初始化新权重，因为没有在exp_A中训练
```

### 2. 多层源测试 (sae_test_t2v_1_3b.py)

新增参数：
```bash
--layer_sources "15:run1,29:run2"  # 指定每层的源
```

**示例：**
```bash
python wan/sae_test_t2v_1_3b.py \
    --hook_layers "15,29" \
    --layer_sources "15:sae_runs/exp_style,29:sae_runs/exp_content"
```

---

## 新增模块

### 离线训练 (wan/sae/offline_training/)

**流程：**
```bash
# 1. 采集激活值
python -m wan.sae.offline_training.activation_collector \
    --checkpoint_dir ./Wan2.1-T2V-1.3B \
    --prompt_dir ./prompts \
    --output_dir offline_data/run1 \
    --hook_layers "15,29"

# 2. 离线训练（无需运行DiT，速度更快）
python -m wan.sae.offline_training.train_offline \
    --data_dir offline_data/run1 \
    --run_dir sae_runs/offline_exp1 \
    --epochs 10

# 3. 离线测试
python -m wan.sae.offline_training.test_offline \
    --data_dir offline_data/run1 \
    --run_dir sae_runs/offline_exp1
```

**与现有逻辑兼容：**
- 使用相同的 `SAEConfig` 和 `SparseAutoEncoder`
- 使用相同的 `SAERunLocator` 管理ckpt
- 生成相同格式的 `sae_latest.pt` 和 `sae_config.json`

### 概念提取 (wan/sae/interpretability/)

**使用：**
```bash
# 准备正负提示词文件
# - concepts/violence_positive.txt
# - concepts/violence_negative.txt

python -m wan.sae.interpretability.concept_extractor \
    --run_dir sae_runs/exp1 \
    --positive_file concepts/violence_positive.txt \
    --negative_file concepts/violence_negative.txt \
    --concept_name violence \
    --hook_layers "15,29"
```

**输出：**
```
concept_vectors/
├── violence_block_out.layer15.npy   # 向量数据
└── violence_block_out.layer15.json  # 元信息和统计
```

**与现有逻辑兼容：**
- 直接加载现有的 `sae_latest.pt`
- 使用标准的 `SAERunLocator` 定位ckpt

### 干预生成 (wan/sae/steering/)

**配置文件 (steering_config.json)：**
```json
{
  "prompt": "Two people arguing",
  "interventions": [
    {
      "concept_name": "violence",
      "layer_key": "block_out.layer15",
      "strength": -0.5,
      "method": "additive"
    }
  ]
}
```

**使用：**
```bash
python -m wan.sae.steering.steering_generator \
    --config steering_config.json \
    --run_dir sae_runs/exp1 \
    --concept_dir concept_vectors
```

**与现有逻辑兼容：**
- 加载现有的SAE ckpt
- 加载概念提取模块生成的 `.npy` 文件

---

## 兼容性保证

### ckpt格式兼容
所有模块使用相同的ckpt格式：
```python
# 保存格式
torch.save({
    "state_dict": sae.state_dict(),
    "step": step
}, "sae_latest.pt")

# 配置格式
{
  "sae": {
    "d_model": 1536,
    "d_hidden": 6144,
    "activation": "relu",
    "sparsity": "topk",
    "top_k": 64
  },
  "hook": {
    "hook_mode": "block_out",
    "layer_idx": 15
  }
}
```

### 目录结构兼容
```
sae_runs/exp_name/
├── train_state.json              # 训练状态（新旧格式兼容）
├── block_out.layer15/
│   ├── sae_config.json          # SAE配置
│   └── sae_latest.pt            # 权重（标准格式）
└── block_out.layer29/
    └── ...
```

---

## 完整工作流示例

### 场景：训练SAE检测NSFW内容并用于安全过滤

**Step 1: 在线训练基础层**
```bash
python wan/sae_train_t2v_1_3b.py \
    --hook_layers "15" \
    --run_dir sae_runs/exp_nsfw_base \
    --steps 2000
```

**Step 2: 扩展训练到深层**
```bash
python wan/sae_train_t2v_1_3b.py \
    --resume_from sae_runs/exp_nsfw_base \
    --run_dir sae_runs/exp_nsfw_deep \
    --hook_layers "15,29" \
    --additional_layers "29" \
    --steps 2000
```

**Step 3: 采集激活值用于概念提取**
```bash
# 采集NSFW提示词激活值
python -m wan.sae.offline_training.activation_collector \
    --prompt_dir nsfw_prompts \
    --output_dir offline_data/nsfw

# 采集安全提示词激活值
python -m wan.sae.offline_training.activation_collector \
    --prompt_dir safe_prompts \
    --output_dir offline_data/safe
```

**Step 4: 提取NSFW概念向量**
```bash
python -m wan.sae.interpretability.concept_extractor \
    --run_dir sae_runs/exp_nsfw_deep \
    --positive_file nsfw_prompts.txt \
    --negative_file safe_prompts.txt \
    --concept_name nsfw \
    --hook_layers "15,29"
```

**Step 5: 使用概念向量进行安全干预**
```bash
# 创建steering_config.json
{
  "prompt": "User provided prompt",
  "interventions": [
    {
      "concept_name": "nsfw",
      "layer_key": "block_out.layer15",
      "strength": -0.8,      // 强力抑制NSFW内容
      "method": "additive",
      "timestep_range": [0, 30]
    }
  ]
}

# 运行干预生成
python -m wan.sae.steering.steering_generator \
    --config steering_config.json \
    --run_dir sae_runs/exp_nsfw_deep \
    --concept_dir concept_vectors
```

---

## 文件修改清单

**修改的文件：**
- `wan/sae_train_t2v_1_3b.py` - 添加增强恢复参数
- `wan/sae_test_t2v_1_3b.py` - 添加多层源支持

**新增的文件：**
- `wan/sae/offline_training/activation_collector.py`
- `wan/sae/offline_training/train_offline.py`
- `wan/sae/offline_training/test_offline.py`
- `wan/sae/offline_training/__init__.py`
- `wan/sae/offline_training/README.md`
- `wan/sae/interpretability/concept_extractor.py`
- `wan/sae/interpretability/__init__.py`
- `wan/sae/interpretability/README.md`
- `wan/sae/steering/steering_generator.py`
- `wan/sae/steering/__init__.py`
- `wan/sae/steering/README.md`
- `wan/sae/NEW_MODULES_OVERVIEW.md`
- `wan/sae/enhanced_configs.md` (文档)

**删除的文件：**
- `wan/sae/enhanced_configs.py` (冗余，功能已合并到现有文件)
