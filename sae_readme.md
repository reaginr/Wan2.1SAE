## Wan 1.3B T2V + SAE 说明文档

本说明文档针对当前仓库中新增的 SAE 相关代码，重点解释：

- **训练/推理脚本**：`wan/sae_train_t2v_1_3b.py`、`wan/sae_test_t2v_1_3b.py`
- **SAE 模块结构**：`wan/modules/sae_new.py`
- **Prompt 读取与清洗**：`wan/sae/prompt_io.py`
- **DiT hook 机制**：`wan/sae/hooking.py`
- **checkpoint 命名与定位**：`wan/sae/sae_run_naming.py`

并结合我们的讨论，总结常见 Q&A 和一个**普通多层 SAE 训练**与**SAE 推理**的完整流程。

---

## 一、总体设计思路

- **目标**：在 **Wan 1.3B 文生视频模型 (T2V)** 的 DiT 主干（`WanModel`）中，插入稀疏自编码器（SAE），对隐藏状态进行可解释化分析。
- **基本策略**：
  - **训练阶段**：
    - 使用与真实生成一致的 **多时间步 diffusion 采样流程**（例如 30–50 个 time step）。
    - 在每个 time step 上对选定层（自注意力 / 跨注意力 / 整个 block 输出）做 hook。
    - 对 hook 得到的隐藏状态，训练一个或多个 SAE；**只更新 SAE 参数，不更新 DiT**。
    - 不调用 `VAE.decode`，不生成视频，减少计算与显存。
  - **测试/推理阶段**：
    - 使用同样的 hook 方式与 prompt 数据集。
    - 加载训练好的 SAE，输出中间稀疏特征 `z`，供后续分析与可视化。
- **显存控制**：
  - 每个 time step 只保留当前 step 的隐藏状态，训练 SAE 后立刻释放。
  - 提供可控的 token 采样/截断（`max_tokens_per_key`）与可选的缓存清理。

---

## 二、SAE 模块结构（`wan/modules/sae_new.py`）

### 1. SAEConfig

```python
@dataclass
class SAEConfig:
    d_model: int
    d_hidden: int
    activation: str = "relu"
    sparsity: str = "topk"  # "topk" | "l1"
    top_k: int = 64
    l1_lambda: float = 1e-3
```

- **d_model**：输入维度，等于 DiT 的 `dim`（在 1.3B T2V 中为 1536）。
- **d_hidden**：SAE 隐空间维度，通常为 d_model 的数倍（例如 6144）。
- **activation**：编码器激活函数，支持 `relu/gelu/silu`。
- **sparsity**：
  - `topk`：每个样本只保留前 `top_k` 个最大激活（其余置零）。
  - `l1`：通过 L1 正则鼓励稀疏。

### 2. 稀疏化与前向

核心函数：

- `_apply_activation`：选择激活函数。
- `topk_sparsify`：对 `[N, D]` 的激活按行做 top‑k，返回：
  - `z_sparse`：稀疏后的向量
  - `topk_idx, topk_val`：索引和值（便于解释/可视化）。

前向流程：

```python
z, topk_idx, topk_val = sae.encode(x)  # x: [N, d_model]
x_hat = sae.decode(z)                  # [N, d_model]
loss = mse(x_hat, x) + sparsity_loss
```

其中：

- `sparsity_loss`：
  - 若 `topk`：默认为 0（只通过结构强制稀疏）。
  - 若 `l1`：为 `l1_lambda * z.abs().mean()`。

---

## 三、Prompt 读取与清洗（`wan/sae/prompt_io.py`）

### 1. 输入格式

`prompt_dir` 目录下包含多个 txt 文件，**每行一个 NSFW T2V 提示词**。  
原数据可能存在：

- 中文乱码 / 编码错误：如 `contestresemb靻岋拷mclauabud艧naked`
- 错误的特殊字符：如 `nude neebera aaaaaa foxdrawn :'needed way exhib!!) +) nakednachfractutimed`

### 2. 清洗策略

`PromptCleanConfig`：

- 删除控制字符。
- 规范空白（多空格合并）。
- 只保留可打印 ASCII（过滤乱码）。
- 必须包含英文字符。
- 长度在 `[min_len, max_len]` 范围内（默认 8~400）。

主入口：

- `load_prompts_from_dir(prompt_dir, clean_cfg, limit)`：
  - 递归读取目录下所有 `*.txt` 文件。
  - 按行清洗，过滤掉异常/乱码行。
  - 返回 prompt 列表。
- `batch_iter(items, batch_size, shuffle, seed)`：
  - 将 prompt 列表分 batch，支持固定随机种子 shuffle（方便复现与 resume）。

---

## 四、DiT Hook 机制（`wan/sae/hooking.py`）

### 1. Hook 模式与层数

定义：

- `HookMode = Literal["self_attn", "cross_attn", "self_and_cross", "block_out"]`

说明：

- `self_attn`：只 hook 每个 block 的自注意力输出 `block.self_attn`。
- `cross_attn`：只 hook `block.cross_attn` 输出。
- `self_and_cross`：两者都 hook（会得到两个 key）。
- `block_out`：hook 整个 transformer block 的输出（非常适合作为 SAE 的 residual 表征）。

层数：

- `layer_idx=0` 表示第 0 层 block；`layer_idx=29` 表示第 30 层（1.3B 中共有 30 层）。
- 支持列表形式，如 `"0,5,10,29"`。

### 2. 注册与清理

- `register_dit_hooks(model, hook_layers, hook_mode, on_tensor)`：
  - 对 `WanModel.blocks` 中指定层注册 `forward_hook`。
  - 每次某层被前向调用时，会调用 `on_tensor(key, tensor)`：
    - `key` 形如：`"self_attn.layer0"`, `"block_out.layer29"`。
    - `tensor` 形状 `[B, L, C]`。
  - 返回 hook handle 列表。
- `remove_hooks(handles)`：
  - 训练/推理结束后，统一移除 hooks，防止重复积累。

### 3. Token 维度整理

- `pack_hook_batch(raw, max_tokens_per_key)`：
  - `raw`：`dict[key -> Tensor[B,L,C]]`。
  - 输出：`dict[key -> Tensor[N,C]]`，其中：
    - `N = min(B*L, max_tokens_per_key)`。
  - 可以限制每个 key 的 token 数，防止显存爆炸。

---

## 五、Checkpoint 命名与训练状态（`wan/sae/sae_run_naming.py`）

### 1. SAERunLocator

```python
SAERunLocator(run_dir, hook_mode, layer_idx)
```

- `key()`：`"{hook_mode}.layer{layer_idx}"`，如 `"block_out.layer29"`。
- `artifact_dir()`：`run_dir / key`。
- `config_path()`：`artifact_dir / "sae_config.json"`。
- `latest_ckpt_path()`：`artifact_dir / "sae_latest.pt"`。
- `ckpt_path(step)`：`artifact_dir / f"sae_step{step}.pt"`。

### 2. 全局训练状态

- `train_state_path(run_dir)`：返回 `run_dir / "train_state.json"`。
- `train_state.json` 内容示例：

```json
{
  "step": 1200,
  "max_steps": 2000,
  "hook_mode": "block_out",
  "hook_layers": [0, 15, 29],
  "sae_config": { "...": "..." },
  "sampling_steps": 30,
  "sample_solver": "unipc",
  "seed": 42
}
```

**作用**：

- 记录训练已进行到的 step 数。
- 记录关键 hyper-parameters，便于 resume 与调试。

---

## 六、SAE 训练脚本（`wan/sae_train_t2v_1_3b.py`）

### 1. 功能概述

- 针对 **Wan 1.3B T2V**，在 DiT 中 hook 指定层，训练多个 SAE。
- 每个训练 step：
  - 取一批 prompt（大小为 `batch_prompts`）。
  - 进行完整 diffusion 时间步采样（`sampling_steps`），但不 decode 成视频。
  - 在每个时间步对 hook 的特征训练 SAE，一步 SGD 后立刻释放中间特征。
- 具备：
  - **详细日志输出（logging）**
  - **异常处理（try/except）**
  - **中断恢复（--resume）**

### 2. 关键参数（命令行）

- 数据与模型：
  - `--checkpoint_dir`：Wan 模型权重目录（T2V 1.3B）。
  - `--prompt_dir`：包含多个 NSFW prompt txt 的目录。
  - `--run_dir`：SAE 输出目录（checkpoint 与状态均在此目录下）。
- Hook 设置：
  - `--hook_mode`：`self_attn | cross_attn | self_and_cross | block_out`。
  - `--hook_layers`：例如 `"0,15,29"`。
- 训练流程：
  - `--batch_prompts`：每步使用多少个 prompt。
  - `--max_prompts`：最多加载多少 prompt（上限）。
  - `--steps`：训练步数（外层 step，一步对应一个 prompt batch 完整采样）。
  - `--sampling_steps`：每个 batch 的 diffusion 时间步数（建议 30–50）。
  - `--sample_solver`：`unipc` / `dpm++`。
  - `--shift`：噪声日程 shift。
- SAE 结构：
  - `--d_model`：输入维度（默认 1536）。
  - `--d_hidden`：隐空间维度（例如 6144）。
  - `--activation`：`relu / gelu / silu`。
  - `--sparsity`：`topk / l1`。
  - `--top_k`：top‑k 稀疏的 k 值。
  - `--l1_lambda`：L1 正则权重。
- 训练控制：
  - `--save_every`：每多少 step 保存一次 checkpoint。
  - `--lr`：学习率。
  - `--device_id`：GPU id。
  - `--seed`：随机种子（用于 prompt shuffle 与噪声）。
  - `--max_tokens_per_key`：每个 hook key 最多使用多少 token 参与训练。
  - `--empty_cache_every`：每多少 step 调一次 `torch.cuda.empty_cache()`，0 表示不调。
  - `--resume`：从 `run_dir` 中现有状态恢复训练。
- CFG 与显存优化：
  - `--use_cfg`：是否使用 classifier-free guidance（更贴近真实生成，但计算翻倍）。
  - `--guide_scale`：CFG scale。
  - `--negative_prompt`：负提示词（空则用默认 neg prompt）。
  - `--offload_text_encoder`：每步将 T5 encoder 放回 CPU 以节省显存。

### 3. 训练流程（伪代码简化版）

**Step 1：初始化**

1. 读取并清洗所有 prompt：`prompts = load_prompts_from_dir(...)`。
2. 按 `batch_prompts` 分 batch，固定 seed shuffle：
  - `prompt_batches = list(batch_iter(prompts, batch_size, shuffle=True, seed))`。
3. 初始化 `WanT2V`（只用 text_encoder + WanModel + VAE 的 z_dim 信息）。
4. 根据 `vae_stride` 与 `patch_size` 计算：
  - latent 形状 `[C_latent, T_latent, H_latent, W_latent]`。
  - `seq_len`（DiT 的最大 token 数）。
5. 为每个 `hook_mode + layer_idx`：
  - 创建或加载对应的 SAE（若 `--resume` 且有 latest ckpt，则加载）。
  - 创建对应的 AdamW 优化器。
  - 保存/更新每个 key 的 `sae_config.json`。
6. 若 `--resume` 且存在 `train_state.json`，读取 `step` 作为起点。

**Step 2：每个训练 step（对应一批 prompt）**

对 `it` 从 `start_step` 到 `steps-1`：

1. 取一个 batch 的 prompts：`batch_prompts = prompt_batches[it % len(prompt_batches)]`。
2. 文本编码：`context = text_encoder(batch_prompts)`。
3. 初始化噪声 latent 列表 `latents`（每个 prompt 一份）。
4. 根据 `sampling_steps` 构造时间步序列 `timesteps`（UniPC 或 DPM++）。
5. 如使用 CFG，则额外为 `n_prompt` 编码 `context_null`。
6. 注册 DiT hooks（按 `hook_mode` 与 `hook_layers`）。
7. 对 `for t in timesteps`（完整时间步）：
  - 用 `timestep = [t]*B` 调用 DiT forward：
    - `noise_pred_cond = model(latents, t, context)`
    - 若 CFG：`noise_pred_uncond = model(latents, t, context_null)`，并组合。
  - 将当前 timestep 产生的 hook 特征 `raw[key]` 整理成 `[N,C]`：
    - `hook_batch = pack_hook_batch(raw, max_tokens_per_key)`。
  - 对每个 key 的 SAE：
    - `loss = mse(x_hat, x) + sparsity_loss`。
    - 反向传播与优化，仅对 SAE 参数。
  - 使用 scheduler 更新每个 prompt 的 latent：
    - `latents[i] = scheduler.step(pred[i], t, latents[i])`。
  - 释放当前 timestep 的中间张量（`del pred, noise_pred_cond, hook_batch`）。
8. 移除 hooks，并释放 scheduler 与 latent 列表。
9. `step += 1`。
10. 更新 `train_state.json`：
  - 写入当前 `step` 与配置。
11. 若满足 `save_every`，对每个 key 保存：
  - `sae_step{step}.pt` 与 `sae_latest.pt`。
12. 若设定了 `empty_cache_every`，定期清理显存缓存。

**Step 3：异常与中断处理**

- 训练主循环被 `KeyboardInterrupt` 中断时：
  - 记录日志。
  - 保存当前 `train_state.json`（带上 `step`）。
  - 保存每个 SAE 的 `sae_latest.pt`。
  - 提示可以用 `--resume` 继续训练。
- 出现其它异常时：
  - `logger.exception` 打印完整 traceback。
  - 同样保存 `train_state.json` 与所有 SAE 的最新权重，并附带 `"error"` 字段。

---

## 七、SAE 测试脚本（`wan/sae_test_t2v_1_3b.py`）

### 1. 功能概述

- 加载训练好的 SAE（通过 hook_mode + layer_idx 定位）。
- 使用与训练相同的 prompt 目录与 hook 设置。
- 对每个 prompt：
  - 进行一次 DiT forward（同样可设置 `sampling_steps`，但当前测试脚本仅跑一轮 forward 收集特征）。
  - 通过 SAE 得到中间稀疏激活 `z`。
  - 保存：
    - `prompt`
    - `hook_type`（self_attn/cross_attn/block_out）
    - `layer_idx`
    - `z_mean`（对 token 维求均值后的 `[d_hidden]` 向量）
    - 若为 top‑k SAE，则额外保存单个 token 的 `topk_idx/topk_val`。

### 2. 输出格式

- `--output_path` 指定的 `.pt` 文件，内容形如：

```python
{
  "results": [
    {
      "prompt": "...",
      "hook_type": "block_out",
      "layer_idx": 29,
      "z_mean": Tensor[d_hidden],
      "topk_idx_token0": Tensor[top_k],   # 可选
      "topk_val_token0": Tensor[top_k]    # 可选
    },
    ...
  ]
}
```

后续可据此做降维（PCA/UMAP）、聚类、特征选择等。

---

## 八、扩展：修改 SAE 采样方法与结构（例如替换为 Transformer）

### 1. 修改 SAE 采样方法

- 目前 `pack_hook_batch` 是“截断前 N 个 token”的策略，你可以改写为：
  - 随机采样 token 子集（按行/按整段）。
  - 按条件（例如高注意力权重位置）采样。
- 修改位置：
  - `wan/sae/hooking.py` 中的 `pack_hook_batch`。

### 2. 修改 SAE 结构为 Transformer

- 你可以在 `wan/modules/sae_new.py` 中：
  - 保留 `SAEConfig`，增加新字段（如 `num_layers`、`num_heads`）。
  - 新建一个 `TransformerSAE` 类：
    - encoder：若干层 Transformer Block，将输入 `[N, d_model]` 视为 `[N, 1, d_model]` 或用额外位置信息。
    - decoder：对隐空间 token 做反向映射。
  - 在训练脚本中，将：

```python
sae = SparseAutoEncoder(sae_cfg).to(device)
```

替换为：

```python
sae = TransformerSAE(sae_cfg).to(device)
```

或根据 `args.sae_type` 分支选择不同结构。

---

## 九、从中断中恢复训练

### 1. 中断保存内容

当训练被 Ctrl+C 或异常中断时，脚本会：

- 在 `run_dir/train_state.json` 中记录：
  - 当前 `step`。
  - 训练总步数 `max_steps`。
  - hook 设置与 SAE 配置等。
- 对每个 `hook_mode + layer_idx` 的 SAE 保存：
  - 最新权重到 `run_dir/{hook_mode}.layer{idx}/sae_latest.pt`。

### 2. 恢复训练方法

1. 保持原来训练命令的参数不变，增加 `--resume` 选项。
2. 程序会：
  - 读取 `train_state.json` 获取 `step`，从下一步开始继续。
  - 为每个 SAE key 检查 `sae_latest.pt`，若存在则加载，否则初始化新 SAE。
3. 由于 prompt batching 使用固定的 `seed` 和 `batch_iter`，resume 后会：
  - 再次构造相同的 `prompt_batches` 顺序。
  - 从 `step` 对应的 batch 开始训练，从而**跳过已经完整训练过的 batch**。

---

## 十、常见问题 Q&A（节选自我们之前的讨论）

### Q1：为什么训练 SAE 时仍要完整跑 30–50 个时间步？

- 因为扩散生成本质是多步演化过程：
  - 真实生成中，每个 time step 的隐藏状态都参与了最终视频质量。
  - 只在第一个 time step（或随机单步）采样，会导致训练数据与真实生成轨迹严重偏离。
- 为了让 SAE 真正解释“模型在整个去噪过程中的内部机制”，训练阶段需要：
  - 使用相同的 time step 调度器（UniPC / DPM++）。
  - 对完整时间步序列上的隐藏状态进行训练。

### Q2：训练 SAE 时是否必须使用 CFG（classifier-free guidance）？

- 不必须，但有差异：
  - **不开 CFG**：
    - 每个 time step 只跑一次 DiT（条件）。
    - 更快、更省显存，但与“真实生成”略有分布差异。
  - **开 CFG**：
    - 每个 time step 跑两次 DiT（cond + uncond）。
    - 更贴近真实生成时的 latent 轨迹。
    - 显存和计算量约翻倍。
- 建议：
  - 如果你的研究目标是“解释真实生成中的行为”，建议开启 CFG，并在后续版本中将 cond/uncond 区分存储。

### Q3：为什么只在每个 timestep 后立刻训练 SAE，而不是把所有 timestep 的特征积累后统一训练？

- 原因是 **显存控制**：
  - 若保存所有 time step 的 hidden states，再统一训练，会占用 n * k * L * C 的空间（n 个 prompt、k 个 time step、L 个 token）。
  - 通过“每个 time step 立即使用特征训练 SAE，并删除临时张量”，可以让显存峰值只与当前 step 有关。

### Q4：`torch.cuda.empty_cache()` 会不会把模型权重从显存卸载？

- 不会。
- 该函数只会把 **已释放但仍在 PyTorch 缓存中的内存块** 交还给 CUDA 驱动。
- 仍被任何 tensor 引用的显存不会被释放（例如 model 参数、优化器状态等）。

---

## 十一、两个完整流程示例

### 1. 普通多层 SAE 训练流程

**目标**：对 DiT 的第 0、15、29 层 `block_out` 训练 SAE。

示例命令：

```bash
python wan/sae_train_t2v_1_3b.py ^
  --checkpoint_dir "path/to/wan_1.3b_checkpoints" ^
  --prompt_dir "path/to/nsfw_prompts_folder" ^
  --run_dir "sae_runs/run1" ^
  --hook_mode block_out ^
  --hook_layers "0,15,29" ^
  --batch_prompts 4 ^
  --max_prompts 2000 ^
  --steps 2000 ^
  --sampling_steps 30 ^
  --d_model 1536 ^
  --d_hidden 6144 ^
  --sparsity topk ^
  --top_k 64 ^
  --lr 1e-3
```

**训练过程中发生的事**（简述）：

1. 从 `prompt_dir` 读取并清洗出若干千条 NSFW prompt。
2. 构建 WanT2V 与 WanModel，计算 latent 形状与 seq_len。
3. 为 3 个 key：
  - `block_out.layer0`、`block_out.layer15`、`block_out.layer29`
   初始化各自 SAE 与优化器，并写入相应的 `sae_config.json`。
4. 对每个训练 step（2000 步）：
  - 取 4 条 prompt，编码为文本 embedding。
  - 初始化噪声 latent，并构造 30 个 time step。
  - 对每个 time step：
    - 调用 DiT forward，获取指定层的 `[B,L,1536]`，整理为 `[N,1536]`。
    - 对每个 key 的 SAE 进行一次前向 + 反向 + 更新。
    - 用 scheduler 更新 latent，进入下一 time step。
  - 每 `save_every` 步保存一次 checkpoint。
  - 保存/更新 `train_state.json`。

### 2. SAE 推理（测试）流程

**目标**：使用训练好的 SAE，对同样的 prompt 数据集进行批量分析。

示例命令：

```bash
python wan/sae_test_t2v_1_3b.py ^
  --checkpoint_dir "path/to/wan_1.3b_checkpoints" ^
  --prompt_dir "path/to/nsfw_prompts_folder" ^
  --run_dir "sae_runs/run1" ^
  --hook_mode block_out ^
  --hook_layers "0,15,29" ^
  --output_path "sae_runs/run1/test_out.pt"
```

流程概述：

1. 加载 `run_dir` 下每个 key 对应的 `sae_config.json` 与 `sae_latest.pt`。
2. 按 batch 迭代 prompt，编码文本，构造 latent 与 time step。
3. hook 指定层输出，并通过 SAE 编码得到 `z`。
4. 对 token 维求均值得到 `z_mean`（每个 prompt 一个向量）。
5. 将 `(prompt, hook_type, layer_idx, z_mean, topk 信息)` 保存到 `test_out.pt`。

后续你可以用任意分析/可视化工具对这些结果继续处理。

---

如需进一步扩展（例如按 CFG 区分 cond/uncond 的 SAE、增加更复杂的可视化脚本等），可以在现有框架上平滑迭代。当前代码已经提供了：**可扩展 SAE 结构、可控 hook 模式、多 time step 训练、显存管理、中断恢复与日志输出** 等基础能力。+