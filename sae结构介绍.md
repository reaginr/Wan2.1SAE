# SAE（稀疏自编码器）结构详解

本文档详细介绍 Wan2.1 项目中 SAE 的代码实现结构，包括网络架构、损失函数和稀疏化策略。

---

## 1. 整体架构

SAE 采用经典的**编码器-解码器**结构（`wan/modules/sae_new.py:56-119`）：

```
输入 x [N, d_model]
    ↓
编码器: W_enc [d_model, d_hidden] + activation
    ↓
隐层 z [N, d_hidden] (稀疏化)
    ↓
解码器: W_dec [d_hidden, d_model]
    ↓
重建 x̂ [N, d_model]
```

---

## 2. 核心组件

### 2.1 SAEConfig 配置类

**文件位置**: `wan/modules/sae_new.py:11-21`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 1536 | DiT 隐藏层维度，也是 SAE 输入/输出维度 |
| `d_hidden` | 6144 | SAE 扩展维度（通常为 4x~8x d_model）|
| `activation` | "relu" | 编码器激活函数（relu/gelu/silu）|
| `sparsity` | "topk" | 稀疏化策略："topk" 或 "l1" |
| `top_k` | 64 | topk 稀疏时保留的激活数量（约 d_hidden 的 1%）|
| `l1_lambda` | 1e-3 | L1 稀疏正则权重（sparsity=l1 时生效）|

### 2.2 网络层定义

**文件位置**: `wan/modules/sae_new.py:68-69`

```python
self.encoder = nn.Linear(config.d_model, config.d_hidden, bias=False)
self.decoder = nn.Linear(config.d_hidden, config.d_model, bias=False)
```

**注意**：编码器和解码器都**没有偏置项**（bias=False），这是经典 SAE 的设计选择。

---

## 3. 重建损失（Reconstruction Loss）

**是的，代码中有明确的重建损失**。

### 3.1 损失计算逻辑

**文件位置**: `wan/modules/sae_new.py:98-119`

```python
def forward(self, x: torch.Tensor, return_loss: bool = False):
    z, _, _ = self.encode(x)      # 编码并稀疏化
    x_hat = self.decode(z)         # 解码重建

    if not return_loss:
        return x_hat, z

    # 重建损失：MSE
    recon_loss = F.mse_loss(x_hat, x)

    # 稀疏损失（仅 L1 模式需要）
    if self.config.sparsity == "topk":
        sparsity_loss = 0  # topk 天然稀疏，无需额外惩罚
    else:
        sparsity_loss = self.config.l1_lambda * z.abs().mean()

    loss = recon_loss + sparsity_loss
    return x_hat, z, loss
```

### 3.2 损失组成

| 损失类型 | 计算公式 | 适用模式 |
|----------|----------|----------|
| 重建损失 | `MSE(x̂, x)` | 所有模式 |
| 稀疏损失 | `λ · mean(\|z\|)` | 仅 L1 模式 |

---

## 4. 两种稀疏化策略

### 4.1 策略一：Top-K 稀疏（默认推荐）

**文件位置**: `wan/modules/sae_new.py:34-53`

```python
def topk_sparsify(z: torch.Tensor, k: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    保留每行前 k 个最大激活，其余置零

    输入: z [N, D]
    输出:
        z_sparse [N, D] - 稀疏化后的张量
        topk_idx [N, k] - 激活特征的索引
        topk_val [N, k] - 激活特征的值
    """
    topk_val, topk_idx = torch.topk(z, k=k, dim=1)
    z_sparse = torch.zeros_like(z)
    z_sparse.scatter_(1, topk_idx, topk_val)
    return z_sparse, topk_idx, topk_val
```

**优点**：
- 可解释性强：每个样本固定只有 k 个活跃特征
- 便于分析：能精确知道哪些特征被激活
- 无需额外稀疏损失（硬约束）

### 4.2 策略二：L1 正则化稀疏

```python
sparsity_loss = self.config.l1_lambda * z.abs().mean()
```

**优点**：
- 整体重建质量可能更好
- 非零激活数量自适应（软约束）

---

## 5. 训练中的损失计算

**文件位置**: `wan/sae_train_t2v_1_3b.py:717-736`

```python
# 训练 SAE
z, x_recon, loss = sae(feats, return_loss=True)

# 计算详细指标
with torch.no_grad():
    recon_mse = ((x_recon - feats) ** 2).mean().item()  # 重建 MSE
    l2_norm = sum(p.pow(2).sum().item() for p in sae.parameters())  # 权重 L2
    sparsity = (z.abs() > 1e-6).float().mean().item()   # 实际稀疏度
    num_activations = (z.abs() > 1e-6).sum().item() / z.shape[0]  # 平均激活数
```

### 5.1 训练监控指标

| 指标 | 说明 |
|------|------|
| `loss` | 总损失（MSE + 稀疏损失）|
| `recon_mse` | 重建 MSE |
| `l2_norm` | 权重 L2 范数 |
| `sparsity` | 实际稀疏比例 |
| `num_activations` | 平均激活特征数 |

---

## 6. 设计特点总结

| 特性 | 实现 | 学术意义 |
|------|------|----------|
| **无偏置** | `bias=False` | 强制特征表示围绕原点对称 |
| **ReLU 激活** | `F.relu()` | 产生非负激活，便于稀疏解释 |
| **Top-K 硬稀疏** | `topk_sparsify()` | 明确控制特征激活数量，可解释性强 |
| **MSE 重建损失** | `F.mse_loss(x_hat, x)` | 经典自编码器目标，重建输入 |
| **可选 L1 正则** | `l1_lambda * z.abs().mean()` | 软稀疏约束（L1 模式）|

---

## 7. 两种模式对比

| 模式 | 损失公式 | 稀疏控制方式 | 适用场景 |
|------|----------|--------------|----------|
| **Top-K** | `MSE(x, x̂)` | 硬约束：每样本只保留 k 个 | 特征可视化、可解释性研究 |
| **L1** | `MSE(x, x̂) + λ·\|z\|₁` | 软约束：惩罚小激活 | 追求更好的整体重建质量 |

**代码默认使用 Top-K 模式**（`sparsity="topk"`, `top_k=64`），这在神经网络可解释性研究中更常用，因为它能产生明确的"这个样本由哪 k 个特征表示"的解释。

---

## 8. 关键文件位置

| 文件 | 说明 |
|------|------|
| `wan/modules/sae_new.py` | SAE 网络结构定义 |
| `wan/sae_train_t2v_1_3b.py` | SAE 训练脚本 |
| `wan/sae_test_t2v_1_3b.py` | SAE 测试脚本 |
| `wan/sae/sae_run_naming.py` | Checkpoint 管理 |

---

## 9. 默认配置参数

```python
model_params = {
    "d_model": 1536,        # Wan 1.3B 模型维度
    "d_hidden": 6144,       # 4x 扩展（推荐 4x~8x）
    "activation": "relu",   # 编码器激活函数
    "sparsity": "topk",     # 稀疏化策略
    "top_k": 64,            # 约 d_hidden 的 1%
    "l1_lambda": 1e-3,      # L1 正则权重
}
```
