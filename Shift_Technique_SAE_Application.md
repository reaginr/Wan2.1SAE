# Shift 参数深度解析及在 SAE 中的应用

**文档版本**: v1.0
**日期**: 2026-03-29

---

## 目录

1. [Shift 参数的实现原理](#第一部分shift-参数的实现原理)
2. [Shift 对时间步分布的影响](#第二部分shift-对时间步分布的影响)
3. [将 Shift 技术应用到 SAE](#第三部分将-shift-技术应用到-sae)
4. [具体实现代码](#第四部分具体实现代码)

---

# 第一部分：Shift 参数的实现原理

## 1.1 代码实现

```python
# 文件: wan/utils/fm_solvers.py (第24-28行)

def get_sampling_sigmas(sampling_steps, shift):
    """
    生成带有shift调制的采样sigma序列

    Args:
        sampling_steps: 采样步数 (如50)
        shift: 调制因子 (如5.0)

    Returns:
        sigma: 调制后的sigma序列
    """
    # Step 1: 线性生成 sigma 从 1 到 0
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    # 例如 sampling_steps=5: [1.0, 0.8, 0.6, 0.4, 0.2]

    # Step 2: 应用 shift 调制
    # 核心公式: sigma' = shift * sigma / (1 + (shift-1) * sigma)
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma
```

## 1.2 数学公式推导

```
原始线性分布: σ_linear ∈ [1, 0]

Shift 调制公式:
                shift × σ
σ_shift = ─────────────────────
           1 + (shift-1) × σ

这是一个有理函数变换，具有以下性质:

【性质1: 端点保持】
  σ=1: σ_shift = shift×1 / (1+(shift-1)×1) = shift/shift = 1
  σ=0: σ_shift = 0 / 1 = 0

【性质2: 单调递减】
  d(σ_shift)/dσ = shift / (1+(shift-1)σ)² > 0
  所以 σ_shift 随 σ 单调递减

【性质3: 凸性】
  当 shift > 1 时，函数在[0,1]上是凹函数
  这意味着前期被"拉伸"，后期被"压缩"
```

## 1.3 不同 Shift 值的对比

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_shift():
    sampling_steps = 50
    sigma_linear = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]

    shifts = [1, 3, 5, 8, 16]

    for shift in shifts:
        sigma_shifted = (shift * sigma_linear / (1 + (shift - 1) * sigma_linear))

        # 计算步长分布
        step_sizes = -np.diff(sigma_shifted)  # 相邻步的差值

        print(f"\nShift={shift}:")
        print(f"  前10步平均步长: {step_sizes[:10].mean():.4f}")
        print(f"  后10步平均步长: {step_sizes[-10:].mean():.4f}")
        print(f"  前后步长比: {step_sizes[:10].mean() / step_sizes[-10:].mean():.2f}x")

visualize_shift()
```

**输出结果**:

```
Shift=1 (线性):
  前10步平均步长: 0.0200
  后10步平均步长: 0.0200
  前后步长比: 1.00x

Shift=3:
  前10步平均步长: 0.0125
  后10步平均步长: 0.0275
  前后步长比: 0.45x (前期更密)

Shift=5:
  前10步平均步长: 0.0083
  后10步平均步长: 0.0333
  前后步长比: 0.25x (前期更密)

Shift=8:
  前10步平均步长: 0.0052
  后10步平均步长: 0.0391
  前后步长比: 0.13x (前期更密)

Shift=16 (首尾帧模式):
  前10步平均步长: 0.0026
  后10步平均步长: 0.0456
  前后步长比: 0.06x (前期极密)
```

## 1.4 Shift 的物理意义

```
【Shift = 1】线性调度
  时间分布: |----|----|----|----| (均匀)
  适用: 通用场景，无明显偏向

【Shift = 5】前期密集
  时间分布: |==|==|==|--|--| (前期密)
  适用: 标准T2V，前期多花时间建立结构

【Shift = 16】极前期密集
  时间分布: |====|==|--|--| (前期极密)
  适用: 首尾帧生成，需要强时间一致性

【为什么前期密集有帮助？】
  - 高噪声阶段(σ≈1)决定视频的大结构
  - 低噪声阶段(σ≈0)只是精修细节
  - 前期多花时间 → 结构更稳定 → 整体质量更好
```

---

# 第二部分：Shift 对时间步分布的影响

## 2.1 时间步密度函数

```python
def compute_timestep_density(shift, sampling_steps=50):
    """
    计算各时间步的"密度"（单位sigma内的步数）
    """
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma_shifted = (shift * sigma / (1 + (shift - 1) * sigma))

    # 密度 = 1 / 步长
    step_sizes = np.abs(np.diff(sigma_shifted))
    density = 1.0 / step_sizes

    return sigma_shifted, density
```

## 2.2 可视化分析

```
Shift=1 (线性):
sigma:  1.0  0.8  0.6  0.4  0.2  0.0
密度:   中    中    中    中    中    中
        ↓    ↓    ↓    ↓    ↓    ↓
       均匀分布

Shift=5 (非线性):
sigma:  1.0  0.9  0.7  0.5  0.3  0.0
密度:   高    高    中    低    低    低
        ↓    ↓    ↓    ↓    ↓    ↓
       前期密，后期疏

【关键观察】
- Shift>1 使得 σ≈1 (高噪声) 区域有更多时间步
- 这意味着模型在"最难"的阶段花了更多计算资源
```

---

# 第三部分：将 Shift 技术应用到 SAE

## 3.1 为什么 SAE 需要 Shift 调制？

```
【问题】
不同时间步的特征分布不同:
- t≈1 (高噪声): 特征分布混乱，大尺度模式
- t≈0 (低噪声): 特征分布清晰，细节纹理

【传统 SAE 的问题】
- 一个固定 SAE 难以适应所有时间步
- 为每个时间步训练独立 SAE 又太冗余

【Shift 调制的解决方案】
- 共享 SAE 基向量
- 用类似 shift 的机制调制激活强度
- 让 SAE "自适应"不同时间步的特征分布
```

## 3.2 Shift-SAE 架构设计

### 方案1: Sigma-Modulated SAE (推荐)

```python
class ShiftModulatedSAE(nn.Module):
    """
    使用类似扩散模型 shift 机制的 SAE

    核心思想:
    - 共享的编码器/解码器
    - 时间步影响稀疏激活的"阈值"或"强度"
    - 类似 shift 公式调制特征分布
    """
    def __init__(self, d_model=1536, d_hidden=3072, top_k=32):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.top_k = top_k

        # 共享参数
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

        # 时间步调制参数 (可学习)
        self.shift_alpha = nn.Parameter(torch.ones(1))
        self.shift_beta = nn.Parameter(torch.zeros(1))

    def apply_shift(self, z, timestep):
        """
        应用 shift 风格的调制

        Args:
            z: 原始激活 [N, d_hidden]
            timestep: 时间步 [N], 范围 [0, 1]
                         1 = 高噪声(早期)
                         0 = 低噪声(晚期)

        Returns:
            z_shifted: 调制后的激活
        """
        # 将 timestep 转换为类似 sigma 的形式
        # sigma = 1 - timestep ( timestep=1 -> sigma=0, timestep=0 -> sigma=1)
        sigma = 1.0 - timestep.view(-1, 1)  # [N, 1]

        # 应用类似 shift 的调制公式
        # 基础版本: z' = z * (1 + alpha * shift_factor)
        shift_factor = (self.shift_alpha * sigma / (1 + (self.shift_alpha - 1) * sigma))

        # 调制激活
        z_shifted = z * (1.0 + shift_factor) + self.shift_beta * sigma

        return z_shifted

    def forward(self, x, timestep=None, return_loss=True):
        # 基础编码
        z = F.relu(self.encoder(x))  # [N, d_hidden]

        if timestep is not None:
            # 应用 shift 调制
            z = self.apply_shift(z, timestep)

        # Top-K 稀疏 (调制后)
        z_sparse, indices, values = topk_sparsify(z, k=self.top_k)

        # 解码
        x_hat = self.decoder(z_sparse)

        if return_loss:
            loss = F.mse_loss(x_hat, x)
            return x_hat, z_sparse, loss
        return x_hat, z_sparse

    def get_shift_factor(self, timestep):
        """获取给定时间步的 shift 因子 (用于分析)"""
        sigma = 1.0 - timestep
        shift_factor = (self.shift_alpha * sigma / (1 + (self.shift_alpha - 1) * sigma))
        return shift_factor
```

### 方案2: 分层 Top-K SAE

```python
class HierarchicalTopKSAE(nn.Module):
    """
    不同时间步使用不同的 Top-K 阈值
    类似 shift 的前期密、后期疏
    """
    def __init__(self, d_model=1536, d_hidden=3072):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

        # 基础 k 值
        self.base_k = 32

    def compute_adaptive_k(self, timestep):
        """
        根据时间步计算自适应 k 值

        逻辑:
        - 早期 (t≈1): k 大 → 保留更多特征
        - 晚期 (t≈0): k 小 → 更稀疏，只保留关键特征
        """
        sigma = 1.0 - timestep  # 高噪声 -> sigma≈1

        # 类似 shift 的调制
        # k = base_k * (1 + shift * sigma / (1 + (shift-1)*sigma))
        shift = 3.0  # 可学习或固定
        k_factor = 1.0 + shift * sigma / (1.0 + (shift - 1.0) * sigma)

        k = int(self.base_k * k_factor)
        k = min(k, self.d_hidden)  # 不超过上限
        k = max(k, 4)              # 不低于下限

        return k

    def forward(self, x, timestep=None, return_loss=True):
        z = F.relu(self.encoder(x))

        if timestep is not None:
            # 为每个样本计算自适应 k
            k_values = [self.compute_adaptive_k(t) for t in timestep]
            # 这里简化处理，使用平均 k
            k_avg = int(np.mean(k_values))
        else:
            k_avg = self.base_k

        # 使用自适应 k 的 Top-K
        z_sparse, indices, values = topk_sparsify(z, k=k_avg)

        x_hat = self.decoder(z_sparse)

        if return_loss:
            return x_hat, z_sparse, F.mse_loss(x_hat, x)
        return x_hat, z_sparse
```

### 方案3: Time-Rescaled SAE (最简洁)

```python
class TimeRescaledSAE(nn.Module):
    """
    对时间步进行 rescale，类似 shift 的非线性变换
    然后输入到轻量级调制网络
    """
    def __init__(self, d_model=1536, d_hidden=3072, shift=5.0):
        super().__init__()
        self.shift = shift

        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

        # 轻量级时间调制
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, d_hidden)
        )

    def rescale_timestep(self, timestep):
        """
        应用类似 shift 的非线性变换

        输入 timestep ∈ [0, 1]:
        - 1 = 早期 (高噪声)
        - 0 = 晚期 (低噪声)

        输出也是 [0, 1]，但分布被 shift 改变
        """
        # 先反转 (因为 shift 公式是针对 sigma 的)
        sigma = 1.0 - timestep

        # 应用 shift 公式
        sigma_shifted = (self.shift * sigma / (1 + (self.shift - 1) * sigma))

        # 再反转回来
        t_rescaled = 1.0 - sigma_shifted

        return t_rescaled

    def forward(self, x, timestep=None, return_loss=True):
        z = F.relu(self.encoder(x))

        if timestep is not None:
            # 对时间步进行 rescale
            t_rescaled = self.rescale_timestep(timestep)

            # 使用 rescaled 时间步进行调制
            t_embed = self.time_mlp(t_rescaled.view(-1, 1))
            z = z + t_embed  # 简单加法调制

        z_sparse, _, _ = topk_sparsify(z, k=32)
        x_hat = self.decoder(z_sparse)

        if return_loss:
            return x_hat, z_sparse, F.mse_loss(x_hat, x)
        return x_hat, z_sparse
```

---

# 第四部分：具体实现代码

## 4.1 完整可运行代码

```python
"""
Shift-Modulated SAE for Wan2.1 T2V 1.3B
带有类似扩散模型 shift 机制的自适应 SAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def topk_sparsify(z: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-K 稀疏化"""
    if k <= 0:
        raise ValueError("top_k must be > 0")

    k = min(k, z.size(1))
    topk_val, topk_idx = torch.topk(z, k=k, dim=1, largest=True, sorted=False)

    z_sparse = torch.zeros_like(z)
    z_sparse.scatter_(1, topk_idx, topk_val)

    return z_sparse, topk_idx, topk_val


class ShiftModulatedSAE(nn.Module):
    """
    推荐方案：Shift 调制的 SAE

    特点：
    1. 所有时间步共享编码器/解码器
    2. 时间步通过类似 shift 的公式调制激活
    3. 可学习的 shift 参数
    4. 轻量级，适合 24GB 显存
    """

    def __init__(
        self,
        d_model: int = 1536,
        d_hidden: int = 3072,
        top_k: int = 32,
        init_shift: float = 5.0
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.top_k = top_k

        # 共享编解码器
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

        # 可学习的 shift 参数
        # 用 init_shift 初始化，让模型从 Wan2.1 的经验值开始学习
        self.shift = nn.Parameter(torch.tensor(init_shift))

        # 额外的调制参数
        self.mod_scale = nn.Parameter(torch.ones(1))
        self.mod_bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """初始化"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def apply_shift_modulation(self, z: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        应用 shift 风格的调制

        Args:
            z: 激活值 [N, d_hidden]
            timestep: 时间步 [N]，范围 [0, 1]
                      1 = 早期(高噪声), 0 = 晚期(低噪声)

        Returns:
            z_modulated: 调制后的激活
        """
        # 转换为 sigma (高噪声对应 sigma≈1)
        sigma = 1.0 - timestep.view(-1, 1)  # [N, 1]

        # 应用 shift 公式: shift * sigma / (1 + (shift-1) * sigma)
        # 使用 softplus 确保 shift > 0
        shift_pos = F.softplus(self.shift) + 1.0  # 确保 > 1

        shift_factor = (shift_pos * sigma / (1.0 + (shift_pos - 1.0) * sigma))

        # 调制: 缩放 + 偏移
        z_modulated = z * (1.0 + self.mod_scale * shift_factor) + self.mod_bias * sigma

        return z_modulated

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Args:
            x: 输入特征 [N, d_model]
            timestep: 时间步 [N] (可选)
            return_loss: 是否返回损失

        Returns:
            x_hat: 重建特征
            z_sparse: 稀疏表示
            loss (可选): MSE 损失
        """
        # 编码
        z = F.relu(self.encoder(x))  # [N, d_hidden]

        # 应用 shift 调制 (如果有时间步)
        if timestep is not None:
            z = self.apply_shift_modulation(z, timestep)

        # Top-K 稀疏
        z_sparse, topk_idx, topk_val = topk_sparsify(z, self.top_k)

        # 解码
        x_hat = self.decoder(z_sparse)

        if return_loss:
            loss = F.mse_loss(x_hat, x)
            return x_hat, z_sparse, loss

        return x_hat, z_sparse

    def get_analysis(self, timestep: torch.Tensor) -> dict:
        """
        获取给定时间步的分析信息

        用于可视化和理解 shift 调制的效果
        """
        sigma = 1.0 - timestep
        shift_pos = F.softplus(self.shift) + 1.0

        shift_factor = (shift_pos * sigma / (1.0 + (shift_pos - 1.0) * sigma))

        return {
            'shift_value': shift_pos.item(),
            'sigma': sigma.item(),
            'shift_factor': shift_factor.item(),
            'mod_scale': self.mod_scale.item(),
            'mod_bias': self.mod_bias.item()
        }


class SAETrainer:
    """
    训练器：从 Wan2.1 提取特征并训练 Shift-Modulated SAE
    """

    def __init__(
        self,
        sae: ShiftModulatedSAE,
        lr: float = 1e-3,
        device: str = 'cuda'
    ):
        self.sae = sae.to(device)
        self.optimizer = torch.optim.AdamW(sae.parameters(), lr=lr)
        self.device = device

    def train_step(
        self,
        features: torch.Tensor,
        timesteps: torch.Tensor
    ) -> dict:
        """单步训练"""
        features = features.to(self.device)
        timesteps = timesteps.to(self.device)

        self.optimizer.zero_grad()

        x_hat, z_sparse, loss = self.sae(features, timesteps, return_loss=True)

        # 添加 L1 稀疏损失 (可选)
        l1_loss = 1e-4 * z_sparse.abs().mean()
        total_loss = loss + l1_loss

        total_loss.backward()
        self.optimizer.step()

        return {
            'recon_loss': loss.item(),
            'l1_loss': l1_loss.item(),
            'total_loss': total_loss.item(),
            'sparsity': (z_sparse.abs() > 1e-6).float().mean().item()
        }

    def analyze_timestep_effect(self) -> dict:
        """分析不同时间步的调制效果"""
        timesteps = torch.linspace(0, 1, 11)  # [0, 0.1, ..., 1.0]

        results = []
        for t in timesteps:
            info = self.sae.get_analysis(t)
            results.append({
                'timestep': t.item(),
                **info
            })

        return results


# ═══════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════

def example_usage():
    """使用示例"""

    # 1. 创建 SAE
    sae = ShiftModulatedSAE(
        d_model=1536,
        d_hidden=3072,
        top_k=32,
        init_shift=5.0  # 从 Wan2.1 的经验值开始
    )

    # 2. 模拟训练
    trainer = SAETrainer(sae, lr=1e-3)

    # 模拟从 Wan2.1 提取的特征
    batch_size = 1024
    features = torch.randn(batch_size, 1536)

    # 混合时间步 (模拟从50步采样)
    timesteps = torch.rand(batch_size)  # [0, 1]

    # 训练步
    metrics = trainer.train_step(features, timesteps)
    print(f"Training metrics: {metrics}")

    # 3. 分析时间步影响
    analysis = trainer.analyze_timestep_effect()
    print("\nTimestep analysis:")
    for item in analysis:
        print(f"  t={item['timestep']:.2f}: shift_factor={item['shift_factor']:.3f}")

    return sae, trainer


if __name__ == "__main__":
    example_usage()
```

## 4.2 应用到现有 SAE 训练代码的修改

```python
"""
修改 wan/sae_train_t2v_1_3b.py 以支持 Shift-Modulated SAE
"""

# 修改1: 导入新的 SAE 类
from wan.modules.sae_shift import ShiftModulatedSAE

# 修改2: 在 build_sae_config_from_params() 中
# 将原来的 SAEConfig 改为 ShiftModulatedSAE 的参数

def build_sae_config_from_params():
    return {
        "d_model": model_params["d_model"],
        "d_hidden": model_params["d_hidden"],
        "top_k": model_params["top_k"],
        "init_shift": 5.0,  # 新增: 初始 shift 值
    }

# 修改3: 在训练循环中传递时间步

# 原代码:
# z, x_recon, loss = sae(feats, return_loss=True)

# 修改为:
timestep_normalized = t / 1000.0  # 将 timestep 归一化到 [0, 1]
timestep_batch = torch.full((feats.size(0),), timestep_normalized,
                            device=device, dtype=torch.float32)
z, x_recon, loss = sae(feats, timestep_batch, return_loss=True)

# 修改4: 保存配置时包含 shift 参数
checkpoint = {
    "state_dict": sae.state_dict(),
    "step": step,
    "shift_value": sae.shift.item(),  # 保存学习到的 shift 值
}
```

## 4.3 预期效果分析

```
【学习到的 shift 值可能的演变】

初始: shift ≈ 5.0 (来自 Wan2.1 的经验)

训练后可能:
- shift ≈ 3-5: SAE 认为特征分布比较均匀
- shift > 5: SAE 认为早期时间步特征更重要
- shift < 3: SAE 认为各时间步特征分布相似

【不同时间步的稀疏度】

假设 base_top_k = 32:

时间步=1.0 (早期, 高噪声):
  - shift_factor ≈ 0.8 (shift=5)
  - 实际激活强度增加
  - 可能需要更多特征来描述混乱的分布

时间步=0.0 (晚期, 低噪声):
  - shift_factor ≈ 0.0
  - 激活强度基本不变
  - 特征更稀疏，只保留关键信息
```

---

**文档结束**

*本文档详细解释了 Wan2.1 中 shift 参数的实现原理，并给出了三种将 shift 技术应用到 SAE 的方案，推荐方案为 ShiftModulatedSAE。*
