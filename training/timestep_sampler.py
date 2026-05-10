"""
Layer-Aware Truncated Gaussian Timestep Sampler

核心功能：
1. Truncated Gaussian sampling (禁止 uniform)
2. Layer-aware distribution
3. Asymmetric low-t penalty
4. Vectorized torch implementation
5. Deterministic seed support

设计原则：
- 训练目标不是 reconstruction 最优，而是 stable sparse feature decomposition
- Timestep distribution must reflect diffusion semantic hierarchy
- Avoid collapse regions dominating SAE learning

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from training.sampling_config import (
    LayerTimestepConfig,
    LAYER_TIMESTEP_CONFIGS,
    MIN_TIMESTEP,
    MAX_TIMESTEP,
)


# ============================================================================
# Truncated Gaussian Sampler
# ============================================================================

class TruncatedGaussianSampler:
    """
    Truncated Gaussian 采样器

    实现 rejection-free 的截断高斯采样

    为什么必须使用 Truncated Gaussian:
    - t < 150: Diffusion 已收敛，representation collapse
    - t > 800: 纯噪声 latent，无 semantic structure
    - Uniform sampling 会导致 SAE 学习错误的分布
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        min_t: int = MIN_TIMESTEP,
        max_t: int = MAX_TIMESTEP,
        low_t_decay_alpha: float = -0.15,
        device: str = "cpu",
    ):
        """
        参数:
            mu: Gaussian 中心
            sigma: Gaussian 标准差
            min_t: 最小有效 timestep
            max_t: 最大有效 timestep
            low_t_decay_alpha: 低 timestep 衰减因子 (负值)
            device: 计算设备
        """
        self.mu = mu
        self.sigma = sigma
        self.min_t = min_t
        self.max_t = max_t
        self.low_t_decay_alpha = low_t_decay_alpha
        self.device = device

        # 预计算归一化常数 (用于 PDF)
        self._precompute_normalization()

    def _precompute_normalization(self):
        """预计算截断区间的归一化常数"""
        # 使用 error function 近似计算截断区间的积分
        z_min = (self.min_t - self.mu) / self.sigma
        z_max = (self.max_t - self.mu) / self.sigma

        # Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
        self._phi_min = 0.5 * (1 + math.erf(z_min / math.sqrt(2)))
        self._phi_max = 0.5 * (1 + math.erf(z_max / math.sqrt(2)))
        self._norm_const = self._phi_max - self._phi_min

    def sample(
        self,
        n_samples: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        采样 timestep

        使用 inverse CDF 方法，rejection-free

        参数:
            n_samples: 采样数量
            generator: 随机数生成器 (支持 deterministic)

        返回:
            timesteps: [n_samples] 采样的 timestep
        """
        # 使用 inverse CDF 方法采样截断高斯
        # 这是 rejection-free 的精确方法

        # 1. 采样均匀分布
        u = torch.rand(n_samples, device=self.device, generator=generator)

        # 2. 映射到截断区间的 CDF 值
        # P(T <= t | min_t <= T <= max_t) = (Φ((t-μ)/σ) - Φ((min_t-μ)/σ)) / Z
        phi_u = self._phi_min + u * self._norm_const

        # 3. Inverse CDF: t = μ + σ * Φ^(-1)(phi_u)
        # 使用 inverse error function
        # Φ^(-1)(p) = sqrt(2) * erf^(-1)(2p - 1)
        erf_inv_arg = 2 * phi_u - 1
        erf_inv_arg = erf_inv_arg.clamp(-0.9999, 0.9999)  # 数值稳定性

        z = math.sqrt(2) * torch.erfinv(erf_inv_arg)
        timesteps = self.mu + self.sigma * z

        # 4. Clamp 到有效区间 (数值精度保证)
        timesteps = timesteps.clamp(self.min_t, self.max_t)

        # 5. 转换为整数
        timesteps = timesteps.round().long()

        return timesteps

    def sample_with_low_t_penalty(
        self,
        n_samples: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        带低 timestep 惩罚的采样

        使用 asymmetric Gaussian weighting:
        p(t) ∝ exp(- (t - μ)^2 / (2σ^2)) * decay_factor(t)

        其中 decay_factor(t) = 1 + α * normalize(t)
        α ∈ [-0.2, 0] 是负偏置，降低 low-t 权重

        参数:
            n_samples: 采样数量
            generator: 随机数生成器

        返回:
            timesteps: [n_samples]
        """
        # 1. 先采样基础截断高斯 (2x 数量)
        base_samples = self.sample(n_samples * 2, generator)

        # 2. 计算每个样本的权重
        weights = self._compute_asymmetric_weights(base_samples)

        # 3. 根据权重重新采样
        weights_normalized = weights / weights.sum()
        indices = torch.multinomial(weights_normalized, n_samples, replacement=False,
                                     generator=generator)

        return base_samples[indices]

    def _compute_asymmetric_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        计算 asymmetric Gaussian 权重

        p(t) ∝ exp(- (t - μ)^2 / (2σ^2)) * (1 + α * normalize(t))
        """
        t = timesteps.float()

        # 基础 Gaussian 权重
        gaussian_weight = torch.exp(-((t - self.mu) ** 2) / (2 * self.sigma ** 2))

        # 归一化 timestep 到 [0, 1]
        t_normalized = (t - self.min_t) / (self.max_t - self.min_t)

        # Low-t decay factor
        # α < 0 意味着 low-t 权重降低
        decay_factor = 1.0 + self.low_t_decay_alpha * t_normalized

        # 合成权重
        weights = gaussian_weight * decay_factor

        # 确保权重为正
        weights = weights.clamp(min=1e-10)

        return weights

    def pdf(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算 PDF 值

        参数:
            t: [N] timestep 值

        返回:
            pdf: [N] 概率密度值
        """
        t = t.float()

        # 标准高斯 PDF
        z = (t - self.mu) / self.sigma
        gaussian_pdf = torch.exp(-0.5 * z ** 2) / (self.sigma * math.sqrt(2 * math.pi))

        # 截断归一化
        truncated_pdf = gaussian_pdf / self._norm_const

        # Mask 超出范围的值
        mask = (t >= self.min_t) & (t <= self.max_t)
        truncated_pdf = truncated_pdf * mask.float()

        return truncated_pdf


# ============================================================================
# Layer-Aware Timestep Sampler
# ============================================================================

class LayerAwareTimestepSampler:
    """
    Layer-Conditioned Truncated Gaussian Timestep Sampler

    核心设计：
    - 每个 layer 有独立的 (μ, σ) 配置
    - 基于 diffusion semantic hierarchy 设计
    - 禁止 uniform sampling
    - 禁止越界采样

    Layer 分布设计原理：
    ┌────────────────────────────────────────────────────────────┐
    │ Layer 14 (μ=650, σ=120): 结构特征，需要较高噪声级别        │
    │   主采样区间: 400-800                                      │
    │   原因: 浅层捕获空间结构，需要更多噪声保留结构信息          │
    ├────────────────────────────────────────────────────────────┤
    │ Layer 19 (μ=550, σ=110): 语义特征，中等噪声级别            │
    │   主采样区间: 300-700                                      │
    │   原因: 中深层开始捕获语义，平衡结构与语义                  │
    ├────────────────────────────────────────────────────────────┤
    │ Layer 24 (μ=420, σ=100): 高级语义，较低噪声级别            │
    │   主采样区间: 200-600                                      │
    │   原因: 深层语义更明确，低噪声级别更清晰                    │
    ├────────────────────────────────────────────────────────────┤
    │ Layer 29 (μ=300, σ=80): 最终表征，最低噪声级别             │
    │   主采样区间: 150-450                                      │
    │   原因: 最深层特征最稀疏，低噪声保留细节                    │
    └────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        layer_configs: Optional[Dict[int, LayerTimestepConfig]] = None,
        device: str = "cpu",
        seed: int = 42,
    ):
        """
        参数:
            layer_configs: 各层的配置 {layer_idx: LayerTimestepConfig}
            device: 计算设备
            seed: 随机种子
        """
        self.layer_configs = layer_configs or LAYER_TIMESTEP_CONFIGS
        self.device = device
        self.seed = seed

        # 创建各层的采样器
        self._samplers: Dict[int, TruncatedGaussianSampler] = {}
        for layer_idx, config in self.layer_configs.items():
            self._samplers[layer_idx] = TruncatedGaussianSampler(
                mu=config.mu,
                sigma=config.sigma,
                min_t=config.min_t,
                max_t=config.max_t,
                low_t_decay_alpha=config.low_t_decay_alpha,
                device=device,
            )

        # 随机数生成器
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)

        # 采样历史 (用于监控)
        self._sampling_history: Dict[int, List[int]] = {layer: [] for layer in self.layer_configs}

    def sample_timestep(
        self,
        layer_id: int,
        batch_size: int,
        use_low_t_penalty: bool = True,
    ) -> torch.Tensor:
        """
        为指定层采样 timestep

        参数:
            layer_id: 层索引
            batch_size: batch 大小
            use_low_t_penalty: 是否使用低 timestep 惩罚

        返回:
            timesteps: [batch_size] 采样的 timestep
        """
        if layer_id not in self._samplers:
            # 如果没有配置，使用默认中间值
            sampler = TruncatedGaussianSampler(
                mu=500, sigma=100,
                min_t=MIN_TIMESTEP, max_t=MAX_TIMESTEP,
                device=self.device,
            )
        else:
            sampler = self._samplers[layer_id]

        # 采样
        if use_low_t_penalty:
            timesteps = sampler.sample_with_low_t_penalty(batch_size, self._generator)
        else:
            timesteps = sampler.sample(batch_size, self._generator)

        # 记录历史
        if layer_id in self._sampling_history:
            self._sampling_history[layer_id].extend(timesteps.tolist())

        return timesteps

    def sample_all_layers(
        self,
        batch_size: int,
        use_low_t_penalty: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        为所有层采样 timestep

        参数:
            batch_size: 每层的 batch 大小
            use_low_t_penalty: 是否使用低 timestep 惩罚

        返回:
            timesteps_by_layer: {layer_idx: [batch_size]}
        """
        result = {}
        for layer_id in self.layer_configs:
            result[layer_id] = self.sample_timestep(
                layer_id, batch_size, use_low_t_penalty
            )
        return result

    def get_layer_distribution_params(self, layer_id: int) -> Tuple[float, float]:
        """获取指定层的分布参数"""
        if layer_id in self.layer_configs:
            config = self.layer_configs[layer_id]
            return config.mu, config.sigma
        return 500.0, 100.0  # 默认值

    def get_sampling_history(self) -> Dict[int, List[int]]:
        """获取采样历史"""
        return self._sampling_history.copy()

    def reset_history(self):
        """重置采样历史"""
        self._sampling_history = {layer: [] for layer in self.layer_configs}

    def reset_generator(self, seed: Optional[int] = None):
        """重置随机数生成器"""
        if seed is not None:
            self.seed = seed
        self._generator.manual_seed(self.seed)


# ============================================================================
# 便捷函数
# ============================================================================

def create_layer_aware_sampler(
    device: str = "cpu",
    seed: int = 42,
) -> LayerAwareTimestepSampler:
    """创建 Layer-Aware Timestep Sampler"""
    return LayerAwareTimestepSampler(
        layer_configs=LAYER_TIMESTEP_CONFIGS,
        device=device,
        seed=seed,
    )


def sample_timesteps_for_layer(
    layer_id: int,
    batch_size: int,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    便捷函数：为指定层采样 timestep

    参数:
        layer_id: 层索引
        batch_size: batch 大小
        device: 设备
        seed: 随机种子

    返回:
        timesteps: [batch_size]
    """
    config = LAYER_TIMESTEP_CONFIGS.get(layer_id, LayerTimestepConfig(
        layer_idx=layer_id, mu=500, sigma=100
    ))

    sampler = TruncatedGaussianSampler(
        mu=config.mu,
        sigma=config.sigma,
        min_t=config.min_t,
        max_t=config.max_t,
        low_t_decay_alpha=config.low_t_decay_alpha,
        device=device,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    return sampler.sample_with_low_t_penalty(batch_size, generator)
