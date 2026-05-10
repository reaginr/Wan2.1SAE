"""
Training Token Sampler

训练阶段 Token 采样器

核心原则：
1. 保持真实 activation distribution
2. 仅允许 RMSNorm + weak spatial stride + soft norm bias
3. 禁止: decorrelation, oversampling, hard bucket, similarity rejection

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from training.sampling_config import (
    TrainingSamplingConfig,
    SpatialSamplingConfig,
    NormBiasConfig,
    MIN_TIMESTEP,
    MAX_TIMESTEP,
)


# ============================================================================
# 工具函数
# ============================================================================

def per_token_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token RMSNorm

    必须保留：与初始化阶段一致

    返回:
        x_norm: 归一化后的张量
        rms: RMS 值 (用于元数据记录)
    """
    if x.dim() == 3:
        B, L, C = x.shape
        x = x.reshape(B * L, C)

    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms

    return x_norm, rms.squeeze(-1)


# ============================================================================
# Training Token Sampler
# ============================================================================

class TrainingTokenSampler:
    """
    训练阶段 Token 采样器

    设计目标：保持真实 activation distribution

    允许的操作：
    - RMSNorm (必须，与初始化一致)
    - Mild spatial stride (仅显存控制)
    - Soft norm bias (弱，保持分布)

    禁止的操作：
    - Decorrelation filter
    - Oversampling (oversample_ratio 必须 = 1.0)
    - Similarity rejection
    - Hard norm bucket sampling
    """

    def __init__(self, config: TrainingSamplingConfig):
        """
        参数:
            config: 训练采样配置
        """
        self.config = config

        # 强制验证禁止项
        assert config.decorrelation_enabled == False, "训练阶段禁止 decorrelation"
        assert config.oversample_ratio == 1.0, "训练阶段禁止 oversampling"
        assert config.hard_bucket_enabled == False, "训练阶段禁止 hard bucket"

        # 随机数生成器
        self._generator = torch.Generator()
        self._generator.manual_seed(config.seed)

        # 统计追踪
        self._stats = {
            "total_tokens_processed": 0,
            "total_samples": 0,
        }

    def sample(
        self,
        activations: torch.Tensor,
        timestep: int,
        layer_idx: int,
        grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        训练采样入口

        参数:
            activations: [B, L, C] 或 [N, C] 原始激活
            timestep: 当前时间步 (必须验证在有效区间)
            layer_idx: 层索引 (用于选择 spatial stride)
            grid_size: (F, H, W) 网格尺寸

        返回:
            sampled: [M, C] 采样后的 token (保持真实分布)
            metadata: 元数据
        """
        device = activations.device

        # ===== Step 0: 验证 timestep =====
        if not (MIN_TIMESTEP <= timestep <= MAX_TIMESTEP):
            # 警告但不报错，由调用方决定如何处理
            pass

        # ===== Step 1: RMSNorm (必须) =====
        # 这是唯一允许的变换，必须与初始化一致
        activations_normed, rms = per_token_rms_norm(activations)

        if activations_normed.dim() == 3:
            B, L, C = activations_normed.shape
            activations_flat = activations_normed.reshape(B * L, C)
        else:
            activations_flat = activations_normed
            B = 1
            L = activations_flat.shape[0]
            C = activations_flat.shape[1]

        original_count = len(activations_flat)

        # ===== Step 2: 可选的 Mild Spatial Stride =====
        # 仅用于显存控制，尽量保持原始分辨率
        stride = self.config.spatial.get_stride(layer_idx)
        if stride > 1 and grid_size is not None:
            stride_indices = self._get_spatial_stride_indices(grid_size, stride, B)
            activations_flat = activations_flat[stride_indices]
            # 注意：这是下采样，不是改变分布的筛选

        # ===== Step 3: Soft Norm Bias (可选) =====
        # 仅允许 weak bias，保持真实分布
        if self.config.norm_bias.enabled and len(activations_flat) > self.config.max_tokens_per_batch:
            norms = activations_flat.norm(dim=-1)
            sampled = self._soft_norm_bias_sample(
                activations_flat,
                norms,
                self.config.max_tokens_per_batch,
            )
        elif len(activations_flat) > self.config.max_tokens_per_batch:
            # 随机均匀采样 (无 bias)
            perm = torch.randperm(
                len(activations_flat),
                generator=self._generator,
                device=device
            )
            sampled = activations_flat[perm[:self.config.max_tokens_per_batch]]
        else:
            sampled = activations_flat

        # 更新统计
        self._stats["total_tokens_processed"] += original_count
        self._stats["total_samples"] += 1

        # 元数据
        metadata = {
            "timestep": timestep,
            "layer_idx": layer_idx,
            "input_tokens": original_count,
            "output_tokens": len(sampled),
            "spatial_stride": stride,
            "rms_mean": rms.mean().item() if rms is not None else None,
            "valid_timestep": MIN_TIMESTEP <= timestep <= MAX_TIMESTEP,
        }

        return sampled, metadata

    def _soft_norm_bias_sample(
        self,
        activations: torch.Tensor,
        norms: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        """
        Soft Norm Bias 采样

        使用概率采样而非硬分层：
        prob ∝ exp(bias_strength * normalized_norm)

        注意：bias_strength 应该很小 (0.1 ~ 0.2)
        """
        device = activations.device

        # 归一化 norm 到 [0, 1]
        norm_min = norms.min()
        norm_max = norms.max()
        normalized_norm = (norms - norm_min) / (norm_max - norm_min + 1e-10)

        # 计算采样概率
        # prob ∝ exp(bias_strength * normalized_norm)
        bias_strength = self.config.norm_bias.bias_strength
        logits = bias_strength * normalized_norm

        # 转换为概率
        probs = torch.softmax(logits, dim=0)

        # 概率采样
        indices = torch.multinomial(
            probs,
            num_samples=target_count,
            replacement=False,
            generator=self._generator
        )

        return activations[indices]

    def _get_spatial_stride_indices(
        self,
        grid_size: Tuple[int, int, int],
        stride: int,
        batch_size: int,
    ) -> torch.Tensor:
        """获取空间 stride 采样索引"""
        F, H, W = grid_size
        device = "cpu"

        indices = []
        for b in range(batch_size):
            base = b * F * H * W
            for f in range(F):
                for h in range(0, H, stride):
                    for w in range(0, W, stride):
                        idx = base + f * H * W + h * W + w
                        indices.append(idx)

        return torch.tensor(indices, dtype=torch.long, device=device)

    def reset_generator(self, seed: Optional[int] = None):
        """重置随机数生成器"""
        if seed is not None:
            self.config.seed = seed
        self._generator.manual_seed(self.config.seed)

    def get_statistics(self) -> Dict[str, Any]:
        """获取采样统计"""
        return self._stats.copy()


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_training_token_sampler(
    max_tokens_per_batch: int = 4096,
    seed: int = 42,
) -> TrainingTokenSampler:
    """创建训练 Token 采样器"""
    config = TrainingSamplingConfig(
        max_tokens_per_batch=max_tokens_per_batch,
        seed=seed,
    )
    return TrainingTokenSampler(config)
