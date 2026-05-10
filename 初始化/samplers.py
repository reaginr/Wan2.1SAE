"""
SAE Token Sampler - 严格分离的初始化与训练采样器

核心设计原则：
1. InitTokenSampler: 允许 aggressive decorrelation, diversity engineering
2. TrainTokenSampler: 必须保持真实 activation distribution
3. 两个独立类，不使用 if-else mode 切换

训练阶段禁止：
- decorrelation filter
- oversampling
- similarity rejection
- hard norm bucket sampling

训练仅允许：
- RMSNorm (必须，与初始化一致)
- timestep balancing (可选)
- light spatial downsample (仅显存控制需要时)

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ============================================================================
# 基础工具函数
# ============================================================================

def per_token_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token RMSNorm

    返回:
        x_norm: 归一化后的张量
        rms: 用于反归一化的 rms 值
    """
    if x.dim() == 3:
        B, L, C = x.shape
        x = x.reshape(B * L, C)

    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms

    return x_norm, rms.squeeze(-1)


# ============================================================================
# 初始化阶段采样器
# ============================================================================

@dataclass
class InitSamplerConfig:
    """
    初始化采样器配置

    初始化阶段允许：
    - aggressive decorrelation
    - diversity engineering
    - norm stratified sampling
    - spatial stride
    - oversampling + filtering
    """
    # 目标 token 数
    target_tokens: int = 256000

    # Timestep 覆盖
    timesteps: List[int] = field(default_factory=lambda: [0, 100, 200, 400, 600, 800, 1000])
    samples_per_timestep: int = 40000

    # 空间采样
    spatial_stride: int = 2  # 打破空间局部相关性

    # Norm 分层
    num_norm_buckets: int = 5
    norm_bucket_weights: List[float] = field(
        default_factory=lambda: [0.15, 0.20, 0.25, 0.25, 0.15]
    )

    # 去相关
    decorrelation_threshold: float = 0.3  # aggressive
    oversample_ratio: float = 3.0  # 先采样 3x，再筛选

    # 随机种子
    seed: int = 42


class InitTokenSampler:
    """
    初始化阶段 Token 采样器

    设计目标：最大化特征多样性，为 PCA 提供良好初始化

    允许的操作：
    - Aggressive decorrelation filter
    - Oversampling + similarity rejection
    - Hard norm bucket sampling
    - Spatial stride

    不允许的操作：
    - 改变数据分布的全局统计特性（如 dataset-level normalization）
    """

    def __init__(self, config: InitSamplerConfig):
        self.config = config
        self._generator = torch.Generator()
        self._generator.manual_seed(config.seed)

        # 累积特征（用于去相关）
        self._accumulated_features: Optional[torch.Tensor] = None

    def sample(
        self,
        activations: torch.Tensor,
        timestep: Optional[int] = None,
        grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        采样入口

        参数:
            activations: [B, L, C] 或 [N, C] 原始激活
            timestep: 当前时间步 (用于元数据)
            grid_size: (F, H, W) 网格尺寸

        返回:
            sampled: [M, C] 采样后的 token
            metadata: 采样元数据
        """
        device = activations.device

        # Step 1: RMSNorm (必须)
        activations_normed, rms = per_token_rms_norm(activations)

        if activations_normed.dim() == 3:
            B, L, C = activations_normed.shape
            activations_flat = activations_normed.reshape(B * L, C)
        else:
            activations_flat = activations_normed
            B = 1
            L = activations_flat.shape[0]
            C = activations_flat.shape[1]

        # Step 2: 计算 norm
        norms = activations_flat.norm(dim=-1)

        # Step 3: 空间 stride 采样
        if grid_size is not None and self.config.spatial_stride > 1:
            stride_indices = self._get_spatial_stride_indices(
                grid_size, self.config.spatial_stride, B
            )
            activations_flat = activations_flat[stride_indices]
            norms = norms[stride_indices]

        # Step 4: 过量采样
        n_oversample = int(
            self.config.samples_per_timestep * self.config.oversample_ratio
        )
        n_oversample = min(n_oversample, len(activations_flat))

        # Step 5: Norm 分层采样
        oversample_indices = self._norm_stratified_sample(norms, n_oversample)
        oversampled = activations_flat[oversample_indices]

        # Step 6: 去相关筛选
        if self._accumulated_features is not None and self.config.decorrelation_threshold > 0:
            final_sampled, keep_mask = self._decorrelation_filter(
                oversampled,
                self._accumulated_features,
                self.config.decorrelation_threshold,
            )
        else:
            final_sampled = oversampled
            keep_mask = torch.ones(len(oversampled), dtype=torch.bool)

        # Step 7: 更新累积特征
        if self._accumulated_features is None:
            self._accumulated_features = final_sampled.clone()
        else:
            self._accumulated_features = torch.cat([
                self._accumulated_features,
                final_sampled
            ], dim=0)

        # Step 8: 裁剪到目标数量
        target = self.config.samples_per_timestep
        if len(final_sampled) > target:
            perm = torch.randperm(len(final_sampled), generator=self._generator, device=device)
            final_sampled = final_sampled[perm[:target]]

        # 元数据
        metadata = {
            "timestep": timestep,
            "input_tokens": B * L if activations.dim() == 3 else len(activations),
            "after_stride": len(activations_flat),
            "after_oversample": len(oversampled),
            "after_decorrelation": len(final_sampled[keep_mask]) if keep_mask is not None else len(oversampled),
            "final_tokens": len(final_sampled),
            "rms_mean": rms.mean().item() if rms is not None else None,
        }

        return final_sampled, metadata

    def sample_multi_timestep(
        self,
        activations_by_timestep: Dict[int, torch.Tensor],
        grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        多 timestep 采样

        参数:
            activations_by_timestep: {timestep: [B, L, C]}
            grid_size: (F, H, W)

        返回:
            sampled: [N, C] 所有 timestep 合并后的采样
            metadata: 详细元数据
        """
        all_samples = []
        timestep_metadata = {}

        for timestep in self.config.timesteps:
            if timestep not in activations_by_timestep:
                continue

            sampled, meta = self.sample(
                activations=activations_by_timestep[timestep],
                timestep=timestep,
                grid_size=grid_size,
            )
            all_samples.append(sampled)
            timestep_metadata[timestep] = meta

        # 合并
        final_samples = torch.cat(all_samples, dim=0)

        # 最终裁剪
        if len(final_samples) > self.config.target_tokens:
            perm = torch.randperm(len(final_samples), generator=self._generator)
            final_samples = final_samples[perm[:self.config.target_tokens]]

        metadata = {
            "total_tokens": len(final_samples),
            "timesteps_used": list(timestep_metadata.keys()),
            "timestep_metadata": timestep_metadata,
        }

        return final_samples, metadata

    def reset(self):
        """重置累积状态"""
        self._accumulated_features = None
        self._generator.manual_seed(self.config.seed)

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
            for f in range(0, F, max(1, stride // 2)):
                for h in range(0, H, stride):
                    for w in range(0, W, stride):
                        idx = base + f * H * W + h * W + w
                        indices.append(idx)

        return torch.tensor(indices, dtype=torch.long, device=device)

    def _norm_stratified_sample(
        self,
        norms: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        """Norm 分层采样"""
        device = norms.device
        n_tokens = len(norms)

        # 排序
        sorted_norms, sorted_indices = norms.sort()

        # 计算 bucket 边界
        weights = self.config.norm_bucket_weights
        cumsum = 0
        bucket_boundaries = [0]
        for w in weights[:-1]:
            cumsum += w
            bucket_boundaries.append(int(cumsum * n_tokens))
        bucket_boundaries.append(n_tokens)

        # 从每个 bucket 采样
        sampled_indices = []
        for i in range(self.config.num_norm_buckets):
            start = bucket_boundaries[i]
            end = bucket_boundaries[i + 1]
            bucket_indices = sorted_indices[start:end]

            n_samples = int(target_count * weights[i])
            if len(bucket_indices) <= n_samples:
                sampled_indices.append(bucket_indices)
            else:
                perm = torch.randperm(len(bucket_indices), generator=self._generator, device=device)
                sampled_indices.append(bucket_indices[perm[:n_samples]])

        return torch.cat(sampled_indices)

    def _decorrelation_filter(
        self,
        samples: torch.Tensor,
        existing: torch.Tensor,
        threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """去相关筛选"""
        device = samples.device

        # 归一化
        samples_norm = F.normalize(samples.float(), dim=-1)
        existing_norm = F.normalize(existing.float(), dim=-1)

        # 分块计算相似度
        batch_size = 1000
        n_samples = len(samples_norm)
        keep_mask = torch.ones(n_samples, dtype=torch.bool, device=device)

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = samples_norm[i:end]

            similarity = batch @ existing_norm.T
            max_similarity = similarity.abs().max(dim=-1).values

            keep_mask[i:end] = max_similarity < threshold

        return samples[keep_mask], keep_mask


# ============================================================================
# 训练阶段采样器
# ============================================================================

@dataclass
class TrainSamplerConfig:
    """
    训练采样器配置

    训练阶段必须保持真实 activation distribution

    禁止：
    - decorrelation filter
    - oversampling
    - similarity rejection
    - hard norm bucket sampling

    仅允许：
    - RMSNorm (必须，与初始化一致)
    - timestep balancing (可选)
    - light spatial downsample (仅显存控制)
    """
    # 目标 token 数 (per batch)
    max_tokens_per_batch: int = 4096

    # 是否使用 timestep balancing
    timestep_balancing: bool = True
    timestep_weights: Optional[Dict[int, float]] = None  # None = 均匀

    # 空间下采样 (仅显存控制，尽量小)
    spatial_downsample: bool = False
    spatial_downsample_factor: int = 2  # 仅当显存不足时使用

    # 是否保持原始分布 (推荐 True)
    preserve_distribution: bool = True

    # 随机种子
    seed: int = 42


class TrainTokenSampler:
    """
    训练阶段 Token 采样器

    设计目标：保持真实 activation distribution，避免 distribution shift

    核心原则：
    - 不改变数据分布的任何统计特性
    - RMSNorm 是唯一允许的变换（必须与初始化一致）
    - 采样应该随机且均匀

    不允许的操作：
    - Decorrelation filter
    - Oversampling + filtering
    - Similarity rejection
    - Hard norm bucket sampling
    - 任何改变分布的操作
    """

    def __init__(self, config: TrainSamplerConfig):
        self.config = config
        self._generator = torch.Generator()
        self._generator.manual_seed(config.seed)

        # Timestep 统计 (用于 balancing)
        self._timestep_counts: Dict[int, int] = {}

    def sample(
        self,
        activations: torch.Tensor,
        timestep: Optional[int] = None,
        grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        训练采样入口

        参数:
            activations: [B, L, C] 或 [N, C] 原始激活
            timestep: 当前时间步
            grid_size: (F, H, W) 网格尺寸

        返回:
            sampled: [M, C] 采样后的 token (保持真实分布)
            metadata: 元数据
        """
        device = activations.device

        # Step 1: RMSNorm (必须，与初始化一致)
        # 这是唯一允许的变换
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

        # Step 2: 可选的轻量空间下采样 (仅显存控制)
        if self.config.spatial_downsample and grid_size is not None:
            downsample_indices = self._get_light_downsample_indices(
                grid_size, self.config.spatial_downsample_factor, B
            )
            activations_flat = activations_flat[downsample_indices]

        # Step 3: 随机均匀采样
        # 不使用任何 bias，保持真实分布
        if len(activations_flat) <= self.config.max_tokens_per_batch:
            sampled = activations_flat
        else:
            perm = torch.randperm(
                len(activations_flat),
                generator=self._generator,
                device=device
            )
            sampled = activations_flat[perm[:self.config.max_tokens_per_batch]]

        # 更新 timestep 统计
        if timestep is not None:
            self._timestep_counts[timestep] = self._timestep_counts.get(timestep, 0) + len(sampled)

        # 元数据
        metadata = {
            "timestep": timestep,
            "input_tokens": original_count,
            "output_tokens": len(sampled),
            "rms_mean": rms.mean().item() if rms is not None else None,
            "spatial_downsample_applied": self.config.spatial_downsample,
        }

        return sampled, metadata

    def sample_with_timestep_balancing(
        self,
        activations_by_timestep: Dict[int, torch.Tensor],
        grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        带 timestep balancing 的采样

        确保不同 timestep 的 token 数量均衡

        参数:
            activations_by_timestep: {timestep: [B, L, C]}
            grid_size: (F, H, W)

        返回:
            sampled: [N, C]
            metadata: 元数据
        """
        if not self.config.timestep_balancing:
            # 直接合并
            all_act = torch.cat(list(activations_by_timestep.values()), dim=0)
            return self.sample(all_act, grid_size=grid_size)

        timesteps = list(activations_by_timestep.keys())
        n_timesteps = len(timesteps)

        # 计算每个 timestep 应该采样多少
        tokens_per_timestep = self.config.max_tokens_per_batch // n_timesteps

        all_samples = []
        timestep_metadata = {}

        for timestep in timesteps:
            # 临时调整 max_tokens
            original_max = self.config.max_tokens_per_batch
            self.config.max_tokens_per_batch = tokens_per_timestep

            sampled, meta = self.sample(
                activations=activations_by_timestep[timestep],
                timestep=timestep,
                grid_size=grid_size,
            )

            self.config.max_tokens_per_batch = original_max

            all_samples.append(sampled)
            timestep_metadata[timestep] = meta

        final_samples = torch.cat(all_samples, dim=0)

        metadata = {
            "total_tokens": len(final_samples),
            "timesteps": timesteps,
            "tokens_per_timestep": tokens_per_timestep,
            "timestep_metadata": timestep_metadata,
        }

        return final_samples, metadata

    def get_timestep_statistics(self) -> Dict[int, int]:
        """获取 timestep 采样统计"""
        return self._timestep_counts.copy()

    def reset_statistics(self):
        """重置统计"""
        self._timestep_counts = {}
        self._generator.manual_seed(self.config.seed)

    def _get_light_downsample_indices(
        self,
        grid_size: Tuple[int, int, int],
        factor: int,
        batch_size: int,
    ) -> torch.Tensor:
        """轻量空间下采样索引"""
        F, H, W = grid_size

        indices = []
        for b in range(batch_size):
            base = b * F * H * W
            for f in range(F):
                for h in range(0, H, factor):
                    for w in range(0, W, factor):
                        idx = base + f * H * W + h * W + w
                        indices.append(idx)

        return torch.tensor(indices, dtype=torch.long)


# ============================================================================
# Layer29 Active Ratio 分析
# ============================================================================

def analyze_layer_active_ratio_difference(
    layer_activations: Dict[int, torch.Tensor],
    top_k: int = 128,
    sample_size: int = 5000,
) -> Dict[str, Any]:
    """
    分析 Layer29 与其他层 active ratio 差异的来源

    假设：
    1. Layer29 语义解缠更好 (intrinsic semantic disentanglement)
    2. 初始化问题 (initialization bias)
    3. 特征分布差异 (distribution shift)

    分析指标：
    - Feature diversity (Gini, Entropy)
    - Intrinsic dimension
    - Activation sparsity pattern
    - Cross-layer correlation
    """
    results = {}
    device = "cpu"

    for layer, act in layer_activations.items():
        if act.dim() == 3:
            act = act.reshape(-1, act.shape[-1])

        # RMSNorm
        act_normed, _ = per_token_rms_norm(act)

        # 采样
        n = min(sample_size, len(act_normed))
        perm = torch.randperm(len(act_normed), device=device)[:n]
        act_sample = act_normed[perm]

        # 1. 计算特征 norm 分布
        norms = act_sample.norm(dim=-1)
        norm_cv = (norms.std() / norms.mean()).item()  # 变异系数

        # 2. 计算 TopK 激活模式
        # 模拟 SAE encode
        topk_values, topk_indices = torch.topk(act_sample.abs(), top_k, dim=-1)

        # Firing counts
        firing_counts = torch.zeros(act_sample.shape[-1], device=device)
        firing_counts.scatter_add_(0, topk_indices.flatten(),
                                   torch.ones(topk_indices.numel(), device=device))

        # Active ratio
        active_ratio = (firing_counts > 0).sum().item() / act_sample.shape[-1]

        # Competition entropy
        prob = firing_counts / (firing_counts.sum() + 1e-10)
        prob = prob.clamp(min=1e-10)
        entropy = -(prob * torch.log(prob)).sum().item()
        max_entropy = torch.log(torch.tensor(act_sample.shape[-1], dtype=torch.float32)).item()
        competition_entropy = entropy / max_entropy

        # Gini
        values = firing_counts.float().sort()[0]
        n_features = len(values)
        index = torch.arange(1, n_features + 1, dtype=torch.float32, device=device)
        gini = (2 * (index * values).sum()) / (n_features * values.sum() + 1e-10) - (n_features + 1) / n_features

        # 3. 计算内在维度 (估计)
        # 使用 correlation matrix 的有效秩
        if len(act_sample) > 1000:
            act_subsample = act_sample[:1000]
        else:
            act_subsample = act_sample

        corr_matrix = torch.corrcoef(act_subsample.T)
        eigenvalues = torch.linalg.eigvalsh(corr_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        # 参与比 (participation ratio)
        participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

        results[layer] = {
            "active_ratio": active_ratio,
            "competition_entropy": competition_entropy,
            "gini": gini.item(),
            "norm_cv": norm_cv,
            "participation_ratio": participation_ratio.item(),
            "intrinsic_dimension_estimate": participation_ratio.item(),
        }

    # 分析 Layer29 vs 其他层
    if 29 in results:
        layer29_stats = results[29]
        other_layers = [l for l in results if l != 29]

        avg_other = {
            "active_ratio": sum(results[l]["active_ratio"] for l in other_layers) / len(other_layers),
            "competition_entropy": sum(results[l]["competition_entropy"] for l in other_layers) / len(other_layers),
            "gini": sum(results[l]["gini"] for l in other_layers) / len(other_layers),
            "norm_cv": sum(results[l]["norm_cv"] for l in other_layers) / len(other_layers),
            "participation_ratio": sum(results[l]["participation_ratio"] for l in other_layers) / len(other_layers),
        }

        # 分析结论
        conclusions = []

        # Active ratio 差异
        if layer29_stats["active_ratio"] > avg_other["active_ratio"] * 1.5:
            conclusions.append("Layer29 active ratio 显著高于其他层")

        # Participation ratio (内在维度)
        if layer29_stats["participation_ratio"] > avg_other["participation_ratio"]:
            conclusions.append("Layer29 内在维度更高，特征更解缠")
        else:
            conclusions.append("Layer29 内在维度未显著增加，高 active ratio 可能来自初始化")

        # Norm 变异系数
        if layer29_stats["norm_cv"] < avg_other["norm_cv"]:
            conclusions.append("Layer29 特征更均匀，可能更稳定")

        results["analysis"] = {
            "layer29_stats": layer29_stats,
            "avg_other_layers": avg_other,
            "conclusions": conclusions,
            "hypothesis": (
                "Layer29 高 active ratio 可能来自：\n"
                "1. 该层语义解缠更好 (intrinsic property)\n"
                "2. 深层特征更稀疏激活\n"
                "3. 初始化对该层更适合"
            ),
        }

    return results


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_init_sampler(
    target_tokens: int = 256000,
    seed: int = 42,
) -> InitTokenSampler:
    """创建初始化采样器"""
    config = InitSamplerConfig(
        target_tokens=target_tokens,
        seed=seed,
    )
    return InitTokenSampler(config)


def create_train_sampler(
    max_tokens_per_batch: int = 4096,
    seed: int = 42,
) -> TrainTokenSampler:
    """创建训练采样器"""
    config = TrainSamplerConfig(
        max_tokens_per_batch=max_tokens_per_batch,
        seed=seed,
    )
    return TrainTokenSampler(config)
