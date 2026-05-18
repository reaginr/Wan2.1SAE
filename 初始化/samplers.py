"""
SAE Token Sampler - 严格分离的初始化与训练采样器

核心设计原则：
1. InitTokenSampler: 允许 aggressive decorrelation, diversity engineering
2. TrainTokenSampler: 必须保持真实 activation distribution
3. ParamTestTokenSampler: 参数测试阶段专用，保持时空局部性 + soft norm bias

训练阶段禁止：
- decorrelation filter
- oversampling
- similarity rejection
- hard norm bucket sampling

训练仅允许：
- RMSNorm (必须，与初始化一致)
- timestep balancing (可选)
- light spatial downsample (仅显存控制需要时)

参数测试阶段允许：
- temporal chunk 采样 (保持时间局部性)
- spatial block 采样 (保持空间局部性)
- soft norm bias (轻度偏向高 norm token)
- mild decorrelation (不破坏局部结构)

作者：Claude
日期：2026-05-10 (更新: 2026-05-17)
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


# ============================================================================
# 参数测试阶段采样器 (新增)
# ============================================================================

@dataclass
class ParamTestSamplerConfig:
    """
    参数测试阶段采样器配置

    设计目标：在保持时空局部性的前提下，提供适度的特征多样性

    允许的操作：
    - Temporal chunk 采样 (连续帧)
    - Spatial block 采样 (局部 patch)
    - Soft norm bias (轻度偏向高 norm token)
    - Mild decorrelation (不破坏局部结构)

    禁止的操作：
    - Aggressive decorrelation
    - Oversampling + hard filtering
    - Global random token sampling
    """
    # 目标 token 数 (per timestep)
    tokens_per_timestep: int = 1536  # 可配置 1536~2048

    # Timestep 参数
    num_timesteps_per_prompt: int = 5  # 每个 prompt 采样的 timestep 数

    # Temporal chunk 参数
    temporal_chunk_size: int = 3  # 每个 timestep 采样的连续帧数 (2~3)
    temporal_frames_total: int = 11  # 总帧数 (latent T')

    # Spatial block 参数
    spatial_block_size: int = 8  # 8×8 patch block
    spatial_grid_h: int = 30  # latent 高度
    spatial_grid_w: int = 52  # latent 宽度
    num_spatial_blocks: int = 24  # 采样多少个 spatial block

    # Norm bias 参数
    norm_bias_enabled: bool = True
    norm_bias_strength: float = 0.3  # 0=无偏置, 1=强偏置
    # 采样概率公式: p ∝ norm^(norm_bias_strength)

    # Mild decorrelation 参数
    decorrelation_enabled: bool = True
    decorrelation_threshold: float = 0.7  # 比 init 的 0.3 宽松
    # 仅在 block 内部做轻度去相关

    # 随机种子
    seed: int = 42


class ParamTestTokenSampler:
    """
    参数测试阶段 Token 采样器

    设计目标：验证超参数有效性，保持时空局部性

    采样流程：
    1. Temporal Chunk: 采样连续 2~3 帧，保持时间局部性
    2. Spatial Block: 在每帧内随机采样 8×8 patch block
    3. Token Selection: 在 block 内随机选择，soft norm bias
    4. Mild Decorrelation: 轻度去相关，不破坏局部结构
    5. RMSNorm: 归一化 (必须)

    与其他采样器的区别：
    - vs InitTokenSampler: 更温和，保持局部性，不用 aggressive decorrelation
    - vs TrainTokenSampler: 有结构化采样，有 soft norm bias，有 mild decorrelation
    """

    def __init__(self, config: ParamTestSamplerConfig):
        self.config = config
        self._generator = torch.Generator()
        self._generator.manual_seed(config.seed)

        # 记录已采样的 block 特征 (用于 mild decorrelation)
        self._block_features: Optional[torch.Tensor] = None

    def sample(
        self,
        activations: torch.Tensor,
        timestep: Optional[int] = None,
        grid_size: Optional[Tuple[int, int, int]] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        参数测试阶段采样入口

        参数:
            activations: [B, L, C] 或 [F*H*W, C] 原始激活
            timestep: 当前时间步 (用于元数据)
            grid_size: (F, H, W) 网格尺寸，默认使用配置值
            layer_idx: 层索引 (用于元数据)

        返回:
            sampled: [M, C] 采样后的 token
            metadata: 采样元数据
        """
        device = activations.device

        # 确定 grid 尺寸
        if grid_size is None:
            F_total = self.config.temporal_frames_total
            H = self.config.spatial_grid_h
            W = self.config.spatial_grid_w
        else:
            F_total, H, W = grid_size

        # Step 1: RMSNorm (必须)
        activations_normed, rms = per_token_rms_norm(activations)

        # Flatten
        if activations_normed.dim() == 3:
            B, L, C = activations_normed.shape
            activations_flat = activations_normed.reshape(B * L, C)
        else:
            activations_flat = activations_normed
            B = 1
            original_L = activations_flat.shape[0]
            C = activations_flat.shape[1]

        # 验证 token 数是否匹配 grid
        expected_tokens = F_total * H * W
        if len(activations_flat) < expected_tokens:
            # 如果 token 数不足，使用简化采样
            return self._simple_sample(activations_flat, timestep, device)

        # Step 2: Temporal Chunk 采样
        temporal_chunks = self._sample_temporal_chunks(F_total, device)
        # temporal_chunks: list of (start_frame, end_frame) tuples

        # Step 3: Spatial Block 采样
        all_sampled_tokens = []
        block_metadata = []

        for chunk_start, chunk_end in temporal_chunks:
            chunk_tokens = []

            for f in range(chunk_start, chunk_end):
                # 在该帧内采样 spatial blocks
                frame_tokens, frame_meta = self._sample_spatial_blocks(
                    activations_flat=activations_flat,
                    frame_idx=f,
                    H=H,
                    W=W,
                    F_total=F_total,
                    C=C,
                    device=device,
                )
                chunk_tokens.append(frame_tokens)

            # 合并该 chunk 的所有 token
            chunk_all = torch.cat(chunk_tokens, dim=0)
            all_sampled_tokens.append(chunk_all)

            block_metadata.append({
                "chunk_frames": (chunk_start, chunk_end),
                "n_tokens": len(chunk_all),
            })

        # Step 4: 合并所有 token
        sampled = torch.cat(all_sampled_tokens, dim=0)

        # Step 5: Mild Decorrelation (可选)
        if self.config.decorrelation_enabled and self._block_features is not None:
            sampled, keep_mask = self._mild_decorrelation_filter(
                sampled, self._block_features
            )
        else:
            keep_mask = torch.ones(len(sampled), dtype=torch.bool, device=device)

        # Step 6: 裁剪到目标数量
        target = self.config.tokens_per_timestep
        if len(sampled) > target:
            # 使用 soft norm bias 采样
            sampled = self._soft_norm_bias_sample(sampled, target, device)

        # 更新 block 特征缓存 (用于后续 mild decorrelation)
        if self._block_features is None:
            self._block_features = sampled.clone()
        else:
            # 只保留最近的特征
            self._block_features = torch.cat([
                self._block_features[-min(5000, len(self._block_features)):],
                sampled
            ], dim=0)

        # 元数据
        metadata = {
            "timestep": timestep,
            "layer_idx": layer_idx,
            "input_tokens": len(activations_flat),
            "output_tokens": len(sampled),
            "temporal_chunks": temporal_chunks,
            "block_metadata": block_metadata,
            "rms_mean": rms.mean().item() if rms is not None else None,
            "decorrelation_applied": self.config.decorrelation_enabled,
        }

        return sampled, metadata

    def _sample_temporal_chunks(
        self,
        F_total: int,
        device: torch.device,
    ) -> List[Tuple[int, int]]:
        """
        采样连续的时间 chunks

        返回:
            [(start_frame, end_frame), ...] 列表
        """
        chunk_size = self.config.temporal_chunk_size
        num_chunks = self.config.num_timesteps_per_prompt

        chunks = []
        available_starts = list(range(0, F_total - chunk_size + 1))

        # 使用 Python random 避免设备不匹配问题
        import random
        random.seed(self.config.seed)

        for _ in range(min(num_chunks, len(available_starts))):
            if not available_starts:
                break

            # 随机选择一个起始帧
            idx = random.randint(0, len(available_starts) - 1)
            start = available_starts.pop(idx)
            end = min(start + chunk_size, F_total)

            chunks.append((start, end))

        return chunks

    def _sample_spatial_blocks(
        self,
        activations_flat: torch.Tensor,
        frame_idx: int,
        H: int,
        W: int,
        F_total: int,
        C: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        在单帧内采样 spatial blocks

        返回:
            tokens: [N, C] 采样的 token
            metadata: block 信息
        """
        block_size = self.config.spatial_block_size
        num_blocks = self.config.num_spatial_blocks

        # 计算该帧的起始索引
        frame_start = frame_idx * H * W

        # 计算可以放置多少个 block
        num_blocks_h = H // block_size
        num_blocks_w = W // block_size

        if num_blocks_h == 0 or num_blocks_w == 0:
            # 如果 block 太大，随机采样 (不使用 generator 避免 device 问题)
            indices = torch.randperm(H * W, device=device)[:min(64, H * W)]
            frame_tokens = activations_flat[frame_start:frame_start + H * W]
            return frame_tokens[indices], {"blocks_sampled": 0}

        # 随机选择 block 位置
        all_block_positions = []
        for bh in range(num_blocks_h):
            for bw in range(num_blocks_w):
                all_block_positions.append((bh, bw))

        # 随机采样 num_blocks 个 block (使用 Python random 避免 device 问题)
        import random
        random.seed(self.config.seed + frame_idx)
        n_sample_blocks = min(num_blocks, len(all_block_positions))
        selected_indices = random.sample(range(len(all_block_positions)), n_sample_blocks)

        selected_blocks = [all_block_positions[i] for i in selected_indices]

        # 从每个 block 中采样 token
        all_tokens = []
        for bh, bw in selected_blocks:
            # Block 的起始位置
            block_start_h = bh * block_size
            block_start_w = bw * block_size

            # Block 内的所有 token 索引
            block_indices = []
            for dh in range(block_size):
                for dw in range(block_size):
                    h = block_start_h + dh
                    w = block_start_w + dw
                    if h < H and w < W:
                        idx = frame_start + h * W + w
                        block_indices.append(idx)

            # 在 block 内随机采样 (可以加 soft norm bias)
            block_indices = torch.tensor(block_indices, dtype=torch.long, device=device)

            # 每个 block 采样约 8 个 token
            n_per_block = max(1, block_size * block_size // 8)

            if len(block_indices) <= n_per_block:
                sampled_indices = block_indices
            else:
                # 随机采样 (不使用 generator 避免 device 问题)
                perm_block = torch.randperm(len(block_indices), device=device)
                sampled_indices = block_indices[perm_block[:n_per_block]]

            all_tokens.append(activations_flat[sampled_indices])

        tokens = torch.cat(all_tokens, dim=0)

        metadata = {
            "blocks_sampled": len(selected_blocks),
            "tokens_per_block": n_per_block if selected_blocks else 0,
        }

        return tokens, metadata

    def _soft_norm_bias_sample(
        self,
        tokens: torch.Tensor,
        target_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Soft norm bias 采样

        高 norm token 的采样概率略大，但不完全排除低 norm token
        """
        if not self.config.norm_bias_enabled:
            # 不使用 generator 避免 device 问题
            perm = torch.randperm(len(tokens), device=device)
            return tokens[perm[:target_count]]

        # 计算 norm
        norms = tokens.norm(dim=-1)

        # 计算采样概率 (soft bias)
        # p ∝ norm^(norm_bias_strength)
        # 归一化到 [0, 1]
        norms_normalized = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)

        # 应用 bias
        strength = self.config.norm_bias_strength
        weights = (norms_normalized + 0.1) ** strength  # +0.1 避免 0 权重
        weights = weights / weights.sum()

        # 根据权重采样 (不使用 generator 避免 device 问题)
        indices = torch.multinomial(
            weights,
            num_samples=min(target_count, len(tokens)),
            replacement=False,
        )

        return tokens[indices]

    def _mild_decorrelation_filter(
        self,
        samples: torch.Tensor,
        existing: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mild decorrelation 筛选

        比 init 阶段宽松，阈值更高 (0.7 vs 0.3)
        """
        device = samples.device

        # 归一化
        samples_norm = F.normalize(samples.float(), dim=-1)
        existing_norm = F.normalize(existing.float(), dim=-1)

        # 分块计算相似度
        batch_size = 1000
        n_samples = len(samples_norm)
        keep_mask = torch.ones(n_samples, dtype=torch.bool, device=device)

        threshold = self.config.decorrelation_threshold

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = samples_norm[i:end]

            # 计算 max similarity
            similarity = batch @ existing_norm.T
            max_similarity = similarity.abs().max(dim=-1).values

            # 只过滤高度相似的样本
            keep_mask[i:end] = max_similarity < threshold

        return samples[keep_mask], keep_mask

    def _simple_sample(
        self,
        activations_flat: torch.Tensor,
        timestep: Optional[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        简化采样 (当 token 数不足时使用)
        """
        target = min(self.config.tokens_per_timestep, len(activations_flat))
        # 不使用 generator 避免 device 问题
        perm = torch.randperm(len(activations_flat), device=device)
        sampled = activations_flat[perm[:target]]

        metadata = {
            "timestep": timestep,
            "input_tokens": len(activations_flat),
            "output_tokens": len(sampled),
            "mode": "simple_fallback",
        }

        return sampled, metadata

    def reset(self):
        """重置采样器状态"""
        self._block_features = None
        # 重置时不需要手动设置种子，因为使用 Python random


# ============================================================================
# 截断高斯 Timestep 采样器 (用于层特定参数)
# ============================================================================

class TruncatedGaussianTimestepSampler:
    """
    截断高斯 Timestep 采样器

    根据 TODO_list_v4 规范：
    - t ~ TruncatedGaussian(μ_layer, σ, clamp=[150, 800])
    - 层特定参数：
      - Layer 14: μ=650, σ=80
      - Layer 19: μ=550, σ=80
      - Layer 24: μ=450, σ=70
      - Layer 29: μ=350, σ=60
    """

    # 默认层特定参数 (来自 TODO_list_v4)
    DEFAULT_LAYER_PARAMS = {
        14: {"mu": 650, "sigma": 80, "min_t": 150, "max_t": 800},
        19: {"mu": 550, "sigma": 80, "min_t": 150, "max_t": 800},
        24: {"mu": 450, "sigma": 70, "min_t": 150, "max_t": 800},
        29: {"mu": 350, "sigma": 60, "min_t": 150, "max_t": 800},
    }

    def __init__(
        self,
        layer_params: Optional[Dict[int, Dict[str, float]]] = None,
        seed: int = 42,
    ):
        """
        参数:
            layer_params: 层特定参数，格式 {layer_idx: {"mu": ..., "sigma": ..., "min_t": ..., "max_t": ...}}
            seed: 随机种子
        """
        self.layer_params = layer_params or self.DEFAULT_LAYER_PARAMS
        self.seed = seed
        import numpy as np
        np.random.seed(seed)

    def sample(
        self,
        layer_idx: int,
        n_samples: int = 1,
    ) -> List[int]:
        """
        为指定层采样 timestep

        返回:
            list of int timestep values
        """
        import numpy as np

        params = self.layer_params.get(layer_idx, {"mu": 500, "sigma": 100, "min_t": 150, "max_t": 800})
        mu = params["mu"]
        sigma = params["sigma"]
        min_t = params["min_t"]
        max_t = params["max_t"]

        samples = []
        while len(samples) < n_samples:
            t = np.random.normal(mu, sigma)
            if min_t <= t <= max_t:
                samples.append(int(round(t)))

        return samples

    def sample_multi_layer(
        self,
        layer_indices: List[int],
        samples_per_layer: int = 5,
    ) -> Dict[int, List[int]]:
        """
        为多个层采样 timestep

        返回:
            {layer_idx: [t1, t2, ...]}
        """
        return {
            layer: self.sample(layer, samples_per_layer)
            for layer in layer_indices
        }


# ============================================================================
# 便捷工厂函数 (扩展)
# ============================================================================

def create_param_test_sampler(
    tokens_per_timestep: int = 1536,
    num_timesteps_per_prompt: int = 5,
    seed: int = 42,
) -> ParamTestTokenSampler:
    """创建参数测试阶段采样器"""
    config = ParamTestSamplerConfig(
        tokens_per_timestep=tokens_per_timestep,
        num_timesteps_per_prompt=num_timesteps_per_prompt,
        seed=seed,
    )
    return ParamTestTokenSampler(config)


def create_timestep_sampler(
    layer_params: Optional[Dict[int, Dict[str, float]]] = None,
    seed: int = 42,
) -> TruncatedGaussianTimestepSampler:
    """创建截断高斯 timestep 采样器"""
    return TruncatedGaussianTimestepSampler(layer_params, seed)


# ============================================================================
# 统一采样接口 (新增)
# ============================================================================

class UnifiedSampler:
    """
    统一采样接口

    通过 mode 参数选择采样策略：
    - 'init': 初始化阶段采样 (InitTokenSampler)
    - 'train': 训练阶段采样 (TrainTokenSampler)
    - 'param_test': 参数测试阶段采样 (ParamTestTokenSampler)

    使用示例:
        sampler = UnifiedSampler(mode='param_test')
        tokens, meta = sampler.sample(activations, timestep=500, layer_idx=14)
    """

    def __init__(
        self,
        mode: str = 'train',
        seed: int = 42,
        **kwargs,
    ):
        """
        参数:
            mode: 'init' | 'train' | 'param_test'
            seed: 随机种子
            **kwargs: 传递给具体采样器的配置参数
        """
        self.mode = mode
        self.seed = seed

        if mode == 'init':
            self._sampler = InitTokenSampler(InitSamplerConfig(seed=seed, **kwargs))
        elif mode == 'train':
            self._sampler = TrainTokenSampler(TrainSamplerConfig(seed=seed, **kwargs))
        elif mode == 'param_test':
            self._sampler = ParamTestTokenSampler(ParamTestSamplerConfig(seed=seed, **kwargs))
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'init', 'train', or 'param_test'")

    def sample(
        self,
        activations: torch.Tensor,
        timestep: Optional[int] = None,
        grid_size: Optional[Tuple[int, int, int]] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        采样入口

        参数:
            activations: [B, L, C] 或 [N, C] 原始激活
            timestep: 当前时间步
            grid_size: (F, H, W) 网格尺寸
            layer_idx: 层索引

        返回:
            sampled: [M, C] 采样后的 token
            metadata: 采样元数据
        """
        if self.mode == 'param_test':
            return self._sampler.sample(activations, timestep, grid_size, layer_idx)
        else:
            return self._sampler.sample(activations, timestep, grid_size)

    def reset(self):
        """重置采样器状态"""
        self._sampler.reset()


def create_unified_sampler(
    mode: str = 'train',
    seed: int = 42,
    **kwargs,
) -> UnifiedSampler:
    """创建统一采样器"""
    return UnifiedSampler(mode=mode, seed=seed, **kwargs)
