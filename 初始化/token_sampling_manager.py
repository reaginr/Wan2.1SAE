"""
Token Sampling Manager - 统一初始化与训练阶段的数据采样框架

目标：避免 initialization-training distribution shift

设计原则：
1. 初始化阶段：aggressive decorrelation，确保特征多样性
2. 训练阶段：mild decorrelation，保持真实数据分布
3. 统一采样逻辑，可复用

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


class SamplingMode(Enum):
    """采样模式"""
    INIT = "init"      # 初始化模式：aggressive decorrelation
    TRAIN = "train"    # 训练模式：mild decorrelation


@dataclass
class TemporalSamplingConfig:
    """
    时间维度采样配置

    视频数据的时间维度特征：
    - 早期 timestep (t→1000): 高噪声，特征高度随机化
    - 中期 timestep (300→700): 结构信息开始显现
    - 晚期 timestep (0→300): 接近最终生成结果，特征最稳定
    """
    # 初始化模式
    init_timesteps: List[int] = field(default_factory=lambda: [0, 100, 200, 400, 600, 800, 1000])
    init_samples_per_timestep: int = 5000  # 每个 timestep 采样多少 token

    # 训练模式
    train_timesteps: List[int] = field(default_factory=lambda: list(range(0, 1000, 50)))
    train_samples_per_batch: int = 4096

    # 时间采样权重 (不同 timestep 的重要性)
    # 初始化时应该更均匀覆盖，训练时可以偏向某些区域
    init_timestep_weights: Optional[List[float]] = None
    train_timestep_weights: Optional[List[float]] = None


@dataclass
class SpatialSamplingConfig:
    """
    空间维度采样配置

    视频数据的空间局部性：
    - 相邻像素/patch 高度相关
    - 需要空间 stride 打破局部相关性
    - 中心区域 vs 边缘区域可能有不同重要性
    """
    # 空间 stride
    init_spatial_stride: int = 2   # 初始化：更大 stride，减少局部相关
    train_spatial_stride: int = 1  # 训练：保持原始分辨率

    # 空间区域采样
    use_center_bias: bool = False  # 是否对中心区域加权
    center_bias_ratio: float = 0.3  # 中心区域占比

    # 2D grid 参数 (会在运行时根据实际 latent shape 更新)
    height_tokens: int = 30   # 480 / 8 / 2 = 30
    width_tokens: int = 52    # 832 / 8 / 2 = 52


@dataclass
class NormStratifiedConfig:
    """
    Norm 分层采样配置

    为什么需要 norm stratified sampling:
    - DiT activation 的 norm 分布不均匀
    - 高 norm token 通常携带重要语义信息
    - 低 norm token 可能是噪声或背景
    - 均匀采样可能导致语义信息丢失
    """
    # 是否启用
    enabled: bool = True

    # 分层数量
    num_buckets: int = 5

    # 初始化模式：更激进的分层
    init_bucket_weights: List[float] = field(default_factory=lambda: [0.15, 0.20, 0.25, 0.25, 0.15])
    # bucket 0: 最低 norm, bucket 4: 最高 norm
    # 初始化时中间层权重较大，避免极端值主导

    # 训练模式：软偏置
    train_soft_bias: bool = True
    train_bias_strength: float = 0.3  # 0 = 无偏置, 1 = 强偏置

    # 采样前是否先 RMSNorm
    # 初始化时应该在 RMSNorm 后采样，因为 SAE 输入是归一化的
    apply_rms_norm_before_sampling: bool = True


@dataclass
class DecorrelationConfig:
    """
    去相关采样配置

    为什么需要 decorrelation:
    - 视频数据存在强局部相关性（时空连续性）
    - 相邻 token 高度相似
    - 直接采样会导致特征冗余
    - PCA 初始化对冗余数据敏感
    """
    # 是否启用
    enabled: bool = True

    # 初始化模式：aggressive
    init_method: str = "pca_residual"  # "random" | "pca_residual" | "gram_schmidt"
    init_target_redundancy: float = 0.3  # 目标余弦相似度

    # 训练模式：mild
    train_method: str = "random"  # 训练时不需要强去相关
    train_target_redundancy: float = 0.7  # 允许更高相似度

    # 采样数量 vs 目标数量
    oversample_ratio: float = 3.0  # 先采样 3x，再筛选


@dataclass
class TokenSamplingConfig:
    """完整的 Token 采样配置"""
    mode: SamplingMode = SamplingMode.INIT

    # 目标采样数量
    target_tokens: int = 256000  # 初始化时的总 token 数

    # 子配置
    temporal: TemporalSamplingConfig = field(default_factory=TemporalSamplingConfig)
    spatial: SpatialSamplingConfig = field(default_factory=SpatialSamplingConfig)
    norm_stratified: NormStratifiedConfig = field(default_factory=NormStratifiedConfig)
    decorrelation: DecorrelationConfig = field(default_factory=DecorrelationConfig)

    # 随机种子
    seed: int = 42


class TokenSamplingManager:
    """
    Token 采样管理器

    统一初始化与训练阶段的采样逻辑，避免 distribution shift

    使用方法：
        config = TokenSamplingConfig(mode=SamplingMode.INIT)
        sampler = TokenSamplingManager(config)

        # 从原始 activation 采样
        sampled_tokens, metadata = sampler.sample(
            activations=raw_activations,
            timesteps=timesteps,
            grid_sizes=grid_sizes,
        )
    """

    def __init__(self, config: TokenSamplingConfig):
        self.config = config
        self._generator = torch.Generator()
        self._generator.manual_seed(config.seed)

        # 缓存统计信息
        self._stats_cache: Dict[str, Any] = {}

    def sample(
        self,
        activations: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        grid_sizes: Optional[torch.Tensor] = None,
        existing_features: Optional[torch.Tensor] = None,
        return_metadata: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        主采样入口

        参数:
            activations: [B, L, C] 或 [N, C] 原始激活
            timesteps: [B] 每个 sample 的时间步
            grid_sizes: [B, 3] (F, H, W) 网格尺寸
            existing_features: [K, C] 已有特征（用于去相关）
            return_metadata: 是否返回元数据

        返回:
            sampled: [M, C] 采样后的 token
            metadata: 采样元数据
        """
        # 确保输入格式正确
        if activations.dim() == 3:
            B, L, C = activations.shape
            activations_flat = activations.reshape(B * L, C)
        else:
            activations_flat = activations
            B = 1
            L = activations_flat.shape[0]

        device = activations_flat.device

        # Step 1: RMSNorm (如果配置要求)
        if self.config.norm_stratified.apply_rms_norm_before_sampling:
            activations_normed = self._rms_norm(activations_flat)
        else:
            activations_normed = activations_flat

        # Step 2: 计算 token norms
        norms = activations_normed.norm(dim=-1)

        # Step 3: Norm 分层采样
        if self.config.norm_stratified.enabled:
            sampled_indices = self._norm_stratified_sample(
                norms=norms,
                target_count=int(self.config.target_tokens * self.config.decorrelation.oversample_ratio),
            )
        else:
            # 均匀随机采样
            n_oversample = int(self.config.target_tokens * self.config.decorrelation.oversample_ratio)
            sampled_indices = torch.randperm(len(norms), generator=self._generator, device=device)[:n_oversample]

        sampled = activations_normed[sampled_indices]

        # Step 4: 去相关筛选
        if self.config.decorrelation.enabled and existing_features is not None:
            sampled = self._decorrelation_filter(
                samples=sampled,
                existing=existing_features,
            )

        # Step 5: 裁剪到目标数量
        if len(sampled) > self.config.target_tokens:
            sampled = sampled[:self.config.target_tokens]

        # 收集元数据
        metadata = None
        if return_metadata:
            metadata = {
                "input_shape": (B, L, activations.shape[-1]),
                "output_shape": sampled.shape,
                "sampled_ratio": len(sampled) / (B * L),
                "norm_stats": {
                    "mean": norms.mean().item(),
                    "std": norms.std().item(),
                    "min": norms.min().item(),
                    "max": norms.max().item(),
                },
            }

        return sampled, metadata

    def sample_for_initialization(
        self,
        activations_by_timestep: Dict[int, torch.Tensor],
        grid_sizes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        初始化专用采样接口

        从多个 timestep 的激活中采样，确保时间覆盖均匀

        参数:
            activations_by_timestep: {timestep: [B, L, C]} 各时间步的激活
            grid_sizes: [B, 3] 网格尺寸

        返回:
            sampled: [N, C] 采样后的 token
            metadata: 详细元数据
        """
        assert self.config.mode == SamplingMode.INIT

        all_samples = []
        timestep_stats = {}

        for t, act in activations_by_timestep.items():
            # 每个时间步单独采样
            samples, meta = self.sample(
                activations=act,
                timesteps=torch.tensor([t] * act.shape[0]),
                grid_sizes=grid_sizes,
                existing_features=torch.cat(all_samples, dim=0) if all_samples else None,
            )

            all_samples.append(samples)
            timestep_stats[t] = meta

        # 合并所有时间步的采样
        final_samples = torch.cat(all_samples, dim=0)

        # 最终裁剪
        if len(final_samples) > self.config.target_tokens:
            perm = torch.randperm(len(final_samples), generator=self._generator)
            final_samples = final_samples[perm[:self.config.target_tokens]]

        metadata = {
            "total_samples": len(final_samples),
            "timesteps_used": list(activations_by_timestep.keys()),
            "timestep_stats": timestep_stats,
        }

        return final_samples, metadata

    def sample_for_training(
        self,
        activations: torch.Tensor,
        timestep: int,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        训练专用采样接口

        训练阶段使用更温和的采样策略

        参数:
            activations: [B, L, C] 当前 batch 的激活
            timestep: 当前时间步
            batch_idx: batch 索引

        返回:
            sampled: [M, C] 采样后的 token
            metadata: 元数据
        """
        assert self.config.mode == SamplingMode.TRAIN

        # 训练模式：较少的预处理
        if activations.dim() == 3:
            B, L, C = activations.shape
            activations_flat = activations.reshape(B * L, C)
        else:
            activations_flat = activations
            B = 1

        # 简单 RMSNorm
        activations_normed = self._rms_norm(activations_flat)

        # 随机采样
        n_samples = min(self.config.target_tokens, len(activations_normed))
        perm = torch.randperm(len(activations_normed), generator=self._generator, device=activations_normed.device)
        sampled = activations_normed[perm[:n_samples]]

        metadata = {
            "timestep": timestep,
            "batch_idx": batch_idx,
            "input_tokens": B * activations.shape[1] if activations.dim() == 3 else len(activations),
            "output_tokens": len(sampled),
        }

        return sampled, metadata

    def _rms_norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Per-token RMSNorm"""
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x / rms

    def _norm_stratified_sample(
        self,
        norms: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        """
        Norm 分层采样实现

        将 token 按范数分成多个 bucket，每个 bucket 按权重采样
        """
        device = norms.device
        n_tokens = len(norms)

        # 计算 bucket 边界 (percentile-based)
        if self.config.mode == SamplingMode.INIT:
            weights = self.config.norm_stratified.init_bucket_weights
        else:
            # 训练模式：均匀权重
            weights = [1.0 / self.config.norm_stratified.num_buckets] * self.config.norm_stratified.num_buckets

        # 计算 percentile 边界
        sorted_norms, sorted_indices = norms.sort()
        bucket_boundaries = [0]
        cumsum = 0
        for w in weights[:-1]:
            cumsum += w
            boundary = int(cumsum * n_tokens)
            bucket_boundaries.append(boundary)
        bucket_boundaries.append(n_tokens)

        # 从每个 bucket 采样
        sampled_indices_list = []
        for i in range(self.config.norm_stratified.num_buckets):
            start = bucket_boundaries[i]
            end = bucket_boundaries[i + 1]
            bucket_indices = sorted_indices[start:end]

            # 该 bucket 应该采样多少
            n_bucket_samples = int(target_count * weights[i])

            if len(bucket_indices) <= n_bucket_samples:
                sampled_indices_list.append(bucket_indices)
            else:
                # 随机采样
                perm = torch.randperm(len(bucket_indices), generator=self._generator, device=device)
                sampled_indices_list.append(bucket_indices[perm[:n_bucket_samples]])

        return torch.cat(sampled_indices_list)

    def _decorrelation_filter(
        self,
        samples: torch.Tensor,
        existing: torch.Tensor,
    ) -> torch.Tensor:
        """
        去相关筛选

        移除与已有特征高度相似的样本
        """
        device = samples.device

        # 归一化
        samples_norm = F.normalize(samples, dim=-1)
        existing_norm = F.normalize(existing, dim=-1)

        # 计算相似度矩阵 (分块避免 OOM)
        batch_size = 1000
        n_samples = len(samples_norm)
        keep_mask = torch.ones(n_samples, dtype=torch.bool, device=device)

        target_redundancy = (
            self.config.decorrelation.init_target_redundancy
            if self.config.mode == SamplingMode.INIT
            else self.config.decorrelation.train_target_redundancy
        )

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = samples_norm[i:end]

            # 与已有特征计算相似度
            similarity = batch @ existing_norm.T  # [batch, K]
            max_similarity = similarity.abs().max(dim=-1).values

            # 标记高相似度样本
            keep_mask[i:end] = max_similarity < target_redundancy

        return samples[keep_mask]

    def get_spatial_indices(
        self,
        grid_size: Tuple[int, int, int],  # (F, H, W)
        stride: Optional[int] = None,
    ) -> torch.Tensor:
        """
        获取空间采样索引

        根据空间 stride 返回要采样的 token 索引
        """
        F, H, W = grid_size

        if stride is None:
            stride = (
                self.config.spatial.init_spatial_stride
                if self.config.mode == SamplingMode.INIT
                else self.config.spatial.train_spatial_stride
            )

        # 生成采样索引
        indices = []
        for f in range(0, F, max(1, stride // 2)):  # 时间维度步长较小
            for h in range(0, H, stride):
                for w in range(0, W, stride):
                    idx = f * H * W + h * W + w
                    indices.append(idx)

        return torch.tensor(indices, dtype=torch.long)

    def reset_generator(self, seed: Optional[int] = None):
        """重置随机数生成器"""
        if seed is not None:
            self.config.seed = seed
        self._generator.manual_seed(self.config.seed)


# ============================================================================
# 统计分析工具
# ============================================================================

class ActivationStatisticsAnalyzer:
    """
    激活值统计分析工具

    分析内容：
    - token norm 分布
    - 时间维度冗余性
    - 空间局部性
    - 层级激活秩
    - PCA 频谱分析
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results: Dict[str, Any] = {}

    def analyze_token_norm_distribution(
        self,
        activations: torch.Tensor,
        name: str = "activations",
    ) -> Dict[str, Any]:
        """
        分析 token norm 分布

        输出:
            - percentile (p5, p25, p50, p75, p95, p99)
            - histogram
            - 统计量 (mean, std, skewness, kurtosis)
        """
        if activations.dim() == 3:
            activations = activations.reshape(-1, activations.shape[-1])

        norms = activations.norm(dim=-1).float()

        # Percentiles
        percentiles = {}
        for p in [5, 25, 50, 75, 95, 99]:
            percentiles[f"p{p}"] = torch.quantile(norms, p / 100).item()

        # Histogram
        hist_counts, hist_edges = torch.histogram(norms, bins=50)
        histogram = {
            "counts": hist_counts.tolist(),
            "edges": hist_edges.tolist(),
        }

        # 统计量
        mean = norms.mean().item()
        std = norms.std().item()

        # Skewness and Kurtosis
        normalized = (norms - mean) / (std + 1e-8)
        skewness = (normalized ** 3).mean().item()
        kurtosis = (normalized ** 4).mean().item() - 3  # excess kurtosis

        result = {
            "name": name,
            "n_tokens": len(norms),
            "percentiles": percentiles,
            "histogram": histogram,
            "mean": mean,
            "std": std,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

        self.results[f"norm_dist_{name}"] = result
        return result

    def analyze_temporal_redundancy(
        self,
        activations_by_timestep: Dict[int, torch.Tensor],
        sample_size: int = 10000,
    ) -> Dict[str, Any]:
        """
        分析时间维度冗余性

        计算不同 timestep 之间 token 的平均余弦相似度
        """
        timesteps = sorted(activations_by_timestep.keys())
        n_timesteps = len(timesteps)

        # 采样计算
        sampled_by_t = {}
        for t in timesteps:
            act = activations_by_timestep[t]
            if act.dim() == 3:
                act = act.reshape(-1, act.shape[-1])

            n = min(sample_size, len(act))
            perm = torch.randperm(len(act), device=self.device)[:n]
            sampled_by_t[t] = F.normalize(act[perm].to(self.device).float(), dim=-1)

        # 计算 timestep 间的平均相似度
        similarity_matrix = torch.zeros(n_timesteps, n_timesteps)

        for i, t1 in enumerate(timesteps):
            for j, t2 in enumerate(timesteps):
                if i <= j:
                    # 采样计算相似度
                    n_pairs = min(10000, len(sampled_by_t[t1]))
                    idx1 = torch.randperm(len(sampled_by_t[t1]), device=self.device)[:n_pairs]
                    idx2 = torch.randperm(len(sampled_by_t[t2]), device=self.device)[:n_pairs]

                    sim = (sampled_by_t[t1][idx1] * sampled_by_t[t2][idx2]).sum(dim=-1).abs().mean().item()
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        # 分析
        # 相邻 timestep 平均相似度
        adjacent_sim = torch.diag(similarity_matrix, diagonal=1).mean().item()

        # 远距离 timestep 平均相似度
        far_indices = torch.triu_indices(n_timesteps, n_timesteps, offset=3)
        far_sim = similarity_matrix[far_indices[0], far_indices[1]].mean().item() if far_indices.shape[1] > 0 else 0.0

        result = {
            "timesteps": timesteps,
            "similarity_matrix": similarity_matrix.tolist(),
            "adjacent_timestep_similarity": adjacent_sim,
            "far_timestep_similarity": far_sim,
            "temporal_coherence": adjacent_sim - far_sim,  # 时间一致性指标
        }

        self.results["temporal_redundancy"] = result
        return result

    def analyze_spatial_locality(
        self,
        activations: torch.Tensor,
        grid_size: Tuple[int, int, int],  # (F, H, W)
        sample_size: int = 5000,
    ) -> Dict[str, Any]:
        """
        分析空间局部性

        计算相邻 token vs 远距离 token 的相似度差异
        """
        F, H, W = grid_size

        if activations.dim() == 3:
            activations = activations.reshape(-1, activations.shape[-1])

        device = self.device
        activations_norm = F.normalize(activations[:F * H * W].to(device).float(), dim=-1)

        # 采样 token 对
        n_tokens = len(activations_norm)

        # 相邻对 (4-邻域)
        adjacent_sims = []
        for _ in range(min(sample_size, n_tokens // 4)):
            idx = torch.randint(0, n_tokens, (1,), device=device).item()
            f, h, w = idx // (H * W), (idx % (H * W)) // W, idx % W

            # 4-邻域
            neighbors = []
            if h > 0:
                neighbors.append(f * H * W + (h - 1) * W + w)
            if h < H - 1:
                neighbors.append(f * H * W + (h + 1) * W + w)
            if w > 0:
                neighbors.append(f * H * W + h * W + (w - 1))
            if w < W - 1:
                neighbors.append(f * H * W + h * W + (w + 1))

            if neighbors:
                for nidx in neighbors:
                    sim = (activations_norm[idx] * activations_norm[nidx]).sum().item()
                    adjacent_sims.append(abs(sim))

        # 远距离对
        far_sims = []
        for _ in range(sample_size):
            idx1 = torch.randint(0, n_tokens, (1,), device=device).item()
            idx2 = torch.randint(0, n_tokens, (1,), device=device).item()

            # 确保距离足够远
            f1, h1, w1 = idx1 // (H * W), (idx1 % (H * W)) // W, idx1 % W
            f2, h2, w2 = idx2 // (H * W), (idx2 % (H * W)) // W, idx2 % W
            dist = abs(f1 - f2) + abs(h1 - h2) + abs(w1 - w2)

            if dist > 5:  # 距离阈值
                sim = (activations_norm[idx1] * activations_norm[idx2]).sum().item()
                far_sims.append(abs(sim))

        adjacent_mean = sum(adjacent_sims) / len(adjacent_sims) if adjacent_sims else 0.0
        far_mean = sum(far_sims) / len(far_sims) if far_sims else 0.0

        result = {
            "grid_size": grid_size,
            "adjacent_similarity_mean": adjacent_mean,
            "adjacent_similarity_std": torch.tensor(adjacent_sims).std().item() if adjacent_sims else 0.0,
            "far_similarity_mean": far_mean,
            "far_similarity_std": torch.tensor(far_sims).std().item() if far_sims else 0.0,
            "locality_index": adjacent_mean - far_mean,  # 正值表示存在局部相关性
        }

        self.results["spatial_locality"] = result
        return result

    def analyze_effective_rank(
        self,
        activations: torch.Tensor,
        threshold: float = 0.99,
    ) -> Dict[str, Any]:
        """
        分析有效秩 (Effective Rank)

        有效秩定义：累计方差达到 threshold 需要的主成分数
        """
        if activations.dim() == 3:
            activations = activations.reshape(-1, activations.shape[-1])

        # 使用 SVD 估计
        activations = activations.to(self.device).float()

        # 采样避免 OOM
        if len(activations) > 50000:
            perm = torch.randperm(len(activations), device=self.device)[:50000]
            activations = activations[perm]

        # 中心化
        mean = activations.mean(dim=0, keepdim=True)
        activations_centered = activations - mean

        # SVD
        U, S, V = torch.linalg.svd(activations_centered, full_matrices=False)

        # 计算方差解释比例
        total_var = (S ** 2).sum()
        explained_var_ratio = (S ** 2) / total_var
        cumulative = explained_var_ratio.cumsum(0)

        # 有效秩
        effective_rank = (cumulative < threshold).sum().item() + 1

        # 完整秩分析
        rank_50 = (cumulative < 0.50).sum().item() + 1
        rank_90 = (cumulative < 0.90).sum().item() + 1
        rank_95 = (cumulative < 0.95).sum().item() + 1
        rank_99 = (cumulative < 0.99).sum().item() + 1

        result = {
            "total_dimensions": activations.shape[-1],
            "effective_rank_50": rank_50,
            "effective_rank_90": rank_90,
            "effective_rank_95": rank_95,
            "effective_rank_99": rank_99,
            "singular_values": S[:100].tolist(),  # 前 100 个奇异值
            "explained_variance_ratio": explained_var_ratio[:100].tolist(),
        }

        self.results["effective_rank"] = result
        return result

    def analyze_pca_spectrum(
        self,
        activations: torch.Tensor,
        n_components: int = 100,
    ) -> Dict[str, Any]:
        """
        PCA 频谱分析

        分析主成分的方差分布
        """
        if activations.dim() == 3:
            activations = activations.reshape(-1, activations.shape[-1])

        activations = activations.to(self.device).float()

        # 采样避免 OOM
        if len(activations) > 50000:
            perm = torch.randperm(len(activations), device=self.device)[:50000]
            activations = activations[perm]

        # 中心化
        mean = activations.mean(dim=0, keepdim=True)
        activations_centered = activations - mean

        # PCA
        U, S, V = torch.linalg.svd(activations_centered, full_matrices=False)

        n_components = min(n_components, len(S))
        explained_var = (S[:n_components] ** 2) / (len(activations) - 1)
        total_var = activations_centered.var(dim=0).sum()
        explained_var_ratio = explained_var / total_var

        # 累计方差
        cumulative = explained_var_ratio.cumsum(0)

        result = {
            "n_components": n_components,
            "explained_variance": explained_var.tolist(),
            "explained_variance_ratio": explained_var_ratio.tolist(),
            "cumulative_variance_ratio": cumulative.tolist(),
            "top_10_variance": cumulative[9].item() if n_components >= 10 else None,
            "top_50_variance": cumulative[49].item() if n_components >= 50 else None,
            "top_100_variance": cumulative[99].item() if n_components >= 100 else None,
        }

        self.results["pca_spectrum"] = result
        return result

    def analyze_layer_differences(
        self,
        activations_by_layer: Dict[int, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        分析不同层的激活差异
        """
        layers = sorted(activations_by_layer.keys())

        layer_stats = {}
        for layer in layers:
            act = activations_by_layer[layer]
            if act.dim() == 3:
                act = act.reshape(-1, act.shape[-1])

            # 基本统计
            norms = act.norm(dim=-1)
            layer_stats[layer] = {
                "norm_mean": norms.mean().item(),
                "norm_std": norms.std().item(),
                "norm_p50": torch.quantile(norms.float(), 0.5).item(),
                "norm_p95": torch.quantile(norms.float(), 0.95).item(),
            }

        # 层间相似度
        similarity_matrix = torch.zeros(len(layers), len(layers))
        for i, l1 in enumerate(layers):
            for j, l2 in enumerate(layers):
                if i <= j:
                    act1 = activations_by_layer[l1]
                    act2 = activations_by_layer[l2]

                    if act1.dim() == 3:
                        act1 = act1.reshape(-1, act1.shape[-1])
                    if act2.dim() == 3:
                        act2 = act2.reshape(-1, act2.shape[-1])

                    # 采样计算相似度
                    n = min(5000, min(len(act1), len(act2)))
                    perm1 = torch.randperm(len(act1), device=self.device)[:n]
                    perm2 = torch.randperm(len(act2), device=self.device)[:n]

                    act1_sample = F.normalize(act1[perm1].to(self.device).float(), dim=-1)
                    act2_sample = F.normalize(act2[perm2].to(self.device).float(), dim=-1)

                    sim = (act1_sample * act2_sample).sum(dim=-1).abs().mean().item()
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        result = {
            "layers": layers,
            "layer_stats": layer_stats,
            "inter_layer_similarity": similarity_matrix.tolist(),
        }

        self.results["layer_differences"] = result
        return result

    def get_summary(self) -> Dict[str, Any]:
        """获取所有分析结果的摘要"""
        return self.results.copy()

    def print_summary(self):
        """打印分析结果摘要"""
        print("\n" + "=" * 70)
        print("Activation Statistics Analysis Summary")
        print("=" * 70)

        if "norm_dist_activations" in self.results:
            r = self.results["norm_dist_activations"]
            print(f"\n[Token Norm Distribution]")
            print(f"  Mean: {r['mean']:.4f}, Std: {r['std']:.4f}")
            print(f"  Percentiles: P5={r['percentiles']['p5']:.2f}, "
                  f"P50={r['percentiles']['p50']:.2f}, P95={r['percentiles']['p95']:.2f}")
            print(f"  Skewness: {r['skewness']:.4f}, Kurtosis: {r['kurtosis']:.4f}")

        if "temporal_redundancy" in self.results:
            r = self.results["temporal_redundancy"]
            print(f"\n[Temporal Redundancy]")
            print(f"  Adjacent Timestep Similarity: {r['adjacent_timestep_similarity']:.4f}")
            print(f"  Far Timestep Similarity: {r['far_timestep_similarity']:.4f}")
            print(f"  Temporal Coherence: {r['temporal_coherence']:.4f}")

        if "spatial_locality" in self.results:
            r = self.results["spatial_locality"]
            print(f"\n[Spatial Locality]")
            print(f"  Adjacent Similarity: {r['adjacent_similarity_mean']:.4f}")
            print(f"  Far Similarity: {r['far_similarity_mean']:.4f}")
            print(f"  Locality Index: {r['locality_index']:.4f}")

        if "effective_rank" in self.results:
            r = self.results["effective_rank"]
            print(f"\n[Effective Rank]")
            print(f"  Rank@50%: {r['effective_rank_50']}, "
                  f"Rank@90%: {r['effective_rank_90']}, "
                  f"Rank@99%: {r['effective_rank_99']}")

        print("\n" + "=" * 70)


# ============================================================================
# 便捷函数
# ============================================================================

def create_init_sampler(
    target_tokens: int = 256000,
    seed: int = 42,
) -> TokenSamplingManager:
    """创建初始化模式的采样器"""
    config = TokenSamplingConfig(
        mode=SamplingMode.INIT,
        target_tokens=target_tokens,
        seed=seed,
    )
    return TokenSamplingManager(config)


def create_train_sampler(
    target_tokens: int = 4096,
    seed: int = 42,
) -> TokenSamplingManager:
    """创建训练模式的采样器"""
    config = TokenSamplingConfig(
        mode=SamplingMode.TRAIN,
        target_tokens=target_tokens,
        seed=seed,
    )
    return TokenSamplingManager(config)
