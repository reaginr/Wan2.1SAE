"""
Sampling Statistics Monitor

监控训练阶段的采样分布，确保符合预期

监控指标：
1. Timestep histogram
2. Per-layer entropy
3. Sampling distribution drift
4. Coherence correlation with timestep

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SamplingSnapshot:
    """采样快照"""
    timestep: int
    layer_idx: int
    n_tokens: int
    norm_mean: float
    norm_std: float
    active_ratio: Optional[float] = None
    timestamp: float = 0.0


class SamplingStatisticsMonitor:
    """
    采样统计监控器

    监控训练阶段的采样质量，确保：
    1. Timestep 分布符合 truncated Gaussian
    2. 各层采样均衡
    3. 没有分布漂移
    """

    def __init__(self, min_timestep: int = 150, max_timestep: int = 800):
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep

        # 采样历史
        self._timestep_history: List[int] = []
        self._layer_timestep_history: Dict[int, List[int]] = defaultdict(list)
        self._snapshots: List[SamplingSnapshot] = []

        # 统计缓存
        self._stats_cache: Optional[Dict[str, Any]] = None

    def record_sample(
        self,
        timestep: int,
        layer_idx: int,
        activations: Optional[torch.Tensor] = None,
        n_tokens: Optional[int] = None,
    ):
        """
        记录一次采样

        参数:
            timestep: 时间步
            layer_idx: 层索引
            activations: 激活数据 (可选，用于详细分析)
            n_tokens: token 数量
        """
        import time

        # 记录 timestep
        self._timestep_history.append(timestep)
        self._layer_timestep_history[layer_idx].append(timestep)

        # 创建快照
        snapshot = SamplingSnapshot(
            timestep=timestep,
            layer_idx=layer_idx,
            n_tokens=n_tokens or 0,
            norm_mean=0.0,
            norm_std=0.0,
            timestamp=time.time(),
        )

        # 如果提供激活，计算详细统计
        if activations is not None:
            if activations.dim() == 3:
                activations = activations.reshape(-1, activations.shape[-1])

            norms = activations.norm(dim=-1)
            snapshot.norm_mean = norms.mean().item()
            snapshot.norm_std = norms.std().item()

        self._snapshots.append(snapshot)

        # 清除缓存
        self._stats_cache = None

    def compute_timestep_histogram(
        self,
        n_bins: int = 20,
    ) -> Dict[str, Any]:
        """
        计算 timestep 直方图

        返回:
            histogram: 包含 counts, edges, expected_distribution
        """
        if not self._timestep_history:
            return {"counts": [], "edges": [], "violation_rate": 0.0}

        timesteps = np.array(self._timestep_history)

        # 计算直方图
        counts, edges = np.histogram(timesteps, bins=n_bins, range=(0, 1000))

        # 计算越界率
        violations = np.sum((timesteps < self.min_timestep) | (timesteps > self.max_timestep))
        violation_rate = violations / len(timesteps)

        # 计算期望分布 (uniform within valid range 作为对比)
        expected_per_bin = len(timesteps) / n_bins

        return {
            "counts": counts.tolist(),
            "edges": edges.tolist(),
            "violation_rate": float(violation_rate),
            "valid_ratio": 1.0 - violation_rate,
            "expected_per_bin": float(expected_per_bin),
        }

    def compute_layer_entropy(self) -> Dict[int, float]:
        """
        计算各层的 timestep 分布熵

        高熵 = 分布均匀
        低熵 = 分布集中

        期望：各层熵值适中，反映 truncated Gaussian 分布
        """
        layer_entropies = {}

        for layer_idx, timesteps in self._layer_timestep_history.items():
            if not timesteps:
                continue

            timesteps = np.array(timesteps)

            # 计算分布
            bins = np.arange(0, 1001, 50)  # 50 为 bin 宽度
            counts, _ = np.histogram(timesteps, bins=bins)

            # 归一化为概率
            probs = counts / counts.sum()
            probs = probs[probs > 0]  # 移除零值

            # 计算熵
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(bins) - 1)

            layer_entropies[layer_idx] = {
                "entropy": float(entropy),
                "max_entropy": float(max_entropy),
                "normalized_entropy": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
            }

        return layer_entropies

    def detect_distribution_drift(
        self,
        window_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        检测分布漂移

        比较近期采样与历史采样的分布差异
        """
        if len(self._timestep_history) < window_size * 2:
            return {
                "drift_detected": False,
                "reason": "Insufficient data",
                "kl_divergence": 0.0,
                "mean_drift": 0.0,
                "std_drift": 0.0,
            }

        timesteps = np.array(self._timestep_history)

        # 分割为历史和近期
        historical = timesteps[:-window_size]
        recent = timesteps[-window_size:]

        # 计算均值和标准差
        hist_mean, hist_std = historical.mean(), historical.std()
        recent_mean, recent_std = recent.mean(), recent.std()

        # 计算漂移指标
        mean_drift = abs(recent_mean - hist_mean)
        std_drift = abs(recent_std - hist_std)

        # 使用 KL 散度 (近似)
        bins = np.arange(0, 1001, 50)
        hist_counts, _ = np.histogram(historical, bins=bins)
        recent_counts, _ = np.histogram(recent, bins=bins)

        hist_probs = hist_counts / hist_counts.sum() + 1e-10
        recent_probs = recent_counts / recent_counts.sum() + 1e-10

        kl_divergence = np.sum(hist_probs * np.log(hist_probs / recent_probs))

        # 判断是否漂移
        drift_threshold = 0.5  # KL 散度阈值
        drift_detected = kl_divergence > drift_threshold

        return {
            "drift_detected": drift_detected,
            "kl_divergence": float(kl_divergence),
            "mean_drift": float(mean_drift),
            "std_drift": float(std_drift),
            "historical_mean": float(hist_mean),
            "recent_mean": float(recent_mean),
        }

    def compute_coherence_timestep_correlation(
        self,
        coherence_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        计算 coherence 与 timestep 的相关性

        参数:
            coherence_values: 对应每个样本的 coherence 值

        返回:
            correlation: 相关系数
            interpretation: 解释
        """
        if coherence_values is None or len(coherence_values) != len(self._timestep_history):
            return {
                "correlation": None,
                "interpretation": "No coherence data provided"
            }

        timesteps = np.array(self._timestep_history)
        coherence = np.array(coherence_values)

        # 计算相关系数
        if len(timesteps) > 1:
            correlation = np.corrcoef(timesteps, coherence)[0, 1]
        else:
            correlation = 0.0

        # 解释
        if correlation > 0.3:
            interpretation = "Higher coherence at higher timesteps (more structure)"
        elif correlation < -0.3:
            interpretation = "Higher coherence at lower timesteps (more semantic)"
        else:
            interpretation = "No strong correlation between coherence and timestep"

        return {
            "correlation": float(correlation),
            "interpretation": interpretation,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取完整统计信息"""
        if self._stats_cache is not None:
            return self._stats_cache

        stats = {
            "total_samples": len(self._timestep_history),
            "timestep_histogram": self.compute_timestep_histogram(),
            "layer_entropy": self.compute_layer_entropy(),
            "layer_timestep_stats": {},
        }

        # 各层统计
        for layer_idx, timesteps in self._layer_timestep_history.items():
            if not timesteps:
                continue

            timesteps = np.array(timesteps)
            stats["layer_timestep_stats"][layer_idx] = {
                "count": len(timesteps),
                "mean": float(timesteps.mean()),
                "std": float(timesteps.std()),
                "min": int(timesteps.min()),
                "max": int(timesteps.max()),
                "valid_ratio": float(
                    np.sum((timesteps >= self.min_timestep) & (timesteps <= self.max_timestep)) / len(timesteps)
                ),
            }

        self._stats_cache = stats
        return stats

    def print_report(self):
        """打印统计报告"""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("Sampling Statistics Report")
        print("=" * 70)

        # Timestep 分布
        hist = stats["timestep_histogram"]
        print(f"\n[Timestep Distribution]")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Valid ratio: {hist['valid_ratio']:.2%}")
        print(f"  Violation rate: {hist['violation_rate']:.2%}")

        # 各层熵
        print(f"\n[Layer Entropy]")
        for layer_idx, entropy_info in stats["layer_entropy"].items():
            print(f"  Layer {layer_idx}: normalized_entropy = {entropy_info['normalized_entropy']:.4f}")

        # 各层统计
        print(f"\n[Layer Timestep Statistics]")
        for layer_idx, layer_stats in stats["layer_timestep_stats"].items():
            print(f"  Layer {layer_idx}:")
            print(f"    Mean: {layer_stats['mean']:.1f}, Std: {layer_stats['std']:.1f}")
            print(f"    Range: [{layer_stats['min']}, {layer_stats['max']}]")
            print(f"    Valid ratio: {layer_stats['valid_ratio']:.2%}")

        print("\n" + "=" * 70)

    def reset(self):
        """重置监控器"""
        self._timestep_history = []
        self._layer_timestep_history = defaultdict(list)
        self._snapshots = []
        self._stats_cache = None


# ============================================================================
# 便捷函数
# ============================================================================

def create_monitor(
    min_timestep: int = 150,
    max_timestep: int = 800,
) -> SamplingStatisticsMonitor:
    """创建采样监控器"""
    return SamplingStatisticsMonitor(min_timestep, max_timestep)
