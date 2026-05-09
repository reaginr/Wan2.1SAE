#!/usr/bin/env python3
"""
SAE 初始化质量检查脚本 - 超宽 Latent 工程化版本

核心原则:
- 禁止任何 O(N²) 全量分析
- 所有统计必须采用 sampled / streaming / chunked / approximate 方式
- 支持 hidden_dim >= 24576 的超宽 SAE

核心检测:
1. 基础检查: Reconstruction MSE, Tied init, Decoder norm, Dead neurons
2. Feature Cosine Similarity (采样): 检测 duplicated features
3. Mutual Coherence: 比 mean cosine 更重要
4. TopK Dynamics (采样): 模拟 TopK 竞争健康度
5. Gini Coefficient: 检测 feature monopoly
6. SVD Spectrum: 检测低秩塌缩
7. 初始化评分系统

使用方法:
    python -m 初始化.sae_quality_check --init_dir ./sae_init --cache_dir ./cache --layer all
    python -m 初始化.sae_quality_check --init_dir ./sae_init --cache_dir ./cache --layer all --save_plots
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

# 可选导入 matplotlib
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available, plots will be disabled")

# 可选导入 psutil (用于 CPU 内存追踪)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================================
# 常量配置
# ============================================================================

# 内存安全阈值
MEMORY_LIMIT_MB = 2048  # 2GB

# 采样配置
DEFAULT_NUM_PAIRS = 500_000  # 采样 50 万对
DEFAULT_BATCH_SIZE = 4096

# Histogram 配置
HISTOGRAM_BINS = 200
HISTOGRAM_RANGE = (-1.0, 1.0)


# ============================================================================
# Memory Tracker 工具类
# ============================================================================

class MemoryTracker:
    """
    内存追踪工具类

    功能:
    1. 追踪 GPU 显存峰值 (如果使用 CUDA)
    2. 追踪 CPU 内存峰值 (如果 psutil 可用)
    3. 分阶段记录内存使用
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.use_cuda = device.startswith("cuda") and torch.cuda.is_available()

        # 峰值记录
        self._peak_gpu_mb = 0.0
        self._peak_cpu_mb = 0.0

        # 分阶段记录
        self._stage_memory: Dict[str, Dict[str, float]] = {}

        # 初始 CPU 内存
        self._initial_cpu_mb = self._get_process_memory_mb()

    def _get_process_memory_mb(self) -> float:
        """获取当前进程内存使用 (MB)"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        return 0.0

    def _get_gpu_memory_mb(self) -> float:
        """获取当前 GPU 显存使用 (MB)"""
        if self.use_cuda:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def reset_peak(self) -> None:
        """重置峰值记录"""
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats()
        self._peak_cpu_mb = self._get_process_memory_mb()
        self._peak_gpu_mb = self._get_gpu_memory_mb()

    def update_peak(self) -> Tuple[float, float]:
        """
        更新峰值记录

        返回:
            (gpu_peak_mb, cpu_peak_mb)
        """
        # GPU 峰值
        if self.use_cuda:
            current_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self._peak_gpu_mb = max(self._peak_gpu_mb, current_gpu)

        # CPU 峰值
        current_cpu = self._get_process_memory_mb()
        self._peak_cpu_mb = max(self._peak_cpu_mb, current_cpu)

        return self._peak_gpu_mb, self._peak_cpu_mb

    def record_stage(self, stage_name: str) -> Dict[str, float]:
        """
        记录当前阶段的内存使用

        参数:
            stage_name: 阶段名称

        返回:
            当前内存使用字典
        """
        self.update_peak()

        current_gpu = self._get_gpu_memory_mb()
        current_cpu = self._get_process_memory_mb()

        self._stage_memory[stage_name] = {
            "gpu_current_mb": current_gpu,
            "gpu_peak_mb": self._peak_gpu_mb,
            "cpu_current_mb": current_cpu,
            "cpu_delta_mb": current_cpu - self._initial_cpu_mb,
        }

        return self._stage_memory[stage_name]

    def get_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        return {
            "peak_gpu_mb": self._peak_gpu_mb,
            "peak_cpu_mb": self._peak_cpu_mb,
            "cpu_delta_mb": self._get_process_memory_mb() - self._initial_cpu_mb,
            "stages": self._stage_memory.copy(),
        }

    def print_summary(self) -> None:
        """打印内存使用摘要"""
        print(f"\n[Memory Tracker Summary]")
        print(f"  Peak GPU Memory: {self._peak_gpu_mb:.1f} MB")
        print(f"  Peak CPU Memory: {self._peak_cpu_mb:.1f} MB")
        print(f"  CPU Delta: {self._get_process_memory_mb() - self._initial_cpu_mb:.1f} MB")

        if self._stage_memory:
            print(f"\n  Stages:")
            for stage, mem in self._stage_memory.items():
                gpu_str = f"GPU: {mem['gpu_current_mb']:.1f}/{mem['gpu_peak_mb']:.1f} MB" if self.use_cuda else "GPU: N/A"
                print(f"    {stage}: {gpu_str}, CPU: {mem['cpu_current_mb']:.1f} MB (Δ{mem['cpu_delta_mb']:+.1f} MB)")


# ============================================================================
# Streaming Statistics 工具类
# ============================================================================

class RunningStats:
    """
    流式统计工具类

    支持 online 计算:
    - mean
    - variance
    - std
    - max
    - histogram (用于 percentile 估计)

    避免保存全部中间 tensor
    """

    def __init__(self, num_bins: int = HISTOGRAM_BINS, value_range: Tuple[float, float] = HISTOGRAM_RANGE):
        self.num_bins = num_bins
        self.value_range = value_range

        # Running statistics
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0  # for Welford's algorithm
        self._max = float('-inf')

        # Histogram for percentile estimation
        self._histogram = torch.zeros(num_bins, dtype=torch.long)

    def update(self, values: torch.Tensor) -> None:
        """
        更新统计 (Welford's online algorithm)

        参数:
            values: 一批新数据 [batch_size]
        """
        values = values.float()
        batch_count = values.numel()

        if batch_count == 0:
            return

        # Update count
        old_count = self._count
        self._count += batch_count

        # Update mean and variance (Welford's algorithm)
        batch_mean = values.mean().item()
        delta = batch_mean - self._mean
        self._mean += delta * batch_count / self._count

        if batch_count > 1:
            batch_var = values.var().item()
            self._M2 += batch_var * (batch_count - 1)
        else:
            delta2 = values - self._mean
            self._M2 += (delta * delta2).sum().item()

        # Update max
        self._max = max(self._max, values.max().item())

        # Update histogram (向量化操作，避免 Python 循环)
        min_val, max_val = self.value_range
        bin_width = (max_val - min_val) / self.num_bins
        bin_indices = ((values - min_val) / bin_width).long().clamp(0, self.num_bins - 1)

        # 使用 torch.bincount 进行向量化统计，比 Python 循环快 20-100 倍
        bin_counts = torch.bincount(bin_indices, minlength=self.num_bins)
        self._histogram += bin_counts

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        if self._count < 2:
            return 0.0
        return self._M2 / (self._count - 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    @property
    def max(self) -> float:
        return self._max

    def estimate_percentile(self, p: float) -> float:
        """
        基于 histogram 估计百分位数

        参数:
            p: 百分位 (0.0 ~ 1.0)

        返回:
            估计的百分位数值
        """
        if self._count == 0:
            return 0.0

        target_count = int(p * self._count)
        cumsum = self._histogram.cumsum(0)

        # 找到第一个 >= target_count 的 bin
        bin_idx = (cumsum >= target_count).nonzero()
        if len(bin_idx) == 0:
            bin_idx = self.num_bins - 1
        else:
            bin_idx = bin_idx[0].item()

        # 线性插值
        min_val, max_val = self.value_range
        bin_width = (max_val - min_val) / self.num_bins
        return min_val + bin_idx * bin_width

    def get_histogram(self) -> Tuple[torch.Tensor, int]:
        """返回 histogram 和总计数"""
        return self._histogram.clone(), self._count

    def reset(self) -> None:
        """重置所有统计"""
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0
        self._max = float('-inf')
        self._histogram.zero_()


# ============================================================================
# 内存安全机制
# ============================================================================

class MemorySafeAnalyzer:
    """
    内存安全分析器

    功能:
    1. 运行前自动估算 tensor 大小
    2. 超过阈值自动切换 sampled mode
    3. 输出内存使用日志
    """

    @staticmethod
    def estimate_pairwise_memory(d_hidden: int, dtype_bytes: int = 4) -> int:
        """
        估算全量 pairwise 计算所需内存

        参数:
            d_hidden: 隐藏层维度
            dtype_bytes: 数据类型字节数 (float32 = 4)

        返回:
            估算内存 (MB)
        """
        num_pairs = d_hidden * (d_hidden - 1) // 2
        memory_bytes = num_pairs * dtype_bytes
        return memory_bytes / (1024 * 1024)  # MB

    @staticmethod
    def should_use_sampled(d_hidden: int, threshold_mb: float = MEMORY_LIMIT_MB) -> Tuple[bool, float]:
        """
        判断是否应该使用采样模式

        返回:
            (should_sample, estimated_memory_mb)
        """
        estimated_mb = MemorySafeAnalyzer.estimate_pairwise_memory(d_hidden)
        return estimated_mb > threshold_mb, estimated_mb

    @staticmethod
    def print_memory_status(d_hidden: int, threshold_mb: float = MEMORY_LIMIT_MB) -> None:
        """打印内存状态日志"""
        estimated_mb = MemorySafeAnalyzer.estimate_pairwise_memory(d_hidden)
        should_sample = estimated_mb > threshold_mb

        print(f"\n[MemorySafe]")
        print(f"  d_hidden: {d_hidden}")
        print(f"  Estimated full pairwise tensor: {estimated_mb:.1f} MB")
        print(f"  Threshold: {threshold_mb} MB")

        if should_sample:
            print(f"  [SWITCH] Switching to sampled statistics mode")
        else:
            print(f"  [OK] Full analysis mode")


# ============================================================================
# 采样相似度分析器
# ============================================================================

class SampledSimilarityAnalyzer:
    """
    采样相似度分析器

    使用 Monte Carlo 采样替代全量 pairwise cosine matrix

    功能:
    1. Feature Cosine Similarity (采样)
    2. Mutual Coherence (比 mean cosine 更重要)
    3. Histogram-based percentile estimation
    """

    def __init__(
        self,
        Wdec: torch.Tensor,
        num_pairs: int = DEFAULT_NUM_PAIRS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "cpu",
    ):
        """
        参数:
            Wdec: Decoder 权重 [d_model, d_hidden]
            num_pairs: 采样 pair 数量
            batch_size: 批处理大小
            device: 计算设备
        """
        self.Wdec = Wdec.float().to(device)
        self.Wdec_norm = F.normalize(self.Wdec, dim=0)  # 预先归一化
        self.d_model = Wdec.shape[0]
        self.d_hidden = Wdec.shape[1]
        self.num_pairs = num_pairs
        self.batch_size = batch_size
        self.device = device

    def analyze(self, verbose: bool = True) -> Dict[str, Any]:
        """
        执行采样相似度分析

        返回:
            统计结果字典
        """
        if verbose:
            print(f"\n[InitDiag] Feature Cosine Similarity 分析 (采样模式)...")

            # 检查内存状态
            should_sample, estimated_mb = MemorySafeAnalyzer.should_use_sampled(self.d_hidden)

            if should_sample:
                print(f"  [MemorySafe] Estimated full tensor: {estimated_mb:.0f} MB > {MEMORY_LIMIT_MB} MB")
                print(f"  [MemorySafe] Using sampled mode")
            else:
                print(f"  [MemorySafe] Estimated full tensor: {estimated_mb:.0f} MB")

            total_pairs = self.d_hidden * (self.d_hidden - 1) // 2
            print(f"  采样 {self.num_pairs:,} 对 (总数约 {total_pairs:,})")

        start_time = time.time()

        # Running statistics
        running_stats = RunningStats()
        running_abs_stats = RunningStats()  # for mutual coherence

        # Monte Carlo 采样
        num_processed = 0
        chunk_size = self.batch_size

        while num_processed < self.num_pairs:
            batch_size = min(chunk_size, self.num_pairs - num_processed)

            # 随机采样 feature pairs
            idx_i = torch.randint(0, self.d_hidden, (batch_size,), device=self.device)
            idx_j = torch.randint(0, self.d_hidden, (batch_size,), device=self.device)

            # 排除自比较 (i == j)
            mask = idx_i != idx_j
            if not mask.any():
                continue

            idx_i = idx_i[mask]
            idx_j = idx_j[mask]

            # 计算余弦相似度
            cos_values = (self.Wdec_norm[:, idx_i] * self.Wdec_norm[:, idx_j]).sum(dim=0)

            # 更新统计
            running_stats.update(cos_values)
            running_abs_stats.update(cos_values.abs())

            num_processed += batch_size

        elapsed = time.time() - start_time

        # 提取结果
        results = {
            "cosine_mean": running_stats.mean,
            "cosine_std": running_stats.std,
            "cosine_p95": running_stats.estimate_percentile(0.95),
            "cosine_p99": running_stats.estimate_percentile(0.99),
            "cosine_max": running_stats.max,
            "mutual_coherence": running_abs_stats.max,  # max |cos|
            "num_sampled_pairs": num_processed,
            "analysis_time_sec": elapsed,
        }

        if verbose:
            print(f"  [InitDiag] CosSim Mean: {results['cosine_mean']:.4f}")
            print(f"  [InitDiag] CosSim Std : {results['cosine_std']:.4f}")
            print(f"  [InitDiag] CosSim P95 : {results['cosine_p95']:.4f}")
            print(f"  [InitDiag] CosSim P99 : {results['cosine_p99']:.4f}")
            print(f"  [InitDiag] CosSim Max : {results['cosine_max']:.4f}")
            print(f"  [InitDiag] Mutual Coherence: {results['mutual_coherence']:.4f}")
            print(f"  [Approximate] Sampled: {num_processed:,} pairs, Time: {elapsed:.2f}s")

            # 判定
            mu = results['mutual_coherence']
            cos_mean = results['cosine_mean']
            cos_p95 = results['cosine_p95']
            cos_max = results['cosine_max']

            # Mutual Coherence 判定
            if mu < 0.2:
                mc_status = "✓ 优秀"
            elif mu < 0.35:
                mc_status = "✓ 可接受"
            elif mu > 0.5:
                mc_status = "⚠ 危险"
            else:
                mc_status = "⚠ 警告"

            print(f"  [InitDiag] Mutual Coherence Status: {mc_status}")

            # Feature Diversity 判定
            if cos_mean < 0.05 and cos_p95 < 0.2 and cos_max < 0.5:
                print(f"  [InitDiag] ✓ 特征多样性优秀")
            elif cos_p95 > 0.4:
                print(f"  [InitDiag] ⚠ 存在较多相似特征")
            elif cos_max > 0.8:
                print(f"  [InitDiag] ⚠ 存在高相似度特征对")
            else:
                print(f"  [InitDiag] ✓ 特征多样性良好")

        return results


# ============================================================================
# 评分数据类
# ============================================================================

@dataclass
class InitializationScore:
    """初始化评分结果"""
    total_score: float = 0.0
    reconstruction_score: float = 0.0
    diversity_score: float = 0.0
    mutual_coherence_score: float = 0.0
    topk_score: float = 0.0
    gini_score: float = 0.0
    spectrum_score: float = 0.0

    status: str = "UNKNOWN"
    issues: List[str] = field(default_factory=list)

    # 近似统计信息
    sampled_pairs: int = 0
    peak_memory_mb: float = 0.0

    # 内存详情 (各阶段)
    memory_stages: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "reconstruction_score": self.reconstruction_score,
            "diversity_score": self.diversity_score,
            "mutual_coherence_score": self.mutual_coherence_score,
            "topk_score": self.topk_score,
            "gini_score": self.gini_score,
            "spectrum_score": self.spectrum_score,
            "status": self.status,
            "issues": self.issues,
            "approximate_stats": {
                "sampled_pairs": self.sampled_pairs,
                "peak_memory_mb": self.peak_memory_mb,
                "memory_stages": self.memory_stages,
            },
        }


# ============================================================================
# 初始化诊断分析器 (超宽 Latent 工程化版本)
# ============================================================================

class InitializationAnalyzer:
    """
    SAE 初始化诊断分析器

    所有分析均采用 sampled/streaming 模式，支持超宽 SAE (hidden_dim >= 24576)
    """

    def __init__(
        self,
        Wdec: torch.Tensor,
        Wenc: torch.Tensor,
        bpre: torch.Tensor,
        x_norm: torch.Tensor,
        top_k: int = 128,
        num_pairs: int = DEFAULT_NUM_PAIRS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "cpu",
    ):
        self.Wdec = Wdec.float().to(device)
        self.Wenc = Wenc.float().to(device)
        self.bpre = bpre.float().to(device)
        self.x_norm = x_norm.float().to(device)
        self.top_k = top_k
        self.num_pairs = num_pairs
        self.batch_size = batch_size
        self.device = device

        self.d_model = Wdec.shape[0]
        self.d_hidden = Wdec.shape[1]

        # 打印内存状态
        MemorySafeAnalyzer.print_memory_status(self.d_hidden)

        # 内存追踪器
        self._memory_tracker = MemoryTracker(device=device)
        self._memory_tracker.record_stage("init")

        # 缓存结果
        self._cosine_stats = None
        self._topk_stats = None
        self._gini = None
        self._svd_stats = None

    def analyze_feature_similarity(self, verbose: bool = True) -> Dict[str, Any]:
        """分析 Feature Cosine Similarity (采样模式)"""
        self._memory_tracker.reset_peak()

        analyzer = SampledSimilarityAnalyzer(
            Wdec=self.Wdec,
            num_pairs=self.num_pairs,
            batch_size=self.batch_size,
            device=self.device,
        )
        self._cosine_stats = analyzer.analyze(verbose=verbose)

        # 记录内存
        mem = self._memory_tracker.record_stage("feature_similarity")
        if verbose:
            print(f"  [Memory] GPU Peak: {mem['gpu_peak_mb']:.1f} MB, CPU Delta: {mem['cpu_delta_mb']:+.1f} MB")

        return self._cosine_stats

    def analyze_topk_dynamics(
        self,
        n_samples: int = 2048,
        n_overlap_samples: int = 500,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        模拟 TopK 竞争健康度 (采样模式)

        分析:
        1. Feature Firing Frequency
        2. TopK Overlap (采样)
        3. Activation Entropy
        """
        if verbose:
            print(f"\n[InitDiag] TopK Dynamics 模拟...")

        start_time = time.time()

        n_total = self.x_norm.shape[0]
        if n_samples > n_total:
            n_samples = n_total

        # 采样
        idx = torch.randperm(n_total, device=self.device)[:n_samples]
        x_sample = self.x_norm[idx]

        # SAE Encode
        x_centered = x_sample - self.bpre
        z = x_centered @ self.Wenc.T

        # TopK
        top_k = min(self.top_k, self.d_hidden)
        topk_values, topk_indices = torch.topk(z, top_k, dim=1)

        # ================================
        # A. Feature Firing Frequency
        # ================================
        firing_counts = torch.zeros(self.d_hidden, device=self.device)
        firing_counts.scatter_add_(0, topk_indices.flatten(),
                                   torch.ones(topk_indices.numel(), device=self.device))

        firing_freq = firing_counts / n_samples

        # 使用 RunningStats 避免 full quantile
        freq_stats = RunningStats()
        freq_stats.update(firing_freq)

        freq_mean = freq_stats.mean
        freq_std = freq_stats.std
        freq_top1pct = freq_stats.estimate_percentile(0.99)
        freq_bottom1pct = freq_stats.estimate_percentile(0.01)

        # ================================
        # B. Activation Entropy
        # ================================
        prob = firing_freq / (firing_freq.sum() + 1e-10)
        prob = prob.clamp(min=1e-10)
        entropy = -(prob * torch.log(prob)).sum().item()
        max_entropy = torch.log(torch.tensor(self.d_hidden, dtype=torch.float32)).item()
        normalized_entropy = entropy / max_entropy

        # ================================
        # C. TopK Overlap (采样模式)
        # ================================
        overlap_stats = RunningStats(value_range=(0.0, 1.0), num_bins=100)

        topk_indices_cpu = topk_indices.cpu()
        for _ in range(n_overlap_samples):
            i, j = torch.randint(0, n_samples, (2,))
            set_i = set(topk_indices_cpu[i].tolist())
            set_j = set(topk_indices_cpu[j].tolist())
            overlap = len(set_i & set_j) / top_k
            overlap_stats.update(torch.tensor([overlap]))

        overlap_mean = overlap_stats.mean
        overlap_p95 = overlap_stats.estimate_percentile(0.95)

        elapsed = time.time() - start_time

        results = {
            "firing_freq_mean": freq_mean,
            "firing_freq_std": freq_std,
            "firing_freq_top1pct": freq_top1pct,
            "firing_freq_bottom1pct": freq_bottom1pct,
            "activation_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "topk_overlap_mean": overlap_mean,
            "topk_overlap_p95": overlap_p95,
            "firing_counts": firing_counts.cpu(),
            "analysis_time_sec": elapsed,
        }

        if verbose:
            print(f"  [InitDiag] Firing Freq Mean: {freq_mean:.6f}")
            print(f"  [InitDiag] Firing Freq Std : {freq_std:.6f}")
            print(f"  [InitDiag] Activation Entropy: {normalized_entropy:.4f} (normalized)")
            print(f"  [InitDiag] TopK Overlap Mean: {overlap_mean:.4f}")
            print(f"  [InitDiag] TopK Overlap P95 : {overlap_p95:.4f}")
            print(f"  [Approximate] Time: {elapsed:.2f}s")

            if normalized_entropy > 0.8 and overlap_mean < 0.3:
                print(f"  [InitDiag] ✓ TopK 竞争健康")
            elif normalized_entropy < 0.5:
                print(f"  [InitDiag] ⚠ Feature 使用不均衡")
            elif overlap_mean > 0.5:
                print(f"  [InitDiag] ⚠ TopK Overlap 过高")
            else:
                print(f"  [InitDiag] ✓ TopK 竞争良好")

        # 记录内存
        mem = self._memory_tracker.record_stage("topk_dynamics")
        if verbose:
            print(f"  [Memory] GPU Peak: {mem['gpu_peak_mb']:.1f} MB, CPU Delta: {mem['cpu_delta_mb']:+.1f} MB")

        self._topk_stats = results
        return results

    def analyze_gini_coefficient(
        self,
        firing_counts: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """计算 Gini 系数"""
        if verbose:
            print(f"\n[InitDiag] Gini Coefficient 计算...")

        if firing_counts is None:
            if self._topk_stats is None:
                self.analyze_topk_dynamics(verbose=False)
            firing_counts = self._topk_stats["firing_counts"]

        values = firing_counts.float().sort()[0]
        n = len(values)

        index = torch.arange(1, n + 1, dtype=torch.float32)
        gini = (2 * (index * values).sum()) / (n * values.sum() + 1e-10) - (n + 1) / n
        gini = gini.item()

        zero_count = (values == 0).sum().item()
        zero_ratio = zero_count / n

        results = {
            "gini_coefficient": gini,
            "zero_firing_count": zero_count,
            "zero_firing_ratio": zero_ratio,
        }

        if verbose:
            print(f"  [InitDiag] Gini: {gini:.4f}")
            print(f"  [InitDiag] Zero Firing Features: {zero_count} / {n} ({zero_ratio:.2%})")

            if gini < 0.3:
                print(f"  [InitDiag] ✓ Feature 使用均衡")
            elif gini > 0.7:
                print(f"  [InitDiag] ⚠ Feature 极度不均衡")
            elif gini > 0.5:
                print(f"  [InitDiag] ⚠ Feature 使用不均衡")
            else:
                print(f"  [InitDiag] ✓ Feature 使用较均衡")

        self._gini = results
        return results

    def analyze_svd_spectrum(
        self,
        n_components: int = 256,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        分析 SVD 谱

        使用 torch.pca_lowrank() 避免 full covariance materialization
        """
        if verbose:
            print(f"\n[InitDiag] SVD Spectrum 分析...")

        start_time = time.time()

        # 采样
        n_samples = min(10000, self.x_norm.shape[0])
        x_sample = self.x_norm[:n_samples]

        # 中心化
        x_centered = x_sample - self.bpre

        # 使用 randomized SVD (避免 full covariance)
        U, S, V = torch.pca_lowrank(x_centered, q=min(n_components, min(n_samples, self.d_model)))

        # 解释方差
        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_variance
        cumulative_variance = explained_variance_ratio.cumsum(0)

        # 提取关键点
        variance_at_k = {}
        for k in [32, 64, 128, 256]:
            if k <= len(cumulative_variance):
                variance_at_k[f"top{k}"] = cumulative_variance[k-1].item()

        elapsed = time.time() - start_time

        results = {
            "singular_values": S.cpu(),
            "explained_variance_ratio": explained_variance_ratio.cpu(),
            "cumulative_variance": cumulative_variance.cpu(),
            "variance_at_k": variance_at_k,
            "analysis_time_sec": elapsed,
        }

        if verbose:
            print(f"  [InitDiag] SVD Spectrum (Randomized PCA):")
            for k, var in variance_at_k.items():
                print(f"    {k}: {var*100:.2f}%")
            print(f"  [Approximate] Time: {elapsed:.2f}s")

            if "top128" in variance_at_k and variance_at_k["top128"] > 0.9:
                print(f"  [InitDiag] ⚠ 残差空间可能过于低秩")
            else:
                print(f"  [InitDiag] ✓ 残差空间秩正常")

        # 记录内存
        mem = self._memory_tracker.record_stage("svd_spectrum")
        if verbose:
            print(f"  [Memory] GPU Peak: {mem['gpu_peak_mb']:.1f} MB, CPU Delta: {mem['cpu_delta_mb']:+.1f} MB")

        self._svd_stats = results
        return results

    def compute_initialization_score(
        self,
        mse_ratio: float,
        verbose: bool = True,
    ) -> InitializationScore:
        """
        计算初始化综合评分 (0-100)

        评分维度:
        1. Reconstruction Quality (20分)
        2. Feature Diversity (20分)
        3. Mutual Coherence (15分)  # 新增
        4. TopK Competition (20分)
        5. Gini Coefficient (15分)
        6. SVD Spectrum (10分)
        """
        score = InitializationScore()

        # 确保所有分析已完成
        if self._cosine_stats is None:
            self.analyze_feature_similarity(verbose=False)
        if self._topk_stats is None:
            self.analyze_topk_dynamics(verbose=False)
        if self._gini is None:
            self.analyze_gini_coefficient(verbose=False)
        if self._svd_stats is None:
            self.analyze_svd_spectrum(verbose=False)

        # ================================
        # 1. Reconstruction Quality (20分)
        # ================================
        if mse_ratio < 10:
            score.reconstruction_score = 20.0
        elif mse_ratio > 30:
            score.reconstruction_score = 0.0
        else:
            score.reconstruction_score = 20.0 * (1 - (mse_ratio - 10) / 20)

        # ================================
        # 2. Feature Diversity (20分)
        # ================================
        cos_mean = self._cosine_stats["cosine_mean"]
        cos_p95 = self._cosine_stats["cosine_p95"]
        cos_max = self._cosine_stats["cosine_max"]

        div_score = 0.0
        if cos_mean < 0.05:
            div_score += 8
        elif cos_mean < 0.1:
            div_score += 4

        if cos_p95 < 0.2:
            div_score += 8
        elif cos_p95 < 0.4:
            div_score += 4

        if cos_max < 0.5:
            div_score += 4
        elif cos_max < 0.8:
            div_score += 2

        score.diversity_score = div_score

        # ================================
        # 3. Mutual Coherence (15分) - 新增
        # ================================
        mu = self._cosine_stats["mutual_coherence"]

        if mu < 0.2:
            score.mutual_coherence_score = 15.0
        elif mu < 0.35:
            score.mutual_coherence_score = 10.0
        elif mu > 0.5:
            score.mutual_coherence_score = 0.0
            score.issues.append(f"Mutual Coherence 过高: {mu:.4f}")
        else:
            score.mutual_coherence_score = 5.0

        # ================================
        # 4. TopK Competition (20分)
        # ================================
        entropy = self._topk_stats["normalized_entropy"]
        overlap = self._topk_stats["topk_overlap_mean"]

        topk_score = 0.0
        if entropy > 0.8:
            topk_score += 10
        elif entropy > 0.6:
            topk_score += 5

        if overlap < 0.3:
            topk_score += 10
        elif overlap < 0.5:
            topk_score += 5

        score.topk_score = topk_score

        # ================================
        # 5. Gini Coefficient (15分)
        # ================================
        gini = self._gini["gini_coefficient"]

        if gini < 0.3:
            score.gini_score = 15.0
        elif gini > 0.7:
            score.gini_score = 0.0
        else:
            score.gini_score = 15.0 * (1 - (gini - 0.3) / 0.4)

        # ================================
        # 6. SVD Spectrum (10分)
        # ================================
        var_k = self._svd_stats["variance_at_k"]
        top128_var = var_k.get("top128", 0.5)

        if top128_var > 0.9:
            score.spectrum_score = 2.0
            score.issues.append("SVD: 可能低秩塌缩")
        elif top128_var < 0.7:
            score.spectrum_score = 10.0
        else:
            score.spectrum_score = 6.0

        # ================================
        # 总分计算
        # ================================
        score.total_score = (
            score.reconstruction_score +
            score.diversity_score +
            score.mutual_coherence_score +
            score.topk_score +
            score.gini_score +
            score.spectrum_score
        )

        # 状态判定
        if score.total_score >= 80:
            score.status = "GOOD"
        elif score.total_score >= 60:
            score.status = "WARNING"
        else:
            score.status = "DANGEROUS"

        # 收集问题
        if cos_p95 > 0.4:
            score.issues.append(f"Feature Cosine P95 过高: {cos_p95:.4f}")
        if cos_max > 0.8:
            score.issues.append(f"Feature Cosine Max 过高: {cos_max:.4f}")
        if entropy < 0.5:
            score.issues.append(f"Activation Entropy 过低: {entropy:.4f}")
        if gini > 0.5:
            score.issues.append(f"Gini 系数过高: {gini:.4f}")

        # 记录近似统计信息
        score.sampled_pairs = self._cosine_stats.get("num_sampled_pairs", 0)

        # 记录最终阶段的内存
        mem_summary = self._memory_tracker.get_summary()
        score.peak_memory_mb = mem_summary["peak_gpu_mb"]
        score.memory_stages = mem_summary["stages"]

        if verbose:
            print(f"\n{'='*70}")
            print(f"[Initialization Diagnostics]")
            print(f"{'='*70}")
            print(f"  Init Score: {score.total_score:.1f}/100")
            print(f"  Status: {score.status}")
            print(f"\n  分项得分:")
            print(f"    Reconstruction    : {score.reconstruction_score:.1f}/20")
            print(f"    Diversity         : {score.diversity_score:.1f}/20")
            print(f"    Mutual Coherence  : {score.mutual_coherence_score:.1f}/15")
            print(f"    TopK Competition  : {score.topk_score:.1f}/20")
            print(f"    Gini              : {score.gini_score:.1f}/15")
            print(f"    Spectrum          : {score.spectrum_score:.1f}/10")

            print(f"\n  [Approximate Statistics]")
            print(f"    Sampled Pairs: {score.sampled_pairs:,}")
            print(f"    Mutual Coherence: {mu:.4f}")

            # 打印内存摘要
            self._memory_tracker.print_summary()

            if score.issues:
                print(f"\n  问题列表:")
                for issue in score.issues:
                    print(f"    - {issue}")
            else:
                print(f"\n  ✓ 无明显问题")
            print(f"{'='*70}")

        return score

    def save_plots(self, output_dir: str, prefix: str = "") -> List[str]:
        """生成并保存可视化图表 (采样模式)"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠ matplotlib 不可用，跳过图表生成")
            return []

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_files = []

        # ================================
        # 1. Cosine Histogram (采样)
        # ================================
        if self._cosine_stats is None:
            self.analyze_feature_similarity(verbose=False)

        # 重新采样用于绘图
        Wdec_norm = F.normalize(self.Wdec, dim=0)
        sample_size = min(100_000, self.num_pairs)

        idx_i = torch.randint(0, self.d_hidden, (sample_size,))
        idx_j = torch.randint(0, self.d_hidden, (sample_size,))
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        cos_samples = (Wdec_norm[:, idx_i] * Wdec_norm[:, idx_j]).sum(dim=0).cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(cos_samples, bins=100, density=True, alpha=0.7, color='steelblue')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title(f'{prefix}Feature Cosine Similarity Distribution (Sampled)')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)

        filepath = output_path / f"{prefix}cosine_histogram.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(filepath))

        # ================================
        # 2. Cosine Heatmap (采样)
        # ================================
        sample_n = min(256, self.d_hidden)  # 减小采样数
        sample_idx = torch.randperm(self.d_hidden)[:sample_n]
        Wdec_sample = Wdec_norm[:, sample_idx]
        cos_matrix = (Wdec_sample.T @ Wdec_sample).cpu().numpy()

        plt.figure(figsize=(8, 7))
        im = plt.imshow(cos_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, label='Cosine Similarity')
        plt.title(f'{prefix}Feature Cosine Heatmap (sampled {sample_n})')

        filepath = output_path / f"{prefix}cosine_heatmap.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(filepath))

        # ================================
        # 3. Firing Frequency Histogram
        # ================================
        if self._topk_stats is None:
            self.analyze_topk_dynamics(verbose=False)

        firing_counts = self._topk_stats["firing_counts"].numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(firing_counts[firing_counts > 0], bins=100, alpha=0.7, color='steelblue')
        plt.xlabel('Firing Count')
        plt.ylabel('Number of Features')
        plt.title(f'{prefix}Feature Firing Frequency')
        plt.grid(True, alpha=0.3)

        filepath = output_path / f"{prefix}firing_frequency_hist.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(filepath))

        # ================================
        # 4. SVD Spectrum
        # ================================
        if self._svd_stats is None:
            self.analyze_svd_spectrum(verbose=False)

        S = self._svd_stats["singular_values"].numpy()
        cum_var = self._svd_stats["cumulative_variance"].numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].semilogy(S[:min(256, len(S))], 'b-')
        axes[0].set_xlabel('Component')
        axes[0].set_ylabel('Singular Value (log)')
        axes[0].set_title(f'{prefix}Singular Value Spectrum')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(cum_var[:min(256, len(cum_var))], 'g-')
        axes[1].axhline(y=0.9, color='r', linestyle='--', label='90%')
        axes[1].set_xlabel('Component')
        axes[1].set_ylabel('Cumulative Variance')
        axes[1].set_title(f'{prefix}Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        filepath = output_path / f"{prefix}svd_spectrum.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(filepath))

        # ================================
        # 5. TopK Overlap Histogram
        # ================================
        overlap_mean = self._topk_stats["topk_overlap_mean"]

        plt.figure(figsize=(10, 6))
        plt.bar(['Mean Overlap'], [overlap_mean], color='steelblue', alpha=0.7)
        plt.ylabel('Overlap Ratio')
        plt.title(f'{prefix}TopK Overlap: {overlap_mean:.3f}')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')

        filepath = output_path / f"{prefix}topk_overlap.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(filepath))

        # ================================
        # 6. Feature Usage Gini
        # ================================
        if self._gini is None:
            self.analyze_gini_coefficient(verbose=False)

        firing_counts_sorted = np.sort(firing_counts[firing_counts > 0])
        if len(firing_counts_sorted) > 0:
            n = len(firing_counts_sorted)
            cumulative_share = np.cumsum(firing_counts_sorted) / firing_counts_sorted.sum()

            plt.figure(figsize=(10, 6))
            plt.plot(np.linspace(0, 1, n), cumulative_share, 'b-', label='Lorenz Curve')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
            plt.fill_between(np.linspace(0, 1, n), np.linspace(0, 1, n), cumulative_share,
                            alpha=0.3, color='red')
            plt.xlabel('Cumulative Share of Features')
            plt.ylabel('Cumulative Share of Firing')
            plt.title(f'{prefix}Feature Usage (Gini={self._gini["gini_coefficient"]:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)

            filepath = output_path / f"{prefix}feature_usage_gini.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(str(filepath))

        return saved_files


# ============================================================================
# 基础检查函数
# ============================================================================

def check_sae_initialization(
    init_file: str,
    cache_dir: str,
    layer_idx: int,
    top_k: int = 128,
    num_pairs: int = DEFAULT_NUM_PAIRS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_plots: bool = False,
    plot_dir: str = "./plots/init_analysis",
) -> dict:
    """
    检查 SAE 初始化质量 (超宽 Latent 工程化版本)
    """
    print("=" * 70)
    print(f"SAE 初始化质量检查 - Layer {layer_idx}")
    print("=" * 70)

    # ========== 加载初始化文件 ==========
    print(f"\n[加载] {init_file}")
    data = torch.load(init_file, map_location="cpu")

    Wdec = data["Wdec"].float()
    Wenc = data["Wenc"].float()
    bpre = data["bpre"].float()

    print(f"  Wdec shape: {Wdec.shape}")
    print(f"  Wenc shape: {Wenc.shape}")
    print(f"  bpre shape: {bpre.shape}")

    d_hidden = Wdec.shape[1]

    results = {}

    # ========== 检查 1: Decoder column norm ==========
    print(f"\n[检查 1] Decoder column norm")
    col_norms = Wdec.norm(dim=0)
    norm_mean = col_norms.mean().item()
    norm_std = col_norms.std().item()
    norm_max_dev = (col_norms - 1).abs().max().item()

    results["decoder_norm_mean"] = norm_mean
    results["decoder_norm_std"] = norm_std
    results["decoder_norm_max_dev"] = norm_max_dev

    print(f"  mean: {norm_mean:.6f}, std: {norm_std:.6f}, max_dev: {norm_max_dev:.2e}")

    if norm_max_dev < 1e-3:
        print(f"  ✓ 优秀")
    elif norm_max_dev < 5e-3:
        print(f"  ✓ 可接受")
    else:
        print(f"  ⚠ 偏差过大")

    # ========== 检查 2: Tied 初始化 ==========
    print(f"\n[检查 2] Tied 初始化")
    tied_match = torch.allclose(Wenc, Wdec.T, atol=1e-6)
    results["tied_initialization"] = tied_match
    print(f"  {'✓ 通过' if tied_match else '⚠ 不匹配'}")

    # ========== 加载原始激活 ==========
    cache_file = Path(cache_dir) / f"layer{layer_idx}.pt"
    print(f"\n[加载] 原始激活: {cache_file}")

    x = torch.load(cache_file, map_location="cpu").float()
    print(f"  shape: {x.shape}")

    # ========== Per-token RMSNorm ==========
    print(f"\n[预处理] Per-token RMSNorm")
    eps = 1e-6
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    print(f"  x_norm mean: {x_norm.mean():.4f}, std: {x_norm.std():.4f}")

    # ========== 检查 3: Reconstruction MSE ==========
    print(f"\n[检查 3] Reconstruction MSE")

    n_sample = min(10000, x_norm.shape[0])
    x_sample = x_norm[:n_sample]

    x_centered = x_sample - bpre
    z = F.relu(x_centered @ Wenc.T)
    x_hat_centered = z @ Wdec.T
    x_hat = x_hat_centered + bpre

    mse = F.mse_loss(x_hat, x_sample).item()
    variance = x_sample.var().item()
    mse_ratio = mse / variance

    results["reconstruction_mse"] = mse
    results["mse_to_variance_ratio"] = mse_ratio

    print(f"  MSE: {mse:.4f}, 方差: {variance:.4f}, 比值: {mse_ratio:.2f}")
    print(f"  {'✓ 通过' if mse_ratio < 20 else '⚠ MSE过高'}")

    # ========== 检查 4: Dead neurons ==========
    print(f"\n[检查 4] Dead neurons")
    z_active = (z > 0).any(dim=0)
    dead_count = (~z_active).sum().item()
    dead_ratio = dead_count / d_hidden

    results["dead_neuron_count"] = dead_count
    results["dead_neuron_ratio"] = dead_ratio

    print(f"  count: {dead_count} / {d_hidden} ({dead_ratio:.2%})")
    print(f"  {'✓ 通过' if dead_ratio <= 0.05 else '⚠ 死神经元过多'}")

    # ========== 检查 5: PCA 方差覆盖 ==========
    if "pca_stats" in data:
        print(f"\n[检查 5] PCA variance coverage")
        explained_variance_ratio = data["pca_stats"]["explained_variance_ratio"]
        cum_var = explained_variance_ratio.cumsum(0)

        for k in [64, 128, 256, 512, 1024, 1536]:
            if len(cum_var) >= k:
                print(f"  top{k:4d}: {cum_var[k-1].item() * 100:.2f}%")

        results["pca_variance"] = {
            f"top{k}": cum_var[k-1].item() if len(cum_var) >= k else None
            for k in [64, 128, 256, 512, 1024, 1536]
        }

    # ========== 工业级诊断 ==========
    print(f"\n{'='*70}")
    print(f"工业级诊断分析 (采样模式)")
    print(f"{'='*70}")

    analyzer = InitializationAnalyzer(
        Wdec=Wdec,
        Wenc=Wenc,
        bpre=bpre,
        x_norm=x_norm,
        top_k=top_k,
        num_pairs=num_pairs,
        batch_size=batch_size,
    )

    cosine_stats = analyzer.analyze_feature_similarity()
    topk_stats = analyzer.analyze_topk_dynamics()
    gini_stats = analyzer.analyze_gini_coefficient()
    svd_stats = analyzer.analyze_svd_spectrum()

    score = analyzer.compute_initialization_score(mse_ratio)

    results["diagnostics"] = {
        "cosine_similarity": cosine_stats,
        "topk_dynamics": {k: v for k, v in topk_stats.items() if k != "firing_counts"},
        "gini": gini_stats,
        "svd_spectrum": {k: v for k, v in svd_stats.items()
                        if k not in ["singular_values", "explained_variance_ratio", "cumulative_variance"]},
        "score": score.to_dict(),
    }

    # ========== 可视化图表 ==========
    if save_plots:
        print(f"\n[生成] 可视化图表...")
        saved_files = analyzer.save_plots(output_dir=plot_dir, prefix=f"layer{layer_idx}_")
        results["plot_files"] = saved_files
        for f in saved_files:
            print(f"  保存: {f}")

    # ========== 总结 ==========
    all_passed = True
    issues = []

    if norm_max_dev > 5e-3:
        all_passed = False
        issues.append(f"Decoder norm deviation: {norm_max_dev:.2e}")
    if not tied_match:
        all_passed = False
        issues.append("Tied initialization failed")
    if mse_ratio >= 20:
        all_passed = False
        issues.append(f"MSE/方差比: {mse_ratio:.2f}")
    if dead_ratio > 0.05:
        all_passed = False
        issues.append(f"Dead neuron ratio: {dead_ratio:.2%}")

    issues.extend(score.issues)

    results["all_passed"] = all_passed and (score.status != "DANGEROUS")
    results["issues"] = issues

    print(f"\n{'='*70}")
    print(f"质量检查总结: {'✓ 通过' if results['all_passed'] else '⚠ 存在问题'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SAE 初始化质量检查 (超宽 Latent 工程化版本)")

    parser.add_argument("--init_dir", type=str, default="./sae_init")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--layer", type=str, default="all")
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--num_pairs", type=int, default=DEFAULT_NUM_PAIRS,
                        help=f"采样 pair 数量 (默认: {DEFAULT_NUM_PAIRS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--plot_dir", type=str, default="./plots/init_analysis")

    args = parser.parse_args()

    if args.layer.lower() == "all":
        layers = [14, 19, 24, 29]
    else:
        layers = [int(x.strip()) for x in args.layer.split(",")]

    all_results = {}

    for layer_idx in layers:
        init_file = Path(args.init_dir) / f"sae_init_layer{layer_idx}.pt"

        if not init_file.exists():
            print(f"\n⚠ 跳过 Layer {layer_idx}: 文件不存在")
            continue

        result = check_sae_initialization(
            init_file=str(init_file),
            cache_dir=args.cache_dir,
            layer_idx=layer_idx,
            top_k=args.top_k,
            num_pairs=args.num_pairs,
            batch_size=args.batch_size,
            save_plots=args.save_plots,
            plot_dir=args.plot_dir,
        )

        all_results[f"layer{layer_idx}"] = result

    # 汇总
    print("\n" + "=" * 70)
    print("批量检查汇总")
    print("=" * 70)

    for layer_key, result in all_results.items():
        status = "✓" if result.get("all_passed", False) else "⚠"
        score = result.get("diagnostics", {}).get("score", {}).get("total_score", "N/A")
        mse = result.get("reconstruction_mse", "N/A")

        if isinstance(score, float):
            score = f"{score:.1f}"
        if isinstance(mse, float):
            mse = f"{mse:.2f}"

        print(f"  {layer_key}: {status} | Score={score} | MSE={mse}")

    passed = sum(1 for r in all_results.values() if r.get("all_passed", False))
    print(f"\n  通过: {passed}/{len(all_results)}")
    print("=" * 70)

    if args.save_plots:
        output_path = Path(args.plot_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        serializable_results = {}
        for layer_key, result in all_results.items():
            serializable_results[layer_key] = {
                k: v for k, v in result.items() if k != "plot_files"
            }

        results_file = output_path / "diagnostics_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  详细结果保存至: {results_file}")


if __name__ == "__main__":
    main()
