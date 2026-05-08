#!/usr/bin/env python3
"""
SAE 初始化质量检查脚本 - 工业级诊断版本

用于检查已保存的初始化文件质量，包含完整的诊断分析。

核心检测:
1. 基础检查: Reconstruction MSE, Tied init, Decoder norm, Dead neurons
2. Feature Cosine Similarity: 检测 duplicated features
3. TopK Dynamics: 模拟 TopK 竞争健康度
4. Gini Coefficient: 检测 feature monopoly
5. SVD Spectrum: 检测低秩塌缩
6. 初始化评分系统

使用方法:
    python -m 初始化.sae_quality_check --init_dir ./sae_init --cache_dir ./cache --layer 14
    python -m 初始化.sae_quality_check --init_dir ./sae_init --cache_dir ./cache --layer all
    python -m 初始化.sae_quality_check --init_dir ./sae_init --cache_dir ./cache --layer all --save_plots
"""

from __future__ import annotations

import argparse
import json
import sys
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


# ============================================================================
# 评分数据类
# ============================================================================

@dataclass
class InitializationScore:
    """初始化评分结果"""
    total_score: float = 0.0          # 总分 0-100
    reconstruction_score: float = 0.0  # 重建质量分
    diversity_score: float = 0.0       # 特征多样性分
    topk_score: float = 0.0            # TopK竞争分
    gini_score: float = 0.0            # Gini系数分
    spectrum_score: float = 0.0        # SVD谱分

    status: str = "UNKNOWN"            # GOOD / WARNING / DANGEROUS
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "reconstruction_score": self.reconstruction_score,
            "diversity_score": self.diversity_score,
            "topk_score": self.topk_score,
            "gini_score": self.gini_score,
            "spectrum_score": self.spectrum_score,
            "status": self.status,
            "issues": self.issues,
        }


# ============================================================================
# 初始化诊断分析器
# ============================================================================

class InitializationAnalyzer:
    """
    SAE 初始化诊断分析器

    功能:
    1. Feature Cosine Similarity 分析
    2. TopK Dynamics 模拟
    3. Gini Coefficient 计算
    4. SVD Spectrum 分析
    5. 可视化图表生成
    6. 初始化评分
    """

    def __init__(
        self,
        Wdec: torch.Tensor,
        Wenc: torch.Tensor,
        bpre: torch.Tensor,
        x_norm: torch.Tensor,
        top_k: int = 128,
        device: str = "cpu",
    ):
        """
        参数:
            Wdec: Decoder 权重 [d_model, d_hidden]
            Wenc: Encoder 权重 [d_hidden, d_model]
            bpre: 几何中位数 [d_model]
            x_norm: RMSNorm 后的激活 [N, d_model]
            top_k: TopK 稀疏度
            device: 计算设备
        """
        self.Wdec = Wdec.float().to(device)
        self.Wenc = Wenc.float().to(device)
        self.bpre = bpre.float().to(device)
        self.x_norm = x_norm.float().to(device)
        self.top_k = top_k
        self.device = device

        self.d_model = Wdec.shape[0]
        self.d_hidden = Wdec.shape[1]

        # 缓存结果
        self._cosine_stats = None
        self._topk_stats = None
        self._gini = None
        self._svd_stats = None

    # ------------------------------------------------------------------------
    # 1. Feature Cosine Similarity 分析
    # ------------------------------------------------------------------------

    def analyze_feature_similarity(
        self,
        chunk_size: int = 1024,
        sample_for_heatmap: int = 512,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        分析 Feature 之间的余弦相似度

        检测 duplicated features / PCA expansion collapse

        参数:
            chunk_size: 分块计算大小 (避免 OOM)
            sample_for_heatmap: heatmap 采样数量
            verbose: 是否输出详细信息

        返回:
            统计结果字典
        """
        if verbose:
            print(f"\n[InitDiag] Feature Cosine Similarity 分析...")

        Wdec_norm = F.normalize(self.Wdec, dim=0)  # [d_model, d_hidden]
        d_hidden = self.d_hidden

        # 分块计算 pairwise cosine similarity
        # 只计算上三角 (不含对角线)
        all_cosines = []

        for i in range(0, d_hidden, chunk_size):
            end_i = min(i + chunk_size, d_hidden)
            chunk_i = Wdec_norm[:, i:end_i]  # [d_model, chunk_size]

            for j in range(i + 1, d_hidden, chunk_size):
                end_j = min(j + chunk_size, d_hidden)
                chunk_j = Wdec_norm[:, j:end_j]  # [d_model, chunk_size]

                # 计算余弦相似度矩阵 [chunk_i, chunk_j]
                cos_matrix = chunk_i.T @ chunk_j
                all_cosines.append(cos_matrix.flatten())

        # 合并所有余弦值
        all_cosines = torch.cat(all_cosines)

        # 统计
        cos_mean = all_cosines.mean().item()
        cos_std = all_cosines.std().item()
        cos_p95 = torch.quantile(all_cosines, 0.95).item()
        cos_p99 = torch.quantile(all_cosines, 0.99).item()
        cos_max = all_cosines.max().item()

        results = {
            "cosine_mean": cos_mean,
            "cosine_std": cos_std,
            "cosine_p95": cos_p95,
            "cosine_p99": cos_p99,
            "cosine_max": cos_max,
        }

        if verbose:
            print(f"  [InitDiag] CosSim Mean: {cos_mean:.4f}")
            print(f"  [InitDiag] CosSim Std : {cos_std:.4f}")
            print(f"  [InitDiag] CosSim P95 : {cos_p95:.4f}")
            print(f"  [InitDiag] CosSim P99 : {cos_p99:.4f}")
            print(f"  [InitDiag] CosSim Max : {cos_max:.4f}")

            # 判定
            if cos_mean < 0.05 and cos_p95 < 0.2 and cos_max < 0.5:
                print(f"  [InitDiag] ✓ 特征多样性优秀")
            elif cos_p95 > 0.4:
                print(f"  [InitDiag] ⚠ 存在较多相似特征")
            elif cos_max > 0.8:
                print(f"  [InitDiag] ⚠ 存在高相似度特征对")
            else:
                print(f"  [InitDiag] ✓ 特征多样性良好")

        self._cosine_stats = results
        return results

    # ------------------------------------------------------------------------
    # 2. TopK Dynamics 模拟
    # ------------------------------------------------------------------------

    def analyze_topk_dynamics(
        self,
        n_samples: int = 2048,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        模拟 TopK 竞争健康度

        分析:
        1. Feature Firing Frequency
        2. TopK Overlap
        3. Activation Entropy

        参数:
            n_samples: 采样 token 数
            verbose: 是否输出详细信息

        返回:
            统计结果字典
        """
        if verbose:
            print(f"\n[InitDiag] TopK Dynamics 模拟...")

        # 采样
        n_total = self.x_norm.shape[0]
        if n_samples > n_total:
            n_samples = n_total

        idx = torch.randperm(n_total, device=self.device)[:n_samples]
        x_sample = self.x_norm[idx]  # [n_samples, d_model]

        # SAE Encode
        x_centered = x_sample - self.bpre
        z = x_centered @ self.Wenc.T  # [n_samples, d_hidden]

        # TopK
        top_k = min(self.top_k, self.d_hidden)
        topk_values, topk_indices = torch.topk(z, top_k, dim=1)

        # ================================
        # A. Feature Firing Frequency
        # ================================
        firing_counts = torch.zeros(self.d_hidden, device=self.device)
        firing_counts.scatter_add_(0, topk_indices.flatten(),
                                   torch.ones(topk_indices.numel(), device=self.device))

        firing_freq = firing_counts / n_samples  # 每个feature被选中的频率

        freq_mean = firing_freq.mean().item()
        freq_std = firing_freq.std().item()
        freq_top1pct = torch.quantile(firing_freq, 0.99).item()
        freq_bottom1pct = torch.quantile(firing_freq, 0.01).item()

        # ================================
        # B. Activation Entropy
        # ================================
        # 归一化 firing frequency 作为概率分布
        prob = firing_freq / firing_freq.sum()
        prob = prob.clamp(min=1e-10)  # 避免 log(0)
        entropy = -(prob * torch.log(prob)).sum().item()

        # 最大熵: 均匀分布
        max_entropy = torch.log(torch.tensor(self.d_hidden, dtype=torch.float32)).item()
        normalized_entropy = entropy / max_entropy

        # ================================
        # C. TopK Overlap
        # ================================
        # 随机采样 pair 计算 overlap
        n_overlap_samples = min(500, n_samples // 2)
        overlaps = []

        for _ in range(n_overlap_samples):
            i, j = torch.randint(0, n_samples, (2,), device=self.device)
            set_i = set(topk_indices[i].tolist())
            set_j = set(topk_indices[j].tolist())
            overlap = len(set_i & set_j) / top_k
            overlaps.append(overlap)

        overlap_mean = sum(overlaps) / len(overlaps)
        overlap_p95 = sorted(overlaps)[int(len(overlaps) * 0.95)]

        results = {
            "firing_freq_mean": freq_mean,
            "firing_freq_std": freq_std,
            "firing_freq_top1pct": freq_top1pct,
            "firing_freq_bottom1pct": freq_bottom1pct,
            "activation_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "topk_overlap_mean": overlap_mean,
            "topk_overlap_p95": overlap_p95,
            "firing_counts": firing_counts.cpu(),  # 用于 Gini 计算
        }

        if verbose:
            print(f"  [InitDiag] Firing Freq Mean: {freq_mean:.6f}")
            print(f"  [InitDiag] Firing Freq Std : {freq_std:.6f}")
            print(f"  [InitDiag] Firing Freq Top1%: {freq_top1pct:.6f}")
            print(f"  [InitDiag] Activation Entropy: {normalized_entropy:.4f} (normalized)")
            print(f"  [InitDiag] TopK Overlap Mean: {overlap_mean:.4f}")
            print(f"  [InitDiag] TopK Overlap P95 : {overlap_p95:.4f}")

            # 判定
            if normalized_entropy > 0.8 and overlap_mean < 0.3:
                print(f"  [InitDiag] ✓ TopK 竞争健康")
            elif normalized_entropy < 0.5:
                print(f"  [InitDiag] ⚠ Feature 使用不均衡")
            elif overlap_mean > 0.5:
                print(f"  [InitDiag] ⚠ TopK Overlap 过高，可能存在 feature monopoly")
            else:
                print(f"  [InitDiag] ✓ TopK 竞争良好")

        self._topk_stats = results
        return results

    # ------------------------------------------------------------------------
    # 3. Gini Coefficient
    # ------------------------------------------------------------------------

    def analyze_gini_coefficient(
        self,
        firing_counts: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        计算 Gini 系数

        检测 feature 使用是否极度不均衡

        参数:
            firing_counts: firing 频率 (如果为 None, 需要先运行 TopK 分析)
            verbose: 是否输出详细信息

        返回:
            统计结果字典
        """
        if verbose:
            print(f"\n[InitDiag] Gini Coefficient 计算...")

        if firing_counts is None:
            if self._topk_stats is None:
                self.analyze_topk_dynamics(verbose=False)
            firing_counts = self._topk_stats["firing_counts"]

        # 排序
        values = firing_counts.float().sort()[0]
        n = len(values)

        # Gini coefficient formula
        # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
        index = torch.arange(1, n + 1, dtype=torch.float32)
        gini = (2 * (index * values).sum()) / (n * values.sum() + 1e-10) - (n + 1) / n
        gini = gini.item()

        # 额外统计: 有多少 feature 从未被选中
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

            # 判定
            if gini < 0.3:
                print(f"  [InitDiag] ✓ Feature 使用均衡")
            elif gini > 0.7:
                print(f"  [InitDiag] ⚠ Feature 极度不均衡，可能存在 feature monopoly")
            elif gini > 0.5:
                print(f"  [InitDiag] ⚠ Feature 使用不均衡")
            else:
                print(f"  [InitDiag] ✓ Feature 使用较均衡")

        self._gini = results
        return results

    # ------------------------------------------------------------------------
    # 4. SVD Spectrum 分析
    # ------------------------------------------------------------------------

    def analyze_svd_spectrum(
        self,
        n_components: int = 256,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        分析 SVD 谱

        检测 PCA 是否低秩塌缩

        参数:
            n_components: 分析的主成分数
            verbose: 是否输出详细信息

        返回:
            统计结果字典
        """
        if verbose:
            print(f"\n[InitDiag] SVD Spectrum 分析...")

        # 对原始激活矩阵 (已 RMSNorm) 进行 SVD
        # 采样以加速
        n_samples = min(10000, self.x_norm.shape[0])
        x_sample = self.x_norm[:n_samples]

        # 中心化
        x_centered = x_sample - self.bpre

        # SVD
        U, S, V = torch.svd(x_centered, some=True)

        # 解释方差
        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_variance
        cumulative_variance = explained_variance_ratio.cumsum(0)

        # 提取关键点
        top_k_list = [32, 64, 128, 256]
        variance_at_k = {}
        for k in top_k_list:
            if k <= len(cumulative_variance):
                variance_at_k[f"top{k}"] = cumulative_variance[k-1].item()

        # 计算有效秩 (effective rank)
        # Effective rank = exp(entropy of normalized singular values)
        s_normalized = S / S.sum()
        s_normalized = s_normalized.clamp(min=1e-10)
        entropy = -(s_normalized * torch.log(s_normalized)).sum().item()
        effective_rank = min(n_samples, self.d_model)  # 简化版

        results = {
            "singular_values": S.cpu(),
            "explained_variance_ratio": explained_variance_ratio.cpu(),
            "cumulative_variance": cumulative_variance.cpu(),
            "variance_at_k": variance_at_k,
            "effective_rank_estimate": effective_rank,
        }

        if verbose:
            print(f"  [InitDiag] SVD Spectrum:")
            for k, var in variance_at_k.items():
                print(f"    {k}: {var*100:.2f}%")

            # 判定: 检查是否低秩塌缩
            if "top128" in variance_at_k and variance_at_k["top128"] > 0.9:
                print(f"  [InitDiag] ⚠ 残差空间可能过于低秩")
                print(f"  [InitDiag]   可能存在 token sampling collapse")
            else:
                print(f"  [InitDiag] ✓ 残差空间秩正常")

        self._svd_stats = results
        return results

    # ------------------------------------------------------------------------
    # 5. 综合评分
    # ------------------------------------------------------------------------

    def compute_initialization_score(
        self,
        mse_ratio: float,
        verbose: bool = True,
    ) -> InitializationScore:
        """
        计算初始化综合评分 (0-100)

        评分维度:
        1. Reconstruction Quality (25分)
        2. Feature Diversity (25分)
        3. TopK Competition (20分)
        4. Gini Coefficient (15分)
        5. SVD Spectrum (15分)

        参数:
            mse_ratio: MSE/方差比
            verbose: 是否输出详细信息

        返回:
            InitializationScore 对象
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
        # 1. Reconstruction Quality (25分)
        # ================================
        # MSE ratio < 10: 满分
        # MSE ratio > 30: 0分
        if mse_ratio < 10:
            score.reconstruction_score = 25.0
        elif mse_ratio > 30:
            score.reconstruction_score = 0.0
        else:
            score.reconstruction_score = 25.0 * (1 - (mse_ratio - 10) / 20)

        # ================================
        # 2. Feature Diversity (25分)
        # ================================
        cos_mean = self._cosine_stats["cosine_mean"]
        cos_p95 = self._cosine_stats["cosine_p95"]
        cos_max = self._cosine_stats["cosine_max"]

        # mean < 0.05: +10
        # p95 < 0.2: +10
        # max < 0.5: +5
        div_score = 0.0
        if cos_mean < 0.05:
            div_score += 10
        elif cos_mean < 0.1:
            div_score += 5

        if cos_p95 < 0.2:
            div_score += 10
        elif cos_p95 < 0.4:
            div_score += 5

        if cos_max < 0.5:
            div_score += 5
        elif cos_max < 0.8:
            div_score += 2

        score.diversity_score = div_score

        # ================================
        # 3. TopK Competition (20分)
        # ================================
        entropy = self._topk_stats["normalized_entropy"]
        overlap = self._topk_stats["topk_overlap_mean"]

        # entropy > 0.8: +10
        # overlap < 0.3: +10
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
        # 4. Gini Coefficient (15分)
        # ================================
        gini = self._gini["gini_coefficient"]

        # gini < 0.3: 满分
        # gini > 0.7: 0分
        if gini < 0.3:
            score.gini_score = 15.0
        elif gini > 0.7:
            score.gini_score = 0.0
        else:
            score.gini_score = 15.0 * (1 - (gini - 0.3) / 0.4)

        # ================================
        # 5. SVD Spectrum (15分)
        # ================================
        var_k = self._svd_stats["variance_at_k"]
        top128_var = var_k.get("top128", 0.5)

        # top128 > 90%: 可能低秩塌缩, 扣分
        # top128 < 70%: 秩正常, 满分
        if top128_var > 0.9:
            score.spectrum_score = 5.0
            score.issues.append("SVD: 可能低秩塌缩")
        elif top128_var < 0.7:
            score.spectrum_score = 15.0
        else:
            score.spectrum_score = 10.0

        # ================================
        # 总分计算
        # ================================
        score.total_score = (
            score.reconstruction_score +
            score.diversity_score +
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

        if verbose:
            print(f"\n{'='*70}")
            print(f"[Initialization Diagnostics]")
            print(f"{'='*70}")
            print(f"  Init Score: {score.total_score:.1f}/100")
            print(f"  Status: {score.status}")
            print(f"\n  分项得分:")
            print(f"    Reconstruction: {score.reconstruction_score:.1f}/25")
            print(f"    Diversity     : {score.diversity_score:.1f}/25")
            print(f"    TopK          : {score.topk_score:.1f}/20")
            print(f"    Gini          : {score.gini_score:.1f}/15")
            print(f"    Spectrum      : {score.spectrum_score:.1f}/15")

            if score.issues:
                print(f"\n  问题列表:")
                for issue in score.issues:
                    print(f"    - {issue}")
            else:
                print(f"\n  ✓ 无明显问题")
            print(f"{'='*70}")

        return score

    # ------------------------------------------------------------------------
    # 6. 可视化图表生成
    # ------------------------------------------------------------------------

    def save_plots(
        self,
        output_dir: str,
        prefix: str = "",
    ) -> List[str]:
        """
        生成并保存可视化图表

        参数:
            output_dir: 输出目录
            prefix: 文件名前缀 (如 "layer14_")

        返回:
            保存的文件路径列表
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠ matplotlib 不可用，跳过图表生成")
            return []

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # ================================
        # 1. Cosine Histogram
        # ================================
        if self._cosine_stats is None:
            self.analyze_feature_similarity(verbose=False)

        # 重新采样 cosines 用于绘图
        Wdec_norm = F.normalize(self.Wdec, dim=0)
        d_hidden = self.d_hidden
        sample_size = min(100000, d_hidden * (d_hidden - 1) // 2)

        # 随机采样 feature pairs
        idx_i = torch.randint(0, d_hidden, (sample_size,))
        idx_j = torch.randint(0, d_hidden, (sample_size,))
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        cos_samples = (Wdec_norm[:, idx_i] * Wdec_norm[:, idx_j]).sum(dim=0).cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(cos_samples, bins=100, density=True, alpha=0.7, color='steelblue')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title(f'{prefix}Feature Cosine Similarity Distribution')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)

        filepath = output_path / f"{prefix}cosine_histogram.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(filepath))

        # ================================
        # 2. Cosine Heatmap (采样)
        # ================================
        sample_n = min(512, d_hidden)
        sample_idx = torch.randperm(d_hidden)[:sample_n]
        Wdec_sample = Wdec_norm[:, sample_idx]
        cos_matrix = (Wdec_sample.T @ Wdec_sample).cpu().numpy()

        plt.figure(figsize=(10, 8))
        im = plt.imshow(cos_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, label='Cosine Similarity')
        plt.title(f'{prefix}Feature Cosine Similarity Heatmap (sampled {sample_n})')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Index')

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
        plt.title(f'{prefix}Feature Firing Frequency Distribution')
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

        # Singular values (log scale)
        axes[0].semilogy(S[:256], 'b-')
        axes[0].set_xlabel('Component')
        axes[0].set_ylabel('Singular Value (log scale)')
        axes[0].set_title(f'{prefix}Singular Value Spectrum')
        axes[0].grid(True, alpha=0.3)

        # Cumulative variance
        axes[1].plot(cum_var[:256], 'g-')
        axes[1].axhline(y=0.9, color='r', linestyle='--', label='90%')
        axes[1].axhline(y=0.95, color='orange', linestyle='--', label='95%')
        axes[1].set_xlabel('Component')
        axes[1].set_ylabel('Cumulative Variance Ratio')
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
        # 重新计算 overlap 用于绘图
        n_overlap_samples = 1000
        overlaps = []

        top_k = min(self.top_k, self.d_hidden)
        n_samples = min(2048, self.x_norm.shape[0])
        idx = torch.randperm(self.x_norm.shape[0], device=self.device)[:n_samples]
        x_sample = self.x_norm[idx]
        x_centered = x_sample - self.bpre
        z = x_centered @ self.Wenc.T
        _, topk_indices = torch.topk(z, top_k, dim=1)
        topk_indices = topk_indices.cpu()

        for _ in range(n_overlap_samples):
            i, j = np.random.randint(0, n_samples, 2)
            set_i = set(topk_indices[i].tolist())
            set_j = set(topk_indices[j].tolist())
            overlap = len(set_i & set_j) / top_k
            overlaps.append(overlap)

        plt.figure(figsize=(10, 6))
        plt.hist(overlaps, bins=50, alpha=0.7, color='steelblue')
        plt.xlabel('TopK Overlap Ratio')
        plt.ylabel('Count')
        plt.title(f'{prefix}TopK Overlap Distribution')
        plt.axvline(x=np.mean(overlaps), color='red', linestyle='--',
                   label=f'Mean: {np.mean(overlaps):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        filepath = output_path / f"{prefix}topk_overlap_hist.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(filepath))

        # ================================
        # 6. Feature Usage Gini
        # ================================
        if self._gini is None:
            self.analyze_gini_coefficient(verbose=False)

        firing_counts_sorted = np.sort(firing_counts[firing_counts > 0])
        n = len(firing_counts_sorted)
        cumulative_share = np.cumsum(firing_counts_sorted) / firing_counts_sorted.sum()

        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, n), cumulative_share, 'b-', label='Lorenz Curve')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
        plt.fill_between(np.linspace(0, 1, n), np.linspace(0, 1, n), cumulative_share,
                        alpha=0.3, color='red')
        plt.xlabel('Cumulative Share of Features')
        plt.ylabel('Cumulative Share of Firing')
        plt.title(f'{prefix}Feature Usage Distribution (Gini={self._gini["gini_coefficient"]:.3f})')
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
    save_plots: bool = False,
    plot_dir: str = "./plots/init_analysis",
) -> dict:
    """
    检查 SAE 初始化质量 (工业级诊断版本)

    检查项:
    1. 基础检查: Reconstruction MSE, Dead neurons, Decoder norm, Tied init
    2. Feature Cosine Similarity: 检测 duplicated features
    3. TopK Dynamics: 模拟 TopK 竞争健康度
    4. Gini Coefficient: 检测 feature monopoly
    5. SVD Spectrum: 检测低秩塌缩
    6. 初始化评分系统
    """
    print("=" * 70)
    print(f"SAE 初始化质量检查 (工业级诊断) - Layer {layer_idx}")
    print("=" * 70)

    # ========== 加载初始化文件 ==========
    print(f"\n[加载] {init_file}")
    data = torch.load(init_file, map_location="cpu")

    Wdec = data["Wdec"].float()    # [1536, 12288]
    Wenc = data["Wenc"].float()    # [12288, 1536]
    bpre = data["bpre"].float()    # [1536]

    print(f"  Wdec shape: {Wdec.shape}")
    print(f"  Wenc shape: {Wenc.shape}")
    print(f"  bpre shape: {bpre.shape}")

    d_hidden = Wdec.shape[1]

    results = {}

    # ========== 检查 1: Decoder column norm ==========
    print(f"\n[检查 1] Decoder column norm (应为 1.0)")
    col_norms = Wdec.norm(dim=0)  # [12288]
    norm_mean = col_norms.mean().item()
    norm_std = col_norms.std().item()
    norm_max_dev = (col_norms - 1).abs().max().item()

    results["decoder_norm_mean"] = norm_mean
    results["decoder_norm_std"] = norm_std
    results["decoder_norm_max_dev"] = norm_max_dev

    print(f"  mean: {norm_mean:.6f}")
    print(f"  std: {norm_std:.6f}")
    print(f"  max deviation from 1: {norm_max_dev:.2e}")

    if norm_max_dev < 1e-3:
        print(f"  ✓ 通过 (优秀)")
    elif norm_max_dev < 5e-3:
        print(f"  ✓ 通过 (可接受)")
    else:
        print(f"  ⚠ 偏差过大")

    # ========== 检查 2: Tied 初始化 ==========
    print(f"\n[检查 2] Tied 初始化 (Wenc == Wdec.T)")
    tied_match = torch.allclose(Wenc, Wdec.T, atol=1e-6)
    results["tied_initialization"] = tied_match

    if tied_match:
        print(f"  ✓ 通过")
    else:
        diff = (Wenc - Wdec.T).abs().max().item()
        print(f"  ⚠ 不匹配, max diff: {diff:.2e}")

    # ========== 加载原始激活 ==========
    cache_file = Path(cache_dir) / f"layer{layer_idx}.pt"
    print(f"\n[加载] 原始激活: {cache_file}")

    x = torch.load(cache_file, map_location="cpu").float()  # [256000, 1536]
    print(f"  shape: {x.shape}")

    # ========== Per-token RMSNorm ==========
    print(f"\n[预处理] Per-token RMSNorm")
    eps = 1e-6
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    print(f"  x_norm mean: {x_norm.mean():.4f}, std: {x_norm.std():.4f}")

    # ========== 检查 3: Reconstruction MSE ==========
    print(f"\n[检查 3] Reconstruction MSE")
    print(f"  流程: x_norm -> 中心化 -> 编码(ReLU) -> 解码 -> 去中心化 -> x_hat")
    print(f"  注意: ReLU会截断负值，初始MSE较高是正常的")

    # 采样计算
    n_sample = min(10000, x_norm.shape[0])
    x_sample = x_norm[:n_sample]

    # 正确的前向传播:
    x_centered = x_sample - bpre
    z = F.relu(x_centered @ Wenc.T)
    x_hat_centered = z @ Wdec.T
    x_hat = x_hat_centered + bpre

    # 计算 MSE
    mse = F.mse_loss(x_hat, x_sample).item()
    results["reconstruction_mse"] = mse

    # 计算 MSE 相对于方差的比率
    variance = x_sample.var().item()
    mse_ratio = mse / variance
    results["mse_to_variance_ratio"] = mse_ratio

    print(f"  MSE: {mse:.4f}")
    print(f"  输入方差: {variance:.4f}")
    print(f"  MSE/方差比: {mse_ratio:.2f} (阈值: 20)")

    if mse_ratio < 20:
        print(f"  ✓ 通过 (MSE/方差 < 20)")
    else:
        print(f"  ⚠ MSE过高 (MSE/方差 >= 20)")

    # ========== 检查 4: Dead neuron ratio ==========
    print(f"\n[检查 4] Dead neurons")
    z_active = (z > 0).any(dim=0)
    dead_count = (~z_active).sum().item()
    dead_ratio = dead_count / d_hidden

    results["dead_neuron_count"] = dead_count
    results["dead_neuron_ratio"] = dead_ratio

    print(f"  count: {dead_count} / {d_hidden}")
    print(f"  ratio: {dead_ratio:.2%}")

    if dead_ratio <= 0.05:
        print(f"  ✓ 通过 (阈值: 5%)")
    else:
        print(f"  ⚠ 死神经元过多 (阈值: 5%)")

    # ========== 检查 5: PCA 方差覆盖 ==========
    if "pca_stats" in data:
        print(f"\n[检查 5] PCA variance coverage")
        explained_variance_ratio = data["pca_stats"]["explained_variance_ratio"]
        cum_var = explained_variance_ratio.cumsum(0)

        for k in [64, 128, 256, 512, 1024, 1536]:
            if len(cum_var) >= k:
                ratio = cum_var[k-1].item() * 100
                print(f"  top{k:4d}: {ratio:.2f}%")

        results["pca_variance"] = {
            "top64": cum_var[63].item() if len(cum_var) >= 64 else None,
            "top128": cum_var[127].item() if len(cum_var) >= 128 else None,
            "top256": cum_var[255].item() if len(cum_var) >= 256 else None,
            "top512": cum_var[511].item() if len(cum_var) >= 512 else None,
            "top1024": cum_var[1023].item() if len(cum_var) >= 1024 else None,
            "top1536": cum_var[1535].item() if len(cum_var) >= 1536 else None,
        }

    # ========== 工业级诊断 ==========
    print(f"\n{'='*70}")
    print(f"工业级诊断分析")
    print(f"{'='*70}")

    # 创建分析器
    analyzer = InitializationAnalyzer(
        Wdec=Wdec,
        Wenc=Wenc,
        bpre=bpre,
        x_norm=x_norm,
        top_k=top_k,
    )

    # 运行所有分析
    cosine_stats = analyzer.analyze_feature_similarity()
    topk_stats = analyzer.analyze_topk_dynamics()
    gini_stats = analyzer.analyze_gini_coefficient()
    svd_stats = analyzer.analyze_svd_spectrum()

    # 计算综合评分
    score = analyzer.compute_initialization_score(mse_ratio)

    # 保存诊断结果
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
        saved_files = analyzer.save_plots(
            output_dir=plot_dir,
            prefix=f"layer{layer_idx}_",
        )
        results["plot_files"] = saved_files
        for f in saved_files:
            print(f"  保存: {f}")

    # ========== 总结 ==========
    print(f"\n{'='*70}")
    print(f"质量检查总结:")

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
        issues.append(f"Reconstruction MSE/方差比: {mse_ratio:.2f} (阈值: 20)")
    if dead_ratio > 0.05:
        all_passed = False
        issues.append(f"Dead neuron ratio: {dead_ratio:.2%}")

    # 添加诊断问题
    issues.extend(score.issues)

    results["all_passed"] = all_passed and (score.status != "DANGEROUS")
    results["issues"] = issues

    if results["all_passed"]:
        print(f"  ✓ 所有检查通过")
    else:
        print(f"  ⚠ 存在问题:")
        for issue in issues:
            print(f"      - {issue}")

    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SAE 初始化质量检查 (工业级诊断)")

    parser.add_argument("--init_dir", type=str, default="./sae_init",
                        help="初始化文件目录")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="激活缓存目录")
    parser.add_argument("--layer", type=str, default="all",
                        help="层索引，如 '14' 或 'all' 或 '14,19,24,29'")
    parser.add_argument("--top_k", type=int, default=128,
                        help="TopK 稀疏度 (默认: 128)")
    parser.add_argument("--save_plots", action="store_true",
                        help="生成并保存可视化图表")
    parser.add_argument("--plot_dir", type=str, default="./plots/init_analysis",
                        help="图表输出目录")

    args = parser.parse_args()

    # 确定要检查的层
    if args.layer.lower() == "all":
        layers = [14, 19, 24, 29]
    else:
        layers = [int(x.strip()) for x in args.layer.split(",")]

    # 批量检查
    all_results = {}

    for layer_idx in layers:
        init_file = Path(args.init_dir) / f"sae_init_layer{layer_idx}.pt"

        if not init_file.exists():
            print(f"\n⚠ 跳过 Layer {layer_idx}: 文件不存在 {init_file}")
            continue

        result = check_sae_initialization(
            init_file=str(init_file),
            cache_dir=args.cache_dir,
            layer_idx=layer_idx,
            top_k=args.top_k,
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
        mse = result.get("reconstruction_mse", "N/A")
        dead = result.get("dead_neuron_ratio", "N/A")
        score = result.get("diagnostics", {}).get("score", {}).get("total_score", "N/A")

        if isinstance(mse, float):
            mse = f"{mse:.4f}"
        if isinstance(dead, float):
            dead = f"{dead:.2%}"
        if isinstance(score, float):
            score = f"{score:.1f}"

        print(f"  {layer_key}: {status} | Score={score} | MSE={mse} | Dead={dead}")

    # 总结
    passed = sum(1 for r in all_results.values() if r.get("all_passed", False))
    total = len(all_results)

    print(f"\n  通过: {passed}/{total}")
    print("=" * 70)

    # 保存详细结果
    if args.save_plots:
        output_path = Path(args.plot_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 移除不可序列化的数据
        serializable_results = {}
        for layer_key, result in all_results.items():
            serializable_results[layer_key] = {
                k: v for k, v in result.items()
                if k not in ["plot_files"]
            }

        results_file = output_path / "diagnostics_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  详细结果保存至: {results_file}")


if __name__ == "__main__":
    main()
