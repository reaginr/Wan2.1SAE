#!/usr/bin/env python3
"""
SAE 混合源初始化脚本 - Competition-Oriented 版本

核心原则:
- 禁止 duplicated PCA expansion
- 禁止 activation monopoly
- 初始化目标: feature competition, angular diversity, sparse differentiability

初始化组成 (hidden_dim=12288, d_model=1536):
1. PCA principal directions: 1536
2. Residual PCA directions: ~512-1024 (根据 explained variance threshold)
3. Random orthogonal directions: 3072
4. Random Gaussian directions: 2560
5. Activation sample directions: 3072 (partial decorrelated)

验收标准:
- Geometry: max cosine < 0.7, P99 cosine < 0.2, mu < 0.3
- Competition: entropy > 0.7, active ratio > 40%
- Distribution: Gini < 0.7

使用方法:
    python -m 初始化.sae_mixed_init --cache_dir ./cache --output_dir ./sae_init --layer all
"""

from __future__ import annotations

import argparse
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


# ============================================================================
# 常量配置
# ============================================================================

# 初始化组成 (hidden_dim=12288, d_model=1536)
N_PCA_PRINCIPAL = 1536           # Top PCA components
N_RANDOM_ORTHOGONAL = 3072       # Random orthogonal directions
N_RANDOM_GAUSSIAN = 2560         # Random Gaussian directions
N_ACTIVATION_SAMPLE_MAX = 3072   # Activation samples (最多 hidden_dim 的 25%)

# Residual PCA 配置
RESIDUAL_VARIANCE_THRESHOLD = 0.80  # 累计方差达到 80% 停止
RESIDUAL_PCA_MIN = 256              # 最少成分数
RESIDUAL_PCA_MAX = 1024             # 最多成分数

# 质量标准 - Geometry
MAX_MUTUAL_COHERENCE = 0.30
MAX_COSINE = 0.70
MAX_P99_COSINE = 0.20

# 质量标准 - Competition
MIN_COMPETITION_ENTROPY = 0.70
MIN_ACTIVE_RATIO = 0.40
MAX_GINI = 0.70

# Mutual Coherence 过滤
COHERENCE_THRESHOLD = 0.70
COHERENCE_TARGET = 0.30
MAX_COHERENCE_ITERATIONS = 200

# 采样配置
N_SAMPLE_PAIRS = 1_000_000  # 用于 cosine 统计


# ============================================================================
# 工具函数
# ============================================================================

def per_token_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-token RMSNorm"""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms


def weiszfeld_geometric_median(
    x: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-5,
    verbose: bool = True,
) -> torch.Tensor:
    """Weiszfeld 算法计算几何中位数"""
    N, D = x.shape
    median = x.mean(dim=0)

    for i in range(max_iter):
        diff = x - median
        dist = diff.norm(dim=1).clamp(min=1e-8)
        weights = 1.0 / dist
        new_median = (weights.unsqueeze(1) * x).sum(dim=0) / weights.sum()

        if (new_median - median).norm() < tol:
            if verbose:
                print(f"  [Geometric Median] 收敛于第 {i+1} 次迭代")
            return new_median
        median = new_median

    return median


def randomized_pca(
    x: torch.Tensor,
    n_components: int,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized PCA"""
    N, D = x.shape
    n_components = min(n_components, min(N, D))

    if verbose:
        print(f"  [PCA] 输入: [{N}, {D}], n_components={n_components}")

    U, S, V = torch.pca_lowrank(x.float(), q=n_components, center=False)

    components = V
    explained_variance = (S[:n_components] ** 2) / (N - 1)
    total_variance = x.var(dim=0).sum()
    explained_variance_ratio = explained_variance / total_variance

    return components, explained_variance, explained_variance_ratio


def residual_pca_adaptive(
    x_residual: torch.Tensor,
    variance_threshold: float = RESIDUAL_VARIANCE_THRESHOLD,
    min_components: int = RESIDUAL_PCA_MIN,
    max_components: int = RESIDUAL_PCA_MAX,
    verbose: bool = True,
) -> Tuple[torch.Tensor, int, float]:
    """
    自适应 Residual PCA

    根据累计方差阈值确定成分数，而非固定数量

    返回:
        components: [D, n_selected]
        n_selected: 选择的成分数
        cumulative_variance: 最终累计方差
    """
    D = x_residual.shape[1]
    max_components = min(max_components, D)

    if verbose:
        print(f"  [Residual PCA] 自适应选择 (阈值: {variance_threshold})")

    # 计算足够多的成分
    components, explained_var, explained_ratio = randomized_pca(
        x_residual, max_components, verbose=False
    )

    # 找到达到阈值的成分数
    cumsum = explained_ratio.cumsum(0)
    n_selected = min_components

    for i in range(min_components, max_components):
        if cumsum[i].item() >= variance_threshold:
            n_selected = i + 1
            break
    else:
        n_selected = max_components

    if verbose:
        actual_variance = cumsum[n_selected - 1].item()
        print(f"  [Residual PCA] 选择 {n_selected} 成分, 累计方差: {actual_variance*100:.2f}%")

    return components[:, :n_selected], n_selected, cumsum[n_selected - 1].item()


def generate_random_orthogonal(
    d_model: int,
    n_vectors: int,
    device: str = "cpu",
) -> torch.Tensor:
    """生成随机正交向量"""
    n_vectors = min(n_vectors, d_model)  # 最多 d_model 个正交向量

    A = torch.randn(d_model, n_vectors, device=device)
    Q, R = torch.linalg.qr(A)
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)

    return Q


def sample_activations_decorrelated(
    x_norm: torch.Tensor,
    n_samples: int,
    W_existing: torch.Tensor,
    device: str = "cpu",
    verbose: bool = True,
) -> torch.Tensor:
    """
    从真实激活中采样向量并进行 partial decorrelation

    参数:
        x_norm: [N, D] RMSNorm 后的激活
        n_samples: 采样数量
        W_existing: [D, K] 已有的 dictionary (PCA + residual + orthogonal)
        device: 设备
        verbose: 是否输出信息

    返回:
        samples: [D, n_samples] 去相关后的采样向量
    """
    N, D = x_norm.shape

    if n_samples > N:
        n_samples = N

    if verbose:
        print(f"  [Activation Sample] 采样 {n_samples} 向量并进行 decorrelation")

    # 分层采样
    norms = x_norm.norm(dim=1)
    sorted_idx = norms.argsort()

    n_high = int(n_samples * 0.4)
    n_medium = int(n_samples * 0.4)
    n_uniform = n_samples - n_high - n_medium

    high_region = sorted_idx[int(N * 0.6):]
    high_idx = high_region[torch.randperm(len(high_region))[:n_high]]

    medium_region = sorted_idx[int(N * 0.3):int(N * 0.7)]
    medium_idx = medium_region[torch.randperm(len(medium_region))[:n_medium]]

    uniform_idx = sorted_idx[torch.randperm(N)[:n_uniform]]

    selected_idx = torch.cat([high_idx, medium_idx, uniform_idx])

    # 提取
    A = x_norm[selected_idx].to(device)  # [n_samples, D]

    # Partial decorrelation: 去除对已有 dictionary 的投影
    # A_decorrelated = A - A @ W_existing @ W_existing.T
    W_existing = W_existing.to(device)
    projection = A @ W_existing  # [n_samples, K]
    A_decorrelated = A - projection @ W_existing.T  # [n_samples, D]

    # 归一化
    A_decorrelated = F.normalize(A_decorrelated, dim=1)

    if verbose:
        # 计算去相关前后的差异
        original_norm = A.norm()
        decorrelated_norm = A_decorrelated.norm()
        print(f"  [Activation Sample] Decorrelation: norm {original_norm:.2f} -> {decorrelated_norm:.2f}")

    return A_decorrelated.T  # [D, n_samples]


# ============================================================================
# Cosine Similarity 采样分析
# ============================================================================

def compute_cosine_stats_sampled(
    W: torch.Tensor,
    n_samples: int = N_SAMPLE_PAIRS,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    采样计算 Cosine Similarity 统计

    返回:
        mean, std, p95, p99, max
    """
    D, K = W.shape
    W_norm = F.normalize(W, dim=0)

    n_samples = min(n_samples, K * (K - 1) // 2)

    # 采样
    idx_i = torch.randint(0, K, (n_samples,), device=device)
    idx_j = torch.randint(0, K, (n_samples,), device=device)
    mask = idx_i != idx_j

    cos_values = (W_norm[:, idx_i[mask]] * W_norm[:, idx_j[mask]]).sum(dim=0).abs()

    # 统计
    cos_mean = cos_values.mean().item()
    cos_std = cos_values.std().item()
    cos_p95 = torch.quantile(cos_values, 0.95).item()
    cos_p99 = torch.quantile(cos_values, 0.99).item()
    cos_max = cos_values.max().item()

    if verbose:
        print(f"  [Cosine Stats] Mean={cos_mean:.4f}, P95={cos_p95:.4f}, P99={cos_p99:.4f}, Max={cos_max:.4f}")

    return {
        "cosine_mean": cos_mean,
        "cosine_std": cos_std,
        "cosine_p95": cos_p95,
        "cosine_p99": cos_p99,
        "cosine_max": cos_max,
    }


def compute_mutual_coherence_sampled(
    W: torch.Tensor,
    n_samples: int = N_SAMPLE_PAIRS,
    device: str = "cpu",
) -> Tuple[float, int, int]:
    """采样计算 Mutual Coherence"""
    D, K = W.shape
    W_norm = F.normalize(W, dim=0)

    max_abs_cos = 0.0
    max_i, max_j = 0, 0

    batch_size = 10000
    n_batches = n_samples // batch_size

    for _ in range(n_batches):
        idx_i = torch.randint(0, K, (batch_size,), device=device)
        idx_j = torch.randint(0, K, (batch_size,), device=device)
        mask = idx_i != idx_j

        cos_values = (W_norm[:, idx_i[mask]] * W_norm[:, idx_j[mask]]).sum(dim=0).abs()
        local_max, local_idx = cos_values.max(dim=0)

        if local_max > max_abs_cos:
            max_abs_cos = local_max.item()
            max_i = idx_i[mask][local_idx].item()
            max_j = idx_j[mask][local_idx].item()

    return max_abs_cos, max_i, max_j


def enforce_mutual_coherence(
    Wdec: torch.Tensor,
    threshold: float = COHERENCE_THRESHOLD,
    target: float = COHERENCE_TARGET,
    max_iterations: int = MAX_COHERENCE_ITERATIONS,
    verbose: bool = True,
) -> torch.Tensor:
    """执行 Mutual Coherence 过滤"""
    D, K = Wdec.shape
    device = Wdec.device

    if verbose:
        print(f"\n  [Coherence Filter] 阈值: {threshold}, 目标: {target}")

    for iteration in range(max_iterations):
        mu, i, j = compute_mutual_coherence_sampled(Wdec, n_samples=500_000, device=device)

        if verbose and iteration % 20 == 0:
            print(f"    迭代 {iteration}: mu = {mu:.4f}")

        if mu <= target:
            if verbose:
                print(f"    ✓ 达到目标: mu = {mu:.4f} <= {target}")
            break

        if mu > threshold:
            # 重新随机初始化其中一个向量
            new_vec = torch.randn(D, device=device)
            new_vec = F.normalize(new_vec, dim=0)
            Wdec[:, i] = new_vec

    # 最终归一化
    Wdec = F.normalize(Wdec, dim=0)

    return Wdec


# ============================================================================
# Competition Entropy 计算
# ============================================================================

def compute_competition_metrics(
    Wdec: torch.Tensor,
    bpre: torch.Tensor,
    x_norm: torch.Tensor,
    top_k: int = 128,
    n_samples: int = 2048,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    计算 Competition-oriented 指标

    返回:
        competition_entropy: 归一化熵
        active_ratio: 活跃 feature 比例
        gini: Gini 系数
    """
    D, K = Wdec.shape
    device = Wdec.device

    if verbose:
        print(f"\n  [Competition Metrics] TopK={top_k}, Samples={n_samples}")

    # 采样
    n_total = x_norm.shape[0]
    idx = torch.randperm(n_total, device=device)[:n_samples]
    x_sample = x_norm[idx]

    # SAE Encode
    Wenc = Wdec.T
    x_centered = x_sample - bpre.to(device)
    z = x_centered @ Wenc

    # TopK
    top_k = min(top_k, K)
    _, topk_indices = torch.topk(z.abs(), top_k, dim=1)

    # Firing counts
    firing_counts = torch.zeros(K, device=device)
    firing_counts.scatter_add_(0, topk_indices.flatten(),
                               torch.ones(topk_indices.numel(), device=device))

    # Competition Entropy
    prob = firing_counts / (firing_counts.sum() + 1e-10)
    prob = prob.clamp(min=1e-10)
    entropy = -(prob * torch.log(prob)).sum().item()
    max_entropy = torch.log(torch.tensor(K, dtype=torch.float32)).item()
    competition_entropy = entropy / max_entropy

    # Active Ratio
    active_count = (firing_counts > 0).sum().item()
    active_ratio = active_count / K

    # Gini
    values = firing_counts.float().sort()[0]
    n = len(values)
    index = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    gini = (2 * (index * values).sum()) / (n * values.sum() + 1e-10) - (n + 1) / n
    gini = gini.item()

    if verbose:
        print(f"    Competition Entropy: {competition_entropy:.4f} (阈值: {MIN_COMPETITION_ENTROPY})")
        print(f"    Active Ratio: {active_ratio:.2%} (阈值: {MIN_ACTIVE_RATIO:.0%})")
        print(f"    Gini: {gini:.4f} (阈值: {MAX_GINI})")

    return {
        "competition_entropy": competition_entropy,
        "active_ratio": active_ratio,
        "gini": gini,
    }


# ============================================================================
# 混合源初始化
# ============================================================================

def mixed_source_initialization(
    x_norm: torch.Tensor,
    d_hidden: int = 12288,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Competition-Oriented 混合源初始化
    """
    N, D = x_norm.shape
    device = x_norm.device

    if verbose:
        print(f"\n{'='*70}")
        print(f"[Competition-Oriented Mixed Initialization]")
        print(f"{'='*70}")
        print(f"  输入: [{N}, {D}], d_hidden={d_hidden}")

    components_list = []
    stats = {}

    # ========== 1. 几何中位数 ==========
    if verbose:
        print(f"\n[1/6] 计算几何中位数...")

    bpre = weiszfeld_geometric_median(x_norm, verbose=verbose)
    x_centered = x_norm - bpre

    # ========== 2. PCA Principal (1536) ==========
    if verbose:
        print(f"\n[2/6] PCA Principal Directions ({N_PCA_PRINCIPAL})...")

    pca_components, pca_var, pca_ratio = randomized_pca(x_centered, N_PCA_PRINCIPAL, verbose)
    W_pca = pca_components.clone()
    components_list.append(W_pca)

    stats["n_pca"] = W_pca.shape[1]
    stats["pca_top128_variance"] = pca_ratio[:128].sum().item()

    if verbose:
        print(f"  形状: {W_pca.shape}")

    # ========== 3. Residual PCA (自适应) ==========
    if verbose:
        print(f"\n[3/6] Residual PCA (自适应)...")

    x_reconstructed = x_centered @ W_pca @ W_pca.T
    x_residual = x_centered - x_reconstructed

    residual_norm_ratio = x_residual.norm() / x_centered.norm()
    stats["residual_norm_ratio"] = residual_norm_ratio.item()

    if verbose:
        print(f"  残差范数比: {residual_norm_ratio:.4f}")

    W_residual, n_residual, residual_variance = residual_pca_adaptive(
        x_residual,
        variance_threshold=RESIDUAL_VARIANCE_THRESHOLD,
        verbose=verbose,
    )
    components_list.append(W_residual)

    stats["n_residual"] = n_residual
    stats["residual_variance"] = residual_variance

    # ========== 4. Random Orthogonal (3072) ==========
    if verbose:
        print(f"\n[4/6] Random Orthogonal Directions ({N_RANDOM_ORTHOGONAL})...")

    W_ortho = generate_random_orthogonal(D, N_RANDOM_ORTHOGONAL, device)
    components_list.append(W_ortho)

    stats["n_ortho"] = W_ortho.shape[1]

    if verbose:
        print(f"  形状: {W_ortho.shape}")

    # ========== 5. Random Gaussian (2560) ==========
    if verbose:
        print(f"\n[5/6] Random Gaussian Directions ({N_RANDOM_GAUSSIAN})...")

    W_gauss = torch.randn(D, N_RANDOM_GAUSSIAN, device=device)
    W_gauss = F.normalize(W_gauss, dim=0)
    components_list.append(W_gauss)

    stats["n_gauss"] = N_RANDOM_GAUSSIAN

    if verbose:
        print(f"  形状: {W_gauss.shape}")

    # ========== 6. Activation Sample (decorrelated) ==========
    # 计算剩余需要的 activation 数量
    current_total = sum(c.shape[1] for c in components_list)
    n_activation = d_hidden - current_total

    # 限制最大值
    n_activation = min(n_activation, N_ACTIVATION_SAMPLE_MAX)

    if verbose:
        print(f"\n[6/6] Activation Samples ({n_activation}, decorrelated)...")

    # 合并已有 dictionary 用于 decorrelation
    W_existing = torch.cat(components_list, dim=1)

    W_activation = sample_activations_decorrelated(
        x_norm, n_activation, W_existing, device, verbose
    )
    components_list.append(W_activation)

    stats["n_activation"] = n_activation

    # ========== 合并 ==========
    if verbose:
        print(f"\n[合并所有来源]...")

    Wdec = torch.cat(components_list, dim=1)

    if verbose:
        print(f"  合并后形状: {Wdec.shape}")
        print(f"  来源: PCA={stats['n_pca']}, Residual={stats['n_residual']}, "
              f"Ortho={stats['n_ortho']}, Gauss={stats['n_gauss']}, "
              f"Activation={stats['n_activation']}")

    # 验证维度
    assert Wdec.shape[1] == d_hidden, f"维度不匹配: {Wdec.shape[1]} != {d_hidden}"

    # 归一化
    Wdec = F.normalize(Wdec, dim=0)

    # Global shuffle
    perm = torch.randperm(d_hidden)
    Wdec = Wdec[:, perm]

    # Global jitter (仅打破对称性)
    Wdec = Wdec + 0.01 * torch.randn_like(Wdec)
    Wdec = F.normalize(Wdec, dim=0)

    if verbose:
        print(f"  已执行 global shuffle + jitter")

    # ========== Mutual Coherence 过滤 ==========
    Wdec = enforce_mutual_coherence(Wdec, verbose=verbose)

    stats["total_features"] = Wdec.shape[1]

    if verbose:
        print(f"\n{'='*70}")
        print(f"[初始化完成]")
        print(f"{'='*70}")

    return Wdec, bpre, stats


# ============================================================================
# 质量验证 (Competition-Oriented)
# ============================================================================

def validate_initialization(
    Wdec: torch.Tensor,
    bpre: torch.Tensor,
    x_norm: torch.Tensor,
    top_k: int = 128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Competition-Oriented 质量验证

    验收标准:
    - Geometry: max_cosine < 0.7, P99_cosine < 0.2, mu < 0.3
    - Competition: entropy > 0.7, active_ratio > 40%
    - Distribution: Gini < 0.7
    """
    D, K = Wdec.shape
    device = Wdec.device

    if verbose:
        print(f"\n{'='*70}")
        print(f"[Competition-Oriented 质量验证]")
        print(f"{'='*70}")

    results = {}

    # ========== 1. Geometry 指标 ==========
    if verbose:
        print(f"\n[1/2] Geometry 指标...")

    cos_stats = compute_cosine_stats_sampled(Wdec, n_samples=N_SAMPLE_PAIRS, device=device, verbose=verbose)
    results.update(cos_stats)

    mu, _, _ = compute_mutual_coherence_sampled(Wdec, n_samples=N_SAMPLE_PAIRS, device=device)
    results["mutual_coherence"] = mu

    if verbose:
        print(f"  Mutual Coherence: {mu:.4f} (阈值: {MAX_MUTUAL_COHERENCE})")

    # ========== 2. Competition 指标 ==========
    if verbose:
        print(f"\n[2/2] Competition 指标...")

    comp_metrics = compute_competition_metrics(Wdec, bpre, x_norm, top_k=top_k, verbose=verbose)
    results.update(comp_metrics)

    # ========== 3. Decoder Norm ==========
    col_norms = Wdec.norm(dim=0)
    results["decoder_norm_mean"] = col_norms.mean().item()
    results["decoder_norm_std"] = col_norms.std().item()
    results["decoder_norm_max_dev"] = (col_norms - 1).abs().max().item()

    # ========== 验收判定 ==========
    geometry_passed = (
        results["cosine_max"] < MAX_COSINE and
        results["cosine_p99"] < MAX_P99_COSINE and
        results["mutual_coherence"] < MAX_MUTUAL_COHERENCE
    )

    competition_passed = (
        results["competition_entropy"] >= MIN_COMPETITION_ENTROPY and
        results["active_ratio"] >= MIN_ACTIVE_RATIO
    )

    distribution_passed = results["gini"] < MAX_GINI

    all_passed = geometry_passed and competition_passed and distribution_passed

    results["geometry_passed"] = geometry_passed
    results["competition_passed"] = competition_passed
    results["distribution_passed"] = distribution_passed
    results["all_passed"] = all_passed

    # ========== 总结 ==========
    if verbose:
        print(f"\n{'='*70}")
        print(f"[验收结果]")
        print(f"{'='*70}")

        print(f"\n  Geometry:")
        geo_status = "✓ 通过" if geometry_passed else "⚠ 失败"
        print(f"    max_cosine: {results['cosine_max']:.4f} < {MAX_COSINE} {'✓' if results['cosine_max'] < MAX_COSINE else '⚠'}")
        print(f"    P99_cosine: {results['cosine_p99']:.4f} < {MAX_P99_COSINE} {'✓' if results['cosine_p99'] < MAX_P99_COSINE else '⚠'}")
        print(f"    mu: {results['mutual_coherence']:.4f} < {MAX_MUTUAL_COHERENCE} {'✓' if results['mutual_coherence'] < MAX_MUTUAL_COHERENCE else '⚠'}")
        print(f"    {geo_status}")

        print(f"\n  Competition:")
        comp_status = "✓ 通过" if competition_passed else "⚠ 失败"
        print(f"    entropy: {results['competition_entropy']:.4f} > {MIN_COMPETITION_ENTROPY} {'✓' if results['competition_entropy'] >= MIN_COMPETITION_ENTROPY else '⚠'}")
        print(f"    active_ratio: {results['active_ratio']:.2%} > {MIN_ACTIVE_RATIO:.0%} {'✓' if results['active_ratio'] >= MIN_ACTIVE_RATIO else '⚠'}")
        print(f"    {comp_status}")

        print(f"\n  Distribution:")
        dist_status = "✓ 通过" if distribution_passed else "⚠ 失败"
        print(f"    Gini: {results['gini']:.4f} < {MAX_GINI} {'✓' if results['gini'] < MAX_GINI else '⚠'}")
        print(f"    {dist_status}")

        print(f"\n  总体: {'✓ 通过' if all_passed else '⚠ 失败'}")
        print(f"{'='*70}")

    return results


# ============================================================================
# 主初始化函数
# ============================================================================

def initialize_sae_layer(
    cache_file: str,
    output_file: str,
    layer_idx: int,
    d_hidden: int = 12288,
    top_k: int = 128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """初始化单个层的 SAE"""
    print(f"\n{'='*70}")
    print(f"SAE 混合源初始化 (Competition-Oriented) - Layer {layer_idx}")
    print(f"{'='*70}")

    start_time = time.time()

    # 加载激活
    print(f"\n[加载] {cache_file}")
    x = torch.load(cache_file, map_location="cpu").float()
    print(f"  形状: {x.shape}")

    # RMSNorm
    print(f"\n[预处理] Per-token RMSNorm")
    x_norm = per_token_rms_norm(x)
    print(f"  x_norm mean: {x_norm.mean():.4f}, std: {x_norm.std():.4f}")

    # 混合源初始化
    Wdec, bpre, init_stats = mixed_source_initialization(
        x_norm, d_hidden=d_hidden, verbose=verbose
    )

    # Encoder 权重 (Tied)
    Wenc = Wdec.T.clone()

    # 质量验证
    quality_results = validate_initialization(
        Wdec, bpre, x_norm, top_k=top_k, verbose=verbose
    )

    # 保存
    print(f"\n[保存] {output_file}")

    output_data = {
        "Wdec": Wdec.cpu(),
        "Wenc": Wenc.cpu(),
        "bpre": bpre.cpu(),
        "config": {
            "d_model": Wdec.shape[0],
            "d_hidden": Wdec.shape[1],
            "init_method": "competition_oriented_mixed",
            **init_stats,
        },
        "quality": quality_results,
        "layer_idx": layer_idx,
    }

    torch.save(output_data, output_file)

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"[完成] Layer {layer_idx}")
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  验收: {'✓ 通过' if quality_results['all_passed'] else '⚠ 失败'}")
    print(f"{'='*70}")

    return {
        "layer_idx": layer_idx,
        "elapsed": elapsed,
        "quality": quality_results,
        "all_passed": quality_results["all_passed"],
    }


def main():
    parser = argparse.ArgumentParser(description="SAE Competition-Oriented 混合初始化")

    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./sae_init")
    parser.add_argument("--layer", type=str, default="all",
                        help="层索引: 'all' 或 '14' 或 '14,19,24,29'")
    parser.add_argument("--d_hidden", type=int, default=12288)
    parser.add_argument("--top_k", type=int, default=128)

    args = parser.parse_args()

    # 确定要初始化的层
    if args.layer.lower() == "all":
        layers = [14, 19, 24, 29]
    else:
        layers = [int(x.strip()) for x in args.layer.split(",")]

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 打印启动信息
    print(f"\n{'='*70}")
    print(f"SAE Competition-Oriented 混合初始化")
    print(f"{'='*70}")
    print(f"  缓存目录: {args.cache_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  待初始化层: {layers}")
    print(f"  d_hidden: {args.d_hidden}")
    print(f"  top_k: {args.top_k}")
    print(f"{'='*70}")

    total_start_time = time.time()
    all_results = {}

    # 依次初始化每一层
    for i, layer_idx in enumerate(layers, 1):
        print(f"\n{'#'*70}")
        print(f"# [{i}/{len(layers)}] 开始初始化 Layer {layer_idx}")
        print(f"{'#'*70}")

        cache_file = Path(args.cache_dir) / f"layer{layer_idx}.pt"
        output_file = output_path / f"sae_init_layer{layer_idx}.pt"

        if not cache_file.exists():
            print(f"\n⚠ 跳过 Layer {layer_idx}: 缓存文件不存在 {cache_file}")
            all_results[f"layer{layer_idx}"] = {"error": "cache_not_found", "all_passed": False}
            continue

        try:
            result = initialize_sae_layer(
                cache_file=str(cache_file),
                output_file=str(output_file),
                layer_idx=layer_idx,
                d_hidden=args.d_hidden,
                top_k=args.top_k,
            )
            all_results[f"layer{layer_idx}"] = result
        except Exception as e:
            print(f"\n⚠ Layer {layer_idx} 初始化失败: {e}")
            all_results[f"layer{layer_idx}"] = {"error": str(e), "all_passed": False}
            continue

        # 每层完成后打印进度
        print(f"\n>>> Layer {layer_idx} 完成，进度: {i}/{len(layers)}")

    total_elapsed = time.time() - total_start_time

    # ==================== 最终汇总 ====================
    print(f"\n{'='*70}")
    print("初始化完成汇总")
    print(f"{'='*70}")
    print(f"  总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    # 详细结果表格
    print(f"\n  {'Layer':<10} {'Status':<8} {'mu':<8} {'entropy':<10} {'active':<10} {'gini':<8} {'cos_max':<10}")
    print(f"  {'-'*64}")

    for layer_key, result in all_results.items():
        if "error" in result:
            print(f"  {layer_key:<10} {'⚠ 错误':<8} {result['error']}")
        else:
            status = "✓ 通过" if result["all_passed"] else "⚠ 失败"
            q = result["quality"]
            print(f"  {layer_key:<10} {status:<8} {q['mutual_coherence']:.4f}   "
                  f"{q['competition_entropy']:.4f}     "
                  f"{q['active_ratio']:.1%}      "
                  f"{q['gini']:.4f}   "
                  f"{q['cosine_max']:.4f}")

    # 统计
    passed = sum(1 for r in all_results.values() if r.get("all_passed", False))
    total = len(all_results)

    print(f"\n  验收通过: {passed}/{total}")

    if passed == total:
        print(f"\n  ✓ 所有层初始化成功！")
    else:
        print(f"\n  ⚠ 部分层初始化未通过验收，请检查日志")

    print(f"{'='*70}")

    # 保存汇总结果
    summary_file = output_path / "init_summary.json"

    # 清理不可序列化的数据
    serializable_results = {}
    for layer_key, result in all_results.items():
        if isinstance(result, dict):
            serializable_results[layer_key] = {
                k: v for k, v in result.items()
                if not isinstance(v, torch.Tensor)
            }
            # 处理 quality 中的 tensor
            if "quality" in serializable_results[layer_key]:
                q = serializable_results[layer_key]["quality"]
                serializable_results[layer_key]["quality"] = {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in q.items()
                }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_elapsed_sec": total_elapsed,
            "passed": passed,
            "total": total,
            "layers": serializable_results,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  汇总保存至: {summary_file}")


if __name__ == "__main__":
    main()
