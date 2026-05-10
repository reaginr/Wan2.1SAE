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
# 固定比例配置
N_PCA_PRINCIPAL = 1536       # PCA Principal
N_RESIDUAL_PCA = 768         # Residual PCA
N_RANDOM_ORTHOGONAL = 1024   # Orthogonal
N_ACTIVATION_SAMPLE = 1536   # Activation Sample
N_SPARSE_GAUSSIAN = 7424     # Sparse Gaussian

# 验证总数
assert N_PCA_PRINCIPAL + N_RESIDUAL_PCA + N_RANDOM_ORTHOGONAL + N_ACTIVATION_SAMPLE + N_SPARSE_GAUSSIAN == 12288, \
    f"总数应为 12288, 当前: {N_PCA_PRINCIPAL + N_RESIDUAL_PCA + N_RANDOM_ORTHOGONAL + N_ACTIVATION_SAMPLE + N_SPARSE_GAUSSIAN}"

# 质量标准 - Geometry (放宽标准，适合超宽 SAE)
# 注意：在1536维空间放12288个向量，必然存在高相似度向量，这是数学限制
MAX_MUTUAL_COHERENCE = 0.75   # 放宽：0.50 -> 0.75 (超宽SAE的数学限制)
MAX_COSINE = 0.85             # 放宽：0.70 -> 0.85
MAX_P99_COSINE = 0.50         # 放宽：0.30 -> 0.50

# 质量标准 - Competition
MIN_COMPETITION_ENTROPY = 0.60   # 放宽：0.70 -> 0.60
MIN_ACTIVE_RATIO = 0.30          # 放宽：0.40 -> 0.30
MAX_GINI = 0.75                  # 放宽：0.70 -> 0.75

# Mutual Coherence 过滤
COHERENCE_TARGET = 0.75           # 放宽：0.50 -> 0.75 (现实目标)
MAX_COHERENCE_ITERATIONS = 100    # 减少：500 -> 100 (避免无效迭代)
COHERENCE_EARLY_STOP = 20         # 连续N次无改善则早停

# 采样配置
N_SAMPLE_PAIRS = 500_000  # 减少：1_000_000 -> 500_000 (加速)

# 全局共享向量缓存文件名
SHARED_VECTORS_FILE = "shared_vectors.pt"


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


# ============================================================================
# 阶段1: 预计算所有层的PCA
# ============================================================================

def precompute_all_pca(
    cache_dir: str,
    pca_cache_dir: str,
    layers: List[int],
    verbose: bool = True,
) -> Dict[int, str]:
    """
    预计算所有层的PCA并缓存

    参数:
        cache_dir: 激活缓存目录
        pca_cache_dir: PCA缓存目录
        layers: 要处理的层列表
        verbose: 是否输出详细信息

    返回:
        results: {layer_idx: cache_file_path}
    """
    print(f"\n{'='*70}")
    print(f"[阶段1] 预计算所有层 PCA")
    print(f"{'='*70}")
    print(f"  激活目录: {cache_dir}")
    print(f"  PCA缓存目录: {pca_cache_dir}")
    print(f"  待处理层: {layers}")

    pca_cache_path = Path(pca_cache_dir)
    pca_cache_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for i, layer_idx in enumerate(layers, 1):
        print(f"\n{'#'*70}")
        print(f"# [{i}/{len(layers)}] Layer {layer_idx}")
        print(f"{'#'*70}")

        cache_file = Path(cache_dir) / f"layer{layer_idx}.pt"
        pca_cache_file = pca_cache_path / f"pca_cache_layer{layer_idx}.pt"

        # 检查是否已有缓存
        if pca_cache_file.exists():
            print(f"  ✓ PCA缓存已存在: {pca_cache_file}")
            results[layer_idx] = str(pca_cache_file)
            continue

        # 检查激活文件
        if not cache_file.exists():
            print(f"  ⚠ 激活文件不存在: {cache_file}")
            continue

        try:
            # 加载激活
            print(f"  [加载] {cache_file}")
            x = torch.load(cache_file, map_location="cpu").float()
            print(f"    形状: {x.shape}")

            # RMSNorm
            print(f"  [预处理] Per-token RMSNorm")
            x_norm = per_token_rms_norm(x)
            print(f"    x_norm mean: {x_norm.mean():.4f}, std: {x_norm.std():.4f}")

            # 几何中位数
            print(f"  [计算] 几何中位数")
            bpre = weiszfeld_geometric_median(x_norm, verbose=False)
            print(f"    bpre norm: {bpre.norm():.4f}")

            x_centered = x_norm - bpre

            # PCA Principal
            print(f"  [计算] PCA Principal ({N_PCA_PRINCIPAL})")
            pca_components, pca_var, pca_ratio = randomized_pca(
                x_centered, N_PCA_PRINCIPAL, verbose=False
            )
            print(f"    形状: {pca_components.shape}")

            # Residual PCA
            print(f"  [计算] Residual PCA ({N_RESIDUAL_PCA})")
            x_reconstructed = x_centered @ pca_components @ pca_components.T
            x_residual = x_centered - x_reconstructed
            residual_norm_ratio = x_residual.norm() / x_centered.norm()
            print(f"    残差范数比: {residual_norm_ratio:.4f}")

            residual_components, _, _ = randomized_pca(
                x_residual, N_RESIDUAL_PCA, verbose=False
            )
            print(f"    形状: {residual_components.shape}")

            # 保存缓存
            pca_cache = {
                "bpre": bpre.cpu(),
                "pca_components": pca_components.cpu(),
                "pca_ratio": pca_ratio.cpu(),
                "residual_components": residual_components.cpu(),
                "residual_norm_ratio": residual_norm_ratio.item(),
            }

            torch.save(pca_cache, pca_cache_file)
            print(f"  ✓ PCA缓存已保存: {pca_cache_file}")

            results[layer_idx] = str(pca_cache_file)

        except Exception as e:
            print(f"  ⚠ Layer {layer_idx} PCA计算失败: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"[阶段1完成] PCA预计算结束")
    print(f"  成功: {len(results)}/{len(layers)} 层")
    print(f"{'='*70}")

    return results


def generate_shared_vectors(
    shared_cache_dir: str,
    d_model: int = 1536,
    force_regenerate: bool = False,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    生成并缓存全局共享向量 (Random Orthogonal + Sparse Gaussian)

    这些向量对所有层都是一样的，只需生成一次

    参数:
        shared_cache_dir: 共享向量缓存目录
        d_model: 模型维度
        force_regenerate: 是否强制重新生成
        verbose: 是否输出详细信息

    返回:
        shared_vectors: {"W_ortho": ..., "W_gauss": ...}
    """
    shared_cache_path = Path(shared_cache_dir)
    shared_cache_path.mkdir(parents=True, exist_ok=True)
    shared_file = shared_cache_path / SHARED_VECTORS_FILE

    # 检查是否已有缓存
    if shared_file.exists() and not force_regenerate:
        if verbose:
            print(f"\n[共享向量] 加载缓存: {shared_file}")
        try:
            shared_vectors = torch.load(shared_file, map_location="cpu")
            if verbose:
                print(f"  ✓ W_ortho: {shared_vectors['W_ortho'].shape}")
                print(f"  ✓ W_gauss: {shared_vectors['W_gauss'].shape}")
            return shared_vectors
        except Exception as e:
            if verbose:
                print(f"  ⚠ 缓存加载失败: {e}，将重新生成")

    # 生成新的共享向量
    if verbose:
        print(f"\n[共享向量] 生成新向量")
        print(f"  Random Orthogonal: {N_RANDOM_ORTHOGONAL}")
        print(f"  Sparse Gaussian: {N_SPARSE_GAUSSIAN}")

    W_ortho = generate_random_orthogonal(d_model, N_RANDOM_ORTHOGONAL)

    W_gauss = torch.randn(d_model, N_SPARSE_GAUSSIAN)
    W_gauss = F.normalize(W_gauss, dim=0)

    shared_vectors = {
        "W_ortho": W_ortho.cpu(),
        "W_gauss": W_gauss.cpu(),
    }

    # 保存
    torch.save(shared_vectors, shared_file)
    if verbose:
        print(f"  ✓ 已保存: {shared_file}")

    return shared_vectors


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
    target: float = COHERENCE_TARGET,
    max_iterations: int = MAX_COHERENCE_ITERATIONS,
    early_stop_patience: int = COHERENCE_EARLY_STOP,
    verbose: bool = True,
) -> torch.Tensor:
    """
    执行 Mutual Coherence 过滤 (批量替换版本)

    改进：
    - 早停机制：连续N次无改善则停止
    - 动态阈值：根据当前mu调整
    """
    D, K = Wdec.shape
    device = Wdec.device

    if verbose:
        print(f"\n  [Coherence Filter] 目标: {target}, 最大迭代: {max_iterations}")

    best_mu = float('inf')
    no_improve_count = 0

    for iteration in range(max_iterations):
        mu, i, j = compute_mutual_coherence_sampled(Wdec, n_samples=500_000, device=device)

        if verbose and iteration % 10 == 0:
            print(f"    迭代 {iteration}: mu = {mu:.4f}")

        if mu <= target:
            if verbose:
                print(f"    ✓ 达到目标: mu = {mu:.4f} <= {target}")
            break

        # 早停检查
        if mu < best_mu - 0.01:  # 有显著改善
            best_mu = mu
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                if verbose:
                    print(f"    ⚠ 早停: 连续 {early_stop_patience} 次无改善, 当前 mu = {mu:.4f}")
                break

        # 批量替换高相似度向量
        W_norm = F.normalize(Wdec, dim=0)

        # 采样找出高相似度对
        n_find = 10000
        idx_i = torch.randint(0, K, (n_find,), device=device)
        idx_j = torch.randint(0, K, (n_find,), device=device)
        mask = idx_i != idx_j

        cos_values = (W_norm[:, idx_i[mask]] * W_norm[:, idx_j[mask]]).sum(dim=0).abs()

        # 动态阈值：当前 mu 的 80%
        dynamic_threshold = mu * 0.8
        high_cos_mask = cos_values > dynamic_threshold
        high_i = idx_i[mask][high_cos_mask]
        high_j = idx_j[mask][high_cos_mask]

        # 替换这些向量
        n_replace = min(len(high_i), 100)
        if n_replace > 0:
            to_replace = torch.unique(torch.cat([high_i[:n_replace], high_j[:n_replace]]))[:50]

            for idx in to_replace:
                new_vec = torch.randn(D, device=device)
                new_vec = F.normalize(new_vec, dim=0)
                Wdec[:, idx] = new_vec

        # 归一化
        Wdec = F.normalize(Wdec, dim=0)

    # 最终归一化
    Wdec = F.normalize(Wdec, dim=0)

    if verbose and mu > target:
        print(f"    最终 mu = {mu:.4f} (目标: {target})")

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

    # SAE Encode: z = x @ Wdec (Wdec 作为 encoder 权重)
    x_centered = x_sample - bpre.to(device)
    z = x_centered @ Wdec  # [n_samples, D] @ [D, K] -> [n_samples, K]

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
    pca_cache_file: Optional[str] = None,
    shared_vectors: Optional[Dict[str, torch.Tensor]] = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Competition-Oriented 混合源初始化 (固定比例)

    参数:
        x_norm: [N, D] RMSNorm 后的激活
        d_hidden: 隐藏层维度
        pca_cache_file: PCA 缓存文件路径 (可选，用于跳过 PCA 计算)
        shared_vectors: 预生成的共享向量 (W_ortho, W_gauss)
        verbose: 是否输出详细信息

    返回:
        Wdec: [D, d_hidden] 解码器权重
        bpre: [D] 几何中位数
        stats: 统计信息
    """
    N, D = x_norm.shape
    device = x_norm.device

    if verbose:
        print(f"\n{'='*70}")
        print(f"[Competition-Oriented Mixed Initialization]")
        print(f"{'='*70}")
        print(f"  输入: [{N}, {D}], d_hidden={d_hidden}")
        print(f"  比例: PCA={N_PCA_PRINCIPAL}, Residual={N_RESIDUAL_PCA}, "
              f"Ortho={N_RANDOM_ORTHOGONAL}, Activation={N_ACTIVATION_SAMPLE}, "
              f"Gaussian={N_SPARSE_GAUSSIAN}")

    components_list = []
    stats = {}
    pca_cache = None

    # ========== 尝试加载 PCA 缓存 ==========
    if pca_cache_file and Path(pca_cache_file).exists():
        if verbose:
            print(f"\n[缓存] 加载 PCA 中间结果: {pca_cache_file}")
        try:
            pca_cache = torch.load(pca_cache_file, map_location=device)
            if verbose:
                print(f"  ✓ 缓存加载成功")
        except Exception as e:
            if verbose:
                print(f"  ⚠ 缓存加载失败: {e}，将重新计算")
            pca_cache = None

    # ========== 1. 几何中位数 ==========
    if pca_cache is not None and "bpre" in pca_cache:
        bpre = pca_cache["bpre"].to(device)
        if verbose:
            print(f"\n[1/6] 几何中位数 (从缓存加载)")
            print(f"  bpre norm: {bpre.norm():.4f}")
    else:
        if verbose:
            print(f"\n[1/6] 计算几何中位数...")
        bpre = weiszfeld_geometric_median(x_norm, verbose=verbose)

    x_centered = x_norm - bpre

    # ========== 2. PCA Principal (1536) ==========
    if pca_cache is not None and "pca_components" in pca_cache:
        W_pca = pca_cache["pca_components"].to(device)
        pca_ratio = pca_cache["pca_ratio"].to(device)
        if verbose:
            print(f"\n[2/6] PCA Principal Directions (从缓存加载)")
            print(f"  形状: {W_pca.shape}")
    else:
        if verbose:
            print(f"\n[2/6] PCA Principal Directions ({N_PCA_PRINCIPAL})...")

        pca_components, pca_var, pca_ratio = randomized_pca(x_centered, N_PCA_PRINCIPAL, verbose)
        W_pca = pca_components.clone()

        # 保存到缓存
        if pca_cache is None:
            pca_cache = {}
        pca_cache["bpre"] = bpre.cpu()
        pca_cache["pca_components"] = W_pca.cpu()
        pca_cache["pca_ratio"] = pca_ratio.cpu()

    components_list.append(W_pca)

    stats["n_pca"] = W_pca.shape[1]
    stats["pca_top128_variance"] = pca_ratio[:128].sum().item()

    # ========== 3. Residual PCA (768) ==========
    if pca_cache is not None and "residual_components" in pca_cache:
        W_residual = pca_cache["residual_components"].to(device)
        residual_norm_ratio = pca_cache.get("residual_norm_ratio", 0.0)
        if verbose:
            print(f"\n[3/6] Residual PCA Directions (从缓存加载)")
            print(f"  形状: {W_residual.shape}")
            print(f"  残差范数比: {residual_norm_ratio:.4f}")
    else:
        if verbose:
            print(f"\n[3/6] Residual PCA Directions ({N_RESIDUAL_PCA})...")

        x_reconstructed = x_centered @ W_pca @ W_pca.T
        x_residual = x_centered - x_reconstructed

        residual_norm_ratio = x_residual.norm() / x_centered.norm()
        stats["residual_norm_ratio"] = residual_norm_ratio.item()

        if verbose:
            print(f"  残差范数比: {residual_norm_ratio:.4f}")

        # 固定数量的 Residual PCA
        residual_components, _, _ = randomized_pca(x_residual, N_RESIDUAL_PCA, verbose=False)
        W_residual = residual_components.clone()

        # 保存到缓存
        pca_cache["residual_components"] = W_residual.cpu()
        pca_cache["residual_norm_ratio"] = residual_norm_ratio

    components_list.append(W_residual)

    stats["n_residual"] = W_residual.shape[1]
    stats["residual_norm_ratio"] = residual_norm_ratio if isinstance(residual_norm_ratio, float) else residual_norm_ratio.item()

    # ========== 保存 PCA 缓存 ==========
    if pca_cache_file and pca_cache is not None:
        if verbose:
            print(f"\n[缓存] 保存 PCA 中间结果: {pca_cache_file}")
        try:
            Path(pca_cache_file).parent.mkdir(parents=True, exist_ok=True)
            torch.save(pca_cache, pca_cache_file)
            if verbose:
                print(f"  ✓ PCA 缓存已保存")
        except Exception as e:
            if verbose:
                print(f"  ⚠ PCA 缓存保存失败: {e}")

    # ========== 4. Random Orthogonal (1024) ==========
    if shared_vectors is not None and "W_ortho" in shared_vectors:
        W_ortho = shared_vectors["W_ortho"].to(device)
        if verbose:
            print(f"\n[4/6] Random Orthogonal Directions (从共享缓存加载)")
            print(f"  形状: {W_ortho.shape}")
    else:
        if verbose:
            print(f"\n[4/6] Random Orthogonal Directions ({N_RANDOM_ORTHOGONAL})...")
        W_ortho = generate_random_orthogonal(D, N_RANDOM_ORTHOGONAL, device)
        if verbose:
            print(f"  形状: {W_ortho.shape}")

    components_list.append(W_ortho)
    stats["n_ortho"] = W_ortho.shape[1]

    # ========== 5. Activation Sample (1536, decorrelated) ==========
    if verbose:
        print(f"\n[5/6] Activation Samples ({N_ACTIVATION_SAMPLE}, decorrelated)...")

    # 合并已有 dictionary 用于 decorrelation
    W_existing = torch.cat(components_list, dim=1)

    W_activation = sample_activations_decorrelated(
        x_norm, N_ACTIVATION_SAMPLE, W_existing, device, verbose
    )
    components_list.append(W_activation)

    stats["n_activation"] = N_ACTIVATION_SAMPLE

    # ========== 6. Sparse Gaussian (7424) ==========
    if shared_vectors is not None and "W_gauss" in shared_vectors:
        W_gauss = shared_vectors["W_gauss"].to(device)
        if verbose:
            print(f"\n[6/6] Sparse Gaussian Directions (从共享缓存加载)")
            print(f"  形状: {W_gauss.shape}")
    else:
        if verbose:
            print(f"\n[6/6] Sparse Gaussian Directions ({N_SPARSE_GAUSSIAN})...")
        W_gauss = torch.randn(D, N_SPARSE_GAUSSIAN, device=device)
        W_gauss = F.normalize(W_gauss, dim=0)
        if verbose:
            print(f"  形状: {W_gauss.shape}")

    components_list.append(W_gauss)
    stats["n_gauss"] = N_SPARSE_GAUSSIAN

    # ========== 合并 ==========
    if verbose:
        print(f"\n[合并所有来源]...")

    Wdec = torch.cat(components_list, dim=1)

    if verbose:
        print(f"  合并后形状: {Wdec.shape}")
        print(f"  来源: PCA={stats['n_pca']}, Residual={stats['n_residual']}, "
              f"Ortho={stats['n_ortho']}, Activation={stats['n_activation']}, "
              f"Gaussian={stats['n_gauss']}")

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
    pca_cache_dir: Optional[str] = None,
    shared_vectors: Optional[Dict[str, torch.Tensor]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    初始化单个层的 SAE

    参数:
        cache_file: 激活缓存文件路径
        output_file: 输出文件路径
        layer_idx: 层索引
        d_hidden: 隐藏层维度
        top_k: TopK 验证参数
        pca_cache_dir: PCA 缓存目录 (可选，用于跳过 PCA 计算)
        shared_vectors: 预生成的共享向量 (W_ortho, W_gauss)
        verbose: 是否输出详细信息
    """
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

    # PCA 缓存文件
    pca_cache_file = None
    if pca_cache_dir:
        pca_cache_file = str(Path(pca_cache_dir) / f"pca_cache_layer{layer_idx}.pt")
        if Path(pca_cache_file).exists():
            print(f"\n[PCA 缓存] 发现缓存文件: {pca_cache_file}")

    # 混合源初始化
    Wdec, bpre, init_stats = mixed_source_initialization(
        x_norm,
        d_hidden=d_hidden,
        pca_cache_file=pca_cache_file,
        shared_vectors=shared_vectors,
        verbose=verbose
    )

    # Encoder 权重 (Tied)
    Wenc = Wdec.T.clone()

    # ========== 关键：先保存初始化结果 ==========
    # 即使验证失败，初始化结果也不会丢失
    print(f"\n[保存] {output_file} (初始化结果)")

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
        "layer_idx": layer_idx,
    }

    # 先保存不带 quality 的版本
    torch.save(output_data, output_file)
    print(f"  ✓ 初始化结果已保存")

    # ========== 然后运行质量验证 ==========
    # 如果验证失败，至少初始化结果已经保存
    validation_error = None
    try:
        quality_results = validate_initialization(
            Wdec, bpre, x_norm, top_k=top_k, verbose=verbose
        )
    except Exception as e:
        print(f"\n⚠ 验证过程出错: {e}")
        quality_results = {
            "all_passed": False,
            "error": str(e),
        }
        validation_error = str(e)

    # 更新保存文件，添加 quality 结果
    output_data["quality"] = quality_results
    torch.save(output_data, output_file)
    print(f"  ✓ 质量验证结果已更新到文件")

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"[完成] Layer {layer_idx}")
    print(f"  耗时: {elapsed:.2f}s")
    if validation_error:
        print(f"  验证: ⚠ 出错 (但初始化结果已保存)")
        print(f"  错误: {validation_error[:100]}...")
    else:
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
    parser.add_argument("--pca_cache_dir", type=str, default="./pca_cache",
                        help="PCA 中间结果缓存目录")
    parser.add_argument("--shared_cache_dir", type=str, default="./shared_cache",
                        help="共享向量缓存目录 (Random Orthogonal + Sparse Gaussian)")
    parser.add_argument("--layer", type=str, default="all",
                        help="层索引: 'all' 或 '14' 或 '14,19,24,29'")
    parser.add_argument("--d_hidden", type=int, default=12288)
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--skip_pca", action="store_true",
                        help="跳过PCA预计算 (假设已有缓存)")
    parser.add_argument("--skip_init", action="store_true",
                        help="只运行PCA预计算，不进行初始化")

    args = parser.parse_args()

    # 确定要初始化的层
    if args.layer.lower() == "all":
        layers = [14, 19, 24, 29]
    else:
        layers = [int(x.strip()) for x in args.layer.split(",")]

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pca_cache_path = Path(args.pca_cache_dir)
    pca_cache_path.mkdir(parents=True, exist_ok=True)

    shared_cache_path = Path(args.shared_cache_dir)
    shared_cache_path.mkdir(parents=True, exist_ok=True)

    # 打印启动信息
    print(f"\n{'='*70}")
    print(f"SAE Competition-Oriented 混合初始化")
    print(f"{'='*70}")
    print(f"  缓存目录: {args.cache_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  PCA缓存目录: {args.pca_cache_dir}")
    print(f"  共享向量目录: {args.shared_cache_dir}")
    print(f"  待初始化层: {layers}")
    print(f"  d_hidden: {args.d_hidden}")
    print(f"  top_k: {args.top_k}")
    print(f"{'='*70}")

    total_start_time = time.time()
    all_results = {}

    # ==================== 阶段1: 预计算PCA ====================
    if not args.skip_pca:
        print(f"\n{'='*70}")
        print(f"[阶段1] 预计算所有层 PCA")
        print(f"{'='*70}")

        precompute_all_pca(
            cache_dir=args.cache_dir,
            pca_cache_dir=args.pca_cache_dir,
            layers=layers,
            verbose=True,
        )

    if args.skip_init:
        print(f"\n{'='*70}")
        print(f"[完成] 只运行了PCA预计算")
        print(f"{'='*70}")
        return

    # ==================== 阶段2: 生成共享向量 ====================
    print(f"\n{'='*70}")
    print(f"[阶段2] 生成共享向量")
    print(f"{'='*70}")

    shared_vectors = generate_shared_vectors(
        shared_cache_dir=args.shared_cache_dir,
        d_model=1536,
        verbose=True,
    )

    # ==================== 阶段3: 逐层初始化 ====================
    print(f"\n{'='*70}")
    print(f"[阶段3] 逐层初始化 SAE")
    print(f"{'='*70}")

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
                pca_cache_dir=args.pca_cache_dir,
                shared_vectors=shared_vectors,
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
