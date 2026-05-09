#!/usr/bin/env python3
"""
SAE 混合源初始化脚本

核心原则:
- 禁止 duplicated PCA expansion (v + ε)
- 采用 mixed-source initialization
- 支持 hidden_dim >= 12288 的超宽 SAE

初始化组成 (hidden_dim=12288):
1. PCA principal directions: 1536
2. Residual PCA directions: 3072
3. Random orthogonal directions: 4096
4. Sampled activation initialization: 3584

使用方法:
    python -m 初始化.sae_mixed_init --cache_dir ./cache --output_dir ./sae_init --layer 14
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

# 初始化组成 (hidden_dim=12288)
N_PCA_PRINCIPAL = 1536      # Top PCA components
N_RESIDUAL_PCA = 3072       # Residual PCA directions
N_RANDOM_ORTHOGONAL = 4096  # Random orthogonal directions
N_ACTIVATION_SAMPLE = 3584  # Sampled activation vectors

# 质量标准
MAX_MUTUAL_COHERENCE = 0.35
MAX_GINI = 0.7
MAX_ZERO_FIRE_RATIO = 0.5
MAX_COSINE = 0.85

# Mutual Coherence 过滤
COHERENCE_THRESHOLD = 0.85
COHERENCE_TARGET = 0.35
MAX_COHERENCE_ITERATIONS = 100


# ============================================================================
# 工具函数
# ============================================================================

def per_token_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-token RMSNorm

    参数:
        x: [N, D] 输入向量
        eps: 数值稳定性

    返回:
        x_norm: [N, D] 归一化后的向量
    """
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms


def weiszfeld_geometric_median(
    x: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-5,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Weiszfeld 算法计算几何中位数

    参数:
        x: [N, D] 输入数据
        max_iter: 最大迭代次数
        tol: 收敛阈值
        verbose: 是否输出信息

    返回:
        median: [D] 几何中位数
    """
    N, D = x.shape

    # 初始估计: 均值
    median = x.mean(dim=0)

    for i in range(max_iter):
        # 计算距离
        diff = x - median  # [N, D]
        dist = diff.norm(dim=1)  # [N]

        # 避免除零
        dist = dist.clamp(min=1e-8)

        # Weiszfeld 更新
        weights = 1.0 / dist
        new_median = (weights.unsqueeze(1) * x).sum(dim=0) / weights.sum()

        # 检查收敛
        if (new_median - median).norm() < tol:
            if verbose:
                print(f"  [Geometric Median] 收敛于第 {i+1} 次迭代")
            median = new_median
            break

        median = new_median

    return median


def randomized_pca(
    x: torch.Tensor,
    n_components: int,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized PCA

    参数:
        x: [N, D] 中心化后的数据
        n_components: 主成分数
        verbose: 是否输出信息

    返回:
        components: [D, n_components] 主方向
        explained_variance: [n_components] 解释方差
        explained_variance_ratio: [n_components] 解释方差比例
    """
    N, D = x.shape

    if verbose:
        print(f"  [PCA] 输入: [{N}, {D}], n_components={n_components}")

    # 使用 torch.pca_lowrank (Randomized PCA)
    U, S, V = torch.pca_lowrank(x.float(), q=min(n_components, min(N, D)), center=False)

    # V: [D, n_components] 主方向
    components = V

    # 解释方差
    explained_variance = (S[:n_components] ** 2) / (N - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    return components[:, :n_components], explained_variance, explained_variance_ratio


def generate_random_orthogonal(
    d_model: int,
    n_vectors: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    生成随机正交向量

    参数:
        d_model: 模型维度
        n_vectors: 向量数
        device: 设备

    返回:
        Q: [d_model, n_vectors] 正交向量 (列单位范数)
    """
    # 随机高斯矩阵
    A = torch.randn(d_model, n_vectors, device=device)

    # QR 分解
    Q, R = torch.linalg.qr(A)

    # 确保对角线为正 (数值稳定性)
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)

    return Q


def sample_activations(
    x_norm: torch.Tensor,
    n_samples: int,
    norm_stratified: bool = True,
    verbose: bool = True,
) -> torch.Tensor:
    """
    从真实激活中采样向量

    参数:
        x_norm: [N, D] RMSNorm 后的激活
        n_samples: 采样数量
        norm_stratified: 是否分层采样
        verbose: 是否输出信息

    返回:
        samples: [D, n_samples] 采样向量 (列单位范数)
    """
    N, D = x_norm.shape

    if n_samples > N:
        if verbose:
            print(f"  [Sample] 请求数量 {n_samples} > 总数 {N}，使用全部数据")
        n_samples = N

    if norm_stratified:
        # 计算 norm 分位数
        norms = x_norm.norm(dim=1)
        sorted_idx = norms.argsort()

        # 分层: 40% high-norm, 40% medium-norm, 20% uniform
        n_high = int(n_samples * 0.4)
        n_medium = int(n_samples * 0.4)
        n_uniform = n_samples - n_high - n_medium

        # High-norm: 最后 40% 区域
        high_region = sorted_idx[int(N * 0.6):]
        high_idx = high_region[torch.randperm(len(high_region))[:n_high]]

        # Medium-norm: 中间 40% 区域
        medium_region = sorted_idx[int(N * 0.3):int(N * 0.7)]
        medium_idx = medium_region[torch.randperm(len(medium_region))[:n_medium]]

        # Uniform: 全局随机
        uniform_idx = sorted_idx[torch.randperm(N)[:n_uniform]]

        # 合并
        selected_idx = torch.cat([high_idx, medium_idx, uniform_idx])

        if verbose:
            print(f"  [Sample] 分层采样: high={n_high}, medium={n_medium}, uniform={n_uniform}")
    else:
        # 简单随机采样
        selected_idx = torch.randperm(N)[:n_samples]

    # 提取并归一化
    samples = x_norm[selected_idx]  # [n_samples, D]
    samples = F.normalize(samples, dim=1)  # 行归一化

    if verbose:
        print(f"  [Sample] 采样数量: {samples.shape[0]}")

    return samples.T  # [D, n_samples]


# ============================================================================
# Mutual Coherence 过滤
# ============================================================================

def compute_mutual_coherence_sampled(
    W: torch.Tensor,
    n_samples: int = 100_000,
    device: str = "cpu",
) -> Tuple[float, int, int]:
    """
    采样计算 Mutual Coherence

    参数:
        W: [D, K] 权重矩阵
        n_samples: 采样数量
        device: 设备

    返回:
        mu: mutual coherence (max |cos|)
        max_i, max_j: 最大相似度对应的索引
    """
    D, K = W.shape
    W_norm = F.normalize(W, dim=0)

    max_abs_cos = 0.0
    max_i, max_j = 0, 0

    for _ in range(n_samples // 1000):
        idx_i = torch.randint(0, K, (1000,), device=device)
        idx_j = torch.randint(0, K, (1000,), device=device)
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
    """
    执行 Mutual Coherence 过滤

    参数:
        Wdec: [D, K] Decoder 权重
        threshold: 高相似度阈值
        target: 目标 mutual coherence
        max_iterations: 最大迭代次数
        verbose: 是否输出信息

    返回:
        Wdec: 过滤后的权重
    """
    D, K = Wdec.shape
    device = Wdec.device

    if verbose:
        print(f"\n  [Coherence Filter] 阈值: {threshold}, 目标: {target}")

    for iteration in range(max_iterations):
        mu, i, j = compute_mutual_coherence_sampled(Wdec, n_samples=100_000, device=device)

        if verbose and iteration % 10 == 0:
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

    return Wdec


# ============================================================================
# 混合源初始化
# ============================================================================

def mixed_source_initialization(
    x_norm: torch.Tensor,
    d_hidden: int = 12288,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    混合源初始化

    参数:
        x_norm: [N, D] RMSNorm 后的激活
        d_hidden: 隐藏层维度
        verbose: 是否输出信息

    返回:
        Wdec: [D, d_hidden] Decoder 权重
        bpre: [D] 几何中位数
        stats: 统计信息
    """
    N, D = x_norm.shape
    device = x_norm.device

    if verbose:
        print(f"\n{'='*70}")
        print(f"[Mixed Source Initialization]")
        print(f"{'='*70}")
        print(f"  输入: [{N}, {D}], d_hidden={d_hidden}")

    # ========== 1. 计算几何中位数 ==========
    if verbose:
        print(f"\n[1/6] 计算几何中位数...")

    bpre = weiszfeld_geometric_median(x_norm, verbose=verbose)

    # 中心化
    x_centered = x_norm - bpre

    # ========== 2. PCA Principal Directions (1536) ==========
    if verbose:
        print(f"\n[2/6] PCA Principal Directions ({N_PCA_PRINCIPAL})...")

    pca_components, pca_var, pca_ratio = randomized_pca(x_centered, N_PCA_PRINCIPAL, verbose)
    W_pca = pca_components.clone()  # [D, 1536]

    if verbose:
        print(f"  形状: {W_pca.shape}")
        print(f"  Top128 累计方差: {pca_ratio[:128].sum().item()*100:.2f}%")

    # ========== 3. Residual PCA (3072) ==========
    if verbose:
        print(f"\n[3/6] Residual PCA Directions ({N_RESIDUAL_PCA})...")

    # 计算残差: X - X @ PCA @ PCA.T
    x_reconstructed = x_centered @ W_pca @ W_pca.T
    x_residual = x_centered - x_reconstructed

    if verbose:
        residual_norm_ratio = x_residual.norm() / x_centered.norm()
        print(f"  残差范数比: {residual_norm_ratio:.4f}")

    # 对残差做 PCA
    residual_components, residual_var, residual_ratio = randomized_pca(
        x_residual, N_RESIDUAL_PCA, verbose
    )
    W_residual = residual_components.clone()  # [D, 3072]

    if verbose:
        print(f"  形状: {W_residual.shape}")

    # ========== 4. Random Orthogonal (4096) ==========
    if verbose:
        print(f"\n[4/6] Random Orthogonal Directions ({N_RANDOM_ORTHOGONAL})...")

    W_random = generate_random_orthogonal(D, N_RANDOM_ORTHOGONAL, device)  # [D, 4096]

    if verbose:
        print(f"  形状: {W_random.shape}")

    # ========== 5. Activation Sampling (3584) ==========
    if verbose:
        print(f"\n[5/6] Activation Sampling ({N_ACTIVATION_SAMPLE})...")

    W_activation = sample_activations(x_norm, N_ACTIVATION_SAMPLE, verbose=verbose)  # [D, 3584]

    # ========== 6. 合并与处理 ==========
    if verbose:
        print(f"\n[6/6] 合并与处理...")

    # 合并所有来源
    Wdec = torch.cat([W_pca, W_residual, W_random, W_activation], dim=1)  # [D, 12288]

    if verbose:
        print(f"  合并后形状: {Wdec.shape}")
        print(f"  来源: PCA={N_PCA_PRINCIPAL}, Residual={N_RESIDUAL_PCA}, "
              f"Random={N_RANDOM_ORTHOGONAL}, Activation={N_ACTIVATION_SAMPLE}")

    # 归一化
    Wdec = F.normalize(Wdec, dim=0)

    # Global shuffle
    perm = torch.randperm(d_hidden)
    Wdec = Wdec[:, perm]

    # Global jitter (打破数值对称性)
    Wdec = Wdec + 0.01 * torch.randn_like(Wdec)
    Wdec = F.normalize(Wdec, dim=0)

    if verbose:
        print(f"  已执行 global shuffle + jitter")

    # ========== 7. Mutual Coherence 过滤 ==========
    Wdec = enforce_mutual_coherence(Wdec, verbose=verbose)

    # ========== 8. 统计 ==========
    stats = {
        "n_pca": N_PCA_PRINCIPAL,
        "n_residual": N_RESIDUAL_PCA,
        "n_random": N_RANDOM_ORTHOGONAL,
        "n_activation": N_ACTIVATION_SAMPLE,
        "pca_top128_variance": pca_ratio[:128].sum().item(),
        "residual_norm_ratio": residual_norm_ratio.item() if isinstance(residual_norm_ratio, torch.Tensor) else residual_norm_ratio,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"[初始化完成]")
        print(f"{'='*70}")

    return Wdec, bpre, stats


# ============================================================================
# 质量验证
# ============================================================================

def validate_initialization(
    Wdec: torch.Tensor,
    bpre: torch.Tensor,
    x_norm: torch.Tensor,
    top_k: int = 128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    验证初始化质量

    硬指标:
    1. mutual coherence < 0.35
    2. Gini < 0.7
    3. zero-fire features < 50%
    4. max cosine < 0.85
    """
    D, K = Wdec.shape
    device = Wdec.device

    if verbose:
        print(f"\n{'='*70}")
        print(f"[质量验证]")
        print(f"{'='*70}")

    results = {}

    # ========== 1. Mutual Coherence ==========
    if verbose:
        print(f"\n[1/4] Mutual Coherence...")

    mu, _, _ = compute_mutual_coherence_sampled(Wdec, n_samples=500_000, device=device)
    results["mutual_coherence"] = mu

    if verbose:
        status = "✓ 通过" if mu < MAX_MUTUAL_COHERENCE else "⚠ 超标"
        print(f"  Mutual Coherence: {mu:.4f} (阈值: {MAX_MUTUAL_COHERENCE}) {status}")

    # ========== 2. Max Cosine ==========
    if verbose:
        print(f"\n[2/4] Max Cosine Similarity...")

    # 采样计算
    W_norm = F.normalize(Wdec, dim=0)
    sample_size = min(100_000, K * (K-1) // 2)
    idx_i = torch.randint(0, K, (sample_size,), device=device)
    idx_j = torch.randint(0, K, (sample_size,), device=device)
    mask = idx_i != idx_j

    cos_values = (W_norm[:, idx_i[mask]] * W_norm[:, idx_j[mask]]).sum(dim=0)
    max_cos = cos_values.max().item()
    results["max_cosine"] = max_cos

    if verbose:
        status = "✓ 通过" if max_cos < MAX_COSINE else "⚠ 超标"
        print(f"  Max Cosine: {max_cos:.4f} (阈值: {MAX_COSINE}) {status}")

    # ========== 3. TopK 模拟 (Gini) ==========
    if verbose:
        print(f"\n[3/4] TopK Dynamics (Gini)...")

    n_samples = min(2048, x_norm.shape[0])
    idx = torch.randperm(x_norm.shape[0], device=device)[:n_samples]
    x_sample = x_norm[idx]

    # SAE Encode
    Wenc = Wdec.T.clone()
    x_centered = x_sample - bpre
    z = x_centered @ Wenc.T

    # TopK
    top_k = min(top_k, K)
    _, topk_indices = torch.topk(z.abs(), top_k, dim=1)

    # Firing counts
    firing_counts = torch.zeros(K, device=device)
    firing_counts.scatter_add_(0, topk_indices.flatten(),
                               torch.ones(topk_indices.numel(), device=device))

    # Gini
    values = firing_counts.float().sort()[0]
    n = len(values)
    index = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    gini = (2 * (index * values).sum()) / (n * values.sum() + 1e-10) - (n + 1) / n
    gini = gini.item()

    # Zero firing ratio
    zero_count = (values == 0).sum().item()
    zero_ratio = zero_count / K

    results["gini"] = gini
    results["zero_fire_ratio"] = zero_ratio

    if verbose:
        gini_status = "✓ 通过" if gini < MAX_GINI else "⚠ 超标"
        zero_status = "✓ 通过" if zero_ratio < MAX_ZERO_FIRE_RATIO else "⚠ 超标"
        print(f"  Gini: {gini:.4f} (阈值: {MAX_GINI}) {gini_status}")
        print(f"  Zero Fire Ratio: {zero_ratio:.2%} (阈值: {MAX_ZERO_FIRE_RATIO:.0%}) {zero_status}")

    # ========== 4. Decoder Norm ==========
    if verbose:
        print(f"\n[4/4] Decoder Column Norm...")

    col_norms = Wdec.norm(dim=0)
    norm_mean = col_norms.mean().item()
    norm_std = col_norms.std().item()
    norm_max_dev = (col_norms - 1).abs().max().item()

    results["decoder_norm_mean"] = norm_mean
    results["decoder_norm_std"] = norm_std
    results["decoder_norm_max_dev"] = norm_max_dev

    if verbose:
        print(f"  Mean: {norm_mean:.6f}, Std: {norm_std:.6f}, Max Dev: {norm_max_dev:.2e}")

    # ========== 总结 ==========
    all_passed = (
        mu < MAX_MUTUAL_COHERENCE and
        max_cos < MAX_COSINE and
        gini < MAX_GINI and
        zero_ratio < MAX_ZERO_FIRE_RATIO
    )
    results["all_passed"] = all_passed

    if verbose:
        print(f"\n{'='*70}")
        if all_passed:
            print(f"  ✓ 所有质量指标通过")
        else:
            print(f"  ⚠ 存在超标指标")
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
    """
    初始化单个层的 SAE

    参数:
        cache_file: 激活缓存文件路径
        output_file: 输出文件路径
        layer_idx: 层索引
        d_hidden: 隐藏层维度
        top_k: TopK 稀疏度
        verbose: 是否输出信息

    返回:
        results: 初始化结果
    """
    print(f"\n{'='*70}")
    print(f"SAE 混合源初始化 - Layer {layer_idx}")
    print(f"{'='*70}")

    start_time = time.time()

    # ========== 加载激活 ==========
    print(f"\n[加载] {cache_file}")
    x = torch.load(cache_file, map_location="cpu").float()
    print(f"  形状: {x.shape}")

    # ========== Per-token RMSNorm ==========
    print(f"\n[预处理] Per-token RMSNorm")
    x_norm = per_token_rms_norm(x)
    print(f"  x_norm mean: {x_norm.mean():.4f}, std: {x_norm.std():.4f}")

    # ========== 混合源初始化 ==========
    Wdec, bpre, init_stats = mixed_source_initialization(
        x_norm, d_hidden=d_hidden, verbose=verbose
    )

    # ========== Encoder 权重 (Tied) ==========
    Wenc = Wdec.T.clone()

    # ========== 质量验证 ==========
    quality_results = validate_initialization(
        Wdec, bpre, x_norm, top_k=top_k, verbose=verbose
    )

    # ========== 保存 ==========
    print(f"\n[保存] {output_file}")

    output_data = {
        "Wdec": Wdec.cpu(),
        "Wenc": Wenc.cpu(),
        "bpre": bpre.cpu(),
        "config": {
            "d_model": Wdec.shape[0],
            "d_hidden": Wdec.shape[1],
            "init_method": "mixed_source",
            "n_pca": N_PCA_PRINCIPAL,
            "n_residual": N_RESIDUAL_PCA,
            "n_random": N_RANDOM_ORTHOGONAL,
            "n_activation": N_ACTIVATION_SAMPLE,
        },
        "init_stats": init_stats,
        "quality": quality_results,
        "layer_idx": layer_idx,
    }

    torch.save(output_data, output_file)

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"[完成] Layer {layer_idx}")
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  质量通过: {'✓' if quality_results['all_passed'] else '⚠'}")
    print(f"{'='*70}")

    return {
        "layer_idx": layer_idx,
        "elapsed": elapsed,
        "quality": quality_results,
        "all_passed": quality_results["all_passed"],
    }


def main():
    parser = argparse.ArgumentParser(description="SAE 混合源初始化")

    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="激活缓存目录")
    parser.add_argument("--output_dir", type=str, default="./sae_init",
                        help="输出目录")
    parser.add_argument("--layer", type=str, default="all",
                        help="层索引: 'all' 或 '14,19,24,29'")
    parser.add_argument("--d_hidden", type=int, default=12288,
                        help="隐藏层维度")
    parser.add_argument("--top_k", type=int, default=128,
                        help="TopK 稀疏度")

    args = parser.parse_args()

    # 确定要初始化的层
    if args.layer.lower() == "all":
        layers = [14, 19, 24, 29]
    else:
        layers = [int(x.strip()) for x in args.layer.split(",")]

    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 批量初始化
    all_results = {}

    for layer_idx in layers:
        cache_file = Path(args.cache_dir) / f"layer{layer_idx}.pt"
        output_file = output_path / f"sae_init_layer{layer_idx}.pt"

        if not cache_file.exists():
            print(f"\n⚠ 跳过 Layer {layer_idx}: 缓存文件不存在 {cache_file}")
            continue

        result = initialize_sae_layer(
            cache_file=str(cache_file),
            output_file=str(output_file),
            layer_idx=layer_idx,
            d_hidden=args.d_hidden,
            top_k=args.top_k,
        )

        all_results[f"layer{layer_idx}"] = result

    # 汇总
    print(f"\n{'='*70}")
    print("批量初始化汇总")
    print(f"{'='*70}")

    for layer_key, result in all_results.items():
        status = "✓" if result["all_passed"] else "⚠"
        mu = result["quality"]["mutual_coherence"]
        gini = result["quality"]["gini"]
        elapsed = result["elapsed"]

        print(f"  {layer_key}: {status} | mu={mu:.3f} | Gini={gini:.3f} | Time={elapsed:.1f}s")

    passed = sum(1 for r in all_results.values() if r["all_passed"])
    print(f"\n  通过: {passed}/{len(all_results)}")
    print(f"{'='*70}")

    # 保存汇总结果
    summary_file = output_path / "init_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  汇总保存至: {summary_file}")


if __name__ == "__main__":
    main()
