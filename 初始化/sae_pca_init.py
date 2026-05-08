#!/usr/bin/env python3
"""
SAE PCA 初始化脚本 - 严格遵循 TODO list_v2 2.3 规范

核心流程:
1. 加载 cache/layerXX.pt [256000, 1536]
2. Per-token RMSNorm (禁止dataset normalization/whitening)
3. 几何中位数 bpre (Weiszfeld算法, FP32, tol=1e-5)
4. 中心化: X_centered = X_norm - bpre
5. Randomized PCA, n_components=1536
6. Overcomplete expansion: 1536 × 8 = 12288
7. Wdec [1536, 12288], Wenc = Wdec.T
8. 保存 sae_init_layerXX.pt
9. 输出初始化质量检查

使用方法:
    python -m 初始化.sae_pca_init --cache_dir ./cache --layer 14 --output_dir ./sae_init

作者: Claude
日期: 2026-05-08
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# 配置
# ============================================================================

@dataclass
class PCAInitConfig:
    """PCA 初始化配置"""
    d_model: int = 1536
    d_hidden: int = 12288          # 8倍扩展
    n_components: int = 1536       # PCA主成分数
    expansion_factor: int = 8      # 每个PCA方向复制8份

    # RMSNorm
    eps: float = 1e-6

    # Weiszfeld算法
    geometric_median_tol: float = 1e-5
    geometric_median_max_iter: int = 100

    # 扰动
    perturbation_ratio: float = 0.01  # 扰动 = v.std() * 0.01


# ============================================================================
# 1. Per-Token RMSNorm
# ============================================================================

def per_token_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token RMSNorm (禁止dataset normalization, 禁止whitening)

    对每个token独立归一化，不是全局归一化

    输入:
        x: [N, D] 或 [B, L, D]

    输出:
        x_norm: 归一化后的张量
        rms: 用于反归一化的rms值

    公式:
        rms = sqrt(mean(x^2, dim=-1) + eps)
        x_norm = x / rms
    """
    if x.dim() == 3:
        B, L, D = x.shape
        x = x.view(-1, D)  # [B*L, D]

    # 计算 rms: [N, 1]
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    # 归一化
    x_norm = x / rms

    return x_norm, rms


# ============================================================================
# 2. 几何中位数 (Weiszfeld算法)
# ============================================================================

def weiszfeld_geometric_median(
    x: torch.Tensor,
    tol: float = 1e-5,
    max_iter: int = 100,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Weiszfeld算法计算几何中位数

    参数:
        x: [N, D] 数据点 (FP32)
        tol: 收敛阈值
        max_iter: 最大迭代次数
        verbose: 是否输出迭代信息

    返回:
        median: [D] 几何中位数

    算法:
        y_{n+1} = sum_i(x_i / ||x_i - y_n||) / sum_i(1 / ||x_i - y_n||)
    """
    # FP32计算
    x = x.float()

    # 初始点: 均值
    y = x.mean(dim=0)

    for iteration in range(max_iter):
        # 计算距离
        diff = x - y.unsqueeze(0)  # [N, D]
        dist = diff.norm(dim=-1)   # [N]

        # 避免除零
        dist = dist.clamp(min=1e-8)

        # 计算权重
        weights = 1.0 / dist  # [N]

        # 加权平均
        y_new = (weights.unsqueeze(1) * x).sum(dim=0) / weights.sum()

        # 检查收敛
        change = (y_new - y).norm().item()

        if verbose and (iteration + 1) % 20 == 0:
            print(f"    Weiszfeld iter {iteration+1}: change={change:.2e}")

        if change < tol:
            if verbose:
                print(f"    Weiszfeld 收敛于 iter {iteration+1}, change={change:.2e}")
            break

        y = y_new

    return y


# ============================================================================
# 3. Randomized PCA
# ============================================================================

def randomized_pca(
    x: torch.Tensor,
    n_components: int = 1536,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized PCA

    参数:
        x: [N, D] 中心化后的数据
        n_components: 主成分数
        verbose: 是否输出信息

    返回:
        components: [D, n_components] 主方向 (单位范数)
        explained_variance: [n_components] 解释方差
        explained_variance_ratio: [n_components] 解释方差比例
    """
    N, D = x.shape

    if verbose:
        print(f"  [PCA] 输入: [{N}, {D}], n_components={n_components}")

    # 使用 torch.pca_lowrank (Randomized PCA)
    U, S, V = torch.pca_lowrank(x.float(), q=n_components, center=False)

    # V: [D, n_components] 主方向 (已单位范数)
    components = V

    # 解释方差
    explained_variance = (S ** 2) / (N - 1)

    # 解释方差比例
    total_variance = x.var(dim=0).sum()
    explained_variance_ratio = explained_variance / total_variance

    # 累计解释方差
    cumulative_ratio = explained_variance_ratio.cumsum(0)

    # 必须打印的累计比例
    if verbose:
        print(f"\n  [PCA] 累计解释方差比例:")
        for k in [64, 128, 256, 512, 1024, 1536]:
            if k <= n_components:
                ratio = cumulative_ratio[k-1].item() * 100
                print(f"    top{k:4d}: {ratio:.2f}%")

    return components, explained_variance, explained_variance_ratio


# ============================================================================
# 4. Overcomplete Expansion
# ============================================================================

def overcomplete_expansion(
    pca_components: torch.Tensor,
    expansion_factor: int = 8,
    perturbation_ratio: float = 0.01,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Overcomplete Expansion

    参数:
        pca_components: [D, n_pca] PCA主方向, D=1536, n_pca=1536
        expansion_factor: 扩展倍数, 8
        perturbation_ratio: 扰动比例, 0.01

    返回:
        Wdec: [D, d_hidden] 解码器权重, D=1536, d_hidden=12288

    算法:
        1536 PCA directions × 8 = 12288 (完全匹配，无随机填充)

        每个PCA方向:
            for j in range(8):
                v_expand = v + N(0, (v.std() * 0.01)^2)
                v_expand = normalize(v_expand)

        最后 global shuffle
    """
    D, n_pca = pca_components.shape
    d_hidden = n_pca * expansion_factor

    if verbose:
        print(f"\n  [Overcomplete Expansion]")
        print(f"    PCA directions: {n_pca}")
        print(f"    Expansion factor: {expansion_factor}")
        print(f"    Target d_hidden: {d_hidden}")
        print(f"    计算: {n_pca} × {expansion_factor} = {d_hidden}")

    # 验证: 1536 × 8 = 12288
    assert d_hidden == 12288, f"d_hidden 应为 12288, 当前 {d_hidden}"
    assert n_pca == 1536, f"n_pca 应为 1536, 当前 {n_pca}"

    Wdec_list = []

    for i in range(n_pca):
        v = pca_components[:, i]  # [D]

        # 每个PCA方向复制 expansion_factor 份
        for j in range(expansion_factor):
            # 扰动: 使用 v.std() * perturbation_ratio
            # 正确做法: torch.randn_like(v) * v.std() * 0.01
            # 错误做法: torch.randn_like(v) * 0.01 (固定值)
            perturbation = torch.randn_like(v) * v.std() * perturbation_ratio
            v_expand = v + perturbation

            # 单位范数归一化
            v_expand = F.normalize(v_expand, dim=0)

            Wdec_list.append(v_expand)

    # 验证维度
    assert len(Wdec_list) == d_hidden, \
        f"扩展后维度错误: {len(Wdec_list)} != {d_hidden}"

    if verbose:
        print(f"    扩展后向量数: {len(Wdec_list)} (无随机填充)")

    # Global shuffle: 打乱所有 12288 列
    perm = torch.randperm(d_hidden)
    Wdec = torch.stack([Wdec_list[i] for i in perm], dim=1)  # [D, d_hidden]

    if verbose:
        print(f"    Wdec shape: {Wdec.shape}")
        print(f"    已执行 global shuffle")

    return Wdec


# ============================================================================
# 5. 初始化质量检查
# ============================================================================

def validate_initialization(
    Wdec: torch.Tensor,
    bpre: torch.Tensor,
    x_norm: torch.Tensor,
    explained_variance_ratio: torch.Tensor,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    初始化质量检查

    检查项:
    1. reconstruction MSE
    2. dead neuron ratio
    3. decoder column norm
    4. PCA variance coverage
    """
    results = {}

    D, d_hidden = Wdec.shape

    # 1. Decoder column norm (应该都是1)
    col_norms = Wdec.norm(dim=0)  # [d_hidden]
    norm_mean = col_norms.mean().item()
    norm_std = col_norms.std().item()
    norm_max_dev = (col_norms - 1).abs().max().item()

    results["decoder_norm_mean"] = norm_mean
    results["decoder_norm_std"] = norm_std
    results["decoder_norm_max_deviation"] = norm_max_dev

    if verbose:
        print(f"\n  [质量检查] Decoder column norm:")
        print(f"    mean: {norm_mean:.6f}")
        print(f"    std: {norm_std:.6f}")
        print(f"    max deviation from 1: {norm_max_dev:.2e}")

    # 2. Reconstruction MSE
    # SAE 正确前向传播:
    # x_norm -> 中心化(x_norm - bpre) -> 编码(ReLU) -> 解码 -> 去中心化(+bpre) -> x_hat
    Wenc = Wdec.T  # [d_hidden, D]

    # 采样一小批计算 MSE
    n_sample = min(10000, x_norm.shape[0])
    x_sample = x_norm[:n_sample]  # [n_sample, D]

    # Step 1: 中心化 (关键步骤!)
    x_centered = x_sample - bpre  # [n_sample, D]

    # Step 2: 编码
    z = F.relu(x_centered @ Wenc.T)  # [n_sample, d_hidden]

    # Step 3: 解码
    x_hat_centered = z @ Wdec.T  # [n_sample, D]

    # Step 4: 去中心化
    x_hat = x_hat_centered + bpre  # [n_sample, D]

    # MSE: 比较 x_hat 和原始 x_norm
    mse = F.mse_loss(x_hat, x_sample).item()
    results["reconstruction_mse"] = mse

    if verbose:
        print(f"\n  [质量检查] Reconstruction MSE: {mse:.6f}")

    # 3. Dead neuron ratio
    # 检查哪些神经元在样本中从未激活
    z_active = (z > 0).any(dim=0)  # [d_hidden]
    dead_count = (~z_active).sum().item()
    dead_ratio = dead_count / d_hidden

    results["dead_neuron_count"] = dead_count
    results["dead_neuron_ratio"] = dead_ratio

    if verbose:
        print(f"\n  [质量检查] Dead neurons:")
        print(f"    count: {dead_count} / {d_hidden}")
        print(f"    ratio: {dead_ratio:.2%}")

    # 4. PCA variance coverage
    # 前 k 个主成分解释的方差比例
    cum_var = explained_variance_ratio.cumsum(0)

    results["pca_variance_top64"] = cum_var[63].item() if len(cum_var) >= 64 else None
    results["pca_variance_top128"] = cum_var[127].item() if len(cum_var) >= 128 else None
    results["pca_variance_top256"] = cum_var[255].item() if len(cum_var) >= 256 else None
    results["pca_variance_top512"] = cum_var[511].item() if len(cum_var) >= 512 else None
    results["pca_variance_top1024"] = cum_var[1023].item() if len(cum_var) >= 1024 else None
    results["pca_variance_top1536"] = cum_var[1535].item() if len(cum_var) >= 1536 else None

    if verbose:
        print(f"\n  [质量检查] PCA variance coverage:")
        print(f"    top64:   {results['pca_variance_top64']*100:.2f}%")
        print(f"    top128:  {results['pca_variance_top128']*100:.2f}%")
        print(f"    top256:  {results['pca_variance_top256']*100:.2f}%")
        print(f"    top512:  {results['pca_variance_top512']*100:.2f}%")
        print(f"    top1024: {results['pca_variance_top1024']*100:.2f}%")
        print(f"    top1536: {results['pca_variance_top1536']*100:.2f}%")

    # 总体评估
    all_passed = True
    issues = []

    if norm_max_dev > 1e-3:
        issues.append(f"Decoder norm deviation too large: {norm_max_dev:.2e}")
        all_passed = False

    if mse > 0.3:
        issues.append(f"Reconstruction MSE too high: {mse:.4f}")
        all_passed = False

    if dead_ratio > 0.05:
        issues.append(f"Dead neuron ratio too high: {dead_ratio:.2%}")
        all_passed = False

    results["all_passed"] = all_passed
    results["issues"] = issues

    if verbose:
        if all_passed:
            print(f"\n  [质量检查] ✓ 所有检查通过")
        else:
            print(f"\n  [质量检查] ⚠ 存在问题:")
            for issue in issues:
                print(f"      - {issue}")

    return results


# ============================================================================
# 6. 主初始化函数
# ============================================================================

def initialize_sae_from_cache(
    cache_dir: str,
    layer_idx: int,
    output_dir: str,
    config: Optional[PCAInitConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    从缓存初始化 SAE

    完整流程:
    1. 加载 cache/layerXX.pt
    2. Per-token RMSNorm
    3. 几何中位数 bpre
    4. 中心化
    5. Randomized PCA
    6. Overcomplete expansion
    7. 质量检查
    8. 保存

    返回:
        stats: 初始化统计信息
    """
    if config is None:
        config = PCAInitConfig()

    start_time = time.time()

    print("=" * 70)
    print(f"SAE PCA 初始化 - Layer {layer_idx}")
    print("=" * 70)

    # ========== Step 1: 加载缓存 ==========
    cache_path = Path(cache_dir)
    layer_file = cache_path / f"layer{layer_idx}.pt"

    if not layer_file.exists():
        raise FileNotFoundError(f"缓存文件不存在: {layer_file}")

    print(f"\n[Step 1] 加载缓存: {layer_file}")
    x = torch.load(layer_file, map_location="cpu")
    print(f"  shape: {x.shape}")
    print(f"  dtype: {x.dtype}")

    # 转换为 float32 用于计算
    x = x.float()

    # ========== Step 2: Per-token RMSNorm ==========
    print(f"\n[Step 2] Per-token RMSNorm")
    print(f"  禁止 dataset normalization")
    print(f"  禁止 whitening")

    x_norm, rms = per_token_rms_norm(x, eps=config.eps)

    print(f"  输入 mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"  输出 mean: {x_norm.mean():.4f}, std: {x_norm.std():.4f}")

    # ========== Step 3: 几何中位数 bpre ==========
    print(f"\n[Step 3] 几何中位数 (Weiszfeld算法)")
    print(f"  计算精度: FP32")
    print(f"  收敛阈值: {config.geometric_median_tol}")

    bpre = weiszfeld_geometric_median(
        x_norm,
        tol=config.geometric_median_tol,
        max_iter=config.geometric_median_max_iter,
        verbose=verbose,
    )

    print(f"  bpre shape: {bpre.shape}")
    print(f"  bpre norm: {bpre.norm():.4f}")

    # ========== Step 4: 中心化 ==========
    print(f"\n[Step 4] 中心化")
    x_centered = x_norm - bpre

    print(f"  x_centered mean: {x_centered.mean():.6f}")

    # ========== Step 5: Randomized PCA ==========
    print(f"\n[Step 5] Randomized PCA")

    components, explained_variance, explained_variance_ratio = randomized_pca(
        x_centered,
        n_components=config.n_components,
        verbose=verbose,
    )

    print(f"  components shape: {components.shape}")

    # ========== Step 6: Overcomplete Expansion ==========
    Wdec = overcomplete_expansion(
        components,
        expansion_factor=config.expansion_factor,
        perturbation_ratio=config.perturbation_ratio,
        verbose=verbose,
    )

    # ========== Step 7: Wenc = Wdec.T ==========
    print(f"\n[Step 7] Wenc = Wdec.T")
    Wenc = Wdec.T.clone()

    print(f"  Wdec shape: {Wdec.shape}")
    print(f"  Wenc shape: {Wenc.shape}")

    # 验证 Tied 初始化
    assert torch.allclose(Wenc, Wdec.T, atol=1e-6), "Wenc != Wdec.T"
    print(f"  ✓ Tied 初始化验证通过")

    # ========== Step 8: 质量检查 ==========
    quality_results = validate_initialization(
        Wdec, bpre, x_norm, explained_variance_ratio,
        verbose=verbose,
    )

    # ========== Step 9: 保存 ==========
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_file = output_path / f"sae_init_layer{layer_idx}.pt"

    print(f"\n[Step 9] 保存到: {save_file}")

    save_dict = {
        "Wdec": Wdec.to(torch.bfloat16),      # [1536, 12288]
        "Wenc": Wenc.to(torch.bfloat16),      # [12288, 1536]
        "bpre": bpre.to(torch.float32),       # [1536]
        "pca_stats": {
            "components": components.to(torch.bfloat16),
            "explained_variance": explained_variance,
            "explained_variance_ratio": explained_variance_ratio,
        },
        "config": {
            "d_model": config.d_model,
            "d_hidden": config.d_hidden,
            "n_components": config.n_components,
            "expansion_factor": config.expansion_factor,
            "eps": config.eps,
        },
        "quality": quality_results,
    }

    torch.save(save_dict, save_file)

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"初始化完成!")
    print(f"  耗时: {elapsed:.1f}秒")
    print(f"  保存至: {save_file}")
    print(f"{'='*70}")

    return {
        "layer": layer_idx,
        "elapsed_time": elapsed,
        "save_path": str(save_file),
        "quality": quality_results,
    }


# ============================================================================
# 7. 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAE PCA 初始化")

    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="激活缓存目录")
    parser.add_argument("--layer", type=str, default="14",
                        help="层索引，如 '14' 或 'all'")
    parser.add_argument("--output_dir", type=str, default="./sae_init",
                        help="输出目录")

    args = parser.parse_args()

    # 确定层
    if args.layer.lower() == "all":
        layers = [14, 19, 24, 29]
    else:
        layers = [int(x.strip()) for x in args.layer.split(",")]

    # 执行初始化
    all_results = {}

    for layer_idx in layers:
        print(f"\n{'#'*70}")
        print(f"# Layer {layer_idx}")
        print(f"{'#'*70}")

        result = initialize_sae_from_cache(
            cache_dir=args.cache_dir,
            layer_idx=layer_idx,
            output_dir=args.output_dir,
        )

        all_results[f"layer{layer_idx}"] = result

    # 保存总结
    summary_file = Path(args.output_dir) / "initialization_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n总结保存至: {summary_file}")


if __name__ == "__main__":
    main()
