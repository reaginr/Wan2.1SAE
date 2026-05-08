#!/usr/bin/env python3
"""
SAE 初始化质量检查脚本 - 独立版本

用于检查已保存的初始化文件质量，无需重新运行初始化。

使用方法:
    python -m 初始化.sae_quality_check --init_file ./sae_init/sae_init_layer14.pt --cache_dir ./cache --layer 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_sae_initialization(
    init_file: str,
    cache_dir: str,
    layer_idx: int,
) -> dict:
    """
    检查 SAE 初始化质量

    检查项:
    1. Reconstruction MSE (正确的中心化流程)
    2. Dead neuron ratio
    3. Decoder column norm
    4. Tied 初始化验证 (Wenc == Wdec.T)
    """
    print("=" * 70)
    print(f"SAE 初始化质量检查 - Layer {layer_idx}")
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
        print(f"  ✓ 通过")
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

    # 采样计算
    n_sample = min(10000, x_norm.shape[0])
    x_sample = x_norm[:n_sample]

    # 正确的前向传播:
    # Step 1: 中心化
    x_centered = x_sample - bpre

    # Step 2: 编码
    z = F.relu(x_centered @ Wenc.T)

    # Step 3: 解码
    x_hat_centered = z @ Wdec.T

    # Step 4: 去中心化
    x_hat = x_hat_centered + bpre

    # 计算 MSE
    mse = F.mse_loss(x_hat, x_sample).item()
    results["reconstruction_mse"] = mse

    print(f"  MSE: {mse:.6f}")

    if mse <= 0.3:
        print(f"  ✓ 通过 (阈值: 0.3)")
    else:
        print(f"  ⚠ MSE 过高 (阈值: 0.3)")

    # ========== 检查 4: Dead neuron ratio ==========
    print(f"\n[检查 4] Dead neurons")
    z_active = (z > 0).any(dim=0)  # [d_hidden]
    dead_count = (~z_active).sum().item()
    d_hidden = Wdec.shape[1]
    dead_ratio = dead_count / d_hidden

    results["dead_neuron_count"] = dead_count
    results["dead_neuron_ratio"] = dead_ratio

    print(f"  count: {dead_count} / {d_hidden}")
    print(f"  ratio: {dead_ratio:.2%}")

    if dead_ratio <= 0.05:
        print(f"  ✓ 通过 (阈值: 5%)")
    else:
        print(f"  ⚠ 死神经元过多 (阈值: 5%)")

    # ========== 检查 5: PCA 方差覆盖 (如果有) ==========
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

    # ========== 总结 ==========
    print(f"\n{'='*70}")
    print(f"质量检查总结:")

    all_passed = True
    issues = []

    if norm_max_dev > 1e-3:
        all_passed = False
        issues.append(f"Decoder norm deviation: {norm_max_dev:.2e}")
    if not tied_match:
        all_passed = False
        issues.append("Tied initialization failed")
    if mse > 0.3:
        all_passed = False
        issues.append(f"Reconstruction MSE: {mse:.4f}")
    if dead_ratio > 0.05:
        all_passed = False
        issues.append(f"Dead neuron ratio: {dead_ratio:.2%}")

    results["all_passed"] = all_passed
    results["issues"] = issues

    if all_passed:
        print(f"  ✓ 所有检查通过")
    else:
        print(f"  ⚠ 存在问题:")
        for issue in issues:
            print(f"      - {issue}")

    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SAE 初始化质量检查")

    parser.add_argument("--init_file", type=str, required=True,
                        help="初始化文件路径，如 ./sae_init/sae_init_layer14.pt")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="激活缓存目录")
    parser.add_argument("--layer", type=int, default=14,
                        help="层索引")

    args = parser.parse_args()

    check_sae_initialization(
        init_file=args.init_file,
        cache_dir=args.cache_dir,
        layer_idx=args.layer,
    )


if __name__ == "__main__":
    main()
