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
    print(f"  注意: ReLU会截断负值，初始MSE较高是正常的")

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

    # 计算 MSE 相对于方差的比率 (更合理的指标)
    variance = x_sample.var().item()
    mse_ratio = mse / variance
    results["mse_to_variance_ratio"] = mse_ratio

    print(f"  MSE: {mse:.4f}")
    print(f"  输入方差: {variance:.4f}")
    print(f"  MSE/方差比: {mse_ratio:.2%}")

    # 初始阶段合理阈值: MSE < 方差 * 20 (即重建误差不超过方差的20倍)
    if mse_ratio < 20:
        print(f"  ✓ 通过 (MSE/方差 < 20)")
    else:
        print(f"  ⚠ MSE过高 (MSE/方差 >= 20)")

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
    # 使用 MSE/方差比作为判断标准 (更合理)
    mse_ratio = results.get("mse_to_variance_ratio", mse)  # 如果没有方差信息，回退到 MSE
    if isinstance(mse_ratio, float) and mse_ratio >= 20:
        all_passed = False
        issues.append(f"Reconstruction MSE/方差比: {mse_ratio:.2f} (阈值: 20)")
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

    parser.add_argument("--init_dir", type=str, default="./sae_init",
                        help="初始化文件目录")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="激活缓存目录")
    parser.add_argument("--layer", type=str, default="all",
                        help="层索引，如 '14' 或 'all' 或 '14,19,24,29'")

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

        if isinstance(mse, float):
            mse = f"{mse:.4f}"
        if isinstance(dead, float):
            dead = f"{dead:.2%}"

        print(f"  {layer_key}: {status} | MSE={mse} | Dead={dead}")

    # 总结
    passed = sum(1 for r in all_results.values() if r.get("all_passed", False))
    total = len(all_results)

    print(f"\n  通过: {passed}/{total}")
    print("=" * 70)


if __name__ == "__main__":
    main()
