#!/usr/bin/env python3
"""
SAE 初始化与训练阶段 Activation 提取示例

演示如何使用 TokenSamplingManager 进行：
1. 初始化阶段的多 timestep 激活提取
2. 训练阶段的单 batch 激活提取
3. 统计分析

使用方法:
    python -m 初始化.activation_analysis_example --mode init --output_dir ./analysis

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from 初始化.token_sampling_manager import (
    TokenSamplingManager,
    TokenSamplingConfig,
    SamplingMode,
    ActivationStatisticsAnalyzer,
    create_init_sampler,
    create_train_sampler,
    TemporalSamplingConfig,
    SpatialSamplingConfig,
    NormStratifiedConfig,
    DecorrelationConfig,
)
from 初始化.dit_activation_extractor import (
    DiTActivationExtractor,
    DiTActivationConfig,
    compute_grid_sizes,
    analyze_activation_statistics,
)


def example_init_mode():
    """初始化模式示例"""
    print("\n" + "=" * 70)
    print("初始化模式示例")
    print("=" * 70)

    # 1. 创建配置
    config = TokenSamplingConfig(
        mode=SamplingMode.INIT,
        target_tokens=256000,
        temporal=TemporalSamplingConfig(
            init_timesteps=[0, 100, 200, 400, 600, 800, 1000],
            init_samples_per_timestep=40000,
        ),
        spatial=SpatialSamplingConfig(
            init_spatial_stride=2,
            height_tokens=30,
            width_tokens=52,
        ),
        norm_stratified=NormStratifiedConfig(
            enabled=True,
            num_buckets=5,
            init_bucket_weights=[0.15, 0.20, 0.25, 0.25, 0.15],
            apply_rms_norm_before_sampling=True,
        ),
        decorrelation=DecorrelationConfig(
            enabled=True,
            init_method="pca_residual",
            init_target_redundancy=0.3,
            oversample_ratio=3.0,
        ),
        seed=42,
    )

    # 2. 创建采样器
    sampler = TokenSamplingManager(config)

    print(f"采样配置:")
    print(f"  模式: {config.mode.value}")
    print(f"  目标 token 数: {config.target_tokens}")
    print(f"  时间步: {config.temporal.init_timesteps}")
    print(f"  空间 stride: {config.spatial.init_spatial_stride}")
    print(f"  Norm 分层: {config.norm_stratified.num_buckets} buckets")
    print(f"  去相关阈值: {config.decorrelation.init_target_redundancy}")

    # 3. 模拟数据 (实际使用时从 DiT 提取)
    print("\n模拟激活数据...")
    d_model = 1536
    n_tokens_per_timestep = 17160  # 11 * 30 * 52
    timesteps = config.temporal.init_timesteps

    activations_by_timestep = {}
    for t in timesteps:
        # 模拟不同 timestep 的特征分布
        noise_level = t / 1000.0
        base = torch.randn(1, n_tokens_per_timestep, d_model)
        # 高噪声 timestep 特征更随机
        activations_by_timestep[t] = base * (1 + noise_level)

    print(f"  生成 {len(timesteps)} 个 timestep 的激活")
    print(f"  每个 timestep: {n_tokens_per_timestep} tokens")

    # 4. 采样
    print("\n执行采样...")
    sampled, metadata = sampler.sample_for_initialization(activations_by_timestep)

    print(f"  采样结果: {sampled.shape}")
    print(f"  元数据:")
    for key, value in metadata.items():
        if key != "timestep_stats":
            print(f"    {key}: {value}")

    return sampler, sampled, metadata


def example_train_mode():
    """训练模式示例"""
    print("\n" + "=" * 70)
    print("训练模式示例")
    print("=" * 70)

    # 1. 创建配置 (更温和的采样)
    config = TokenSamplingConfig(
        mode=SamplingMode.TRAIN,
        target_tokens=4096,  # 每个 batch 较少
        spatial=SpatialSamplingConfig(
            train_spatial_stride=1,  # 保持原始分辨率
        ),
        norm_stratified=NormStratifiedConfig(
            enabled=True,
            train_soft_bias=True,
            train_bias_strength=0.3,
        ),
        decorrelation=DecorrelationConfig(
            enabled=False,  # 训练时不使用去相关
        ),
        seed=42,
    )

    # 2. 创建采样器
    sampler = TokenSamplingManager(config)

    print(f"采样配置:")
    print(f"  模式: {config.mode.value}")
    print(f"  目标 token 数: {config.target_tokens}")
    print(f"  空间 stride: {config.spatial.train_spatial_stride}")
    print(f"  Soft bias: {config.norm_stratified.train_soft_bias}")
    print(f"  去相关: {config.decorrelation.enabled}")

    # 3. 模拟训练 batch
    print("\n模拟训练 batch...")
    batch_size = 4
    n_tokens = 17160
    d_model = 1536

    activations = torch.randn(batch_size, n_tokens, d_model)
    timestep = 500

    print(f"  Batch size: {batch_size}")
    print(f"  Tokens per sample: {n_tokens}")
    print(f"  Timestep: {timestep}")

    # 4. 采样
    print("\n执行采样...")
    sampled, metadata = sampler.sample_for_training(
        activations=activations,
        timestep=timestep,
        batch_idx=0,
    )

    print(f"  采样结果: {sampled.shape}")
    print(f"  元数据: {metadata}")

    return sampler, sampled, metadata


def example_statistics_analysis():
    """统计分析示例"""
    print("\n" + "=" * 70)
    print("统计分析示例")
    print("=" * 70)

    # 创建分析器
    analyzer = ActivationStatisticsAnalyzer(device="cpu")

    # 模拟多层数据
    layers = [14, 19, 24, 29]
    n_tokens = 50000
    d_model = 1536

    print("\n生成模拟数据...")
    activations_by_layer = {}
    for layer in layers:
        # 不同层有不同的特征分布
        base = torch.randn(n_tokens, d_model)
        scale = 1 + (layer / 30) * 0.5  # 深层 norm 更大
        activations_by_layer[f"layer{layer}"] = base * scale

    # 模拟多 timestep 数据
    timesteps = [0, 200, 400, 600, 800, 1000]
    activations_by_timestep = {}
    for t in timesteps:
        activations_by_timestep[t] = {
            "layer14": torch.randn(n_tokens // 6, d_model) * (t / 500)
        }

    # 1. Norm 分布分析
    print("\n[1] Token Norm 分布分析")
    for layer, act in activations_by_layer.items():
        result = analyzer.analyze_token_norm_distribution(act, name=layer)
        print(f"  {layer}: mean={result['mean']:.4f}, std={result['std']:.4f}")

    # 2. 有效秩分析
    print("\n[2] 有效秩分析")
    for layer, act in activations_by_layer.items():
        result = analyzer.analyze_effective_rank(act)
        print(f"  {layer}: Rank@90%={result['effective_rank_90']}, Rank@99%={result['effective_rank_99']}")

    # 3. PCA 频谱分析
    print("\n[3] PCA 频谱分析")
    result = analyzer.analyze_pca_spectrum(activations_by_layer["layer14"], n_components=50)
    print(f"  Top 10 variance: {result['top_10_variance']:.4f}")
    print(f"  Top 50 variance: {result['top_50_variance']:.4f}")

    # 4. 空间局部性分析
    print("\n[4] 空间局部性分析")
    grid_size = (11, 30, 52)  # F, H, W
    result = analyzer.analyze_spatial_locality(
        activations_by_layer["layer14"],
        grid_size=grid_size,
        sample_size=1000,
    )
    print(f"  相邻相似度: {result['adjacent_similarity_mean']:.4f}")
    print(f"  远距离相似度: {result['far_similarity_mean']:.4f}")
    print(f"  局部性指数: {result['locality_index']:.4f}")

    # 5. 打印完整摘要
    print("\n" + "=" * 70)
    analyzer.print_summary()

    return analyzer


def example_integrated_workflow():
    """集成工作流示例"""
    print("\n" + "=" * 70)
    print("集成工作流示例")
    print("=" * 70)

    print("""
完整初始化流程:

1. 准备数据
   ├── 加载激活缓存 (cache/layer{XX}.pt)
   └── 或从 DiT 实时提取

2. 创建采样器
   ├── 配置 TemporalSamplingConfig
   ├── 配置 SpatialSamplingConfig
   ├── 配置 NormStratifiedConfig
   └── 配置 DecorrelationConfig

3. 多 timestep 提取
   ├── 遍历 timesteps [0, 100, ..., 1000]
   ├── 每个 timestep 提取激活
   └── 应用 RMSNorm

4. Token 采样
   ├── Norm 分层采样
   ├── 空间 stride 采样
   ├── 去相关筛选
   └── 合并所有 timestep

5. 统计分析
   ├── Norm 分布
   ├── 有效秩
   ├── PCA 频谱
   └── 空间局部性

6. PCA 初始化
   ├── 计算几何中位数
   ├── Randomized PCA
   ├── Overcomplete Expansion
   └── 质量验证

代码示例:
```python
from 初始化.token_sampling_manager import create_init_sampler
from 初始化.dit_activation_extractor import DiTActivationExtractor
from 初始化.sae_mixed_init import mixed_source_initialization

# 1. 创建采样器
sampler = create_init_sampler(target_tokens=256000)

# 2. 提取激活
extractor = DiTActivationExtractor(model)
activations = extractor.extract_for_initialization(
    latents_list=latents_list,
    timesteps=[0, 100, 200, 400, 600, 800, 1000],
    context_list=context_list,
    seq_len=seq_len,
)

# 3. 采样
sampled, metadata = sampler.sample_for_initialization(activations)

# 4. 分析
analyzer = ActivationStatisticsAnalyzer()
analyzer.analyze_token_norm_distribution(sampled)
analyzer.analyze_effective_rank(sampled)

# 5. PCA 初始化
Wdec, bpre, stats = mixed_source_initialization(sampled)
```
""")


def main():
    parser = argparse.ArgumentParser(description="SAE Activation Sampling 示例")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["init", "train", "analysis", "all", "workflow"],
                        help="运行模式")
    parser.add_argument("--output_dir", type=str, default="./analysis_results",
                        help="输出目录")

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.mode in ["init", "all"]:
        sampler, sampled, metadata = example_init_mode()
        results["init"] = {
            "shape": list(sampled.shape),
            "metadata": {k: v for k, v in metadata.items() if not isinstance(v, dict)},
        }

    if args.mode in ["train", "all"]:
        sampler, sampled, metadata = example_train_mode()
        results["train"] = {
            "shape": list(sampled.shape),
            "metadata": metadata,
        }

    if args.mode in ["analysis", "all"]:
        analyzer = example_statistics_analysis()
        results["analysis"] = analyzer.get_summary()

    if args.mode == "workflow":
        example_integrated_workflow()

    # 保存结果
    if results:
        result_file = output_path / "sampling_results.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
