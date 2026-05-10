#!/usr/bin/env python3
"""
Training Sampling Pipeline Example

演示如何使用新的训练采样模块

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from training import (
    TrainingSamplingPipeline,
    create_training_pipeline,
    TrainingSamplingConfig,
)
from training.timestep_sampler import (
    LayerAwareTimestepSampler,
    create_layer_aware_sampler,
)
from training.sampling_monitor import SamplingStatisticsMonitor
from training.layer_timestep_config import print_layer_config_table


def example_timestep_sampling():
    """示例：Layer-aware Timestep 采样"""
    print("\n" + "=" * 70)
    print("Example 1: Layer-Aware Timestep Sampling")
    print("=" * 70)

    # 创建采样器
    sampler = create_layer_aware_sampler(device="cpu", seed=42)

    # 为各层采样 timestep
    print("\n各层采样的 timestep:")
    for layer_idx in [14, 19, 24, 29]:
        timesteps = sampler.sample_timestep(layer_idx, batch_size=10)
        mu, sigma = sampler.get_layer_distribution_params(layer_idx)
        print(f"  Layer {layer_idx} (μ={mu}, σ={sigma}): {timesteps.tolist()}")

    # 查看采样历史
    history = sampler.get_sampling_history()
    print(f"\n采样历史记录:")
    for layer, ts in history.items():
        print(f"  Layer {layer}: {len(ts)} samples")


def example_token_sampling():
    """示例：Token 采样"""
    print("\n" + "=" * 70)
    print("Example 2: Token Sampling (Distribution Preserving)")
    print("=" * 70)

    # 创建配置
    config = TrainingSamplingConfig(
        max_tokens_per_batch=4096,
        seed=42,
    )

    # 创建流水线
    pipeline = create_training_pipeline(
        max_tokens_per_batch=4096,
        seed=42,
        device="cpu",
    )

    # 模拟激活数据
    batch_size = 2
    n_tokens = 17160  # 11 * 30 * 52
    d_model = 1536

    activations = torch.randn(batch_size, n_tokens, d_model)
    grid_size = (11, 30, 52)

    print(f"\n输入激活: {activations.shape}")
    print(f"网格尺寸: {grid_size}")

    # 为各层采样
    print("\n各层采样结果:")
    for layer_idx in [14, 19, 24, 29]:
        result = pipeline.sample(
            activations=activations,
            layer_idx=layer_idx,
            grid_size=grid_size,
        )
        print(f"  Layer {layer_idx}:")
        print(f"    输出激活: {result.activations.shape}")
        print(f"    采样 timestep: {result.metadata['avg_timestep']}")
        print(f"    空间 stride: {result.metadata['spatial_stride']}")


def example_full_pipeline():
    """示例：完整流水线"""
    print("\n" + "=" * 70)
    print("Example 3: Full Training Sampling Pipeline")
    print("=" * 70)

    # 创建流水线
    pipeline = create_training_pipeline(
        max_tokens_per_batch=4096,
        seed=42,
        device="cpu",
    )

    # 模拟多步训练
    n_steps = 100

    print(f"\n模拟 {n_steps} 步训练...")
    for step in range(n_steps):
        # 模拟激活
        activations = torch.randn(2, 17160, 1536)

        # 为各层采样
        for layer_idx in [14, 19, 24, 29]:
            result = pipeline.sample(
                activations=activations,
                layer_idx=layer_idx,
                grid_size=(11, 30, 52),
            )

    # 打印统计报告
    pipeline.print_report()


def example_monitoring():
    """示例：统计监控"""
    print("\n" + "=" * 70)
    print("Example 4: Statistics Monitoring")
    print("=" * 70)

    monitor = SamplingStatisticsMonitor(min_timestep=150, max_timestep=800)

    # 模拟采样记录
    import random

    for _ in range(1000):
        layer_idx = random.choice([14, 19, 24, 29])

        # 根据 layer 选择合适的 timestep
        if layer_idx == 14:
            timestep = int(random.gauss(650, 120))
        elif layer_idx == 19:
            timestep = int(random.gauss(550, 110))
        elif layer_idx == 24:
            timestep = int(random.gauss(420, 100))
        else:
            timestep = int(random.gauss(300, 80))

        timestep = max(150, min(800, timestep))  # Clamp

        monitor.record_sample(timestep, layer_idx, n_tokens=4096)

    # 获取统计
    stats = monitor.get_statistics()

    print("\n[Timestep Histogram]")
    hist = stats["timestep_histogram"]
    print(f"  Valid ratio: {hist['valid_ratio']:.2%}")

    print("\n[Layer Entropy]")
    for layer_idx, entropy_info in stats["layer_entropy"].items():
        print(f"  Layer {layer_idx}: normalized_entropy = {entropy_info['normalized_entropy']:.4f}")

    # 检测分布漂移
    drift = monitor.detect_distribution_drift()
    print(f"\n[Distribution Drift]")
    print(f"  Detected: {drift['drift_detected']}")
    print(f"  KL Divergence: {drift['kl_divergence']:.4f}")


def main():
    print("\n" + "=" * 70)
    print("Training Sampling Pipeline Examples")
    print("=" * 70)

    # 打印配置表
    print_layer_config_table()

    # 运行示例
    example_timestep_sampling()
    example_token_sampling()
    example_full_pipeline()
    example_monitoring()

    # 打印流水线图
    from training import PIPELINE_DIAGRAM
    print("\n" + "=" * 70)
    print("Pipeline Diagram")
    print("=" * 70)
    print(PIPELINE_DIAGRAM)


if __name__ == "__main__":
    main()
