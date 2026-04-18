"""
生成模拟SAE激活值数据，用于测试t-SNE可视化效果

生成三种类型的数据：
1. well_separated: 正负样本明显分离（Silhouette ≈ 0.7-0.9）
2. partially_separated: 部分重叠（Silhouette ≈ 0.3-0.5）
3. overlapping: 高度重叠（Silhouette ≈ 0.0-0.2）

数据格式与阶段一输出一致：
    {layer_type}_layer{idx}/{category}/{polarity}/activations.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wan.sae.interpretability.activation_io import ActivationIO, SampleMetadata


def generate_well_separated_data(
    n_samples: int,
    d_hidden: int,
    n_timesteps: int = 30,
    n_tokens: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成明显分离的正负样本

    正样本：围绕 +1 的高斯分布
    负样本：围绕 -1 的高斯分布

    预期 Silhouette Score: 0.7-0.9
    """
    rng = np.random.RandomState(seed)

    # 正样本：均值 +1，小方差
    pos_mean = np.ones(d_hidden) * 1.0
    pos_cov = np.eye(d_hidden) * 0.3
    pos_features = rng.multivariate_normal(pos_mean, pos_cov, n_samples)

    # 负样本：均值 -1，小方差
    neg_mean = np.ones(d_hidden) * -1.0
    neg_cov = np.eye(d_hidden) * 0.3
    neg_features = rng.multivariate_normal(neg_mean, neg_cov, n_samples)

    # 扩展为 [N, T, L, D] 格式（时间步和token维度上复制）
    pos_activations = np.tile(
        pos_features[:, np.newaxis, np.newaxis, :],
        (1, n_timesteps, n_tokens, 1)
    ).astype(np.float32)

    neg_activations = np.tile(
        neg_features[:, np.newaxis, np.newaxis, :],
        (1, n_timesteps, n_tokens, 1)
    ).astype(np.float32)

    return pos_activations, neg_activations


def generate_partially_separated_data(
    n_samples: int,
    d_hidden: int,
    n_timesteps: int = 30,
    n_tokens: int = 100,
    seed: int = 43,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成部分重叠的正负样本

    正样本：均值 +0.5
    负样本：均值 -0.5
    方差较大，导致重叠

    预期 Silhouette Score: 0.3-0.5
    """
    rng = np.random.RandomState(seed)

    # 正样本：均值 +0.5，大方差
    pos_mean = np.ones(d_hidden) * 0.5
    pos_cov = np.eye(d_hidden) * 1.2
    pos_features = rng.multivariate_normal(pos_mean, pos_cov, n_samples)

    # 负样本：均值 -0.5，大方差
    neg_mean = np.ones(d_hidden) * -0.5
    neg_cov = np.eye(d_hidden) * 1.2
    neg_features = rng.multivariate_normal(neg_mean, neg_cov, n_samples)

    pos_activations = np.tile(
        pos_features[:, np.newaxis, np.newaxis, :],
        (1, n_timesteps, n_tokens, 1)
    ).astype(np.float32)

    neg_activations = np.tile(
        neg_features[:, np.newaxis, np.newaxis, :],
        (1, n_timesteps, n_tokens, 1)
    ).astype(np.float32)

    return pos_activations, neg_activations


def generate_overlapping_data(
    n_samples: int,
    d_hidden: int,
    n_timesteps: int = 30,
    n_tokens: int = 100,
    seed: int = 44,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成高度重叠的正负样本

    正样本和负样本：均值接近（+0.2 vs -0.2），大方差

    预期 Silhouette Score: 0.0-0.2
    """
    rng = np.random.RandomState(seed)

    # 正样本：均值 +0.2，大方差
    pos_mean = np.ones(d_hidden) * 0.2
    pos_cov = np.eye(d_hidden) * 2.0
    pos_features = rng.multivariate_normal(pos_mean, pos_cov, n_samples)

    # 负样本：均值 -0.2，大方差
    neg_mean = np.ones(d_hidden) * -0.2
    neg_cov = np.eye(d_hidden) * 2.0
    neg_features = rng.multivariate_normal(neg_mean, neg_cov, n_samples)

    pos_activations = np.tile(
        pos_features[:, np.newaxis, np.newaxis, :],
        (1, n_timesteps, n_tokens, 1)
    ).astype(np.float32)

    neg_activations = np.tile(
        neg_features[:, np.newaxis, np.newaxis, :],
        (1, n_timesteps, n_tokens, 1)
    ).astype(np.float32)

    return pos_activations, neg_activations


def generate_sparse_separated_data(
    n_samples: int,
    d_hidden: int,
    n_timesteps: int = 30,
    n_tokens: int = 100,
    sparsity: float = 0.95,
    seed: int = 45,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成稀疏且分离的数据（模拟真实SAE）

    只有10%的特征活跃，其余为0
    活跃特征在正负样本中有明显差异

    预期 Silhouette Score: 0.6-0.8
    """
    rng = np.random.RandomState(seed)

    n_active = int(d_hidden * (1 - sparsity))

    # 生成稀疏特征
    pos_features = np.zeros((n_samples, d_hidden))
    neg_features = np.zeros((n_samples, d_hidden))

    # 选择活跃特征
    active_indices = rng.choice(d_hidden, n_active, replace=False)

    # 正样本：活跃特征为正
    pos_features[:, active_indices] = rng.randn(n_samples, n_active) * 0.5 + 1.0

    # 负样本：活跃特征为负
    neg_features[:, active_indices] = rng.randn(n_samples, n_active) * 0.5 - 1.0

    # 添加少量噪声到非活跃特征
    noise_mask = rng.rand(n_samples, d_hidden) < 0.01
    pos_features[noise_mask] = rng.randn(np.sum(noise_mask)) * 0.1
    neg_features[noise_mask] = rng.randn(np.sum(noise_mask)) * 0.1

    pos_activations = np.tile(
        pos_features[:, np.newaxis, np.newaxis, :],
        (1, n_timesteps, n_tokens, 1)
    ).astype(np.float32)

    neg_activations = np.tile(
        neg_features[:, np.newaxis, np.newaxis, :],
        (1, n_timesteps, n_tokens, 1)
    ).astype(np.float32)

    return pos_activations, neg_activations


def generate_metadata(n_samples: int, category: str, polarity: str) -> List[SampleMetadata]:
    """生成元信息"""
    return [
        SampleMetadata(
            idx=i,
            pair_idx=i,
            prompt=f"{polarity}_prompt_{i:03d}",
            category=category,
            polarity=polarity,
        )
        for i in range(n_samples)
    ]


def save_mock_data(
    output_root: str,
    category: str,
    layer_type: str,
    layer_idx: int,
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
) -> None:
    """保存模拟数据到标准目录结构"""
    io = ActivationIO(output_root)

    # 保存激活值
    io.save_activations(
        layer_type, layer_idx, category, "pos",
        pos_activations, append=False
    )
    io.save_activations(
        layer_type, layer_idx, category, "neg",
        neg_activations, append=False
    )

    # 保存元信息
    pos_metadata = generate_metadata(len(pos_activations), category, "pos")
    neg_metadata = generate_metadata(len(neg_activations), category, "neg")
    io.save_metadata(layer_type, layer_idx, category, "pos", pos_metadata, append=False)
    io.save_metadata(layer_type, layer_idx, category, "neg", neg_metadata, append=False)

    print(f"  正样本: {pos_activations.shape}")
    print(f"  负样本: {neg_activations.shape}")


def generate_all_mock_datasets(
    output_root: str = "simulate/test-tSNE/mock_activations",
    n_samples: int = 200,
    d_hidden: int = 6144,
    n_timesteps: int = 30,
    n_tokens: int = 100,
) -> Dict[str, str]:
    """
    生成所有类型的模拟数据集

    返回类别名称列表
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    categories = []

    # 1. 明显分离的数据
    print("\n" + "="*60)
    print("生成数据集 1: well_separated (预期 Silhouette > 0.7)")
    print("="*60)
    pos, neg = generate_well_separated_data(n_samples, d_hidden, n_timesteps, n_tokens)
    save_mock_data(str(output_root), "well_separated", "sae", 15, pos, neg)
    categories.append("well_separated")

    # 2. 部分重叠的数据
    print("\n" + "="*60)
    print("生成数据集 2: partially_separated (预期 Silhouette 0.3-0.5)")
    print("="*60)
    pos, neg = generate_partially_separated_data(n_samples, d_hidden, n_timesteps, n_tokens)
    save_mock_data(str(output_root), "partially_separated", "sae", 15, pos, neg)
    categories.append("partially_separated")

    # 3. 高度重叠的数据
    print("\n" + "="*60)
    print("生成数据集 3: overlapping (预期 Silhouette < 0.2)")
    print("="*60)
    pos, neg = generate_overlapping_data(n_samples, d_hidden, n_timesteps, n_tokens)
    save_mock_data(str(output_root), "overlapping", "sae", 15, pos, neg)
    categories.append("overlapping")

    # 4. 稀疏分离的数据
    print("\n" + "="*60)
    print("生成数据集 4: sparse_separated (预期 Silhouette 0.6-0.8)")
    print("="*60)
    pos, neg = generate_sparse_separated_data(n_samples, d_hidden, n_timesteps, n_tokens)
    save_mock_data(str(output_root), "sparse_separated", "sae", 15, pos, neg)
    categories.append("sparse_separated")

    # 保存全局配置
    io = ActivationIO(str(output_root))
    io.save_config({
        "mock_data": True,
        "n_samples": n_samples,
        "d_hidden": d_hidden,
        "n_timesteps": n_timesteps,
        "n_tokens": n_tokens,
        "categories": categories,
        "description": "模拟数据用于测试t-SNE可视化效果",
    })

    print("\n" + "="*60)
    print("所有模拟数据生成完成")
    print(f"输出目录: {output_root}")
    print(f"数据集: {categories}")
    print("="*60)

    return {cat: str(output_root) for cat in categories}


def main():
    parser = argparse.ArgumentParser(description="生成模拟SAE激活值数据")
    parser.add_argument("--output_root", type=str, default="simulate/test-tSNE/mock_activations",
                        help="输出根目录")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="每类样本数")
    parser.add_argument("--d_hidden", type=int, default=6144,
                        help="SAE隐藏层维度")
    parser.add_argument("--n_timesteps", type=int, default=30,
                        help="时间步数")
    parser.add_argument("--n_tokens", type=int, default=100,
                        help="token数量")

    args = parser.parse_args()

    generate_all_mock_datasets(
        output_root=args.output_root,
        n_samples=args.n_samples,
        d_hidden=args.d_hidden,
        n_timesteps=args.n_timesteps,
        n_tokens=args.n_tokens,
    )


if __name__ == "__main__":
    main()
