"""
Concept Vector 构建

严格按照 TODO_list_v4.md (紧急版) 规范

构建方法:
1. Mean Difference Vector: v = mean(z_pos) - mean(z_neg)
2. Sparse Feature Vector: 只保留 |d| > 1.0 的 feature

输出:
- sex_vector.pt
- violence_vector.pt

作者：Claude
日期：2026-05-11
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class VectorConfig:
    """Vector 构建配置"""

    # 输入输出
    latent_dir: str = "./outputs/layer29_latents"
    feature_dir: str = "./outputs/concept_features"
    output_dir: str = "./outputs/concept_vectors"

    # SAE 配置
    d_hidden: int = 12288

    # 稀疏化
    min_cohen_d: float = 1.0  # 只保留 |d| > 阈值的 feature


# ============================================================================
# Concept Vector 构建
# ============================================================================

def load_latents(latent_dir: str, category: str) -> List[Dict[str, Any]]:
    """加载 latent 数据"""
    latent_file = Path(latent_dir) / f"{category}_latents.pt"

    if not latent_file.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_file}")

    data = torch.load(latent_file, map_location='cpu')
    return data


def extract_latent_matrix(
    latent_data: List[Dict[str, Any]],
    d_hidden: int = 12288,
) -> np.ndarray:
    """从 latent 数据中提取 feature 激活矩阵"""
    all_features = []

    for record in latent_data:
        for lat_info in record.get("latents", []):
            if "z_sparse" in lat_info:
                z = lat_info["z_sparse"]
            else:
                topk_idx = lat_info["topk_idx"]
                topk_val = lat_info["topk_val"]

                n_tokens = topk_idx.shape[0]
                z = torch.zeros(n_tokens, d_hidden)
                z.scatter_(1, topk_idx, topk_val)

            all_features.append(z.numpy())

    if not all_features:
        return np.array([])

    return np.concatenate(all_features, axis=0)


def build_mean_difference_vector(
    features_pos: np.ndarray,
    features_neg: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    构建均值差异向量

    公式: v = mean(z_pos) - mean(z_neg)

    参数:
        features_pos: [N_pos, d_hidden]
        features_neg: [N_neg, d_hidden]
        normalize: 是否归一化

    返回:
        v: [d_hidden] 概念向量
    """
    mean_pos = np.mean(features_pos, axis=0)
    mean_neg = np.mean(features_neg, axis=0)

    v = mean_pos - mean_neg

    if normalize:
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            v = v / norm

    return v


def build_sparse_feature_vector(
    features_pos: np.ndarray,
    features_neg: np.ndarray,
    discriminative_features: List[int],
    normalize: bool = True,
) -> np.ndarray:
    """
    构建稀疏概念向量

    只保留判别性 feature 的维度，其余置零

    参数:
        features_pos: [N_pos, d_hidden]
        features_neg: [N_neg, d_hidden]
        discriminative_features: 判别性 feature 列表
        normalize: 是否归一化

    返回:
        v: [d_hidden] 稀疏概念向量
    """
    d_hidden = features_pos.shape[1]

    # 先构建完整向量
    v = build_mean_difference_vector(features_pos, features_neg, normalize=False)

    # 创建 mask
    mask = np.zeros(d_hidden, dtype=bool)
    mask[discriminative_features] = True

    # 应用 mask
    v_sparse = v * mask

    # 归一化
    if normalize:
        norm = np.linalg.norm(v_sparse)
        if norm > 1e-8:
            v_sparse = v_sparse / norm

    return v_sparse


def load_discriminative_features(
    feature_dir: str,
    concept: str,
    min_cohen_d: Optional[float] = None,
) -> List[int]:
    """
    加载判别性 feature 列表

    参数:
        feature_dir: feature 分析结果目录
        concept: 概念名
        min_cohen_d: 可选的额外阈值过滤

    返回:
        List[int]: feature ID 列表
    """
    feature_file = Path(feature_dir) / f"{concept}_top_features.json"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    with open(feature_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features = []

    for f_info in data.get("all_discriminative_features", []):
        if min_cohen_d is not None:
            if abs(f_info["cohen_d"]) < min_cohen_d:
                continue

        features.append(f_info["feature_id"])

    return features


# ============================================================================
# 分析流程
# ============================================================================

def build_concept_vectors(
    concept: str,
    positive_category: str,
    negative_category: str,
    config: VectorConfig,
) -> Dict[str, Any]:
    """
    为单个概念构建向量

    返回:
        Dict: 包含 dense 和 sparse 两种向量
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Building concept vector: {concept}")
    logger.info(f"  Positive: {positive_category}")
    logger.info(f"  Negative: {negative_category}")
    logger.info(f"{'='*70}")

    # 加载数据
    logger.info("Loading latents...")
    pos_data = load_latents(config.latent_dir, positive_category)
    neg_data = load_latents(config.latent_dir, negative_category)

    # 提取 feature
    logger.info("Extracting features...")
    features_pos = extract_latent_matrix(pos_data, config.d_hidden)
    features_neg = extract_latent_matrix(neg_data, config.d_hidden)

    if len(features_pos) == 0 or len(features_neg) == 0:
        logger.error("No features extracted!")
        return {}

    logger.info(f"  Positive samples: {len(features_pos)} tokens")
    logger.info(f"  Negative samples: {len(features_neg)} tokens")

    # 构建 dense vector
    logger.info("Building dense mean-difference vector...")
    v_dense = build_mean_difference_vector(features_pos, features_neg)
    logger.info(f"  Dense vector norm: {np.linalg.norm(v_dense):.4f}")
    logger.info(f"  Non-zero dims: {np.sum(v_dense != 0)}")

    # 加载判别性 feature
    logger.info("Loading discriminative features...")
    disc_features = load_discriminative_features(
        config.feature_dir,
        concept,
        min_cohen_d=config.min_cohen_d,
    )
    logger.info(f"  Discriminative features: {len(disc_features)}")

    # 构建 sparse vector
    logger.info("Building sparse feature vector...")
    v_sparse = build_sparse_feature_vector(
        features_pos,
        features_neg,
        disc_features,
        normalize=True,
    )
    logger.info(f"  Sparse vector norm: {np.linalg.norm(v_sparse):.4f}")
    logger.info(f"  Non-zero dims: {np.sum(v_sparse != 0)}")

    # 保存结果
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dense vector
    dense_file = output_path / f"{concept}_vector_dense.pt"
    torch.save({
        "vector": torch.from_numpy(v_dense),
        "concept": concept,
        "positive_category": positive_category,
        "negative_category": negative_category,
        "d_hidden": config.d_hidden,
        "n_discriminative_features": len(disc_features),
    }, dense_file)
    logger.info(f"Saved dense vector to {dense_file}")

    # Sparse vector
    sparse_file = output_path / f"{concept}_vector_sparse.pt"
    torch.save({
        "vector": torch.from_numpy(v_sparse),
        "concept": concept,
        "positive_category": positive_category,
        "negative_category": negative_category,
        "d_hidden": config.d_hidden,
        "discriminative_features": disc_features,
    }, sparse_file)
    logger.info(f"Saved sparse vector to {sparse_file}")

    # 返回结果
    result = {
        "concept": concept,
        "positive_category": positive_category,
        "negative_category": negative_category,
        "n_positive_tokens": len(features_pos),
        "n_negative_tokens": len(features_neg),
        "n_discriminative_features": len(disc_features),
        "dense_vector_norm": float(np.linalg.norm(v_dense)),
        "sparse_vector_norm": float(np.linalg.norm(v_sparse)),
        "sparse_nonzero_dims": int(np.sum(v_sparse != 0)),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Concept Vector Construction")

    # 路径配置
    parser.add_argument("--latent_dir", type=str, default="./outputs/layer29_latents",
                        help="Latent directory")
    parser.add_argument("--feature_dir", type=str, default="./outputs/concept_features",
                        help="Feature analysis directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/concept_vectors",
                        help="Output directory")

    # SAE 配置
    parser.add_argument("--d_hidden", type=int, default=12288,
                        help="SAE hidden dimension")

    # 稀疏化
    parser.add_argument("--min_cohen_d", type=float, default=1.0,
                        help="Minimum Cohen's d for sparse vector")

    # 概念
    parser.add_argument("--concepts", type=str, default="sex,violence",
                        help="Concepts to build (comma-separated)")

    args = parser.parse_args()

    # 创建配置
    config = VectorConfig(
        latent_dir=args.latent_dir,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        d_hidden=args.d_hidden,
        min_cohen_d=args.min_cohen_d,
    )

    # 概念定义
    concept_definitions = {
        "sex": ("sex_positive", "sex_negative"),
        "violence": ("violence_positive", "violence_negative"),
    }

    # 要处理的概念
    concepts = [c.strip() for c in args.concepts.split(",")]

    results = {}

    for concept in concepts:
        if concept not in concept_definitions:
            logger.warning(f"Unknown concept: {concept}")
            continue

        pos_cat, neg_cat = concept_definitions[concept]
        result = build_concept_vectors(concept, pos_cat, neg_cat, config)
        results[concept] = result

    # 打印摘要
    logger.info(f"\n{'='*70}")
    logger.info("VECTOR CONSTRUCTION SUMMARY")
    logger.info(f"{'='*70}")

    for concept, result in results.items():
        if result:
            logger.info(f"\n{concept.upper()}:")
            logger.info(f"  Discriminative features: {result['n_discriminative_features']}")
            logger.info(f"  Sparse non-zero dims: {result['sparse_nonzero_dims']}")

    logger.info("\nVector construction completed!")


if __name__ == "__main__":
    main()
