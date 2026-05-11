"""
Feature 统计分析 - Cohen's d 与判别性 Feature 筛选

严格按照 TODO_list_v4.md (紧急版) 规范

分析流程:
1. 加载各类别 latent
2. 计算 Positive/Negative 激活统计
3. Cohen's d 分析
4. 筛选判别性 Feature
5. 生成排名

输出:
- sex_top_features.json
- violence_top_features.json

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
from scipy import stats
from tqdm import tqdm

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
class AnalysisConfig:
    """分析配置"""

    # 输入输出
    latent_dir: str = "./outputs/layer29_latents"
    output_dir: str = "./outputs/concept_features"

    # SAE 配置
    d_hidden: int = 12288

    # 筛选条件
    min_cohen_d: float = 1.0  # |d| > 1.0
    min_activation_freq: float = 0.01  # 激活频率 > 1%

    # Top-K 保存
    top_k_features: int = 50

    # 概念定义
    concepts: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "sex": ("sex_positive", "sex_negative"),
        "violence": ("violence_positive", "violence_negative"),
    })


# ============================================================================
# Latent 数据加载
# ============================================================================

def load_latents(
    latent_dir: str,
    category: str,
) -> List[Dict[str, Any]]:
    """
    加载 latent 数据

    参数:
        latent_dir: latent 目录
        category: 类别名

    返回:
        List[Dict]: 每个 prompt 的 latent 数据
    """
    latent_file = Path(latent_dir) / f"{category}_latents.pt"

    if not latent_file.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_file}")

    data = torch.load(latent_file, map_location='cpu')
    logger.info(f"Loaded {len(data)} latent records for {category}")

    return data


def extract_latent_features(
    latent_data: List[Dict[str, Any]],
    d_hidden: int = 12288,
) -> np.ndarray:
    """
    从 latent 数据中提取 feature 激活矩阵

    参数:
        latent_data: latent 数据
        d_hidden: SAE 隐藏维度

    返回:
        np.ndarray: [N, d_hidden] feature 激活矩阵
    """
    all_features = []

    for record in latent_data:
        for lat_info in record.get("latents", []):
            # 获取 z_sparse 或从 topk 重建
            if "z_sparse" in lat_info:
                z = lat_info["z_sparse"]  # [n_tokens, d_hidden]
            else:
                # 从 topk 重建
                topk_idx = lat_info["topk_idx"]  # [n_tokens, k]
                topk_val = lat_info["topk_val"]  # [n_tokens, k]

                n_tokens = topk_idx.shape[0]
                z = torch.zeros(n_tokens, d_hidden)
                z.scatter_(1, topk_idx, topk_val)

            all_features.append(z.numpy())

    if not all_features:
        return np.array([])

    features = np.concatenate(all_features, axis=0)  # [N, d_hidden]
    return features


# ============================================================================
# 统计分析
# ============================================================================

def compute_activation_statistics(
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算每个 feature 的激活统计

    参数:
        features: [N, d_hidden] feature 矩阵

    返回:
        mean_act: [d_hidden] 平均激活
        freq: [d_hidden] 激活频率
    """
    # 平均激活
    mean_act = np.mean(features, axis=0)

    # 激活频率 (非零比例)
    freq = np.mean(features > 0, axis=0)

    return mean_act, freq


def compute_cohens_d(
    features_pos: np.ndarray,
    features_neg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算每个 feature 的 Cohen's d

    公式: d = (μ_pos - μ_neg) / pooled_std

    参数:
        features_pos: [N_pos, d_hidden] 正样本 feature
        features_neg: [N_neg, d_hidden] 负样本 feature

    返回:
        cohen_d: [d_hidden] Cohen's d 值
        mean_pos: [d_hidden] 正样本平均激活
        mean_neg: [d_hidden] 负样本平均激活
    """
    # 平均值
    mean_pos = np.mean(features_pos, axis=0)
    mean_neg = np.mean(features_neg, axis=0)

    # 标准差
    std_pos = np.std(features_pos, axis=0, ddof=1)
    std_neg = np.std(features_neg, axis=0, ddof=1)

    # Pooled std
    n_pos = len(features_pos)
    n_neg = len(features_neg)

    pooled_std = np.sqrt(
        ((n_pos - 1) * std_pos**2 + (n_neg - 1) * std_neg**2) /
        (n_pos + n_neg - 2)
    )

    # 避免除零
    pooled_std = np.maximum(pooled_std, 1e-8)

    # Cohen's d
    cohen_d = (mean_pos - mean_neg) / pooled_std

    return cohen_d, mean_pos, mean_neg


def compute_feature_statistics_detailed(
    features_pos: np.ndarray,
    features_neg: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """
    计算每个 feature 的详细统计

    返回:
        Dict[feature_id, Dict[str, float]]
    """
    d_hidden = features_pos.shape[1]

    cohen_d, mean_pos, mean_neg = compute_cohens_d(features_pos, features_neg)

    # 激活频率
    freq_pos = np.mean(features_pos > 0, axis=0)
    freq_neg = np.mean(features_neg > 0, axis=0)

    # 标准差
    std_pos = np.std(features_pos, axis=0, ddof=1)
    std_neg = np.std(features_neg, axis=0, ddof=1)

    results = {}

    for i in range(d_hidden):
        results[i] = {
            "feature_id": i,
            "cohen_d": float(cohen_d[i]),
            "abs_cohen_d": float(abs(cohen_d[i])),
            "mean_pos": float(mean_pos[i]),
            "mean_neg": float(mean_neg[i]),
            "std_pos": float(std_pos[i]),
            "std_neg": float(std_neg[i]),
            "freq_pos": float(freq_pos[i]),
            "freq_neg": float(freq_neg[i]),
        }

    return results


# ============================================================================
# 判别性 Feature 筛选
# ============================================================================

def filter_discriminative_features(
    feature_stats: Dict[int, Dict[str, float]],
    min_cohen_d: float = 1.0,
    min_freq: float = 0.01,
) -> List[Dict[str, float]]:
    """
    筛选判别性 Feature

    条件:
    - |d| > min_cohen_d
    - 激活频率 > min_freq (正样本)

    返回:
        List[Dict]: 筛选后的 feature 列表，按 |d| 排序
    """
    filtered = []

    for fid, stats in feature_stats.items():
        if stats["abs_cohen_d"] >= min_cohen_d and stats["freq_pos"] >= min_freq:
            filtered.append(stats)

    # 按 |d| 降序排序
    filtered.sort(key=lambda x: x["abs_cohen_d"], reverse=True)

    return filtered


def get_top_k_features(
    filtered_features: List[Dict[str, float]],
    top_k: int = 50,
) -> List[Dict[str, float]]:
    """
    获取 Top-K features
    """
    return filtered_features[:top_k]


# ============================================================================
# 分析流程
# ============================================================================

def analyze_concept(
    concept_name: str,
    positive_category: str,
    negative_category: str,
    config: AnalysisConfig,
) -> Dict[str, Any]:
    """
    分析单个概念

    参数:
        concept_name: 概念名 (如 "sex", "violence")
        positive_category: 正样本类别
        negative_category: 负样本类别
        config: 配置

    返回:
        Dict: 分析结果
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing concept: {concept_name}")
    logger.info(f"  Positive: {positive_category}")
    logger.info(f"  Negative: {negative_category}")
    logger.info(f"{'='*70}")

    # 加载数据
    logger.info("Loading latents...")
    pos_data = load_latents(config.latent_dir, positive_category)
    neg_data = load_latents(config.latent_dir, negative_category)

    # 提取 feature
    logger.info("Extracting features...")
    features_pos = extract_latent_features(pos_data, config.d_hidden)
    features_neg = extract_latent_features(neg_data, config.d_hidden)

    if len(features_pos) == 0 or len(features_neg) == 0:
        logger.error("No features extracted!")
        return {}

    logger.info(f"  Positive samples: {len(features_pos)} tokens")
    logger.info(f"  Negative samples: {len(features_neg)} tokens")

    # 计算统计
    logger.info("Computing statistics...")
    feature_stats = compute_feature_statistics_detailed(features_pos, features_neg)

    # 筛选判别性 feature
    logger.info(f"Filtering features (|d| > {config.min_cohen_d}, freq > {config.min_activation_freq})...")
    filtered = filter_discriminative_features(
        feature_stats,
        min_cohen_d=config.min_cohen_d,
        min_freq=config.min_activation_freq,
    )

    logger.info(f"  Found {len(filtered)} discriminative features")

    # Top-K
    top_k = get_top_k_features(filtered, config.top_k_features)

    logger.info(f"\nTop-10 features for {concept_name}:")
    for i, f in enumerate(top_k[:10]):
        logger.info(f"  #{i+1}: Feature {f['feature_id']:4d} | "
                    f"d = {f['cohen_d']:+.3f} | "
                    f"freq_pos = {f['freq_pos']:.3f} | "
                    f"mean_pos = {f['mean_pos']:.4f}")

    # 保存结果
    output_path = Path(config.output_dir) / f"{concept_name}_top_features.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "concept": concept_name,
        "positive_category": positive_category,
        "negative_category": negative_category,
        "n_positive_tokens": len(features_pos),
        "n_negative_tokens": len(features_neg),
        "n_discriminative_features": len(filtered),
        "min_cohen_d_threshold": config.min_cohen_d,
        "min_freq_threshold": config.min_activation_freq,
        "top_features": top_k,
        "all_discriminative_features": filtered,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved results to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Feature Statistical Analysis")

    # 路径配置
    parser.add_argument("--latent_dir", type=str, default="./outputs/layer29_latents",
                        help="Latent directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/concept_features",
                        help="Output directory")

    # SAE 配置
    parser.add_argument("--d_hidden", type=int, default=12288,
                        help="SAE hidden dimension")

    # 筛选条件
    parser.add_argument("--min_cohen_d", type=float, default=1.0,
                        help="Minimum |Cohen's d| threshold")
    parser.add_argument("--min_freq", type=float, default=0.01,
                        help="Minimum activation frequency threshold")

    # Top-K
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top features to save")

    # 概念
    parser.add_argument("--concepts", type=str, default="sex,violence",
                        help="Concepts to analyze (comma-separated)")

    args = parser.parse_args()

    # 创建配置
    config = AnalysisConfig(
        latent_dir=args.latent_dir,
        output_dir=args.output_dir,
        d_hidden=args.d_hidden,
        min_cohen_d=args.min_cohen_d,
        min_activation_freq=args.min_freq,
        top_k_features=args.top_k,
    )

    # 概念定义
    concept_definitions = {
        "sex": ("sex_positive", "sex_negative"),
        "violence": ("violence_positive", "violence_negative"),
    }

    # 要分析的概念
    concepts_to_analyze = [c.strip() for c in args.concepts.split(",")]

    results = {}

    for concept in concepts_to_analyze:
        if concept not in concept_definitions:
            logger.warning(f"Unknown concept: {concept}")
            continue

        pos_cat, neg_cat = concept_definitions[concept]
        result = analyze_concept(concept, pos_cat, neg_cat, config)
        results[concept] = result

    # 打印摘要
    logger.info(f"\n{'='*70}")
    logger.info("ANALYSIS SUMMARY")
    logger.info(f"{'='*70}")

    for concept, result in results.items():
        if result:
            logger.info(f"\n{concept.upper()}:")
            logger.info(f"  Discriminative features: {result['n_discriminative_features']}")
            logger.info(f"  Top feature ID: {result['top_features'][0]['feature_id']}")
            logger.info(f"  Top feature Cohen's d: {result['top_features'][0]['cohen_d']:.3f}")

    logger.info("\nFeature analysis completed!")


if __name__ == "__main__":
    main()
