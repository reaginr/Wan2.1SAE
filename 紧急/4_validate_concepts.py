"""
概念方向验证 - Projection Score & AUC

严格按照 TODO_list_v4.md (紧急版) 规范

验证流程:
1. 加载概念向量
2. 计算 Projection Score: score = z @ v
3. 计算 AUC
4. 生成可视化

目标:
- Sex AUC > 0.85
- Violence AUC > 0.85

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
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

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
class ValidationConfig:
    """验证配置"""

    # 输入路径
    latent_dir: str = "./outputs/layer29_latents"
    vector_dir: str = "./outputs/concept_vectors"
    output_dir: str = "./outputs/validation_results"

    # SAE 配置
    d_hidden: int = 12288

    # 向量类型
    vector_type: str = "sparse"  # "dense" or "sparse"


# ============================================================================
# Projection Score 计算
# ============================================================================

def load_concept_vector(
    vector_dir: str,
    concept: str,
    vector_type: str = "sparse",
) -> np.ndarray:
    """
    加载概念向量

    参数:
        vector_dir: 向量目录
        concept: 概念名
        vector_type: "dense" or "sparse"

    返回:
        np.ndarray: [d_hidden] 概念向量
    """
    vector_file = Path(vector_dir) / f"{concept}_vector_{vector_type}.pt"

    if not vector_file.exists():
        raise FileNotFoundError(f"Vector file not found: {vector_file}")

    data = torch.load(vector_file, map_location='cpu')
    vector = data["vector"].numpy()

    logger.info(f"Loaded {vector_type} vector for {concept}, shape: {vector.shape}")

    return vector


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


def compute_projection_scores(
    features: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """
    计算 projection scores

    公式: score = z @ v

    参数:
        features: [N, d_hidden]
        vector: [d_hidden]

    返回:
        scores: [N] projection scores
    """
    return features @ vector


def compute_auc(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
) -> float:
    """
    计算 AUC

    参数:
        scores_pos: 正样本 projection scores
        scores_neg: 负样本 projection scores

    返回:
        auc: AUC 值
    """
    # 构建标签
    y_true = np.concatenate([
        np.ones(len(scores_pos)),
        np.zeros(len(scores_neg)),
    ])

    # 构建预测分数
    y_score = np.concatenate([scores_pos, scores_neg])

    # 计算 AUC
    auc = roc_auc_score(y_true, y_score)

    return auc


def compute_statistics(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
) -> Dict[str, float]:
    """
    计算 projection score 统计

    返回:
        Dict: 统计信息
    """
    return {
        "mean_pos": float(np.mean(scores_pos)),
        "std_pos": float(np.std(scores_pos)),
        "mean_neg": float(np.mean(scores_neg)),
        "std_neg": float(np.std(scores_neg)),
        "median_pos": float(np.median(scores_pos)),
        "median_neg": float(np.median(scores_neg)),
        "min_pos": float(np.min(scores_pos)),
        "max_pos": float(np.max(scores_pos)),
        "min_neg": float(np.min(scores_neg)),
        "max_neg": float(np.max(scores_neg)),
        "separation": float(np.mean(scores_pos) - np.mean(scores_neg)),
        "t_statistic": float(stats.ttest_ind(scores_pos, scores_neg)[0]),
        "p_value": float(stats.ttest_ind(scores_pos, scores_neg)[1]),
    }


# ============================================================================
# 可视化
# ============================================================================

def plot_projection_histogram(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
    concept: str,
    output_path: str,
):
    """
    绘制 projection score 直方图

    参数:
        scores_pos: 正样本分数
        scores_neg: 负样本分数
        concept: 概念名
        output_path: 输出路径
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制直方图
        bins = np.linspace(
            min(scores_pos.min(), scores_neg.min()),
            max(scores_pos.max(), scores_neg.max()),
            50
        )

        ax.hist(scores_pos, bins=bins, alpha=0.6, label='Positive', color='red', density=True)
        ax.hist(scores_neg, bins=bins, alpha=0.6, label='Negative', color='blue', density=True)

        ax.set_xlabel('Projection Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{concept.upper()} - Projection Score Distribution', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved histogram to {output_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")


def plot_roc_curve(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
    concept: str,
    auc_value: float,
    output_path: str,
):
    """
    绘制 ROC 曲线

    参数:
        scores_pos: 正样本分数
        scores_neg: 负样本分数
        concept: 概念名
        auc_value: AUC 值
        output_path: 输出路径
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        # 构建标签和分数
        y_true = np.concatenate([
            np.ones(len(scores_pos)),
            np.zeros(len(scores_neg)),
        ])
        y_score = np.concatenate([scores_pos, scores_neg])

        # 计算 ROC 曲线
        fpr, tpr, _ = roc_curve(y_true, y_score)

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_value:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{concept.upper()} - ROC Curve', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved ROC curve to {output_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")


# ============================================================================
# 验证流程
# ============================================================================

def validate_concept(
    concept: str,
    positive_category: str,
    negative_category: str,
    config: ValidationConfig,
) -> Dict[str, Any]:
    """
    验证单个概念

    返回:
        Dict: 验证结果
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Validating concept: {concept}")
    logger.info(f"  Vector type: {config.vector_type}")
    logger.info(f"{'='*70}")

    # 加载概念向量
    logger.info("Loading concept vector...")
    vector = load_concept_vector(
        config.vector_dir,
        concept,
        config.vector_type,
    )

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

    # 计算 projection scores
    logger.info("Computing projection scores...")
    scores_pos = compute_projection_scores(features_pos, vector)
    scores_neg = compute_projection_scores(features_neg, vector)

    # 计算统计
    logger.info("Computing statistics...")
    stats_dict = compute_statistics(scores_pos, scores_neg)

    for key, value in stats_dict.items():
        logger.info(f"  {key}: {value:.4f}")

    # 计算 AUC
    logger.info("Computing AUC...")
    auc = compute_auc(scores_pos, scores_neg)
    logger.info(f"  AUC: {auc:.4f}")

    # 判断是否达标
    target_auc = 0.85
    is_passed = auc >= target_auc
    logger.info(f"  Target AUC: {target_auc}")
    logger.info(f"  Status: {'PASSED' if is_passed else 'NOT PASSED'}")

    # 创建输出目录
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 绘制直方图
    plot_projection_histogram(
        scores_pos,
        scores_neg,
        concept,
        str(output_path / f"{concept}_histogram.png"),
    )

    # 绘制 ROC 曲线
    plot_roc_curve(
        scores_pos,
        scores_neg,
        concept,
        auc,
        str(output_path / f"{concept}_roc_curve.png"),
    )

    # 保存结果
    result = {
        "concept": concept,
        "positive_category": positive_category,
        "negative_category": negative_category,
        "vector_type": config.vector_type,
        "n_positive_tokens": len(features_pos),
        "n_negative_tokens": len(features_neg),
        "auc": float(auc),
        "target_auc": target_auc,
        "is_passed": bool(is_passed),
        "statistics": stats_dict,
    }

    result_file = output_path / f"{concept}_validation.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved validation result to {result_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Concept Direction Validation")

    # 路径配置
    parser.add_argument("--latent_dir", type=str, default="./outputs/layer29_latents",
                        help="Latent directory")
    parser.add_argument("--vector_dir", type=str, default="./outputs/concept_vectors",
                        help="Concept vector directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/validation_results",
                        help="Output directory")

    # SAE 配置
    parser.add_argument("--d_hidden", type=int, default=12288,
                        help="SAE hidden dimension")

    # 向量类型
    parser.add_argument("--vector_type", type=str, default="sparse",
                        choices=["dense", "sparse"],
                        help="Vector type to use")

    # 概念
    parser.add_argument("--concepts", type=str, default="sex,violence",
                        help="Concepts to validate (comma-separated)")

    args = parser.parse_args()

    # 创建配置
    config = ValidationConfig(
        latent_dir=args.latent_dir,
        vector_dir=args.vector_dir,
        output_dir=args.output_dir,
        d_hidden=args.d_hidden,
        vector_type=args.vector_type,
    )

    # 概念定义
    concept_definitions = {
        "sex": ("sex_positive", "sex_negative"),
        "violence": ("violence_positive", "violence_negative"),
    }

    # 要验证的概念
    concepts = [c.strip() for c in args.concepts.split(",")]

    results = {}

    for concept in concepts:
        if concept not in concept_definitions:
            logger.warning(f"Unknown concept: {concept}")
            continue

        pos_cat, neg_cat = concept_definitions[concept]
        result = validate_concept(concept, pos_cat, neg_cat, config)
        results[concept] = result

    # 打印摘要
    logger.info(f"\n{'='*70}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*70}")

    for concept, result in results.items():
        if result:
            logger.info(f"\n{concept.upper()}:")
            logger.info(f"  AUC: {result['auc']:.4f}")
            logger.info(f"  Separation: {result['statistics']['separation']:.4f}")
            logger.info(f"  Status: {'PASSED' if result['is_passed'] else 'NOT PASSED'}")

    logger.info("\nValidation completed!")


if __name__ == "__main__":
    main()
