"""
Feature 可解释性分析

严格按照 TODO_list_v4.md (紧急版) 规范

分析流程:
1. 对每个 top feature，检索 top-activation prompts
2. 人工标记 feature 语义

输出:
- Top feature 对应的激活 prompts
- Feature 语义标注

作者：Claude
日期：2026-05-11
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
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
class InterpretConfig:
    """可解释性分析配置"""

    # 输入路径
    latent_dir: str = "./outputs/layer29_latents"
    feature_dir: str = "./outputs/concept_features"
    output_dir: str = "./outputs/feature_interpret"

    # SAE 配置
    d_hidden: int = 12288

    # 分析配置
    top_k_prompts: int = 10  # 每个 feature 显示几个 top prompts
    top_k_features: int = 20  # 分析几个 top features


# ============================================================================
# Feature Activation 检索
# ============================================================================

def load_latents(latent_dir: str, category: str) -> List[Dict[str, Any]]:
    """加载 latent 数据"""
    latent_file = Path(latent_dir) / f"{category}_latents.pt"

    if not latent_file.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_file}")

    data = torch.load(latent_file, map_location='cpu')
    return data


def load_top_features(
    feature_dir: str,
    concept: str,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    加载 top features

    返回:
        List[Dict]: feature 信息列表
    """
    feature_file = Path(feature_dir) / f"{concept}_top_features.json"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    with open(feature_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data["top_features"][:top_k]


def extract_feature_activations(
    latent_data: List[Dict[str, Any]],
    feature_id: int,
    d_hidden: int = 12288,
) -> List[Tuple[str, float, int]]:
    """
    提取单个 feature 的激活信息

    参数:
        latent_data: latent 数据
        feature_id: feature ID
        d_hidden: SAE 隐藏维度

    返回:
        List[Tuple[prompt, activation_value, timestep]]
    """
    activations = []

    for record in latent_data:
        prompt = record.get("prompt", "")

        for lat_info in record.get("latents", []):
            timestep = lat_info.get("timestep", 0)

            # 获取 topk 信息
            topk_idx = lat_info["topk_idx"]  # [n_tokens, k]
            topk_val = lat_info["topk_val"]  # [n_tokens, k]

            # 检查是否包含目标 feature
            mask = topk_idx == feature_id

            if mask.any():
                # 获取激活值
                values = topk_val[mask]
                mean_act = float(values.mean())
                max_act = float(values.max())
                count = int(mask.sum())

                activations.append((prompt, mean_act, max_act, count, timestep))

    return activations


def get_top_activation_prompts(
    activations: List[Tuple[str, float, float, int, int]],
    top_k: int = 10,
    sort_by: str = "mean",  # "mean" or "max" or "count"
) -> List[Dict[str, Any]]:
    """
    获取 top 激活的 prompts

    参数:
        activations: 激活列表
        top_k: 返回数量
        sort_by: 排序方式

    返回:
        List[Dict]: top prompts 信息
    """
    if sort_by == "mean":
        sorted_act = sorted(activations, key=lambda x: x[1], reverse=True)
    elif sort_by == "max":
        sorted_act = sorted(activations, key=lambda x: x[2], reverse=True)
    else:  # count
        sorted_act = sorted(activations, key=lambda x: x[3], reverse=True)

    results = []
    seen_prompts = set()

    for prompt, mean_act, max_act, count, timestep in sorted_act:
        if prompt in seen_prompts:
            continue

        seen_prompts.add(prompt)

        results.append({
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "mean_activation": float(mean_act),
            "max_activation": float(max_act),
            "activation_count": int(count),
            "timestep": int(timestep),
        })

        if len(results) >= top_k:
            break

    return results


# ============================================================================
# Feature 可解释性分析
# ============================================================================

def analyze_feature_interpretability(
    concept: str,
    categories: List[str],
    config: InterpretConfig,
) -> Dict[str, Any]:
    """
    分析单个概念的 feature 可解释性

    参数:
        concept: 概念名
        categories: 要分析的类别列表
        config: 配置

    返回:
        Dict: 分析结果
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing feature interpretability: {concept}")
    logger.info(f"  Categories: {categories}")
    logger.info(f"{'='*70}")

    # 加载 top features
    logger.info("Loading top features...")
    top_features = load_top_features(
        config.feature_dir,
        concept,
        top_k=config.top_k_features,
    )

    logger.info(f"  Found {len(top_features)} top features")

    # 加载各类别的 latent 数据
    logger.info("Loading latents...")
    all_latent_data = []

    for category in categories:
        try:
            data = load_latents(config.latent_dir, category)
            all_latent_data.extend(data)
            logger.info(f"  Loaded {len(data)} records from {category}")
        except FileNotFoundError:
            logger.warning(f"  Category not found: {category}")

    if not all_latent_data:
        logger.error("No latent data found!")
        return {}

    # 分析每个 feature
    logger.info("Analyzing features...")
    feature_analysis = []

    for i, f_info in enumerate(top_features):
        feature_id = f_info["feature_id"]
        cohen_d = f_info["cohen_d"]

        logger.info(f"\n  Feature {feature_id} (d={cohen_d:.3f}):")

        # 提取激活
        activations = extract_feature_activations(
            all_latent_data,
            feature_id,
            config.d_hidden,
        )

        if not activations:
            logger.info("    No activations found")
            continue

        # 获取 top prompts
        top_prompts = get_top_activation_prompts(
            activations,
            top_k=config.top_k_prompts,
            sort_by="mean",
        )

        # 打印 top prompts
        for j, p_info in enumerate(top_prompts[:5]):
            logger.info(f"    #{j+1}: mean={p_info['mean_activation']:.3f} | {p_info['prompt'][:60]}...")

        feature_analysis.append({
            "feature_id": feature_id,
            "cohen_d": cohen_d,
            "abs_cohen_d": f_info["abs_cohen_d"],
            "mean_pos": f_info.get("mean_pos", 0),
            "mean_neg": f_info.get("mean_neg", 0),
            "freq_pos": f_info.get("freq_pos", 0),
            "n_prompts_with_activation": len(activations),
            "top_prompts": top_prompts,
        })

    # 保存结果
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {
        "concept": concept,
        "categories": categories,
        "n_features_analyzed": len(feature_analysis),
        "top_k_prompts_per_feature": config.top_k_prompts,
        "features": feature_analysis,
    }

    result_file = output_path / f"{concept}_feature_interpret.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved interpretability analysis to {result_file}")

    return result


# ============================================================================
# Feature 语义标注模板
# ============================================================================

def generate_annotation_template(
    concept: str,
    feature_analysis: List[Dict[str, Any]],
    output_dir: str,
):
    """
    生成人工标注模板

    用于记录 feature 的语义含义
    """
    output_path = Path(output_dir)
    template_file = output_path / f"{concept}_annotation_template.txt"

    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(f"# {concept.upper()} Feature Semantic Annotation Template\n\n")
        f.write("# Instructions: Fill in the 'semantic' field for each feature\n")
        f.write("# Example semantic labels: nudity, blood, weapon, violence, etc.\n\n")

        for fa in feature_analysis:
            f.write(f"## Feature {fa['feature_id']} (Cohen's d = {fa['cohen_d']:.3f})\n\n")
            f.write(f"Top prompts:\n")

            for i, p_info in enumerate(fa['top_prompts'][:5]):
                f.write(f"  {i+1}. {p_info['prompt']}\n")

            f.write(f"\nSemantic: [FILL IN]\n")
            f.write(f"Confidence: [high/medium/low]\n")
            f.write(f"\n" + "-"*60 + "\n\n")

    logger.info(f"Generated annotation template: {template_file}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Feature Interpretability Analysis")

    # 路径配置
    parser.add_argument("--latent_dir", type=str, default="./outputs/layer29_latents",
                        help="Latent directory")
    parser.add_argument("--feature_dir", type=str, default="./outputs/concept_features",
                        help="Feature analysis directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/feature_interpret",
                        help="Output directory")

    # SAE 配置
    parser.add_argument("--d_hidden", type=int, default=12288,
                        help="SAE hidden dimension")

    # 分析配置
    parser.add_argument("--top_k_prompts", type=int, default=10,
                        help="Top prompts per feature")
    parser.add_argument("--top_k_features", type=int, default=20,
                        help="Number of top features to analyze")

    # 概念
    parser.add_argument("--concepts", type=str, default="sex,violence",
                        help="Concepts to analyze (comma-separated)")

    # 类别
    parser.add_argument("--categories", type=str, default="all",
                        help="Categories to use for prompt retrieval")

    args = parser.parse_args()

    # 创建配置
    config = InterpretConfig(
        latent_dir=args.latent_dir,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        d_hidden=args.d_hidden,
        top_k_prompts=args.top_k_prompts,
        top_k_features=args.top_k_features,
    )

    # 概念和类别定义
    concept_categories = {
        "sex": ["sex_positive", "sex_negative"],
        "violence": ["violence_positive", "violence_negative"],
    }

    # 要分析的概念
    concepts = [c.strip() for c in args.concepts.split(",")]

    results = {}

    for concept in concepts:
        if concept not in concept_categories:
            logger.warning(f"Unknown concept: {concept}")
            continue

        categories = concept_categories[concept]

        if args.categories != "all":
            categories = [c.strip() for c in args.categories.split(",")]

        result = analyze_feature_interpretability(concept, categories, config)
        results[concept] = result

        # 生成标注模板
        if result and "features" in result:
            generate_annotation_template(
                concept,
                result["features"],
                config.output_dir,
            )

    logger.info("\nFeature interpretability analysis completed!")


if __name__ == "__main__":
    main()
