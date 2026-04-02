"""
SAE可解释性分析模块

功能：
1. concept_extractor.py - 通过正负提示词对比提取概念向量

核心功能：
- 加载已训练的SAE和激活值
- 对比正例（包含概念）和负例（不包含概念）的激活模式
- 提取代表概念的方向向量
- 计算特征选择度和统计信息

使用示例：
    python -m wan.sae.interpretability.concept_extractor \
        --run_dir sae_runs/exp1 \
        --positive_file concepts/violence_positive.txt \
        --negative_file concepts/violence_negative.txt \
        --concept_name violence \
        --hook_layers "15,29" \
        --method mean_diff

概念向量格式：
{
    "concept_name": "violence",
    "concept_vector": [...],  # numpy array saved as .npy
    "layer_key": "block_out.layer15",
    "extraction_method": "mean_diff",
    "statistics": {
        "pos_mean_activation": [...],
        "neg_mean_activation": [...],
        "top_k_features": [...],
        "selectivity": {...}
    }
}
"""

from .concept_extractor import (
    ConceptVector,
    extract_concept_vector_mean_diff,
    extract_concept_vector_contrastive,
    compute_feature_selectivity,
)

__all__ = [
    "ConceptVector",
    "extract_concept_vector_mean_diff",
    "extract_concept_vector_contrastive",
    "compute_feature_selectivity",
]
