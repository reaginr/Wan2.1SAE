"""
SAE可解释性分析模块

功能：
1. activation_io.py - 统一的激活值文件I/O接口
2. concept_extractor_stage1.py - 阶段一：激活值采集（GPU必需，需加载WanT2V模型）
3. concept_extractor_stage2.py - 阶段二：概念向量提取（CPU即可，纯NumPy运算）
4. concept_extractor.py - 在线提取（旧版，直接处理提示词，不保存中间结果）
5. visualize_activations.py - 生成热力图可视化激活模式
6. visualize_tsne.py - t-SNE降维可视化，验证正负样本聚类效果

两阶段离线分离式设计：

阶段一（concept_extractor_stage1.py）：
- 加载WanT2V（DiT）模型和训练好的SAE
- 配对处理正负提示词，保证样本对齐
- Hook收集DiT隐藏状态 [B, L, C]
- 实时通过SAE编码得到隐状态z [B, L, d_hidden]
- 保存激活值到分层目录结构: sae_layer{idx}/{category}/{polarity}/

阶段二（concept_extractor_stage2.py）：
- 不需要GPU或模型加载
- 使用numpy.memmap流式加载大文件
- 计算mean_diff概念向量: mean(pos) - mean(neg)
- 保存概念向量和统计信息

使用示例：

1. 阶段一：激活值采集（GPU必需）
    python wan/sae/interpretability/concept_extractor_stage1.py \
        --model_path "./Wan2.1-T2V-1.3B" \
        --sae_run_dir "sae_runs/exp1" \
        --pos_prompts "final_cleaned/pos_prompt_3.txt" \
        --neg_prompts "final_cleaned/neg_prompt_3.txt" \
        --category "violence" \
        --output_root "activations" \
        --sae_layers "15,29" \
        --save_dit_layers "15" \
        --sampling_steps 30

2. 阶段二：概念向量提取（CPU即可）
    python wan/sae/interpretability/concept_extractor_stage2.py \
        --activation_root "activations" \
        --category "violence" \
        --layer_key "sae_layer15" \
        --output_dir "concept_vectors" \
        --method "mean_diff" \
        --normalize \
        --min_threshold 0.01

3. 生成热力图可视化：
    python wan/sae/interpretability/visualize_activations.py \
        --activations_file concept_vectors/per_prompt_activations.npz \
        --output_dir visualizations \
        --plot_types mean_comparison,per_prompt,difference,selectivity

4. t-SNE聚类可视化（验证正负样本分离度）：
    python wan/sae/interpretability/visualize_tsne.py \
        --activation_root "activations" \
        --category "violence" \
        --layer_key "sae_layer15" \
        --output_dir "visualizations" \
        --perplexity 30 \
        --n_iter 1000

激活值文件格式（阶段一输出）：
    activations/
    ├── sae_layer15/
    │   └── violence/
    │       ├── pos/
    │       │   ├── activations.npy      # [N, T, L, D]
    │       │   ├── metadata.json        # 样本元信息
    │       │   └── checkpoint.json      # 断点信息
    │       └── neg/
    ├── dit_layer15/                     # DiT状态（可选）
    └── extraction_config.json           # 全局配置

概念向量格式（阶段二输出）：
    concept_vectors/
    ├── violence_sae_layer15.npy          # [d_hidden] 概念向量
    └── violence_sae_layer15.json         # 元信息和统计

JSON格式内容：
{
    "concept_name": "violence",
    "layer_key": "sae_layer15",
    "method": "mean_diff",
    "vector_shape": [6144],
    "norm": 1.0,
    "top_k_features": [{"index": 0, "value": 0.5}, ...],
    "statistics": {
        "pos_count": 200,
        "neg_count": 200,
        "active_features": 150,
        "total_features": 6144,
        "sparsity": 0.9756
    }
}
"""

# 阶段一和阶段二的公共接口
from .activation_io import (
    ActivationIO,
    SampleMetadata,
    ExtractionCheckpoint,
)

from .concept_extractor_stage1 import (
    PairedActivationCollector,
    ActivationStorage,
    parse_layers,
)

from .concept_extractor_stage2 import (
    RunningMean,
    ConceptExtractor,
)

# 旧版在线提取接口
from .concept_extractor import (
    ConceptVector,
    extract_concept_vector_mean_diff,
    extract_concept_vector_contrastive,
    compute_feature_selectivity,
    load_prompts,
    compute_sae_activations,
)

# 可选导入可视化模块（如果依赖不可用则不导入）
try:
    from .visualize_activations import (
        load_per_prompt_activations,
        plot_mean_comparison,
        plot_per_prompt_heatmap,
        plot_difference_heatmap,
        plot_selectivity_heatmap,
        generate_all_visualizations,
    )
    _has_visualization = True
except ImportError:
    _has_visualization = False

# 可选导入t-SNE可视化模块
try:
    from .visualize_tsne import (
        TSNEVisualizer,
        ClusteringMetrics,
    )
    _has_tsne = True
except ImportError:
    _has_tsne = False

# 阶段一和阶段二的核心类
__all__ = [
    # 公共IO接口
    "ActivationIO",
    "SampleMetadata",
    "ExtractionCheckpoint",
    # 阶段一：采集
    "PairedActivationCollector",
    "ActivationStorage",
    "parse_layers",
    # 阶段二：提取
    "RunningMean",
    "ConceptExtractor",
    # 旧版接口
    "ConceptVector",
    "extract_concept_vector_mean_diff",
    "extract_concept_vector_contrastive",
    "compute_feature_selectivity",
    "load_prompts",
    "compute_sae_activations",
]

if _has_visualization:
    __all__.extend([
        "load_per_prompt_activations",
        "plot_mean_comparison",
        "plot_per_prompt_heatmap",
        "plot_difference_heatmap",
        "plot_selectivity_heatmap",
        "generate_all_visualizations",
    ])

if _has_tsne:
    __all__.extend([
        "TSNEVisualizer",
        "ClusteringMetrics",
    ])
