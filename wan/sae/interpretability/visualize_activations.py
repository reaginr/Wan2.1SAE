"""
SAE激活值可视化模块 - 生成热力图对比正负例激活模式

功能：
1. 特征级热力图：对比正负例在各个SAE特征上的激活强度
2. 提示词级热力图：查看每个提示词的激活模式
3. 差异热力图：突出显示概念特异性特征
4. 选择度热力图：显示特征的选择度分布

输出格式：
- PNG热力图（matplotlib）
- 可交互HTML（plotly，可选）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 无头模式
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed, heatmap generation will be disabled")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


##########################################################################################
# 参数配置区域
##########################################################################################

# --------------------------- 输入配置 ---------------------------
input_params = {
    "activations_file": "concept_vectors/per_prompt_activations.npz",  # per-prompt激活值文件
    "concept_vector_file": "concept_vectors/nsfw_block_out.layer15.json",  # 概念向量文件（可选）
}

# --------------------------- 可视化配置 ---------------------------
visualization_params = {
    # 热力图类型
    "plot_types": ["mean_comparison", "per_prompt", "difference", "selectivity"],

    # 显示的特征数量（top-k最活跃）
    "top_k_features": 100,

    # 显示的提示词数量
    "max_prompts": 50,

    # 归一化方式: "global", "per_feature", "none"
    "normalize": "per_feature",

    # 颜色映射: "RdBu_r", "viridis", "plasma", "coolwarm"
    "colormap": "RdBu_r",

    # 图片尺寸 (英寸)
    "figsize": (12, 8),

    # DPI
    "dpi": 150,
}

# --------------------------- 输出配置 ---------------------------
output_params = {
    "output_dir": "visualizations",
    "output_prefix": "concept_heatmap",
    "formats": ["png", "pdf"],  # 输出格式
    "save_interactive": True,  # 是否保存交互式HTML
}


##########################################################################################
# 核心代码区域
##########################################################################################

def load_per_prompt_activations(npz_path: str) -> Dict[str, np.ndarray]:
    """
    加载每个提示词的激活值

    返回:
        {
            "pos_activations": [N_pos, d_hidden],
            "neg_activations": [N_neg, d_hidden],
            "pos_mean": [d_hidden],
            "neg_mean": [d_hidden],
            "concept_vector": [d_hidden],
        }
    """
    path = Path(npz_path)
    logger.debug(f"[LOAD] 加载 per-prompt activations: {path}")

    if not path.exists():
        raise FileNotFoundError(f"激活值文件不存在: {path}")

    data = dict(np.load(path))

    logger.info(f"已加载激活值:")
    for key, arr in data.items():
        logger.info(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    return data


def normalize_activations(
    activations: np.ndarray,
    method: str = "per_feature"
) -> np.ndarray:
    """
    归一化激活值

    参数:
        activations: [N, d_hidden] 激活矩阵
        method: "global", "per_feature", "none"
    """
    if method == "none":
        return activations

    elif method == "global":
        # 全局归一化到 [0, 1]
        min_val = activations.min()
        max_val = activations.max()
        if max_val > min_val:
            return (activations - min_val) / (max_val - min_val)
        return activations

    elif method == "per_feature":
        # 每个特征单独归一化
        result = np.zeros_like(activations)
        for i in range(activations.shape[1]):
            col = activations[:, i]
            min_val = col.min()
            max_val = col.max()
            if max_val > min_val:
                result[:, i] = (col - min_val) / (max_val - min_val)
            else:
                result[:, i] = col
        return result

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def select_top_features(
    pos_mean: np.ndarray,
    neg_mean: np.ndarray,
    concept_vector: Optional[np.ndarray],
    top_k: int = 100,
    method: str = "by_concept"
) -> np.ndarray:
    """
    选择top-k最相关的特征

    参数:
        method: "by_concept" (按概念向量大小), "by_activation" (按总激活)
    """
    if method == "by_concept" and concept_vector is not None:
        # 按概念向量绝对值排序
        scores = np.abs(concept_vector)
    else:
        # 按正负例总激活排序
        scores = pos_mean + neg_mean

    # 获取top-k索引
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return top_indices


def plot_mean_comparison(
    pos_mean: np.ndarray,
    neg_mean: np.ndarray,
    concept_vector: np.ndarray,
    top_k: int = 100,
    figsize: Tuple[int, int] = (14, 6),
    colormap: str = "RdBu_r",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    绘制正负例平均激活对比热力图

    显示top-k特征在正例和负例中的平均激活值
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return None

    # 选择top-k特征
    top_indices = select_top_features(pos_mean, neg_mean, concept_vector, top_k)

    pos_mean_top = pos_mean[top_indices]
    neg_mean_top = neg_mean[top_indices]

    # 创建对比矩阵 [2, top_k]
    comparison = np.vstack([pos_mean_top, neg_mean_top])

    fig, axes = plt.subplots(2, 2, figsize=figsize,
                             gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [20, 1]})

    # 主热力图
    ax = axes[0, 0]
    im = ax.imshow(comparison, aspect='auto', cmap=colormap,
                   vmin=comparison.min(), vmax=comparison.max())
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Positive', 'Negative'])
    ax.set_xlabel('Feature Index (sorted by concept strength)')
    ax.set_title(f'Mean Activation Comparison (Top {top_k} Features)')

    # 添加数值标注
    for i in range(2):
        for j in range(top_k):
            text = ax.text(j, i, f'{comparison[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=6)

    # 颜色条
    cbar = fig.colorbar(im, cax=axes[0, 1])
    cbar.set_label('Activation Value')

    # 概念向量条形图
    ax2 = axes[1, 0]
    concept_top = concept_vector[top_indices]
    colors = ['red' if v > 0 else 'blue' for v in concept_top]
    ax2.bar(range(top_k), concept_top, color=colors, alpha=0.7)
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Concept Vector Value')
    ax2.set_title('Concept Vector Values (Red=Positive, Blue=Negative)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 隐藏多余的子图
    axes[1, 1].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=visualization_params["dpi"], bbox_inches='tight')
        logger.info(f"Mean comparison heatmap saved: {output_path}")

    return fig


def plot_per_prompt_heatmap(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    concept_vector: np.ndarray,
    top_k: int = 50,
    max_prompts: int = 50,
    figsize: Tuple[int, int] = (14, 10),
    colormap: str = "viridis",
    normalize: str = "per_feature",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    绘制每个提示词的激活热力图

    显示每个提示词在每个特征上的激活值
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return None

    # 选择top-k特征
    pos_mean = pos_activations.mean(axis=0)
    neg_mean = neg_activations.mean(axis=0)
    top_indices = select_top_features(pos_mean, neg_mean, concept_vector, top_k)

    # 限制提示词数量
    pos_activations = pos_activations[:max_prompts, top_indices]
    neg_activations = neg_activations[:max_prompts, top_indices]

    # 归一化
    pos_norm = normalize_activations(pos_activations, method=normalize)
    neg_norm = normalize_activations(neg_activations, method=normalize)

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # 正例热力图
    im1 = axes[0].imshow(pos_norm, aspect='auto', cmap=colormap)
    axes[0].set_title(f'Positive Prompts Activation (Top {top_k} Features, First {max_prompts} Prompts)')
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Prompt Index')
    plt.colorbar(im1, ax=axes[0])

    # 负例热力图
    im2 = axes[1].imshow(neg_norm, aspect='auto', cmap=colormap)
    axes[1].set_title(f'Negative Prompts Activation (Top {top_k} Features, First {max_prompts} Prompts)')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Prompt Index')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=visualization_params["dpi"], bbox_inches='tight')
        logger.info(f"Per-prompt heatmap saved: {output_path}")

    return fig


def plot_difference_heatmap(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    concept_vector: np.ndarray,
    top_k: int = 50,
    max_prompts: int = 50,
    figsize: Tuple[int, int] = (14, 8),
    colormap: str = "RdBu_r",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    绘制差异热力图：显示正负例激活差异

    红色 = 正例激活更强
    蓝色 = 负例激活更强
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return None

    # 选择top-k特征
    pos_mean = pos_activations.mean(axis=0)
    neg_mean = neg_activations.mean(axis=0)
    top_indices = select_top_features(pos_mean, neg_mean, concept_vector, top_k)

    # 限制提示词数量
    pos_activations = pos_activations[:max_prompts, top_indices]
    neg_activations = neg_activations[:max_prompts, top_indices]

    # 计算差异
    difference = pos_activations - neg_activations

    # 合并显示 [2*N, top_k]
    combined = np.vstack([difference, -difference])

    fig, ax = plt.subplots(figsize=figsize)

    vmax = np.abs(combined).max()
    im = ax.imshow(combined, aspect='auto', cmap=colormap, vmin=-vmax, vmax=vmax)

    ax.set_title(f'Activation Difference: Positive - Negative (Top {top_k} Features)')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Prompt Index')

    # 添加分隔线
    ax.axhline(y=max_prompts - 0.5, color='white', linewidth=2)
    ax.text(top_k + 2, max_prompts // 2, 'Pos - Neg', rotation=90, va='center')
    ax.text(top_k + 2, max_prompts + max_prompts // 2, 'Neg - Pos', rotation=90, va='center')

    plt.colorbar(im, ax=ax, label='Activation Difference')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=visualization_params["dpi"], bbox_inches='tight')
        logger.info(f"Difference heatmap saved: {output_path}")

    return fig


def plot_selectivity_heatmap(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    concept_vector: np.ndarray,
    top_k: int = 50,
    figsize: Tuple[int, int] = (14, 6),
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    绘制选择度热力图

    选择度 = P(z_i > 0 | positive) - P(z_i > 0 | negative)
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return None

    # 选择top-k特征
    pos_mean = pos_activations.mean(axis=0)
    neg_mean = neg_activations.mean(axis=0)
    top_indices = select_top_features(pos_mean, neg_mean, concept_vector, top_k)

    # 计算选择度
    pos_binary = (pos_activations[:, top_indices] > 0.01).astype(float)
    neg_binary = (neg_activations[:, top_indices] > 0.01).astype(float)

    pos_freq = pos_binary.mean(axis=0)
    neg_freq = neg_binary.mean(axis=0)
    selectivity = pos_freq - neg_freq

    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 正例激活频率
    im1 = axes[0].imshow(pos_freq.reshape(1, -1), aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[0].set_title('Positive Activation Frequency')
    axes[0].set_xlabel('Feature Index')
    axes[0].set_yticks([])
    plt.colorbar(im1, ax=axes[0])

    # 负例激活频率
    im2 = axes[1].imshow(neg_freq.reshape(1, -1), aspect='auto', cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Negative Activation Frequency')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_yticks([])
    plt.colorbar(im2, ax=axes[1])

    # 选择度
    vmax = max(abs(selectivity.min()), abs(selectivity.max()))
    im3 = axes[2].imshow(selectivity.reshape(1, -1), aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Selectivity (Pos - Neg)')
    axes[2].set_xlabel('Feature Index')
    axes[2].set_yticks([])
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle(f'Feature Selectivity Analysis (Top {top_k} Features)')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=visualization_params["dpi"], bbox_inches='tight')
        logger.info(f"Selectivity heatmap saved: {output_path}")

    return fig


def generate_all_visualizations(
    activations_file: str,
    output_dir: str,
    output_prefix: str = "concept",
    plot_types: List[str] = None,
    top_k_features: int = 100,
    max_prompts: int = 50,
    colormap: str = "RdBu_r",
    normalize: str = "per_feature",
    formats: List[str] = None,
) -> Dict[str, str]:
    """
    生成所有类型的可视化

    返回:
        {plot_type: output_path}
    """
    if plot_types is None:
        plot_types = visualization_params["plot_types"]
    if formats is None:
        formats = output_params["formats"]

    # 加载数据
    data = load_per_prompt_activations(activations_file)
    pos_activations = data["pos_activations"]
    neg_activations = data["neg_activations"]
    pos_mean = data.get("pos_mean", pos_activations.mean(axis=0))
    neg_mean = data.get("neg_mean", neg_activations.mean(axis=0))
    concept_vector = data.get("concept_vector", pos_mean - neg_mean)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = {}

    # 生成各类热力图
    plot_functions = {
        "mean_comparison": lambda: plot_mean_comparison(
            pos_mean, neg_mean, concept_vector,
            top_k=top_k_features, colormap=colormap
        ),
        "per_prompt": lambda: plot_per_prompt_heatmap(
            pos_activations, neg_activations, concept_vector,
            top_k=min(top_k_features, 50), max_prompts=max_prompts,
            colormap="viridis", normalize=normalize
        ),
        "difference": lambda: plot_difference_heatmap(
            pos_activations, neg_activations, concept_vector,
            top_k=min(top_k_features, 50), max_prompts=max_prompts,
            colormap=colormap
        ),
        "selectivity": lambda: plot_selectivity_heatmap(
            pos_activations, neg_activations, concept_vector,
            top_k=top_k_features
        ),
    }

    for plot_type in plot_types:
        if plot_type not in plot_functions:
            logger.warning(f"Unknown plot type: {plot_type}")
            continue

        logger.info(f"Generating {plot_type} visualization...")

        try:
            fig = plot_functions[plot_type]()

            if fig is not None:
                for fmt in formats:
                    output_path = output_dir / f"{output_prefix}_{plot_type}.{fmt}"
                    fig.savefig(output_path, dpi=visualization_params["dpi"], bbox_inches='tight')
                    logger.info(f"Saved: {output_path}")
                    generated_files[plot_type] = str(output_path)

                plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to generate {plot_type}: {e}")
            continue

    return generated_files


def main():
    parser = argparse.ArgumentParser(description="Visualize SAE per-prompt activations as heatmaps")
    parser.add_argument("--activations_file", type=str, default=input_params["activations_file"],
                        help="Path to per_prompt_activations.npz file")
    parser.add_argument("--output_dir", type=str, default=output_params["output_dir"],
                        help="Output directory for visualizations")
    parser.add_argument("--output_prefix", type=str, default="concept",
                        help="Prefix for output file names")
    parser.add_argument("--plot_types", type=str, default="mean_comparison,per_prompt,difference,selectivity",
                        help="Comma-separated list of plot types to generate")
    parser.add_argument("--top_k_features", type=int, default=visualization_params["top_k_features"],
                        help="Number of top features to display")
    parser.add_argument("--max_prompts", type=int, default=visualization_params["max_prompts"],
                        help="Maximum number of prompts to display")
    parser.add_argument("--colormap", type=str, default=visualization_params["colormap"],
                        help="Colormap for heatmaps")
    parser.add_argument("--normalize", type=str, default=visualization_params["normalize"],
                        help="Normalization method: global, per_feature, none")
    parser.add_argument("--formats", type=str, default="png,pdf",
                        help="Comma-separated output formats")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # 检查依赖
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required for visualization. Install with: pip install matplotlib")
        sys.exit(1)

    # 解析参数
    plot_types = [t.strip() for t in args.plot_types.split(",")]
    formats = [f.strip() for f in args.formats.split(",")]

    # 生成可视化
    try:
        generated = generate_all_visualizations(
            activations_file=args.activations_file,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            plot_types=plot_types,
            top_k_features=args.top_k_features,
            max_prompts=args.max_prompts,
            colormap=args.colormap,
            normalize=args.normalize,
            formats=formats,
        )

        logger.info("=" * 60)
        logger.info("Visualization complete!")
        for plot_type, path in generated.items():
            logger.info(f"  {plot_type}: {path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()
