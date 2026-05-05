"""
PCA + t-SNE 全流程全自动优化脚本
严格遵循 scikit-learn 官方规范与 t-SNE 专业调优规则

作者: Claude Code
版本: 2.0
规范: scikit-learn 1.3+ 官方最佳实践
"""

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import numpy.typing as npt

# 设置项目路径
_PROJECT_ROOT = str(Path(__file__).parent)
sys.path.insert(0, _PROJECT_ROOT)

# 导入项目内的激活值读取模块
try:
    from wan.sae.interpretability.activation_io import ActivationIO
except ImportError:
    class SimpleActivationIO:
        """简化版激活值读取器（当主模块不可用时）"""
        def __init__(self, root_dir: str):
            self.root = Path(root_dir)

        def load_activations(self, layer_type: str, layer_idx: int,
                             category: str, polarity: str, mmap: bool = True):
            path = self.root / f"{layer_type}_layer{layer_idx}" / category / polarity / "activations.npy"
            if not path.exists():
                return None
            return np.load(path, mmap_mode='r' if mmap else None)

        def load_metadata(self, layer_type: str, layer_idx: int,
                          category: str, polarity: str):
            path = self.root / f"{layer_type}_layer{layer_idx}" / category / polarity / "metadata.json"
            if not path.exists():
                return []
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

    ActivationIO = SimpleActivationIO

# scikit-learn 导入
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 可视化
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 忽略警告以保持输出整洁
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class TSNEOptimizationResult:
    """t-SNE 优化结果数据类（符合 sklearn 规范）"""
    # 最优参数
    pca_dim: int
    perplexity: float
    learning_rate: Union[str, float]
    n_iter: int
    early_exaggeration: float
    init: str
    random_state: int

    # 优化结果
    silhouette_score: float
    embeddings: np.ndarray
    labels: np.ndarray
    kl_divergence: float
    n_iter_converged: int

    # 数据信息
    n_samples: int
    n_features_original: int
    n_features_reduced: int

    # 辅助信息
    explained_variance_ratio: float = 0.0
    trustworthiness_score: Optional[float] = None


@dataclass
class OptimizationConfig:
    """优化配置参数（严格遵循 sklearn 规范）"""
    # PCA 搜索空间
    pca_dims: List[int] = field(default_factory=lambda: [20, 50, 100])

    # t-SNE 核心参数
    n_components: int = 2
    perplexities: List[float] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30])
    learning_rates: List[Union[str, float]] = field(default_factory=lambda: ['auto', 10, 100, 200, 500, 1000])
    n_iter: int = 2000
    early_exaggerations: List[float] = field(default_factory=lambda: [12.0, 15.0, 20.0])

    # 固定参数
    init: str = 'pca'  # 强制使用 PCA 初始化
    random_state: int = 42
    metric: str = 'euclidean'
    n_jobs: int = -1

    # 约束条件
    max_perplexity_ratio: float = 1/3  # perplexity < n_samples / 3


def parse_layer_key(key: str) -> Tuple[str, int]:
    """解析层key，返回 (layer_type, layer_idx)"""
    if "_layer" not in key:
        raise ValueError(f"无效的 layer_key: {key}，期望格式如 'sae_layer15'")
    parts = key.split("_layer")
    return parts[0], int(parts[1])


def load_binary_classification_data(
    io: ActivationIO,
    layer_type: str,
    layer_idx: int,
    category: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    加载二分类数据（正负样本）

    Parameters
    ----------
    io : ActivationIO
        激活值读取器
    layer_type : str
        层类型（如 'sae'）
    layer_idx : int
        层索引
    category : str
        概念类别

    Returns
    -------
    features : np.ndarray, shape (n_samples, n_features)
        合并后的特征矩阵
    labels : np.ndarray, shape (n_samples,)
        二分类标签（1=正样本, 0=负样本）
    metadata : dict
        数据元信息
    """
    logger.info("=" * 70)
    logger.info("【阶段1】数据加载")
    logger.info("=" * 70)

    # 加载正负样本
    pos_acts = io.load_activations(layer_type, layer_idx, category, "pos", mmap=True)
    neg_acts = io.load_activations(layer_type, layer_idx, category, "neg", mmap=True)

    if pos_acts is None or neg_acts is None:
        raise ValueError(f"数据不存在: {layer_type}_layer{layer_idx}/{category}")

    # 处理 [N, 7, D] 池化格式
    if pos_acts.ndim == 3 and pos_acts.shape[1] == 7:
        logger.info(f"检测到池化格式 [N, 7, D]，提取均值统计量")
        pos_features = np.array(pos_acts[:, 0, :])  # mean
        neg_features = np.array(neg_acts[:, 0, :])
    else:
        pos_features = np.array(pos_acts)
        neg_features = np.array(neg_acts)

    n_pos, n_neg = len(pos_features), len(neg_features)
    n_samples = n_pos + n_neg
    n_features = pos_features.shape[1]

    logger.info(f"正样本: {pos_features.shape}")
    logger.info(f"负样本: {neg_features.shape}")

    # 合并数据
    features = np.vstack([pos_features, neg_features])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    metadata = {
        'n_samples': n_samples,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'n_features': n_features,
        'class_distribution': {'positive': n_pos, 'negative': n_neg}
    }

    logger.info(f"合并后: {features.shape}, 标签分布: pos={n_pos}, neg={n_neg}")

    return features, labels, metadata


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    标准化特征（Z-score 标准化）

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        原始特征

    Returns
    -------
    X_scaled : np.ndarray
        标准化后的特征
    scaler : StandardScaler
        拟合的 scaler 对象
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"标准化: mean=[{X_scaled.mean():.4f}], std=[{X_scaled.std():.4f}]")

    return X_scaled, scaler


def pca_preprocess(
    X: np.ndarray,
    n_components: int,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA, float]:
    """
    PCA 预降维

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        标准化后的特征
    n_components : int
        目标维度
    random_state : int
        随机种子

    Returns
    -------
    X_reduced : np.ndarray
        降维后的特征
    pca : PCA
        拟合的 PCA 对象
    explained_var : float
        保留的方差比例
    """
    n_components = min(n_components, X.shape[0], X.shape[1])

    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    explained_var = np.sum(pca.explained_variance_ratio_)

    logger.info(f"PCA: {X.shape[1]} -> {n_components} 维, 保留方差: {explained_var:.2%}")

    return X_reduced, pca, explained_var


def compute_trustworthiness(
    X_original: np.ndarray,
    X_embedded: np.ndarray,
    n_neighbors: int = 5
) -> float:
    """
    计算 Trustworthiness 指标（高维-低维邻域保持性）

    Parameters
    ----------
    X_original : np.ndarray
        原始高维数据
    X_embedded : np.ndarray
        低维嵌入
    n_neighbors : int
        邻居数量

    Returns
    -------
    trustworthiness : float
        [0, 1] 区间，越接近1表示邻域保持越好
    """
    n_samples = X_original.shape[0]
    n_neighbors = min(n_neighbors, n_samples - 1)

    # 高维空间中的邻居
    nn_high = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
    nn_high.fit(X_original)
    neighbors_high = nn_high.kneighbors(return_distance=False)[:, 1:]

    # 低维空间中的邻居
    nn_low = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
    nn_low.fit(X_embedded)
    neighbors_low = nn_low.kneighbors(return_distance=False)[:, 1:]

    # 计算 Trustworthiness
    trust_sum = 0.0
    for i in range(n_samples):
        high_set = set(neighbors_high[i])
        low_set = set(neighbors_low[i])

        # 在低维是邻居但在高维不是邻居的点
        violations = low_set - high_set
        for j in violations:
            # 找到 j 在高维中的排名
            rank = np.where(neighbors_high[i] == j)[0]
            if len(rank) == 0:
                rank = n_neighbors
            else:
                rank = rank[0]
            trust_sum += max(0, rank - n_neighbors)

    trustworthiness = 1.0 - (2.0 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))) * trust_sum

    return trustworthiness


def optimize_tsne_parameters(
    features: np.ndarray,
    labels: np.ndarray,
    config: OptimizationConfig,
    data_info: Dict[str, Any]
) -> TSNEOptimizationResult:
    """
    全自动 t-SNE 参数优化（严格遵循 sklearn 规范）

    Parameters
    ----------
    features : np.ndarray
        原始高维特征
    labels : np.ndarray
        二分类标签
    config : OptimizationConfig
        优化配置
    data_info : dict
        数据信息

    Returns
    -------
    result : TSNEOptimizationResult
        最优优化结果
    """
    logger.info("\n" + "=" * 70)
    logger.info("【阶段2】t-SNE 全自动参数优化")
    logger.info("=" * 70)

    n_samples = data_info['n_samples']

    # 根据样本量调整 perplexity 上限
    max_perplexity = min(30, int(n_samples * config.max_perplexity_ratio) - 1)
    valid_perplexities = [p for p in config.perplexities if p < max_perplexity]

    logger.info(f"样本数: {n_samples}, Perplexity 搜索范围: {valid_perplexities}")
    logger.info(f"PCA 维度搜索: {config.pca_dims}")
    logger.info(f"Learning rate 搜索: {config.learning_rates}")
    logger.info(f"Early exaggeration 搜索: {config.early_exaggerations}")
    logger.info(f"强制使用 init='{config.init}'")
    logger.info("=" * 70)

    # 存储所有结果
    results_table = []
    best_result = None
    best_silhouette = -1.0

    total_iterations = len(config.pca_dims) * len(valid_perplexities) * len(config.learning_rates) * len(config.early_exaggerations)
    current = 0

    # 标准化原始特征（只需执行一次）
    features_scaled, _ = standardize_features(features)

    for pca_dim in config.pca_dims:
        logger.info(f"\n{'─' * 70}")
        logger.info(f"【PCA 维度: {pca_dim}】")
        logger.info(f"{'─' * 70}")

        # PCA 降维
        try:
            features_reduced, pca_model, explained_var = pca_preprocess(
                features_scaled, pca_dim, config.random_state
            )
        except Exception as e:
            logger.warning(f"PCA {pca_dim} 失败: {e}")
            continue

        for perplexity in valid_perplexities:
            for lr in config.learning_rates:
                for ee in config.early_exaggerations:
                    current += 1
                    lr_str = f"'{lr}'" if isinstance(lr, str) else f"{lr}"
                    logger.info(f"\n[{current}/{total_iterations}] Testing: PCA={pca_dim}, "
                              f"perplexity={perplexity}, lr={lr_str}, ee={ee}")

                    try:
                        # 执行 t-SNE
                        tsne = TSNE(
                            n_components=config.n_components,
                            perplexity=perplexity,
                            learning_rate=lr,
                            n_iter=config.n_iter,
                            early_exaggeration=ee,
                            init=config.init,
                            random_state=config.random_state,
                            metric=config.metric,
                            n_jobs=config.n_jobs,
                            verbose=0
                        )

                        embeddings = tsne.fit_transform(features_reduced)

                        # 计算评估指标
                        sil_score = silhouette_score(embeddings, labels)
                        kl_div = tsne.kl_divergence_
                        n_iter_conv = tsne.n_iter_

                        # 计算 Trustworthiness（邻域保持性）
                        trust_score = compute_trustworthiness(features_reduced, embeddings, n_neighbors=5)

                        logger.info(f"  ✓ Silhouette: {sil_score:.6f} | "
                                  f"KL Div: {kl_div:.4f} | Trust: {trust_score:.4f} | "
                                  f"Iter: {n_iter_conv}")

                        # 记录结果
                        result_entry = {
                            'pca_dim': pca_dim,
                            'perplexity': perplexity,
                            'learning_rate': lr,
                            'early_exaggeration': ee,
                            'silhouette_score': sil_score,
                            'kl_divergence': kl_div,
                            'trustworthiness': trust_score,
                            'n_iter_converged': n_iter_conv,
                            'explained_variance': explained_var
                        }
                        results_table.append(result_entry)

                        # 更新最优结果（以 Silhouette Score 为首要指标）
                        if sil_score > best_silhouette:
                            best_silhouette = sil_score
                            best_result = TSNEOptimizationResult(
                                pca_dim=pca_dim,
                                perplexity=perplexity,
                                learning_rate=lr,
                                n_iter=config.n_iter,
                                early_exaggeration=ee,
                                init=config.init,
                                random_state=config.random_state,
                                silhouette_score=sil_score,
                                embeddings=embeddings,
                                labels=labels,
                                kl_divergence=kl_div,
                                n_iter_converged=n_iter_conv,
                                n_samples=n_samples,
                                n_features_original=features.shape[1],
                                n_features_reduced=pca_dim,
                                explained_variance_ratio=explained_var,
                                trustworthiness_score=trust_score
                            )
                            logger.info(f"  ★ 新的最优结果! Silhouette={sil_score:.6f}")

                    except Exception as e:
                        logger.warning(f"  ✗ 失败: {e}")
                        continue

    # 输出结果汇总表
    logger.info("\n" + "=" * 100)
    logger.info("优化结果汇总表（按 Silhouette Score 排序，前10）")
    logger.info("=" * 100)
    logger.info(f"{'Rank':>4} | {'PCA':>4} | {'Perp':>6} | {'LR':>8} | {'EE':>5} | "
                f"{'Silhouette':>10} | {'Trust':>6} | {'KL Div':>8} | {'Status'}")
    logger.info("-" * 100)

    sorted_results = sorted(results_table, key=lambda x: x['silhouette_score'], reverse=True)

    for rank, r in enumerate(sorted_results[:10], 1):
        lr_disp = f"'{r['learning_rate']}'" if isinstance(r['learning_rate'], str) else f"{r['learning_rate']}"
        status = "★ BEST" if r == sorted_results[0] else ""
        logger.info(f"{rank:>4} | {r['pca_dim']:>4} | {r['perplexity']:>6.1f} | "
                   f"{lr_disp:>8} | {r['early_exaggeration']:>5.1f} | "
                   f"{r['silhouette_score']:>10.6f} | {r['trustworthiness']:>6.4f} | "
                   f"{r['kl_divergence']:>8.4f} | {status}")

    logger.info("=" * 100)

    return best_result


def visualize_optimal_result(
    result: TSNEOptimizationResult,
    output_path: str,
    category: str
) -> None:
    """
    绘制最优 t-SNE 可视化结果

    Parameters
    ----------
    result : TSNEOptimizationResult
        优化结果
    output_path : str
        输出路径
    category : str
        概念类别
    """
    logger.info("\n" + "=" * 70)
    logger.info("【阶段3】生成可视化图")
    logger.info("=" * 70)

    fig, ax = plt.subplots(figsize=(14, 12))

    embeddings = result.embeddings
    labels = result.labels

    pos_mask = labels == 1
    neg_mask = labels == 0

    # 定义颜色方案（高对比度）
    pos_color = '#E53935'  # 红色 - 正样本
    neg_color = '#1E88E5'  # 蓝色 - 负样本

    # 绘制散点（负样本在下层）
    scatter_neg = ax.scatter(
        embeddings[neg_mask, 0],
        embeddings[neg_mask, 1],
        c=neg_color,
        alpha=0.7,
        s=100,
        marker='o',
        edgecolors='white',
        linewidths=1.5,
        label=f'Negative (n={np.sum(neg_mask)})',
        zorder=2
    )

    scatter_pos = ax.scatter(
        embeddings[pos_mask, 0],
        embeddings[pos_mask, 1],
        c=pos_color,
        alpha=0.7,
        s=100,
        marker='o',
        edgecolors='white',
        linewidths=1.5,
        label=f'Positive (n={np.sum(pos_mask)})',
        zorder=3
    )

    # 计算类中心
    pos_center = embeddings[pos_mask].mean(axis=0)
    neg_center = embeddings[neg_mask].mean(axis=0)
    center_distance = np.linalg.norm(pos_center - neg_center)

    # 标记类中心
    ax.scatter(*neg_center, c=neg_color, s=500, marker='X',
               edgecolors='black', linewidths=2.5,
               label='Negative Center', zorder=5)
    ax.scatter(*pos_center, c=pos_color, s=500, marker='X',
               edgecolors='black', linewidths=2.5,
               label='Positive Center', zorder=5)

    # 绘制中心连线
    ax.plot([neg_center[0], pos_center[0]],
            [neg_center[1], neg_center[1]],
            'k--', alpha=0.4, linewidth=2.5,
            label=f'Center Distance: {center_distance:.2f}')

    # 设置标题
    title_text = (
        f"Optimized t-SNE Visualization - {category.upper()}\n"
        f"PCA Dim: {result.pca_dim} | Perplexity: {result.perplexity} | "
        f"LR: {result.learning_rate} | EE: {result.early_exaggeration}\n"
        f"Silhouette Score: {result.silhouette_score:.6f} | "
        f"Trustworthiness: {result.trustworthiness_score:.4f}"
    )

    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=13, fontweight='bold')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=13, fontweight='bold')

    # 图例
    ax.legend(loc='best', fontsize=11, framealpha=0.95,
              edgecolor='gray', fancybox=True, shadow=True)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # 添加统计信息框
    stats_text = (
        f"Samples: {result.n_samples} (Pos: {np.sum(pos_mask)}, Neg: {np.sum(neg_mask)})\n"
        f"Original Dim: {result.n_features_original} → Reduced: {result.n_features_reduced}\n"
        f"Explained Variance: {result.explained_variance_ratio:.2%}\n"
        f"KL Divergence: {result.kl_divergence:.4f}\n"
        f"Iterations to Converge: {result.n_iter_converged}\n"
        f"Center Distance: {center_distance:.2f}"
    )

    props = dict(boxstyle='round,pad=0.7', facecolor='wheat',
                 alpha=0.9, edgecolor='gray', linewidth=2)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=props, fontsize=11,
            family='monospace')

    # 添加分离质量评估
    sil = result.silhouette_score
    if sil > 0.25:
        quality = "Good Separation ✓"
        quality_color = 'green'
    elif sil > 0.05:
        quality = "Moderate Separation ~"
        quality_color = 'orange'
    else:
        quality = "Poor Separation ✗"
        quality_color = 'red'

    ax.text(0.98, 0.02, quality, transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='bottom',
            fontsize=14, fontweight='bold', color=quality_color,
            bbox=dict(boxstyle='round', facecolor='white',
                     edgecolor=quality_color, linewidth=2, alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    logger.info(f"可视化图已保存: {output_path}")
    plt.close()


def save_optimization_report(
    result: TSNEOptimizationResult,
    output_dir: Path,
    category: str,
    layer_key: str
) -> None:
    """
    保存优化报告

    Parameters
    ----------
    result : TSNEOptimizationResult
        优化结果
    output_dir : Path
        输出目录
    category : str
        概念类别
    layer_key : str
        层key
    """
    # 保存最优参数（JSON 格式）
    params = {
        "optimization_info": {
            "category": category,
            "layer_key": layer_key,
            "timestamp": str(np.datetime64('now')),
            "scikit_learn_version": "1.3+"
        },
        "optimal_parameters": {
            "pca_dim": result.pca_dim,
            "perplexity": result.perplexity,
            "learning_rate": result.learning_rate if isinstance(result.learning_rate, str) else float(result.learning_rate),
            "n_iter": result.n_iter,
            "early_exaggeration": result.early_exaggeration,
            "init": result.init,
            "random_state": result.random_state,
            "metric": "euclidean"
        },
        "optimization_metrics": {
            "silhouette_score": float(result.silhouette_score),
            "trustworthiness": float(result.trustworthiness_score) if result.trustworthiness_score else None,
            "kl_divergence": float(result.kl_divergence),
            "n_iter_converged": result.n_iter_converged,
            "explained_variance_ratio": float(result.explained_variance_ratio)
        },
        "data_info": {
            "n_samples": result.n_samples,
            "n_pos": int(np.sum(result.labels == 1)),
            "n_neg": int(np.sum(result.labels == 0)),
            "n_features_original": result.n_features_original,
            "n_features_reduced": result.n_features_reduced
        }
    }

    params_path = output_dir / f"{category}_optimal_params.json"
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    logger.info(f"最优参数已保存: {params_path}")

    # 保存降维后的数据
    data_path = output_dir / f"{category}_optimal_tsne.npz"
    np.savez_compressed(
        data_path,
        embeddings=result.embeddings,
        labels=result.labels,
        pca_dim=result.pca_dim,
        perplexity=result.perplexity,
        learning_rate=result.learning_rate if isinstance(result.learning_rate, str) else float(result.learning_rate),
        early_exaggeration=result.early_exaggeration,
        silhouette_score=result.silhouette_score,
        trustworthiness=result.trustworthiness_score,
        kl_divergence=result.kl_divergence,
        explained_variance_ratio=result.explained_variance_ratio
    )
    logger.info(f"降维数据已保存: {data_path}")


def print_final_report(result: TSNEOptimizationResult, output_dir: Path) -> None:
    """输出最终报告"""
    sil = result.silhouette_score

    if sil > 0.5:
        quality = "优秀"
        interpretation = "类别分离清晰，聚类边界明确"
    elif sil > 0.25:
        quality = "良好"
        interpretation = "类别有一定分离，但存在少量重叠"
    elif sil > 0.05:
        quality = "一般"
        interpretation = "类别部分重叠，分离效果有限"
    else:
        quality = "较差"
        interpretation = "类别严重重叠，几乎无法线性分离"

    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        t-SNE 优化完成报告                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 最优参数组合                                                                  ║
║ ──────────────────────────────────────────────────────────────────────────── ║
║   PCA 维度:           {result.pca_dim:<8}  (保留方差: {result.explained_variance_ratio:.2%})
║   Perplexity:         {result.perplexity:<8}  (样本数约束: <{result.n_samples // 3})
║   Learning Rate:      {str(result.learning_rate):<8}
║   Early Exaggeration: {result.early_exaggeration:<8}
║   Init:               {result.init:<8}  (强制 PCA 初始化)
║   n_iter:             {result.n_iter:<8}  (实际收敛: {result.n_iter_converged})
║   Random State:       {result.random_state:<8}
╠══════════════════════════════════════════════════════════════════════════════╣
║ 优化评估指标                                                                  ║
║ ──────────────────────────────────────────────────────────────────────────── ║
║   Silhouette Score:    {result.silhouette_score:.6f}  [{quality}]
║   Trustworthiness:     {result.trustworthiness_score:.4f}  (邻域保持性)
║   KL Divergence:       {result.kl_divergence:.4f}  (收敛质量)
╠══════════════════════════════════════════════════════════════════════════════╣
║ 调优结论                                                                      ║
║ ──────────────────────────────────────────────────────────────────────────── ║
║   {interpretation}
║                                                                              ║
║   说明: Silhouette Score 越接近 1 表示分离越好，越接近 0 表示重叠越严重。
║         当前结果为 {sil:.6f}，属于"{quality}"级别。
╠══════════════════════════════════════════════════════════════════════════════╣
║ 输出文件                                                                      ║
║ ──────────────────────────────────────────────────────────────────────────── ║
║   参数文件: {output_dir.name}/*_optimal_params.json
║   数据文件: {output_dir.name}/*_optimal_tsne.npz
║   可视化图: {output_dir.name}/*_optimal_tsne.png
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description="PCA + t-SNE 全流程全自动优化 (scikit-learn 规范版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python tsne_full_automation.py --category sex --layer_key sae_layer15
    python tsne_full_automation.py --activation_root ./activations --category violence --layer_key sae_layer29
        """
    )
    parser.add_argument("--activation_root", type=str, default="activations",
                        help="激活值根目录 (默认: activations)")
    parser.add_argument("--category", type=str, required=True,
                        help="概念类别 (如 sex, violence)")
    parser.add_argument("--layer_key", type=str, required=True,
                        help="层key (如 sae_layer15, sae_layer29)")
    parser.add_argument("--output_dir", type=str, default="tsne_optimization_output",
                        help="输出目录 (默认: tsne_optimization_output)")

    args = parser.parse_args()

    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "PCA + t-SNE 全自动优化系统" + " " * 28 + "║")
    logger.info("║" + " " * 18 + "基于 scikit-learn 官方规范" + " " * 28 + "║")
    logger.info("╚" + "═" * 78 + "╝")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析层key
    layer_type, layer_idx = parse_layer_key(args.layer_key)

    logger.info(f"\n概念类别: {args.category}")
    logger.info(f"目标层: {args.layer_key} ({layer_type}, idx={layer_idx})")
    logger.info(f"输出目录: {output_dir.absolute()}")

    # 加载IO
    io = ActivationIO(args.activation_root)

    # 加载数据
    features, labels, data_info = load_binary_classification_data(
        io, layer_type, layer_idx, args.category
    )

    # 配置优化参数
    config = OptimizationConfig(
        pca_dims=[20, 50, 100],
        perplexities=[5, 10, 15, 20, 25, 30],
        learning_rates=['auto', 10, 100, 200, 500, 1000],
        early_exaggerations=[12.0, 15.0, 20.0],
        init='pca',  # 强制使用 PCA 初始化
        n_iter=2000,
        random_state=42,
        metric='euclidean'
    )

    # 执行优化
    best_result = optimize_tsne_parameters(features, labels, config, data_info)

    if best_result is None:
        logger.error("优化失败: 未能找到有效的参数组合")
        sys.exit(1)

    # 保存结果
    save_optimization_report(best_result, output_dir, args.category, args.layer_key)

    # 生成可视化
    plot_path = output_dir / f"{args.category}_optimal_tsne.png"
    visualize_optimal_result(best_result, str(plot_path), args.category)

    # 输出最终报告
    print_final_report(best_result, output_dir)

    logger.info("\n✓ 优化流程全部完成!")


if __name__ == "__main__":
    main()
