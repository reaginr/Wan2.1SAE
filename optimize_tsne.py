"""
全自动 t-SNE 优化可视化脚本

功能：
1. 自动读取项目中的激活值数据
2. PCA 预降维（遍历 20/50/100 维）
3. t-SNE 参数网格搜索（perplexity 5-30）
4. 最大化 Silhouette Score
5. 输出最优参数和可视化图

使用方法：
    python optimize_tsne.py --activation_root "activations" --category "sex" --layer_key "sae_layer15"
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

# 设置项目路径
_PROJECT_ROOT = str(Path(__file__).parent)
sys.path.insert(0, _PROJECT_ROOT)

try:
    from wan.sae.interpretability.activation_io import ActivationIO
except ImportError:
    class SimpleActivationIO:
        def __init__(self, root_dir: str):
            self.root = Path(root_dir)
        def load_activations(self, layer_type: str, layer_idx: int, category: str, polarity: str, mmap: bool = True):
            path = self.root / f"{layer_type}_layer{layer_idx}" / category / polarity / "activations.npy"
            if not path.exists():
                return None
            return np.load(path, mmap_mode='r' if mmap else None)
        def load_metadata(self, layer_type: str, layer_idx: int, category: str, polarity: str):
            path = self.root / f"{layer_type}_layer{layer_idx}" / category / polarity / "metadata.json"
            if not path.exists():
                return []
            with open(path, 'r') as f:
                return json.load(f)
        def print_summary(self, category: str):
            logging.info(f"激活值根目录: {self.root}")
    ActivationIO = SimpleActivationIO

# 检查依赖
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    pca_dim: int
    perplexity: float
    learning_rate: str
    silhouette_score: float
    embeddings: np.ndarray
    labels: np.ndarray


def parse_layer_key(key: str) -> Tuple[str, int]:
    """解析层key"""
    if "_layer" not in key:
        raise ValueError(f"无效的layer_key: {key}")
    parts = key.split("_layer")
    return parts[0], int(parts[1])


def load_data(io: ActivationIO, layer_type: str, layer_idx: int, category: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载正负样本数据

    返回:
        features: [N, D] 合并后的特征
        labels: [N] 标签 (1=pos, 0=neg)
    """
    logger.info("=" * 60)
    logger.info("加载数据")
    logger.info("=" * 60)

    # 加载正负样本
    pos_acts = io.load_activations(layer_type, layer_idx, category, "pos", mmap=True)
    neg_acts = io.load_activations(layer_type, layer_idx, category, "neg", mmap=True)

    if pos_acts is None or neg_acts is None:
        raise ValueError(f"数据不存在")

    # 处理 [N, 7, D] 格式，取 mean (第0维)
    if pos_acts.ndim == 3 and pos_acts.shape[1] == 7:
        pos_features = np.array(pos_acts[:, 0, :])
        neg_features = np.array(neg_acts[:, 0, :])
    else:
        pos_features = np.array(pos_acts)
        neg_features = np.array(neg_acts)

    logger.info(f"正样本: {pos_features.shape}")
    logger.info(f"负样本: {neg_features.shape}")

    # 合并
    features = np.vstack([pos_features, neg_features])
    labels = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))])

    logger.info(f"合并后: {features.shape}, 标签分布: pos={sum(labels==1)}, neg={sum(labels==0)}")

    return features, labels


def preprocess_data(features: np.ndarray, pca_dim: int) -> np.ndarray:
    """
    预处理：标准化 + PCA降维

    Args:
        features: [N, D] 原始特征
        pca_dim: PCA目标维度

    返回:
        reduced_features: [N, pca_dim] 降维后的特征
    """
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # PCA降维
    pca = PCA(n_components=pca_dim, random_state=42)
    features_reduced = pca.fit_transform(features_scaled)

    # 计算保留的方差比例
    variance_ratio = sum(pca.explained_variance_ratio_)
    logger.info(f"  PCA {features.shape[1]} -> {pca_dim} 维, 保留方差: {variance_ratio:.2%}")

    return features_reduced


def optimize_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    pca_dims: List[int] = [20, 50, 100],
    perplexities: List[float] = [5, 10, 15, 20, 25, 30],
    n_iter: int = 2000,
    random_state: int = 42,
) -> OptimizationResult:
    """
    全自动 t-SNE 参数优化

    搜索策略：
    1. 遍历 PCA 维度 (20, 50, 100)
    2. 遍历 perplexity (5-30)
    3. 使用 learning_rate='auto'
    4. 最大化 Silhouette Score

    Args:
        features: [N, D] 原始特征
        labels: [N] 标签
        pca_dims: PCA维度列表
        perplexities: perplexity搜索范围
        n_iter: t-SNE迭代次数
        random_state: 随机种子

    返回:
        OptimizationResult: 最优结果
    """
    logger.info("=" * 60)
    logger.info("开始 t-SNE 参数优化")
    logger.info("=" * 60)
    logger.info(f"PCA维度搜索: {pca_dims}")
    logger.info(f"Perplexity搜索: {perplexities}")
    logger.info(f"Learning rate: auto")
    logger.info(f"n_iter: {n_iter}")

    best_result = None
    best_silhouette = -1
    results_table = []

    total_iterations = len(pca_dims) * len(perplexities)
    current = 0

    for pca_dim in pca_dims:
        logger.info(f"\n{'='*60}")
        logger.info(f"PCA 维度: {pca_dim}")
        logger.info(f"{'='*60}")

        # PCA预处理
        features_reduced = preprocess_data(features, pca_dim)

        for perplexity in perplexities:
            current += 1
            logger.info(f"\n[{current}/{total_iterations}] Testing perplexity={perplexity}")

            try:
                # t-SNE降维
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    learning_rate='auto',
                    n_iter=n_iter,
                    random_state=random_state,
                    verbose=0,
                )

                embeddings = tsne.fit_transform(features_reduced)

                # 计算 Silhouette Score
                sil_score = silhouette_score(embeddings, labels)

                logger.info(f"  Silhouette Score: {sil_score:.4f}")

                # 记录结果
                results_table.append({
                    'pca_dim': pca_dim,
                    'perplexity': perplexity,
                    'silhouette': sil_score,
                })

                # 更新最优结果
                if sil_score > best_silhouette:
                    best_silhouette = sil_score
                    best_result = OptimizationResult(
                        pca_dim=pca_dim,
                        perplexity=perplexity,
                        learning_rate='auto',
                        silhouette_score=sil_score,
                        embeddings=embeddings,
                        labels=labels,
                    )
                    logger.info(f"  ★ 新的最优结果!")

            except Exception as e:
                logger.warning(f"  失败: {e}")
                continue

    # 打印结果表格
    logger.info("\n" + "=" * 80)
    logger.info("优化结果汇总")
    logger.info("=" * 80)
    logger.info(f"{'PCA维':>8} | {'Perplexity':>12} | {'Silhouette':>12} | {'状态'}")
    logger.info("-" * 80)
    for r in results_table:
        status = "★ BEST" if (r['pca_dim'] == best_result.pca_dim and
                              r['perplexity'] == best_result.perplexity) else ""
        logger.info(f"{r['pca_dim']:>8} | {r['perplexity']:>12.1f} | {r['silhouette']:>12.4f} | {status}")
    logger.info("=" * 80)

    return best_result


def plot_optimal_result(result: OptimizationResult, output_path: str, category: str):
    """
    绘制最优 t-SNE 结果

    Args:
        result: 优化结果
        output_path: 输出路径
        category: 概念类别
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib 未安装，跳过绘图")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    embeddings = result.embeddings
    labels = result.labels

    pos_mask = labels == 1
    neg_mask = labels == 0

    # 绘制散点
    ax.scatter(
        embeddings[neg_mask, 0],
        embeddings[neg_mask, 1],
        c='blue',
        alpha=0.6,
        s=80,
        label=f'Negative (n={sum(neg_mask)})',
        edgecolors='white',
        linewidths=1,
    )
    ax.scatter(
        embeddings[pos_mask, 0],
        embeddings[pos_mask, 1],
        c='red',
        alpha=0.6,
        s=80,
        label=f'Positive (n={sum(pos_mask)})',
        edgecolors='white',
        linewidths=1,
    )

    # 计算并标记中心
    pos_center = embeddings[pos_mask].mean(axis=0)
    neg_center = embeddings[neg_mask].mean(axis=0)

    ax.scatter(*neg_center, c='blue', s=300, marker='X', edgecolors='black', linewidths=2, label='Neg Center', zorder=5)
    ax.scatter(*pos_center, c='red', s=300, marker='X', edgecolors='black', linewidths=2, label='Pos Center', zorder=5)

    # 添加中心连线
    ax.plot([neg_center[0], pos_center[0]], [neg_center[1], pos_center[1]],
            'k--', alpha=0.3, linewidth=2, label=f'Center Distance')

    # 设置标题和标签
    ax.set_title(
        f"Optimized t-SNE Visualization - {category.upper()}\n"
        f"PCA={result.pca_dim}, Perplexity={result.perplexity}, LR={result.learning_rate}\n"
        f"Silhouette Score: {result.silhouette_score:.4f}",
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加统计信息文本框
    stats_text = (
        f"Samples: {len(labels)}\n"
        f"Pos: {sum(pos_mask)} | Neg: {sum(neg_mask)}\n"
        f"Center Distance: {np.linalg.norm(pos_center - neg_center):.2f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"最优可视化图保存: {output_path}")
    plt.close()


def save_optimal_params(result: OptimizationResult, output_path: str, category: str, layer_key: str):
    """
    保存最优参数

    Args:
        result: 优化结果
        output_path: 输出路径
        category: 概念类别
        layer_key: 层key
    """
    params = {
        "category": category,
        "layer_key": layer_key,
        "optimal_params": {
            "pca_dim": result.pca_dim,
            "perplexity": result.perplexity,
            "learning_rate": result.learning_rate,
            "n_iter": 2000,
            "n_components": 2,
            "random_state": 42,
        },
        "metrics": {
            "silhouette_score": float(result.silhouette_score),
            "num_samples": len(result.labels),
            "num_pos": int(sum(result.labels == 1)),
            "num_neg": int(sum(result.labels == 0)),
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    logger.info(f"最优参数保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="全自动 t-SNE 优化可视化")
    parser.add_argument("--activation_root", type=str, default="activations",
                        help="激活值根目录")
    parser.add_argument("--category", type=str, required=True,
                        help="概念类别 (如 sex, violence)")
    parser.add_argument("--layer_key", type=str, required=True,
                        help="层key (如 sae_layer15)")
    parser.add_argument("--output_dir", type=str, default="optimize_tsne_output",
                        help="输出目录")

    args = parser.parse_args()

    if not HAS_SKLEARN:
        logger.error("请先安装 scikit-learn: pip install scikit-learn")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("全自动 t-SNE 优化可视化")
    logger.info("=" * 60)
    logger.info(f"类别: {args.category}")
    logger.info(f"层: {args.layer_key}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析层key
    layer_type, layer_idx = parse_layer_key(args.layer_key)

    # 加载IO
    io = ActivationIO(args.activation_root)
    io.print_summary(args.category)

    # 加载数据
    features, labels = load_data(io, layer_type, layer_idx, args.category)

    # 执行优化
    best_result = optimize_tsne(
        features=features,
        labels=labels,
        pca_dims=[20, 50, 100],
        perplexities=[5, 10, 15, 20, 25, 30],
        n_iter=2000,
        random_state=42,
    )

    # 保存最优参数
    params_path = output_dir / f"{args.category}_optimal_params.json"
    save_optimal_params(best_result, str(params_path), args.category, args.layer_key)

    # 保存降维后的数据
    data_path = output_dir / f"{args.category}_optimal_tsne.npz"
    np.savez(
        data_path,
        embeddings=best_result.embeddings,
        labels=best_result.labels,
        pca_dim=best_result.pca_dim,
        perplexity=best_result.perplexity,
        silhouette_score=best_result.silhouette_score,
    )
    logger.info(f"降维数据保存: {data_path}")

    # 绘制最优可视化
    if HAS_MATPLOTLIB:
        plot_path = output_dir / f"{args.category}_optimal_tsne.png"
        plot_optimal_result(best_result, str(plot_path), args.category)

    # 输出最终结果
    logger.info("\n" + "=" * 80)
    logger.info("最优参数与结果")
    logger.info("=" * 80)
    logger.info(f"最优 PCA 维度: {best_result.pca_dim}")
    logger.info(f"最优 Perplexity: {best_result.perplexity}")
    logger.info(f"最优 Learning Rate: {best_result.learning_rate}")
    logger.info(f"最高 Silhouette Score: {best_result.silhouette_score:.4f}")
    logger.info("=" * 80)

    print(f"\n{'='*80}")
    print("优化完成!")
    print(f"{'='*80}")
    print(f"最优 PCA 维度: {best_result.pca_dim}")
    print(f"最优 Perplexity: {best_result.perplexity}")
    print(f"最优 Learning Rate: {best_result.learning_rate}")
    print(f"最高 Silhouette Score: {best_result.silhouette_score:.4f}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
