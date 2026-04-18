"""
SAE激活值 t-SNE 可视化模块

功能：
1. 从阶段一采集的激活值中加载正负样本
2. 在时间步和token维度上池化，得到每样本的SAE特征表示
3. 使用t-SNE降维到2D/3D，可视化聚类效果
4. 计算正负样本的分离度指标（Silhouette Score等）

使用示例：
    python wan/sae/interpretability/visualize_tsne.py \
        --activation_root "activations" \
        --category "violence" \
        --layer_key "sae_layer15" \
        --output_dir "visualizations" \
        --perplexity 30 \
        --n_iter 1000

输出：
    visualizations/
    ├── violence_sae_layer15_tsne.png      # 2D散点图
    ├── violence_sae_layer15_metrics.json   # 聚类指标
    └── violence_sae_layer15_data.npz       # 降维后的数据
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wan.sae.interpretability.activation_io import ActivationIO

logger = logging.getLogger(__name__)

# 尝试导入sklearn
try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed, t-SNE visualization will be disabled")

# 尝试导入matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


##########################################################################################
# 参数配置
##########################################################################################

tsne_params = {
    # activation_root: 阶段一输出的激活值根目录
    "activation_root": "activations",

    # category: 概念类别（与阶段一一致）
    "category": "violence",

    # layer_key: 要可视化的层
    # 格式: "sae_layer15" 或 "dit_layer15"
    "layer_key": "sae_layer15",

    # output_dir: 可视化输出目录
    "output_dir": "visualizations",

    # perplexity: t-SNE困惑度（通常5-50）
    # 学术意义: 控制局部邻域大小，小值关注局部结构，大值关注全局结构
    "perplexity": 30,

    # n_iter: 迭代次数
    "n_iter": 1000,

    # learning_rate: 学习率
    "learning_rate": 200.0,

    # random_state: 随机种子（保证可重复）
    "random_state": 42,

    # n_components: 降维维度（2或3）
    "n_components": 2,

    # pca_components: 预降维到多少维（加速t-SNE，None表示不预降维）
    # 学术意义: SAE特征维度高（如6144），先用PCA降到50-100维可大幅加速t-SNE
    "pca_components": 100,

    # sample_limit: 最多采样多少样本（None表示全部）
    # 用于大数据集的快速预览
    "sample_limit": None,

    # batch_size: 流式处理批次大小
    "batch_size": 32,
}


##########################################################################################
# t-SNE可视化器
##########################################################################################

@dataclass
class ClusteringMetrics:
    """聚类质量指标"""
    silhouette: float  # -1到1，越大越好
    calinski_harabasz: float  # 越大越好
    pos_center: np.ndarray  # 正样本中心
    neg_center: np.ndarray  # 负样本中心
    center_distance: float  # 两中心距离

    def to_dict(self) -> Dict[str, Any]:
        return {
            "silhouette_score": float(self.silhouette),
            "calinski_harabasz_score": float(self.calinski_harabasz),
            "center_distance": float(self.center_distance),
            "pos_center": self.pos_center.tolist(),
            "neg_center": self.neg_center.tolist(),
        }


class TSNEVisualizer:
    """
    t-SNE可视化器

    用于验证正负样本在SAE特征空间中的分离度
    """

    def __init__(
        self,
        io: ActivationIO,
        category: str,
        layer_type: str,
        layer_idx: int,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        learning_rate: float = 200.0,
        random_state: int = 42,
        n_components: int = 2,
        pca_components: Optional[int] = 100,
        sample_limit: Optional[int] = None,
        batch_size: int = 32,
    ):
        self.io = io
        self.category = category
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_components = n_components
        self.pca_components = pca_components
        self.sample_limit = sample_limit
        self.batch_size = batch_size

        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for t-SNE visualization")

    def _load_and_pool_activations(
        self,
        polarity: str,
    ) -> Optional[np.ndarray]:
        """
        加载并池化激活值

        返回: [N, D] 每样本的特征表示（已在时间和token维度上池化）
        """
        acts = self.io.load_activations(
            self.layer_type, self.layer_idx, self.category, polarity, mmap=True
        )

        if acts is None:
            return None

        # acts: [N, T, L, D] -> [N, D]
        N = acts.shape[0]

        # 限制样本数
        if self.sample_limit and N > self.sample_limit:
            indices = np.random.choice(N, self.sample_limit, replace=False)
            acts = acts[indices]
            N = self.sample_limit

        # 流式池化（避免加载全部到内存）
        pooled_list = []
        for i in range(0, N, self.batch_size):
            end_idx = min(i + self.batch_size, N)
            batch = np.array(acts[i:end_idx])  # [B, T, L, D]
            # 在时间和token维度上平均: [B, T, L, D] -> [B, D]
            batch_pooled = batch.reshape(batch.shape[0], -1, batch.shape[-1]).mean(axis=1)
            pooled_list.append(batch_pooled)

        return np.vstack(pooled_list)  # [N, D]

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备数据

        返回:
            features: [N_pos + N_neg, D] 特征矩阵
            labels: [N_pos + N_neg] 标签（1=pos, 0=neg）
        """
        logger.info("=" * 60)
        logger.info("加载并准备数据")
        logger.info("=" * 60)

        # 加载正负样本
        pos_features = self._load_and_pool_activations("pos")
        neg_features = self._load_and_pool_activations("neg")

        if pos_features is None or neg_features is None:
            raise ValueError(f"数据不存在: pos={pos_features is not None}, neg={neg_features is not None}")

        logger.info(f"正样本: {pos_features.shape}")
        logger.info(f"负样本: {neg_features.shape}")

        # 合并
        features = np.vstack([pos_features, neg_features])
        labels = np.concatenate([
            np.ones(len(pos_features)),
            np.zeros(len(neg_features))
        ])

        logger.info(f"合并后: {features.shape}, 标签分布: pos={sum(labels==1)}, neg={sum(labels==0)}")

        return features, labels

    def reduce_dimensionality(self, features: np.ndarray) -> np.ndarray:
        """
        降维

        1. 可选：先用PCA预降维（加速t-SNE）
        2. t-SNE降维到2D/3D
        """
        logger.info("=" * 60)
        logger.info("降维")
        logger.info("=" * 60)

        # 可选：PCA预降维
        if self.pca_components and features.shape[1] > self.pca_components:
            logger.info(f"PCA预降维: {features.shape[1]} -> {self.pca_components}")
            pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            features = pca.fit_transform(features)
            logger.info(f"PCA解释方差比: {pca.explained_variance_ratio_.sum():.2%}")

        # t-SNE降维
        logger.info(f"t-SNE降维: {features.shape[1]} -> {self.n_components}")
        logger.info(f"  perplexity={self.perplexity}, n_iter={self.n_iter}, lr={self.learning_rate}")

        tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            n_iter=self.n_iter,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=1,
            n_jobs=-1,  # 使用所有CPU核心
        )

        embeddings = tsne.fit_transform(features)
        logger.info(f"降维完成: {embeddings.shape}")

        # KL散度（t-SNE的优化目标）
        if hasattr(tsne, 'kl_divergence_'):
            logger.info(f"KL散度: {tsne.kl_divergence_:.4f}")

        return embeddings

    def compute_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> ClusteringMetrics:
        """
        计算聚类质量指标
        """
        logger.info("=" * 60)
        logger.info("计算聚类指标")
        logger.info("=" * 60)

        # Silhouette Score: -1到1，越大表示聚类越好
        silhouette = silhouette_score(embeddings, labels)
        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info(f"  解释: {'良好分离' if silhouette > 0.5 else '有一定分离' if silhouette > 0.25 else '重叠较多'}")

        # Calinski-Harabasz Index: 越大越好
        calinski = calinski_harabasz_score(embeddings, labels)
        logger.info(f"Calinski-Harabasz Index: {calinski:.2f}")

        # 计算正负样本中心距离
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_center = embeddings[pos_mask].mean(axis=0)
        neg_center = embeddings[neg_mask].mean(axis=0)
        center_distance = np.linalg.norm(pos_center - neg_center)
        logger.info(f"中心距离: {center_distance:.4f}")

        return ClusteringMetrics(
            silhouette=silhouette,
            calinski_harabasz=calinski,
            pos_center=pos_center,
            neg_center=neg_center,
            center_distance=center_distance,
        )

    def plot_2d(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        metrics: ClusteringMetrics,
        output_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        绘制2D t-SNE可视化
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib not available")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        pos_mask = labels == 1
        neg_mask = labels == 0

        # 左图：散点图
        ax = axes[0]
        ax.scatter(
            embeddings[neg_mask, 0],
            embeddings[neg_mask, 1],
            c='blue',
            alpha=0.6,
            s=30,
            label=f'Negative (n={sum(neg_mask)})',
            edgecolors='white',
            linewidths=0.5,
        )
        ax.scatter(
            embeddings[pos_mask, 0],
            embeddings[pos_mask, 1],
            c='red',
            alpha=0.6,
            s=30,
            label=f'Positive (n={sum(pos_mask)})',
            edgecolors='white',
            linewidths=0.5,
        )

        # 标记中心
        ax.scatter(*metrics.neg_center, c='blue', s=200, marker='X', edgecolors='black', linewidths=2, label='Neg Center')
        ax.scatter(*metrics.pos_center, c='red', s=200, marker='X', edgecolors='black', linewidths=2, label='Pos Center')

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(f't-SNE Visualization: {self.category}\n({self.layer_type}_layer{self.layer_idx})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 右图：密度图或直方图
        ax = axes[1]

        # 计算每个样本到各自中心的距离
        pos_distances = np.linalg.norm(embeddings[pos_mask] - metrics.pos_center, axis=1)
        neg_distances = np.linalg.norm(embeddings[neg_mask] - metrics.neg_center, axis=1)

        ax.hist(neg_distances, bins=30, alpha=0.6, color='blue', label='Negative', density=True)
        ax.hist(pos_distances, bins=30, alpha=0.6, color='red', label='Positive', density=True)
        ax.axvline(x=metrics.center_distance / 2, color='green', linestyle='--', label='Decision Boundary')
        ax.set_xlabel('Distance to Class Center')
        ax.set_ylabel('Density')
        ax.set_title(f'Distance Distribution\nSilhouette={metrics.silhouette:.3f}, CenterDist={metrics.center_distance:.2f}')
        ax.legend(loc='best')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"保存2D可视化: {output_path}")

        return fig

    def plot_3d(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        output_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        绘制3D t-SNE可视化（保存为多角度视图）
        """
        if not HAS_MATPLOTLIB or self.n_components != 3:
            return None

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(16, 12))

        pos_mask = labels == 1
        neg_mask = labels == 0

        # 多个视角
        elev_azim = [(20, 45), (20, 135), (20, 225), (20, 315)]

        for i, (elev, azim) in enumerate(elev_azim):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')

            ax.scatter(
                embeddings[neg_mask, 0],
                embeddings[neg_mask, 1],
                embeddings[neg_mask, 2],
                c='blue',
                alpha=0.5,
                s=20,
                label='Negative',
            )
            ax.scatter(
                embeddings[pos_mask, 0],
                embeddings[pos_mask, 1],
                embeddings[pos_mask, 2],
                c='red',
                alpha=0.5,
                s=20,
                label='Positive',
            )

            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
            ax.set_title(f'View: elev={elev}, azim={azim}')
            ax.legend()

        plt.suptitle(f'3D t-SNE: {self.category} ({self.layer_type}_layer{self.layer_idx})', fontsize=14)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"保存3D可视化: {output_path}")

        return fig

    def run(
        self,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        运行完整的可视化流程
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 准备数据
        features, labels = self.prepare_data()

        # 降维
        embeddings = self.reduce_dimensionality(features)

        # 计算指标
        metrics = self.compute_metrics(embeddings, labels)

        # 保存结果
        output_name = f"{self.category}_{self.layer_type}_layer{self.layer_idx}"

        # 1. 保存降维后的数据和指标
        data_path = output_dir / f"{output_name}_tsne.npz"
        np.savez(
            data_path,
            embeddings=embeddings,
            labels=labels,
            perplexity=self.perplexity,
            n_iter=self.n_iter,
        )
        logger.info(f"保存降维数据: {data_path}")

        # 2. 保存指标
        metrics_path = output_dir / f"{output_name}_metrics.json"
        result = {
            "category": self.category,
            "layer_key": f"{self.layer_type}_layer{self.layer_idx}",
            "layer_type": self.layer_type,
            "layer_idx": self.layer_idx,
            "num_pos": int(sum(labels == 1)),
            "num_neg": int(sum(labels == 0)),
            "tsne_params": {
                "perplexity": self.perplexity,
                "n_iter": self.n_iter,
                "learning_rate": self.learning_rate,
                "n_components": self.n_components,
                "pca_components": self.pca_components,
            },
            "metrics": metrics.to_dict(),
            "interpretation": {
                "silhouette_good": metrics.silhouette > 0.5,
                "silhouette_moderate": 0.25 < metrics.silhouette <= 0.5,
                "separation_quality": "good" if metrics.silhouette > 0.5 else "moderate" if metrics.silhouette > 0.25 else "poor",
            },
        }
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"保存指标: {metrics_path}")

        # 3. 绘制可视化
        if HAS_MATPLOTLIB:
            if self.n_components == 2:
                fig_path = output_dir / f"{output_name}_tsne.png"
                self.plot_2d(embeddings, labels, metrics, str(fig_path))
            elif self.n_components == 3:
                fig_path = output_dir / f"{output_name}_tsne_3d.png"
                self.plot_3d(embeddings, labels, str(fig_path))

        logger.info("=" * 60)
        logger.info("t-SNE可视化完成")
        logger.info(f"  Silhouette Score: {metrics.silhouette:.4f}")
        logger.info(f"  分离质量: {result['interpretation']['separation_quality']}")
        logger.info("=" * 60)

        return result


##########################################################################################
# 辅助函数
##########################################################################################

def parse_layer_key(key: str) -> Tuple[str, int]:
    """解析层key"""
    if "_layer" not in key:
        raise ValueError(f"无效的layer_key: {key}，应为 'sae_layer15' 或 'dit_layer15'")
    parts = key.split("_layer")
    return parts[0], int(parts[1])


##########################################################################################
# 主流程
##########################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="SAE激活值 t-SNE 可视化 - 验证正负样本聚类效果"
    )

    # 基本参数
    parser.add_argument(
        "--activation_root", type=str, default=tsne_params["activation_root"],
        help="阶段一输出的激活值根目录"
    )
    parser.add_argument(
        "--category", type=str, default=tsne_params["category"],
        help="概念类别"
    )
    parser.add_argument(
        "--layer_key", type=str, default=tsne_params["layer_key"],
        help="要可视化的层，如 'sae_layer15'"
    )
    parser.add_argument(
        "--output_dir", type=str, default=tsne_params["output_dir"],
        help="输出目录"
    )

    # t-SNE参数
    parser.add_argument(
        "--perplexity", type=float, default=tsne_params["perplexity"],
        help="t-SNE困惑度 (5-50)"
    )
    parser.add_argument(
        "--n_iter", type=int, default=tsne_params["n_iter"],
        help="迭代次数"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=tsne_params["learning_rate"],
        help="学习率"
    )
    parser.add_argument(
        "--n_components", type=int, default=tsne_params["n_components"],
        choices=[2, 3],
        help="降维维度 (2或3)"
    )
    parser.add_argument(
        "--pca_components", type=int, default=tsne_params["pca_components"],
        help="PCA预降维维数，None表示不预降维"
    )
    parser.add_argument(
        "--sample_limit", type=int, default=tsne_params["sample_limit"],
        help="最多采样样本数，None表示全部"
    )
    parser.add_argument(
        "--batch_size", type=int, default=tsne_params["batch_size"],
        help="流式处理批次大小"
    )
    parser.add_argument(
        "--random_state", type=int, default=tsne_params["random_state"],
        help="随机种子"
    )

    args = parser.parse_args()

    # 检查依赖
    if not HAS_SKLEARN:
        logger.error("请先安装 scikit-learn: pip install scikit-learn")
        sys.exit(1)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 60)
    logger.info("SAE激活值 t-SNE 可视化")
    logger.info("=" * 60)

    # 解析层key
    layer_type, layer_idx = parse_layer_key(args.layer_key)

    # 初始化IO
    io = ActivationIO(args.activation_root)
    io.print_summary(args.category)

    # 初始化可视化器
    visualizer = TSNEVisualizer(
        io=io,
        category=args.category,
        layer_type=layer_type,
        layer_idx=layer_idx,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        n_components=args.n_components,
        pca_components=args.pca_components,
        sample_limit=args.sample_limit,
        batch_size=args.batch_size,
    )

    # 运行可视化
    result = visualizer.run(args.output_dir)

    # 输出结果摘要
    print(f"\n{'='*60}")
    print("结果摘要")
    print(f"{'='*60}")
    print(f"概念: {result['category']}")
    print(f"层: {result['layer_key']}")
    print(f"样本数: pos={result['num_pos']}, neg={result['num_neg']}")
    print(f"Silhouette Score: {result['metrics']['silhouette_score']:.4f}")
    print(f"  -> 分离质量: {result['interpretation']['separation_quality']}")
    print(f"Calinski-Harabasz: {result['metrics']['calinski_harabasz_score']:.2f}")
    print(f"中心距离: {result['metrics']['center_distance']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
