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

# 将项目根目录添加到 sys.path（支持从任意位置运行）
# 当前文件: wan/sae/interpretability/visualize_tsne.py
# 需要回退4层到项目根目录
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

# 直接导入activation_io，避免触发wan包的其他依赖
try:
    from wan.sae.interpretability.activation_io import ActivationIO
except ImportError:
    # 备用：直接定义简单的IO类
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
            """打印数据摘要"""
            logger.info(f"激活值根目录: {self.root}")
            if not self.root.exists():
                logger.warning(f"目录不存在: {self.root}")
                return
            # 查找所有层
            layer_dirs = list(self.root.glob("*_layer*"))
            logger.info(f"发现 {len(layer_dirs)} 个层目录")
            for layer_dir in layer_dirs:
                cat_dir = layer_dir / category
                if cat_dir.exists():
                    pos_dir = cat_dir / "pos"
                    neg_dir = cat_dir / "neg"
                    pos_npy = pos_dir / "activations.npy" if pos_dir.exists() else None
                    neg_npy = neg_dir / "activations.npy" if neg_dir.exists() else None
                    pos_count = np.load(pos_npy).shape[0] if pos_npy and pos_npy.exists() else 0
                    neg_count = np.load(neg_npy).shape[0] if neg_npy and neg_npy.exists() else 0
                    logger.info(f"  {layer_dir.name}: pos={pos_count}, neg={neg_count}")
    ActivationIO = SimpleActivationIO

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
        # 统计学去噪参数
        winsorization_limits: Tuple[float, float] = (0.01, 0.99),
        use_mad_outlier_removal: bool = True,
        mad_threshold: float = 3.0,
        use_robust_scaler: bool = True,
        use_pca_denoising: bool = True,
        pca_variance_threshold: float = 0.95,
        use_log_transform: bool = False,
        use_gaussian_smooth: bool = False,
        smooth_sigma: float = 1.0,
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

        # 统计学去噪参数
        self.winsorization_limits = winsorization_limits
        self.use_mad_outlier_removal = use_mad_outlier_removal
        self.mad_threshold = mad_threshold
        self.use_robust_scaler = use_robust_scaler
        self.use_pca_denoising = use_pca_denoising
        self.pca_variance_threshold = pca_variance_threshold
        self.use_log_transform = use_log_transform
        self.use_gaussian_smooth = use_gaussian_smooth
        self.smooth_sigma = smooth_sigma

        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for t-SNE visualization")

    # =========================================================================
    # 统计学去噪与鲁棒性方法
    # =========================================================================

    def _winsorize(self, data: np.ndarray, limits: Tuple[float, float] = (0.01, 0.99)) -> np.ndarray:
        """
        Winsorization（缩尾处理）

        将极值限制在特定百分位范围内，保留数据量但减少极值影响。

        Args:
            data: 输入数据 [N, D]
            limits: (下限百分位, 上限百分位)，默认(0.01, 0.99)

        Returns:
            winsorized_data: 处理后的数据
        """
        lower_limit, upper_limit = limits
        lower_bound = np.percentile(data, lower_limit * 100, axis=0, keepdims=True)
        upper_bound = np.percentile(data, upper_limit * 100, axis=0, keepdims=True)
        return np.clip(data, lower_bound, upper_bound)

    def _mad_outlier_detection(self, data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        MAD（Median Absolute Deviation）异常值检测与替换

        使用中位数绝对偏差检测异常值，并用中位数替换。
        比Z-score更鲁棒，不受极端值影响。

        Args:
            data: 输入数据 [N, D]
            threshold: MAD倍数阈值，默认3.0

        Returns:
            cleaned_data: 清洗后的数据
            outlier_mask: 异常值掩码 [N, D]
        """
        median = np.median(data, axis=0, keepdims=True)
        mad = np.median(np.abs(data - median), axis=0, keepdims=True)
        # 避免除以0
        mad = np.where(mad == 0, 1e-10, mad)

        # 计算MAD分数
        modified_z_scores = 0.6745 * (data - median) / mad

        # 检测异常值
        outlier_mask = np.abs(modified_z_scores) > threshold

        # 用中位数替换异常值
        cleaned_data = data.copy()
        cleaned_data[outlier_mask] = np.tile(median, (data.shape[0], 1))[outlier_mask]

        outlier_ratio = outlier_mask.sum() / outlier_mask.size
        logger.info(f"  MAD异常值检测: 发现 {outlier_ratio*100:.2f}% 异常值，已用中位数替换")

        return cleaned_data, outlier_mask

    def _robust_scale(self, data: np.ndarray) -> np.ndarray:
        """
        RobustScaler（鲁棒标准化）

        使用Median和IQR（四分位距）代替Mean和Std，减少异常值影响。
        x_scaled = (x - median) / IQR

        Args:
            data: 输入数据 [N, D]

        Returns:
            scaled_data: 标准化后的数据
        """
        median = np.median(data, axis=0, keepdims=True)
        q1 = np.percentile(data, 25, axis=0, keepdims=True)
        q3 = np.percentile(data, 75, axis=0, keepdims=True)
        iqr = q3 - q1

        # 避免除以0
        iqr = np.where(iqr == 0, 1e-10, iqr)

        scaled_data = (data - median) / iqr
        logger.info(f"  RobustScaler: median范围 [{median.min():.3f}, {median.max():.3f}], IQR范围 [{iqr.min():.3f}, {iqr.max():.3f}]")

        return scaled_data

    def _pca_denoising(self, data: np.ndarray, variance_threshold: float = 0.95) -> np.ndarray:
        """
        PCA去噪

        保留解释大部分方差的主成分，去除噪声成分。

        Args:
            data: 输入数据 [N, D]
            variance_threshold: 保留的方差比例，默认0.95

        Returns:
            denoised_data: 去噪后的数据
        """
        from sklearn.decomposition import PCA

        # 确定保留的成分数
        pca_full = PCA()
        pca_full.fit(data)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        # 执行PCA降维再重构
        pca = PCA(n_components=n_components)
        data_reduced = pca.fit_transform(data)
        denoised_data = pca.inverse_transform(data_reduced)

        logger.info(f"  PCA去噪: {data.shape[1]}维 -> {n_components}维 -> {data.shape[1]}维 (保留{variance_threshold*100:.1f}%方差)")

        return denoised_data

    def _log_transform(self, data: np.ndarray) -> np.ndarray:
        """
        对数变换（log1p）

        对偏态分布进行log1p变换，压缩大值范围。

        Args:
            data: 输入数据 [N, D]

        Returns:
            transformed_data: 变换后的数据
        """
        # 确保数据非负（SAE激活值通常是非负的）
        data_nonneg = np.maximum(data, 0)
        transformed = np.log1p(data_nonneg)
        logger.info(f"  Log1p变换: 原始范围 [{data.min():.3f}, {data.max():.3f}] -> 变换后 [{transformed.min():.3f}, {transformed.max():.3f}]")
        return transformed

    def _gaussian_smooth(self, data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        高斯平滑

        使用高斯滤波减少高频噪声。

        Args:
            data: 输入数据 [N, D]
            sigma: 高斯核标准差

        Returns:
            smoothed_data: 平滑后的数据
        """
        from scipy.ndimage import gaussian_filter1d

        # 对每个样本的特征进行平滑
        smoothed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            smoothed_data[i] = gaussian_filter1d(data[i], sigma=sigma)

        logger.info(f"  高斯平滑: sigma={sigma}")
        return smoothed_data

    def apply_statistical_denoising(self, pos_features: np.ndarray, neg_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用统计学去噪流程

        按顺序应用多种去噪方法：
        1. Winsorization（缩尾处理）
        2. MAD异常值检测与替换
        3. Log变换（可选）
        4. RobustScaler标准化
        5. 高斯平滑（可选）
        6. PCA去噪（可选）

        Args:
            pos_features: 正样本特征 [N_pos, D]
            neg_features: 负样本特征 [N_neg, D]

        Returns:
            pos_cleaned: 清洗后的正样本
            neg_cleaned: 清洗后的负样本
        """
        logger.info("=" * 60)
        logger.info("应用统计学去噪")
        logger.info("=" * 60)

        # 合并数据统一处理
        all_features = np.vstack([pos_features, neg_features])
        n_pos = pos_features.shape[0]

        logger.info(f"原始数据: {all_features.shape}, 范围 [{all_features.min():.3f}, {all_features.max():.3f}]")

        # 1. Winsorization（缩尾处理）
        all_features = self._winsorize(all_features, self.winsorization_limits)
        logger.info(f"Winsorization后: 范围 [{all_features.min():.3f}, {all_features.max():.3f}]")

        # 2. MAD异常值检测与替换
        if self.use_mad_outlier_removal:
            all_features, _ = self._mad_outlier_detection(all_features, self.mad_threshold)
            logger.info(f"MAD清洗后: 范围 [{all_features.min():.3f}, {all_features.max():.3f}]")

        # 3. Log变换（可选）
        if self.use_log_transform:
            all_features = self._log_transform(all_features)

        # 4. RobustScaler标准化
        if self.use_robust_scaler:
            all_features = self._robust_scale(all_features)
            logger.info(f"RobustScaler后: 范围 [{all_features.min():.3f}, {all_features.max():.3f}]")

        # 5. 高斯平滑（可选）
        if self.use_gaussian_smooth:
            all_features = self._gaussian_smooth(all_features, self.smooth_sigma)
            logger.info(f"高斯平滑后: 范围 [{all_features.min():.3f}, {all_features.max():.3f}]")

        # 6. PCA去噪（可选）
        if self.use_pca_denoising:
            all_features = self._pca_denoising(all_features, self.pca_variance_threshold)
            logger.info(f"PCA去噪后: 范围 [{all_features.min():.3f}, {all_features.max():.3f}]")

        # 分离正负样本
        pos_cleaned = all_features[:n_pos]
        neg_cleaned = all_features[n_pos:]

        logger.info("=" * 60)

        return pos_cleaned, neg_cleaned

    def _select_discriminative_features(
        self,
        pos_features: np.ndarray,
        neg_features: np.ndarray,
        top_k: int = 500,
        method: str = "t_test",
    ) -> np.ndarray:
        """
        选择最具区分度的特征维度

        学术意义:
        不是所有SAE维度都对当前概念有区分能力。
        通过统计检验选择区分度最高的维度，可以:
        1. 减少噪音维度的干扰
        2. 提高t-SNE的可解释性
        3. 发现真正编码该概念的SAE特征

        Args:
            pos_features: [N_pos, D] 正样本特征
            neg_features: [N_neg, D] 负样本特征
            top_k: 选择前k个维度
            method: 选择方法 "t_test" | "mean_diff" | "cohen_d"

        Returns:
            selected_features: [N_pos + N_neg, top_k] 筛选后的特征
            selected_indices: [top_k] 选中的维度索引
        """
        from scipy import stats

        logger.info(f"特征选择: {method}, top_k={top_k}")

        if method == "t_test":
            # 独立样本t检验，取负对数p值作为区分度分数
            t_stats, p_values = stats.ttest_ind(pos_features, neg_features, axis=0)
            # 处理nan
            p_values = np.nan_to_num(p_values, nan=1.0)
            scores = -np.log10(p_values + 1e-300)  # 负对数p值，越大越好

        elif method == "mean_diff":
            # 均值差的绝对值
            scores = np.abs(pos_features.mean(axis=0) - neg_features.mean(axis=0))

        elif method == "cohen_d":
            # Cohen's d (效应量)
            mean_diff = pos_features.mean(axis=0) - neg_features.mean(axis=0)
            pooled_std = np.sqrt((pos_features.var(axis=0) + neg_features.var(axis=0)) / 2)
            scores = np.abs(mean_diff) / (pooled_std + 1e-10)

        else:
            raise ValueError(f"未知的选择方法: {method}")

        # 选择top_k维度
        selected_indices = np.argsort(scores)[-top_k:][::-1]

        # 合并正负样本并筛选维度
        all_features = np.vstack([pos_features, neg_features])
        selected_features = all_features[:, selected_indices]

        # 记录统计信息
        logger.info(f"  选中维度数: {top_k}/{len(scores)} ({top_k/len(scores)*100:.1f}%)")
        logger.info(f"  区分度分数范围: [{scores[selected_indices].min():.2f}, {scores[selected_indices].max():.2f}]")
        logger.info(f"  选中维度索引 (前10): {selected_indices[:10]}")

        return selected_features, selected_indices

    def _load_and_pool_activations(
        self,
        polarity: str,
    ) -> Optional[np.ndarray]:
        """
        加载并池化激活值

        支持两种格式：
        1. 实时池化格式: [N, 7, D] - 7个统计量 [mean, std, max, min, median, p95, p05]
        2. 全时间步格式: [N, T, L, D] - 需在线池化

        返回: [N, D] 每样本的特征表示
        """
        acts = self.io.load_activations(
            self.layer_type, self.layer_idx, self.category, polarity, mmap=True
        )

        if acts is None:
            return None

        # 检测数据格式
        if acts.ndim == 3 and acts.shape[1] == 7:
            # 格式1: [N, 7, D] 实时池化统计特征
            # 直接使用第0维 (mean) 作为特征表示
            logger.info(f"  检测到实时池化格式: {acts.shape}, 使用第0维(mean)")
            N = acts.shape[0]

            # 限制样本数
            if self.sample_limit and N > self.sample_limit:
                indices = np.random.choice(N, self.sample_limit, replace=False)
                acts = acts[indices]
                N = self.sample_limit

            # 取mean统计量 (第0维)
            features = np.array(acts[:, 0, :])  # [N, D]
            logger.info(f"  提取mean特征: {features.shape}")
            return features

        elif acts.ndim == 4:
            # 格式2: [N, T, L, D] 全时间步格式
            logger.info(f"  检测到全时间步格式: {acts.shape}, 在线池化")
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

        else:
            raise ValueError(f"未知的激活值格式: {acts.shape}, 期望 [N,7,D] 或 [N,T,L,D]")

    def prepare_data(
        self,
        use_feature_selection: bool = True,
        top_k_features: int = 500,
        selection_method: str = "t_test",
        use_denoising: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        准备数据

        Args:
            use_feature_selection: 是否使用特征选择
            top_k_features: 选择前k个维度
            selection_method: 选择方法 "t_test" | "mean_diff" | "cohen_d"
            use_denoising: 是否应用统计学去噪

        返回:
            features: [N_pos + N_neg, D或top_k] 特征矩阵
            labels: [N_pos + N_neg] 标签（1=pos, 0=neg）
            selected_indices: [top_k] 选中的维度索引（如果使用了特征选择）
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

        # 应用统计学去噪（在特征选择之前）
        if use_denoising:
            pos_features, neg_features = self.apply_statistical_denoising(pos_features, neg_features)

        selected_indices = None

        # 特征选择
        if use_feature_selection and pos_features.shape[1] > top_k_features:
            logger.info("=" * 60)
            logger.info("执行特征选择（筛选高区分度SAE维度）")
            logger.info("=" * 60)
            features, selected_indices = self._select_discriminative_features(
                pos_features, neg_features, top_k_features, selection_method
            )
            labels = np.concatenate([
                np.ones(len(pos_features)),
                np.zeros(len(neg_features))
            ])
        else:
            # 合并所有维度
            features = np.vstack([pos_features, neg_features])
            labels = np.concatenate([
                np.ones(len(pos_features)),
                np.zeros(len(neg_features))
            ])

        logger.info(f"合并后: {features.shape}, 标签分布: pos={sum(labels==1)}, neg={sum(labels==0)}")

        return features, labels, selected_indices

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
        use_feature_selection: bool = True,
        top_k_features: int = 500,
        selection_method: str = "t_test",
        use_denoising: bool = True,
    ) -> Dict[str, Any]:
        """
        运行完整的可视化流程

        Args:
            output_dir: 输出目录
            use_feature_selection: 是否使用特征选择
            top_k_features: 选择前k个维度
            selection_method: 选择方法 "t_test" | "mean_diff" | "cohen_d"
            use_denoising: 是否应用统计学去噪
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 准备数据（带去噪和特征选择）
        features, labels, selected_indices = self.prepare_data(
            use_feature_selection=use_feature_selection,
            top_k_features=top_k_features,
            selection_method=selection_method,
            use_denoising=use_denoising,
        )

        # 降维
        embeddings = self.reduce_dimensionality(features)

        # 计算指标
        metrics = self.compute_metrics(embeddings, labels)

        # 保存结果
        output_name = f"{self.category}_{self.layer_type}_layer{self.layer_idx}"

        # 1. 保存降维后的数据和指标
        data_path = output_dir / f"{output_name}_tsne.npz"
        save_dict = {
            "embeddings": embeddings,
            "labels": labels,
            "perplexity": self.perplexity,
            "n_iter": self.n_iter,
        }
        if selected_indices is not None:
            save_dict["selected_indices"] = selected_indices
        np.savez(data_path, **save_dict)
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
            "denoising": {
                "enabled": use_denoising,
                "winsorization_limits": self.winsorization_limits if use_denoising else None,
                "use_mad_outlier_removal": self.use_mad_outlier_removal if use_denoising else None,
                "mad_threshold": self.mad_threshold if use_denoising else None,
                "use_robust_scaler": self.use_robust_scaler if use_denoising else None,
                "use_pca_denoising": self.use_pca_denoising if use_denoising else None,
                "pca_variance_threshold": self.pca_variance_threshold if use_denoising else None,
                "use_log_transform": self.use_log_transform if use_denoising else None,
                "use_gaussian_smooth": self.use_gaussian_smooth if use_denoising else None,
                "smooth_sigma": self.smooth_sigma if use_denoising else None,
            },
            "feature_selection": {
                "enabled": use_feature_selection,
                "top_k": top_k_features if use_feature_selection else None,
                "method": selection_method if use_feature_selection else None,
                "selected_indices": selected_indices.tolist() if selected_indices is not None else None,
            },
            "tsne_params": {
                "perplexity": self.perplexity,
                "n_iter": self.n_iter,
                "learning_rate": self.learning_rate,
                "n_components": self.n_components,
                "pca_components": self.pca_components,
            },
            "metrics": metrics.to_dict(),
            "interpretation": {
                "silhouette_good": bool(metrics.silhouette > 0.5),
                "silhouette_moderate": bool(0.25 < metrics.silhouette <= 0.5),
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
        if use_feature_selection and selected_indices is not None:
            logger.info(f"  使用特征选择: top {len(selected_indices)} / {6144} 维度")
        logger.info("=" * 60)

        return result

    def run_batch_feature_selection(
        self,
        output_dir: str,
        k_values: List[int] = None,
        selection_method: str = "t_test",
        use_denoising: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """
        批量测试不同数量的特征维度，找出最佳区分组

        Args:
            output_dir: 输出目录
            k_values: 要测试的维度数列表，默认 [400, 350, 300, 250, 200, 150, 100, 50]
            selection_method: 特征选择方法
            use_denoising: 是否应用统计学去噪

        Returns:
            results: {k: result_dict} 每个k值的结果
        """
        if k_values is None:
            k_values = [400, 350, 300, 250, 200, 150, 100, 50]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("批量特征维度测试")
        logger.info("=" * 60)
        logger.info(f"测试维度数: {k_values}")
        logger.info(f"选择方法: {selection_method}")
        logger.info(f"去噪: {'启用' if use_denoising else '禁用'}")
        logger.info("=" * 60)

        # 首先加载所有特征（无选择）
        pos_features = self._load_and_pool_activations("pos")
        neg_features = self._load_and_pool_activations("neg")

        if pos_features is None or neg_features is None:
            raise ValueError("数据不存在")

        logger.info(f"正样本: {pos_features.shape}")
        logger.info(f"负样本: {neg_features.shape}")

        # 应用统计学去噪
        if use_denoising:
            pos_features, neg_features = self.apply_statistical_denoising(pos_features, neg_features)

        # 计算所有维度的分数（只计算一次）
        from scipy import stats
        t_stats, p_values = stats.ttest_ind(pos_features, neg_features, axis=0)
        p_values = np.nan_to_num(p_values, nan=1.0)
        scores = -np.log10(p_values + 1e-300)

        # 按分数排序的维度索引
        sorted_indices = np.argsort(scores)[::-1]

        results = {}
        best_k = None
        best_silhouette = -1

        # 测试每个k值
        for k in k_values:
            logger.info(f"\n{'='*60}")
            logger.info(f"测试 top-{k} 维度")
            logger.info(f"{'='*60}")

            # 选择top-k维度
            selected_indices = sorted_indices[:k]

            # 构建特征
            all_features = np.vstack([pos_features, neg_features])
            features = all_features[:, selected_indices]
            labels = np.concatenate([
                np.ones(len(pos_features)),
                np.zeros(len(neg_features))
            ])

            logger.info(f"  选中维度: {k}/{len(scores)} ({k/len(scores)*100:.1f}%)")
            logger.info(f"  特征形状: {features.shape}")

            # 降维
            embeddings = self.reduce_dimensionality(features)

            # 计算指标
            metrics = self.compute_metrics(embeddings, labels)

            # 保存结果
            result = {
                "k": k,
                "silhouette": float(metrics.silhouette),
                "calinski_harabasz": float(metrics.calinski_harabasz),
                "center_distance": float(metrics.center_distance),
                "selected_indices": selected_indices.tolist(),
                "top_10_indices": selected_indices[:10].tolist(),
            }
            results[k] = result

            # 跟踪最佳
            if metrics.silhouette > best_silhouette:
                best_silhouette = metrics.silhouette
                best_k = k

            # 保存该k值的可视化
            if HAS_MATPLOTLIB and self.n_components == 2:
                fig_path = output_dir / f"{self.category}_top{k}_tsne.png"
                self.plot_2d(embeddings, labels, metrics, str(fig_path))
                logger.info(f"  保存可视化: {fig_path}")

        # 保存汇总结果
        summary_path = output_dir / f"{self.category}_batch_selection_summary.json"
        # 转换numpy类型为Python原生类型（JSON序列化）
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        summary = {
            "category": self.category,
            "layer_key": f"{self.layer_type}_layer{self.layer_idx}",
            "selection_method": selection_method,
            "tested_k_values": k_values,
            "best_k": int(best_k),
            "best_silhouette": float(best_silhouette),
            "all_results": {str(k): convert_to_serializable(v) for k, v in results.items()},
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 打印结果表格
        logger.info("\n" + "=" * 80)
        logger.info("批量测试结果汇总")
        logger.info("=" * 80)
        logger.info(f"{'K值':>8} | {'Silhouette':>12} | {'Calinski-Harabasz':>18} | {'中心距离':>12} | {'状态'}")
        logger.info("-" * 80)
        for k in k_values:
            r = results[k]
            silhouette = r["silhouette"]
            status = "★ BEST" if k == best_k else ""
            logger.info(f"{k:>8} | {silhouette:>12.4f} | {r['calinski_harabasz']:>18.2f} | {r['center_distance']:>12.4f} | {status}")
        logger.info("=" * 80)
        logger.info(f"最佳维度数: {best_k} (Silhouette = {best_silhouette:.4f})")
        logger.info(f"最佳维度组前10: {results[best_k]['top_10_indices']}")
        logger.info(f"汇总结果保存: {summary_path}")
        logger.info("=" * 80)

        return results


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

    # 特征选择参数
    parser.add_argument(
        "--no_feature_selection", action="store_true",
        help="禁用特征选择（默认启用）"
    )
    parser.add_argument(
        "--top_k_features", type=int, default=500,
        help="选择前k个最具区分度的特征 (默认500)"
    )
    parser.add_argument(
        "--selection_method", type=str, default="t_test",
        choices=["t_test", "mean_diff", "cohen_d"],
        help="特征选择方法 (默认t_test)"
    )

    # 批量测试参数
    parser.add_argument(
        "--batch_test", action="store_true",
        help="批量测试不同维度数 (400->50)"
    )
    parser.add_argument(
        "--k_values", type=int, nargs="+",
        default=[400, 350, 300, 250, 200, 150, 100, 50],
        help="批量测试的维度数列表 (默认: 400 350 300 250 200 150 100 50)"
    )

    # 统计学去噪参数
    parser.add_argument(
        "--no_denoising", action="store_true",
        help="禁用统计学去噪（默认启用）"
    )
    parser.add_argument(
        "--winsorization_limits", type=float, nargs=2, default=[0.01, 0.99],
        metavar=("LOWER", "UPPER"),
        help="Winsorization缩尾处理的上下限百分位 (默认: 0.01 0.99)"
    )
    parser.add_argument(
        "--no_mad_removal", action="store_true",
        help="禁用MAD异常值检测与替换（默认启用）"
    )
    parser.add_argument(
        "--mad_threshold", type=float, default=3.0,
        help="MAD异常值检测阈值倍数 (默认: 3.0)"
    )
    parser.add_argument(
        "--no_robust_scaler", action="store_true",
        help="禁用RobustScaler标准化（默认启用）"
    )
    parser.add_argument(
        "--no_pca_denoising", action="store_true",
        help="禁用PCA去噪（默认启用）"
    )
    parser.add_argument(
        "--pca_variance_threshold", type=float, default=0.95,
        help="PCA去噪的方差保留阈值 (默认: 0.95)"
    )
    parser.add_argument(
        "--use_log_transform", action="store_true",
        help="启用Log1p变换（默认禁用）"
    )
    parser.add_argument(
        "--use_gaussian_smooth", action="store_true",
        help="启用高斯平滑（默认禁用）"
    )
    parser.add_argument(
        "--smooth_sigma", type=float, default=1.0,
        help="高斯平滑的sigma参数 (默认: 1.0)"
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
        # 统计学去噪参数
        winsorization_limits=tuple(args.winsorization_limits),
        use_mad_outlier_removal=not args.no_mad_removal,
        mad_threshold=args.mad_threshold,
        use_robust_scaler=not args.no_robust_scaler,
        use_pca_denoising=not args.no_pca_denoising,
        pca_variance_threshold=args.pca_variance_threshold,
        use_log_transform=args.use_log_transform,
        use_gaussian_smooth=args.use_gaussian_smooth,
        smooth_sigma=args.smooth_sigma,
    )

    # 批量测试或单次运行
    if args.batch_test:
        # 批量测试不同维度数
        results = visualizer.run_batch_feature_selection(
            args.output_dir,
            k_values=args.k_values,
            selection_method=args.selection_method,
            use_denoising=not args.no_denoising,
        )

        # 找到最佳k值
        best_k = max(results.keys(), key=lambda k: results[k]["silhouette"])
        best_result = results[best_k]

        print(f"\n{'='*60}")
        print("批量测试结果摘要")
        print(f"{'='*60}")
        print(f"测试维度: {args.k_values}")
        print(f"最佳维度数: {best_k}")
        print(f"  Silhouette: {best_result['silhouette']:.4f}")
        print(f"  Calinski-Harabasz: {best_result['calinski_harabasz']:.2f}")
        print(f"  中心距离: {best_result['center_distance']:.4f}")
        print(f"  前10维度索引: {best_result['top_10_indices']}")
        print(f"{'='*60}")

    else:
        # 单次运行
        result = visualizer.run(
            args.output_dir,
            use_feature_selection=not args.no_feature_selection,
            top_k_features=args.top_k_features,
            selection_method=args.selection_method,
            use_denoising=not args.no_denoising,
        )

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
