"""
SAE概念提取 - 阶段二：概念向量提取（CPU即可，低内存）

功能：
1. 从阶段一采集的激活值中提取概念向量
2. 使用mean_diff方法：concept_vector = mean(pos) - mean(neg)
3. 流式处理大文件，内存友好
4. 支持增量统计（RunningMean）

采样方法说明：
- 本阶段仅进行NumPy运算，不需要模型
- 使用阶段一保存的SAE隐状态z（非DiT状态）
- 时间步和token维度上取平均，得到每样本的概念表示

使用示例：
    python wan/sae/interpretability/concept_extractor_stage2.py \
        --activation_root "activations" \
        --category "violence" \
        --layer_key "sae_layer15" \
        --output_dir "concept_vectors" \
        --method "mean_diff" \
        --normalize \
        --min_threshold 0.01

输出格式：
    concept_vectors/
    ├── violence_sae_layer15.npy          # [d_hidden] 概念向量
    └── violence_sae_layer15.json         # 元信息和统计
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wan.sae.interpretability.activation_io import ActivationIO, SampleMetadata

logger = logging.getLogger(__name__)


##########################################################################################
# 参数配置
##########################################################################################

extract_params = {
    # activation_root: 阶段一输出的激活值根目录
    "activation_root": "activations",

    # category: 概念类别（与阶段一一致）
    "category": "violence",

    # layer_key: 要提取的层
    # 格式: "sae_layer15" 或 "dit_layer15"
    # 注意：通常使用SAE层（sae_layer*），而非DiT层
    "layer_key": "sae_layer15",

    # output_dir: 概念向量输出目录
    "output_dir": "concept_vectors",

    # method: 提取方法
    # "mean_diff": 均值差 (pos_mean - neg_mean)
    "method": "mean_diff",

    # normalize: 是否归一化概念向量
    "normalize": True,

    # batch_size: 流式处理批次大小（内存控制）
    "batch_size": 32,

    # min_threshold: 最小阈值，绝对值小于此值的特征置零
    "min_threshold": 0.01,

    # top_k: 保存Top-K特征的详细信息
    "top_k": 50,
}


##########################################################################################
# 工具类
##########################################################################################

@dataclass
class RunningMean:
    """
    增量计算均值（Welford算法）

    学术意义: O(1)内存复杂度的流式统计，支持任意大小的数据集
    """
    count: int = 0
    mean: Optional[np.ndarray] = None

    def update(self, new_values: npt.NDArray) -> None:
        """
        更新均值

        Args:
            new_values: [B, D] 数组（已在时间和token维度上平均）
        """
        batch_count = new_values.shape[0]
        batch_mean = new_values.mean(axis=0)

        if self.count == 0:
            self.mean = batch_mean
        else:
            # Welford算法
            delta = batch_mean - self.mean
            self.mean += delta * batch_count / (self.count + batch_count)

        self.count += batch_count

    def get_mean(self) -> Optional[np.ndarray]:
        """获取当前均值"""
        return self.mean


class ConceptExtractor:
    """
    概念向量提取器

    使用流式处理从阶段一采集的激活值中提取概念向量
    """

    def __init__(
        self,
        io: ActivationIO,
        category: str,
        layer_type: str,
        layer_idx: int,
        method: str = "mean_diff",
        normalize: bool = True,
        min_threshold: float = 0.01,
        batch_size: int = 32,
    ):
        self.io = io
        self.category = category
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        self.method = method
        self.normalize = normalize
        self.min_threshold = min_threshold
        self.batch_size = batch_size

        # 验证方法
        if method != "mean_diff":
            raise ValueError(f"不支持的提取方法: {method}，目前仅支持 'mean_diff'")

    def _load_and_pool_batch(
        self,
        polarity: str,
        batch_indices: List[int],
    ) -> Optional[np.ndarray]:
        """
        加载并池化一批激活值

        支持两种数据格式:
        - 实时池化格式: [N, 7, D] (7个统计量: mean, std, max, min, median, p95, p05)
        - 旧格式: [N, T, L, D] (全时间步数据)

        Args:
            polarity: "pos" 或 "neg"
            batch_indices: 批次样本索引

        Returns:
            [B, D] 池化后的特征（使用mean统计量），或None
        """
        # 加载激活值（使用内存映射）
        acts = self.io.load_activations(
            self.layer_type, self.layer_idx, self.category, polarity,
            mmap=True
        )

        if acts is None:
            return None

        # 提取指定批次
        batch_acts = acts[batch_indices]

        # 判断数据格式并处理
        if batch_acts.ndim == 3 and batch_acts.shape[1] == 7:
            # 实时池化格式: [B, 7, D]
            # 7个统计量: [mean, std, max, min, median, p95, p05]
            # 第0维是mean (已经在阶段一计算好)
            logger.debug(f"检测到实时池化格式: {batch_acts.shape}, 使用第0维(mean)")
            pooled = batch_acts[:, 0, :]  # [B, D] 取mean

        elif batch_acts.ndim == 4:
            # 旧格式: [B, T, L, D] (全时间步)
            # 需要在线池化
            logger.debug(f"检测到全时间步格式: {batch_acts.shape}, 在线计算mean")
            B = batch_acts.shape[0]
            pooled = batch_acts.reshape(B, -1, batch_acts.shape[-1]).mean(axis=1)

        else:
            raise ValueError(
                f"未知的数据格式: ndim={batch_acts.ndim}, shape={batch_acts.shape}\n"
                f"期望: [N, 7, D] (池化格式) 或 [N, T, L, D] (全时间步)"
            )

        return pooled

    def extract(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        提取概念向量

        Returns:
            (concept_vector, statistics)
            concept_vector: [d_hidden]
            statistics: 统计信息字典
        """
        logger.info("=" * 60)
        logger.info(f"开始提取概念向量: {self.category} / {self.layer_type}_layer{self.layer_idx}")
        logger.info(f"方法: {self.method}")
        logger.info(f"特征统计量: mean (第0维)")
        logger.info(f"特征维度: 6144 (SAE d_hidden)")
        logger.info(f"归一化: {self.normalize}, 阈值: {self.min_threshold}")
        logger.info("=" * 60)

        # 获取样本数量
        num_pos = self.io.get_num_samples(self.layer_type, self.layer_idx, self.category, "pos")
        num_neg = self.io.get_num_samples(self.layer_type, self.layer_idx, self.category, "neg")

        if num_pos == 0 or num_neg == 0:
            raise ValueError(f"样本数量为零: pos={num_pos}, neg={num_neg}")

        logger.info(f"正样本: {num_pos}, 负样本: {num_neg}")

        # 流式计算均值
        pos_stats = RunningMean()
        neg_stats = RunningMean()

        # 处理正样本
        logger.info("处理正样本...")
        pos_start_time = time.time()
        for i in range(0, num_pos, self.batch_size):
            end_idx = min(i + self.batch_size, num_pos)
            batch_indices = list(range(i, end_idx))

            pooled = self._load_and_pool_batch("pos", batch_indices)
            if pooled is not None:
                pos_stats.update(pooled)

            # 每10个batch或最后一批显示进度和ETA
            batch_num = i // self.batch_size + 1
            total_batches = (num_pos + self.batch_size - 1) // self.batch_size
            if batch_num % 10 == 0 or end_idx == num_pos:
                elapsed = time.time() - pos_start_time
                progress = end_idx / num_pos
                if progress > 0:
                    eta = elapsed / progress - elapsed
                    logger.info(f"  已处理 {end_idx}/{num_pos} ({progress*100:.1f}%) | "
                               f"耗时: {elapsed:.1f}s | ETA: {eta:.1f}s")

        # 处理负样本
        logger.info("处理负样本...")
        neg_start_time = time.time()
        for i in range(0, num_neg, self.batch_size):
            end_idx = min(i + self.batch_size, num_neg)
            batch_indices = list(range(i, end_idx))

            pooled = self._load_and_pool_batch("neg", batch_indices)
            if pooled is not None:
                neg_stats.update(pooled)

            # 每10个batch或最后一批显示进度和ETA
            batch_num = i // self.batch_size + 1
            total_batches = (num_neg + self.batch_size - 1) // self.batch_size
            if batch_num % 10 == 0 or end_idx == num_neg:
                elapsed = time.time() - neg_start_time
                progress = end_idx / num_neg
                if progress > 0:
                    eta = elapsed / progress - elapsed
                    logger.info(f"  已处理 {end_idx}/{num_neg} ({progress*100:.1f}%) | "
                               f"耗时: {elapsed:.1f}s | ETA: {eta:.1f}s")

        # 计算概念向量
        pos_mean = pos_stats.get_mean()
        neg_mean = neg_stats.get_mean()

        if pos_mean is None or neg_mean is None:
            raise RuntimeError("均值计算失败")

        concept_vector = pos_mean - neg_mean

        logger.info(f"正例均值范数: {np.linalg.norm(pos_mean):.4f}")
        logger.info(f"负例均值范数: {np.linalg.norm(neg_mean):.4f}")
        logger.info(f"概念向量范数: {np.linalg.norm(concept_vector):.4f}")

        # 阈值过滤
        active_before = np.sum(np.abs(concept_vector) >= self.min_threshold)
        concept_vector[np.abs(concept_vector) < self.min_threshold] = 0
        active_after = np.sum(concept_vector != 0)
        logger.info(f"阈值过滤: {active_before} -> {active_after} 活跃特征 (阈值={self.min_threshold})")

        # 归一化
        final_norm = np.linalg.norm(concept_vector)
        if self.normalize and final_norm > 0:
            concept_vector = concept_vector / final_norm
            logger.info(f"已归一化，原范数: {final_norm:.4f}")

        # 统计信息
        statistics = {
            "pos_count": pos_stats.count,
            "neg_count": neg_stats.count,
            "active_features": int(active_after),
            "total_features": len(concept_vector),
            "sparsity": float(1 - active_after / len(concept_vector)),
            "norm_before_normalize": float(final_norm),
            "norm_after_normalize": float(np.linalg.norm(concept_vector)) if self.normalize else None,
        }

        return concept_vector, statistics

    def get_top_k_features(
        self,
        concept_vector: np.ndarray,
        k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        获取Top-K特征

        Returns:
            [{"index": int, "value": float}, ...]
        """
        top_indices = np.argsort(np.abs(concept_vector))[-k:][::-1]
        return [
            {"index": int(idx), "value": float(concept_vector[idx])}
            for idx in top_indices
        ]


##########################################################################################
# 主流程
##########################################################################################

def parse_layer_key(key: str) -> Tuple[str, int]:
    """
    解析层key

    Args:
        key: "sae_layer15" 或 "dit_layer15"

    Returns:
        (layer_type, layer_idx)
    """
    if "_layer" not in key:
        raise ValueError(f"无效的layer_key: {key}，应为 'sae_layer15' 或 'dit_layer15'")

    parts = key.split("_layer")
    layer_type = parts[0]  # "sae" 或 "dit"
    layer_idx = int(parts[1])

    return layer_type, layer_idx


def main():
    parser = argparse.ArgumentParser(
        description="SAE概念提取 - 阶段二：概念向量提取（CPU即可，低内存）"
    )

    # 必要参数
    parser.add_argument(
        "--activation_root", type=str, default=extract_params["activation_root"],
        help="阶段一输出的激活值根目录"
    )
    parser.add_argument(
        "--category", type=str, default=extract_params["category"],
        help="概念类别（与阶段一一致）"
    )
    parser.add_argument(
        "--layer_key", type=str, default=extract_params["layer_key"],
        help="要提取的层，如 'sae_layer15' 或 'dit_layer15'"
    )
    parser.add_argument(
        "--output_dir", type=str, default=extract_params["output_dir"],
        help="概念向量输出目录"
    )

    # 可选参数
    parser.add_argument(
        "--method", type=str, default=extract_params["method"],
        choices=["mean_diff"],
        help="提取方法"
    )
    parser.add_argument(
        "--normalize", action="store_true", default=extract_params["normalize"],
        help="是否归一化概念向量"
    )
    parser.add_argument(
        "--batch_size", type=int, default=extract_params["batch_size"],
        help="流式处理批次大小（内存控制）"
    )
    parser.add_argument(
        "--min_threshold", type=float, default=extract_params["min_threshold"],
        help="最小阈值，小于此值的特征置零"
    )
    parser.add_argument(
        "--top_k", type=int, default=extract_params["top_k"],
        help="保存Top-K特征的详细信息"
    )

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 60)
    logger.info("阶段二：概念向量提取")
    logger.info("=" * 60)

    # 解析层key
    layer_type, layer_idx = parse_layer_key(args.layer_key)
    logger.info(f"输入目录: {args.activation_root}")
    logger.info(f"概念类别: {args.category}")
    logger.info(f"层类型: {layer_type}, 层索引: {layer_idx}")

    # 初始化IO
    io = ActivationIO(args.activation_root)

    # 打印存储摘要
    io.print_summary(args.category)

    # 检查配置
    config = io.load_config()
    activation_format = None
    if config:
        logger.info("阶段一配置:")
        logger.info(f"  时间步数: {config.get('num_timesteps', 'N/A')}")
        logger.info(f"  CFG: {config.get('use_cfg', 'N/A')}")
        logger.info(f"  采样方法: {config.get('sampling_method', 'N/A')}")

        # 读取数据格式信息
        activation_format = config.get("activation_format", {})
        pool_activations = config.get("pool_activations", False)

        if pool_activations and activation_format:
            logger.info(f"  数据格式: {activation_format.get('shape', 'N/A')}")
            logger.info(f"  说明: {activation_format.get('description', 'N/A')}")
            logger.info(f"  阶段二将使用: {activation_format.get('note', '第0维(mean)')}")
        elif not pool_activations:
            logger.warning(f"  数据格式: 全时间步 [N, T, L, D] - 内存占用大")

    # 检查数据是否存在
    num_pos = io.get_num_samples(layer_type, layer_idx, args.category, "pos")
    num_neg = io.get_num_samples(layer_type, layer_idx, args.category, "neg")

    if num_pos == 0 or num_neg == 0:
        logger.error(f"数据不存在: pos={num_pos}, neg={num_neg}")
        logger.error(f"请检查路径: {args.activation_root}/{args.layer_key}/{args.category}/")
        return

    # 初始化提取器
    extractor = ConceptExtractor(
        io=io,
        category=args.category,
        layer_type=layer_type,
        layer_idx=layer_idx,
        method=args.method,
        normalize=args.normalize,
        min_threshold=args.min_threshold,
        batch_size=args.batch_size,
    )

    # 提取概念向量
    concept_vector, statistics = extractor.extract()

    # 获取Top-K特征
    top_k_features = extractor.get_top_k_features(concept_vector, k=args.top_k)

    # 准备输出
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{args.category}_{args.layer_key}"
    npy_path = output_dir / f"{output_name}.npy"
    json_path = output_dir / f"{output_name}.json"

    # 保存概念向量
    np.save(npy_path, concept_vector)

    # 保存元信息
    result = {
        "concept_name": args.category,
        "category": args.category,
        "layer_key": args.layer_key,
        "layer_type": layer_type,
        "layer_idx": layer_idx,
        "method": args.method,
        "vector_shape": list(concept_vector.shape),
        "norm": float(np.linalg.norm(concept_vector)),
        "top_k_features": top_k_features,
        "statistics": statistics,
        "parameters": {
            "normalize": args.normalize,
            "min_threshold": args.min_threshold,
            "batch_size": args.batch_size,
        },
        "source": {
            "activation_root": args.activation_root,
            "num_pos": num_pos,
            "num_neg": num_neg,
            "activation_format": activation_format or "unknown",
            "stat_used": "mean",  # 明确标注使用的是mean统计量
        },
        "extraction_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("概念提取完成！")
    logger.info(f"概念向量: {npy_path}")
    logger.info(f"元信息: {json_path}")
    logger.info(f"范数: {result['norm']:.4f}")
    logger.info(f"活跃特征: {statistics['active_features']} / {statistics['total_features']}")
    logger.info(f"稀疏度: {statistics['sparsity']:.2%}")
    logger.info("=" * 60)

    # 打印Top-10
    logger.info("Top-10 概念特征:")
    for i, feat in enumerate(top_k_features[:10]):
        logger.info(f"  #{i+1}: 特征 {feat['index']}, 值={feat['value']:.4f}")


if __name__ == "__main__":
    main()
