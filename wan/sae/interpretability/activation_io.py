"""
激活值文件IO管理器

提供统一的激活值文件读写接口，支持：
1. 分层目录结构管理（sae/dit层 -> 类别 -> 极性）
2. 增量保存与加载
3. 元信息（metadata）管理
4. 断点（checkpoint）管理
5. 全局配置管理

设计原则：
- 所有路径操作封装在此类中，上层无需关心文件结构
- 支持内存映射（mmap）读取，处理大文件不OOM
- 自动创建目录结构
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class SampleMetadata:
    """单个样本的元信息"""
    idx: int              # 在同类中的索引
    pair_idx: int         # 配对索引（正负样本共享）
    prompt: str           # 提示词文本
    category: str         # 概念类别
    polarity: str         # "pos" 或 "neg"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "pair_idx": self.pair_idx,
            "prompt": self.prompt,
            "category": self.category,
            "polarity": self.polarity,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SampleMetadata":
        return cls(
            idx=d["idx"],
            pair_idx=d["pair_idx"],
            prompt=d["prompt"],
            category=d["category"],
            polarity=d["polarity"],
        )


@dataclass
class ExtractionCheckpoint:
    """增量采集断点信息"""
    completed_pair_indices: List[int] = field(default_factory=list)
    total_pairs: int = 0
    last_update: float = field(default_factory=lambda: __import__("time").time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completed_pair_indices": self.completed_pair_indices,
            "total_pairs": self.total_pairs,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExtractionCheckpoint":
        return cls(
            completed_pair_indices=d.get("completed_pair_indices", []),
            total_pairs=d.get("total_pairs", 0),
            last_update=d.get("last_update", __import__("time").time()),
        )


class ActivationIO:
    """
    激活值文件IO管理器

    目录结构：
        {root}/
        ├── sae_layer{idx}/
        │   └── {category}/
        │       ├── pos/
        │       │   ├── activations.npy      # [N, 7, D] 统计特征 (实时池化)
        │       │   │                          # 7个统计量: [mean, std, max, min, median, p95, p05]
        │       │   ├── metadata.json        # 元信息列表
        │       │   └── checkpoint.json      # 断点信息
        │       └── neg/
        ├── dit_layer{idx}/
        └── extraction_config.json           # 全局配置

    注：使用实时池化后，激活值形状从 [N, T, L, D] (约23GB/样本) 变为 [N, 7, D] (约168KB/样本)
    """

    def __init__(self, root_dir: str):
        """
        Args:
            root_dir: 激活值存储根目录
        """
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 路径管理
    # =========================================================================

    def _get_layer_dir(self, layer_type: str, layer_idx: int) -> Path:
        """获取层目录路径"""
        return self.root / f"{layer_type}_layer{layer_idx}"

    def _get_category_dir(self, layer_type: str, layer_idx: int, category: str) -> Path:
        """获取类别目录路径"""
        return self._get_layer_dir(layer_type, layer_idx) / category

    def _get_polarity_dir(self, layer_type: str, layer_idx: int, category: str, polarity: str) -> Path:
        """获取极性目录路径"""
        return self._get_category_dir(layer_type, layer_idx, category) / polarity

    def _get_activations_path(self, layer_type: str, layer_idx: int, category: str, polarity: str) -> Path:
        """获取激活值文件路径"""
        return self._get_polarity_dir(layer_type, layer_idx, category, polarity) / "activations.npy"

    def _get_metadata_path(self, layer_type: str, layer_idx: int, category: str, polarity: str) -> Path:
        """获取元信息文件路径"""
        return self._get_polarity_dir(layer_type, layer_idx, category, polarity) / "metadata.json"

    def _get_checkpoint_path(self, layer_type: str, layer_idx: int, category: str, polarity: str) -> Path:
        """获取断点文件路径"""
        return self._get_polarity_dir(layer_type, layer_idx, category, polarity) / "checkpoint.json"

    def _get_config_path(self) -> Path:
        """获取全局配置文件路径"""
        return self.root / "extraction_config.json"

    # =========================================================================
    # 目录操作
    # =========================================================================

    def ensure_dir_structure(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str,
    ) -> Path:
        """
        确保目录结构存在，返回极性目录路径
        """
        polarity_dir = self._get_polarity_dir(layer_type, layer_idx, category, polarity)
        polarity_dir.mkdir(parents=True, exist_ok=True)
        return polarity_dir

    def ensure_layer_structure(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
    ) -> Tuple[Path, Path]:
        """
        确保层的完整目录结构存在（包括pos和neg）
        返回 (pos_dir, neg_dir)
        """
        pos_dir = self.ensure_dir_structure(layer_type, layer_idx, category, "pos")
        neg_dir = self.ensure_dir_structure(layer_type, layer_idx, category, "neg")
        return pos_dir, neg_dir

    # =========================================================================
    # 激活值读写
    # =========================================================================

    def save_activations(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str,
        activations: npt.NDArray,
        append: bool = True,
    ) -> None:
        """
        保存激活值

        Args:
            activations: [N, 7, D] 或要追加的 [1, 7, D]
                7个统计量: [mean, std, max, min, median, p95, p05]
            append: True=追加到现有文件，False=覆盖
        """
        path = self._get_activations_path(layer_type, layer_idx, category, polarity)
        self.ensure_dir_structure(layer_type, layer_idx, category, polarity)

        if append and path.exists():
            # 加载现有数据并追加
            existing = np.load(path)
            combined = np.concatenate([existing, activations], axis=0)
            np.save(path, combined)
            logger.debug(f"追加激活值到 {path}: {existing.shape[0]} + {activations.shape[0]} = {combined.shape[0]}")
        else:
            np.save(path, activations)
            logger.debug(f"保存激活值到 {path}: {activations.shape}")

    def load_activations(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str,
        mmap: bool = False,
    ) -> Optional[npt.NDArray]:
        """
        加载激活值

        Args:
            mmap: True=使用内存映射（大文件不OOM），False=直接加载

        Returns:
            [N, 7, D] 统计特征数组，或None如果文件不存在
            7个统计量: [mean, std, max, min, median, p95, p05]
        """
        path = self._get_activations_path(layer_type, layer_idx, category, polarity)
        if not path.exists():
            return None

        if mmap:
            return np.load(path, mmap_mode='r')
        return np.load(path)

    def iter_activation_batches(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str,
        batch_size: int,
    ) -> Iterator[npt.NDArray]:
        """
        分批迭代激活值（流式读取，内存友好）

        Yields:
            每批 [B, T, L, D] 数组
        """
        path = self._get_activations_path(layer_type, layer_idx, category, polarity)
        if not path.exists():
            return

        # 使用内存映射
        data = np.load(path, mmap_mode='r')
        num_samples = data.shape[0]

        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            # 实际加载到内存
            batch = np.array(data[i:end_idx])
            yield batch

    def get_activation_shape(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str,
    ) -> Optional[Tuple[int, ...]]:
        """
        获取激活值形状（不加载数据）
        """
        path = self._get_activations_path(layer_type, layer_idx, category, polarity)
        if not path.exists():
            return None

        # 使用内存映射只读取元信息
        data = np.load(path, mmap_mode='r')
        return data.shape

    def get_num_samples(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str,
    ) -> int:
        """
        获取样本数量（不加载数据）
        """
        shape = self.get_activation_shape(layer_type, layer_idx, category, polarity)
        return shape[0] if shape else 0

    # =========================================================================
    # 元信息读写
    # =========================================================================

    def save_metadata(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str,
        metadata: List[SampleMetadata],
        append: bool = True,
    ) -> None:
        """
        保存元信息

        Args:
            metadata: 元信息列表
            append: True=追加到现有文件，False=覆盖
        """
        path = self._get_metadata_path(layer_type, layer_idx, category, polarity)
        self.ensure_dir_structure(layer_type, layer_idx, category, polarity)

        if append and path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                existing = [SampleMetadata.from_dict(d) for d in json.load(f)]
            combined = existing + metadata
        else:
            combined = metadata

        with open(path, 'w', encoding='utf-8') as f:
            json.dump([m.to_dict() for m in combined], f, ensure_ascii=False, indent=2)

        logger.debug(f"保存元信息到 {path}: {len(combined)} 条")

    def load_metadata(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str,
    ) -> List[SampleMetadata]:
        """
        加载元信息

        Returns:
            元信息列表，空列表如果文件不存在
        """
        path = self._get_metadata_path(layer_type, layer_idx, category, polarity)
        if not path.exists():
            return []

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [SampleMetadata.from_dict(d) for d in data]

    def get_metadata_dict(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str,
    ) -> List[Dict[str, Any]]:
        """
        加载原始元信息字典（不转换dataclass）
        """
        path = self._get_metadata_path(layer_type, layer_idx, category, polarity)
        if not path.exists():
            return []

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # =========================================================================
    # 断点读写
    # =========================================================================

    def save_checkpoint(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str,
        checkpoint: ExtractionCheckpoint,
    ) -> None:
        """保存断点"""
        path = self._get_checkpoint_path(layer_type, layer_idx, category, polarity)
        self.ensure_dir_structure(layer_type, layer_idx, category, polarity)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)

        logger.debug(f"保存断点到 {path}: {len(checkpoint.completed_pair_indices)}/{checkpoint.total_pairs}")

    def load_checkpoint(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str,
    ) -> Optional[ExtractionCheckpoint]:
        """加载断点，None如果不存在"""
        path = self._get_checkpoint_path(layer_type, layer_idx, category, polarity)
        if not path.exists():
            return None

        with open(path, 'r', encoding='utf-8') as f:
            return ExtractionCheckpoint.from_dict(json.load(f))

    def get_completed_pairs(
        self,
        layer_type: str,
        layer_idx: int,
        category: str,
        polarity: str = "pos",
    ) -> set:
        """
        获取已完成的pair索引集合
        """
        ckpt = self.load_checkpoint(layer_type, layer_idx, category, polarity)
        return set(ckpt.completed_pair_indices) if ckpt else set()

    # =========================================================================
    # 全局配置读写
    # =========================================================================

    def save_config(self, config: Dict[str, Any]) -> None:
        """保存全局配置"""
        path = self._get_config_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"全局配置已保存: {path}")

    def load_config(self) -> Optional[Dict[str, Any]]:
        """加载全局配置，None如果不存在"""
        path = self._get_config_path()
        if not path.exists():
            return None

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def config_exists(self) -> bool:
        """检查全局配置是否存在"""
        return self._get_config_path().exists()

    # =========================================================================
    # 批量操作（方便阶段二使用）
    # =========================================================================

    def list_available_layers(self, category: str) -> List[Tuple[str, int]]:
        """
        列出可用的层（从目录结构推断）

        Returns:
            [("sae", 15), ("sae", 29), ("dit", 15), ...]
        """
        layers = []
        for item in self.root.iterdir():
            if item.is_dir() and (item.name.startswith("sae_layer") or item.name.startswith("dit_layer")):
                layer_type = item.name.split("_layer")[0]
                layer_idx = int(item.name.split("_layer")[1])
                # 检查该层是否有指定类别的数据
                if (item / category).exists():
                    layers.append((layer_type, layer_idx))
        return sorted(set(layers))

    def get_layer_info(self, category: str) -> Dict[str, Any]:
        """
        获取层信息摘要
        """
        layers = self.list_available_layers(category)
        info = {
            "root": str(self.root),
            "category": category,
            "layers": {},
        }

        for layer_type, layer_idx in layers:
            key = f"{layer_type}_layer{layer_idx}"
            info["layers"][key] = {
                "num_pos": self.get_num_samples(layer_type, layer_idx, category, "pos"),
                "num_neg": self.get_num_samples(layer_type, layer_idx, category, "neg"),
            }

        return info

    # =========================================================================
    # 实用方法
    # =========================================================================

    def print_summary(self, category: str) -> None:
        """打印存储摘要"""
        info = self.get_layer_info(category)

        print(f"\n{'='*60}")
        print(f"激活值存储摘要: {info['root']}")
        print(f"类别: {category}")
        print(f"{'='*60}")

        for key, data in info["layers"].items():
            print(f"  {key}:")
            print(f"    正样本: {data['num_pos']}")
            print(f"    负样本: {data['num_neg']}")

        config = self.load_config()
        if config:
            print(f"\n配置信息:")
            print(f"  时间步数: {config.get('num_timesteps', 'N/A')}")
            print(f"  采样方法: {config.get('sampling_method', 'N/A')}")
            print(f"  CFG: {config.get('use_cfg', 'N/A')}")
        print(f"{'='*60}\n")
