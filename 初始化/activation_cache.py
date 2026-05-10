"""
ActivationCache - 统一的激活缓存数据结构

用于保存完整的激活元数据，支持：
- SAE training
- Concept probing
- Feature visualization
- Activation statistics
- Jailbreak concept analysis

数据结构设计：
- 每个记录包含完整的时空和语义元数据
- 支持高效索引和查询
- 内存友好，支持分块加载

作者：Claude
日期：2026-05-10
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch


# ============================================================================
# 核心数据结构
# ============================================================================

@dataclass
class TokenMetadata:
    """
    单个 Token 的元数据

    包含完整的位置和语义信息
    """
    # 位置信息
    batch_idx: int          # 在 batch 中的索引
    token_idx: int          # 全局 token 索引
    frame_idx: int          # 时间帧索引 (F 维度)
    spatial_h: int          # 空间 H 坐标
    spatial_w: int          # 空间 W 坐标

    # 统计信息
    norm: float             # 原始 token norm
    rms: float              # RMSNorm 后的 rms 值

    # 上下文信息
    timestep: int           # 扩散时间步
    layer: int              # DiT 层索引
    prompt_id: str          # 提示词 ID

    # 可选信息
    attention_score: Optional[float] = None  # 注意力分数
    concept_label: Optional[str] = None      # 概念标签 (用于分析)


@dataclass
class ActivationRecord:
    """
    单条激活记录

    包含一个 sample 的所有 token 激活和元数据
    """
    # 唯一标识
    record_id: str
    prompt_id: str
    prompt_text: str

    # 时间信息
    timestep: int
    timestamp: str  # ISO 格式

    # 层信息
    layer: int
    hook_mode: str  # "block_out" | "self_attn" | "cross_attn"

    # 激活数据
    activation_shape: Tuple[int, int]  # (N_tokens, D_model)

    # 元数据 (每个 token)
    token_metadata: List[TokenMetadata]

    # 激活数据 (存储路径或实际数据)
    activation_data_path: Optional[str] = None  # 外部存储路径
    activation_data: Optional[torch.Tensor] = None  # 内嵌数据

    # 统计摘要
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_activation(self, cache_dir: Optional[Path] = None) -> torch.Tensor:
        """获取激活数据"""
        if self.activation_data is not None:
            return self.activation_data

        if self.activation_data_path and cache_dir:
            path = cache_dir / self.activation_data_path
            if path.suffix == ".pt":
                return torch.load(path)
            elif path.suffix == ".npy":
                return torch.from_numpy(np.load(path))

        raise ValueError("No activation data available")


@dataclass
class ActivationCacheConfig:
    """Activation Cache 配置"""
    # 存储配置
    cache_dir: str = "./activation_cache"
    storage_format: str = "numpy"  # "numpy" | "torch" | "hdf5"

    # 分块配置
    max_records_per_file: int = 100
    max_tokens_per_record: int = 20000  # 超过此数量会分块存储

    # 压缩
    compress: bool = True
    compression_level: int = 4  # 1-9

    # 索引配置
    build_index: bool = True
    index_fields: List[str] = field(
        default_factory=lambda: ["timestep", "layer", "prompt_id"]
    )


class ActivationCache:
    """
    激活缓存管理器

    统一管理激活数据的存储、索引和查询

    使用方法：
        cache = ActivationCache(config)

        # 添加记录
        cache.add_record(
            activations=activations,
            timestep=500,
            layer=14,
            prompt_id="p001",
            prompt_text="...",
            grid_size=(11, 30, 52),
        )

        # 查询
        records = cache.query(timestep=500, layer=14)

        # 统计分析
        stats = cache.compute_statistics()
    """

    def __init__(self, config: ActivationCacheConfig):
        self.config = config
        self.cache_path = Path(config.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # 数据存储
        self._records: Dict[str, ActivationRecord] = {}
        self._record_list: List[str] = []

        # 索引
        self._indices: Dict[str, Dict[Any, List[str]]] = {
            field: {} for field in config.index_fields
        }

        # 统计缓存
        self._stats_cache: Optional[Dict[str, Any]] = None

    def add_record(
        self,
        activations: torch.Tensor,
        timestep: int,
        layer: int,
        prompt_id: str,
        prompt_text: str,
        hook_mode: str = "block_out",
        grid_size: Optional[Tuple[int, int, int]] = None,
        batch_idx: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加激活记录

        参数:
            activations: [B, L, C] 或 [N, C] 激活数据
            timestep: 扩散时间步
            layer: DiT 层索引
            prompt_id: 提示词 ID
            prompt_text: 提示词文本
            hook_mode: Hook 模式
            grid_size: (F, H, W) 网格尺寸
            batch_idx: Batch 索引
            extra_metadata: 额外元数据

        返回:
            record_id: 记录 ID
        """
        # RMSNorm 并记录 rms
        if activations.dim() == 3:
            B, L, C = activations.shape
            activations_flat = activations.reshape(B * L, C)
        else:
            activations_flat = activations
            B = 1
            L = activations_flat.shape[0]
            C = activations_flat.shape[1]

        # RMSNorm
        rms = torch.sqrt(activations_flat.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        activations_normed = activations_flat / rms
        rms_values = rms.squeeze(-1)

        # 计算 norm
        norms = activations_flat.norm(dim=-1)

        # 生成 record_id
        record_id = f"{prompt_id}_t{timestep}_l{layer}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        # 构建 token 元数据
        token_metadata = []
        for i in range(len(activations_normed)):
            # 解析空间坐标
            if grid_size is not None:
                F, H, W = grid_size
                frame_idx = (i // (H * W)) % F
                spatial_h = (i % (H * W)) // W
                spatial_w = i % W
            else:
                frame_idx = 0
                spatial_h = 0
                spatial_w = i

            meta = TokenMetadata(
                batch_idx=batch_idx,
                token_idx=i,
                frame_idx=frame_idx,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
                norm=norms[i].item(),
                rms=rms_values[i].item(),
                timestep=timestep,
                layer=layer,
                prompt_id=prompt_id,
            )
            token_metadata.append(meta)

        # 存储激活数据
        data_path = None
        if self.config.storage_format == "numpy":
            data_path = f"data/{record_id}.npy"
            data_file = self.cache_path / data_path
            data_file.parent.mkdir(parents=True, exist_ok=True)

            if self.config.compress:
                np.savez_compressed(
                    data_file.with_suffix(".npz"),
                    activations=activations_normed.numpy(),
                    rms=rms_values.numpy(),
                    norms=norms.numpy(),
                )
                data_path = str(data_file.with_suffix(".npz").relative_to(self.cache_path))
            else:
                np.save(data_file, activations_normed.numpy())
        elif self.config.storage_format == "torch":
            data_path = f"data/{record_id}.pt"
            data_file = self.cache_path / data_path
            data_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "activations": activations_normed,
                "rms": rms_values,
                "norms": norms,
            }, data_file)

        # 计算摘要统计
        summary = {
            "n_tokens": len(activations_normed),
            "d_model": C,
            "norm_mean": norms.mean().item(),
            "norm_std": norms.std().item(),
            "norm_min": norms.min().item(),
            "norm_max": norms.max().item(),
            "rms_mean": rms_values.mean().item(),
        }

        if extra_metadata:
            summary.update(extra_metadata)

        # 创建记录
        record = ActivationRecord(
            record_id=record_id,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            timestep=timestep,
            timestamp=datetime.now().isoformat(),
            layer=layer,
            hook_mode=hook_mode,
            activation_shape=(len(activations_normed), C),
            token_metadata=token_metadata,
            activation_data_path=data_path,
            summary=summary,
        )

        # 存储
        self._records[record_id] = record
        self._record_list.append(record_id)

        # 更新索引
        self._update_indices(record_id, record)

        # 清除统计缓存
        self._stats_cache = None

        return record_id

    def _update_indices(self, record_id: str, record: ActivationRecord):
        """更新索引"""
        # Timestep 索引
        if "timestep" in self._indices:
            if record.timestep not in self._indices["timestep"]:
                self._indices["timestep"][record.timestep] = []
            self._indices["timestep"][record.timestep].append(record_id)

        # Layer 索引
        if "layer" in self._indices:
            if record.layer not in self._indices["layer"]:
                self._indices["layer"][record.layer] = []
            self._indices["layer"][record.layer].append(record_id)

        # Prompt 索引
        if "prompt_id" in self._indices:
            if record.prompt_id not in self._indices["prompt_id"]:
                self._indices["prompt_id"][record.prompt_id] = []
            self._indices["prompt_id"][record.prompt_id].append(record_id)

    def query(
        self,
        timestep: Optional[int] = None,
        layer: Optional[int] = None,
        prompt_id: Optional[str] = None,
    ) -> List[ActivationRecord]:
        """
        查询记录

        支持按 timestep, layer, prompt_id 查询
        """
        # 收集匹配的 record_id
        matching_ids = None

        if timestep is not None:
            ids = set(self._indices["timestep"].get(timestep, []))
            matching_ids = ids if matching_ids is None else matching_ids & ids

        if layer is not None:
            ids = set(self._indices["layer"].get(layer, []))
            matching_ids = ids if matching_ids is None else matching_ids & ids

        if prompt_id is not None:
            ids = set(self._indices["prompt_id"].get(prompt_id, []))
            matching_ids = ids if matching_ids is None else matching_ids & ids

        if matching_ids is None:
            matching_ids = set(self._record_list)

        return [self._records[rid] for rid in matching_ids]

    def get_activations(
        self,
        timestep: Optional[int] = None,
        layer: Optional[int] = None,
        prompt_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        获取激活数据（合并后）
        """
        records = self.query(timestep=timestep, layer=layer, prompt_id=prompt_id)

        activations_list = []
        for record in records:
            act = record.get_activation(self.cache_path)
            activations_list.append(act)

        return torch.cat(activations_list, dim=0)

    def compute_statistics(self) -> Dict[str, Any]:
        """计算缓存统计信息"""
        if self._stats_cache is not None:
            return self._stats_cache

        stats = {
            "total_records": len(self._records),
            "total_tokens": sum(r.activation_shape[0] for r in self._records.values()),
            "timesteps": sorted(set(r.timestep for r in self._records.values())),
            "layers": sorted(set(r.layer for r in self._records.values())),
            "prompts": list(set(r.prompt_id for r in self._records.values())),
        }

        # 每个 timestep 的记录数
        stats["records_per_timestep"] = {
            t: len(ids) for t, ids in self._indices["timestep"].items()
        }

        # 每层的记录数
        stats["records_per_layer"] = {
            l: len(ids) for l, ids in self._indices["layer"].items()
        }

        # Norm 统计
        all_norms = []
        all_rms = []
        for record in self._records.values():
            for meta in record.token_metadata:
                all_norms.append(meta.norm)
                all_rms.append(meta.rms)

        if all_norms:
            stats["norm_statistics"] = {
                "mean": float(np.mean(all_norms)),
                "std": float(np.std(all_norms)),
                "min": float(np.min(all_norms)),
                "max": float(np.max(all_norms)),
                "median": float(np.median(all_norms)),
            }

        self._stats_cache = stats
        return stats

    def save(self, filename: Optional[str] = None):
        """保存缓存到文件"""
        if filename is None:
            filename = f"cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.cache_path / filename

        # 序列化记录
        records_data = {}
        for record_id, record in self._records.items():
            # 转换 token_metadata
            token_meta_list = []
            for tm in record.token_metadata:
                token_meta_list.append(asdict(tm))

            records_data[record_id] = {
                "record_id": record.record_id,
                "prompt_id": record.prompt_id,
                "prompt_text": record.prompt_text,
                "timestep": record.timestep,
                "timestamp": record.timestamp,
                "layer": record.layer,
                "hook_mode": record.hook_mode,
                "activation_shape": list(record.activation_shape),
                "token_metadata": token_meta_list,
                "activation_data_path": record.activation_data_path,
                "summary": record.summary,
            }

        # 保存
        cache_data = {
            "config": asdict(self.config),
            "records": records_data,
            "indices": {k: {kk: list(vv) for kk, vv in v.items()} for k, v in self._indices.items()},
            "statistics": self.compute_statistics(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    @classmethod
    def load(cls, filepath: str) -> "ActivationCache":
        """从文件加载缓存"""
        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = ActivationCacheConfig(**data["config"])
        cache = cls(config)

        # 恢复记录
        for record_id, record_data in data["records"].items():
            token_metadata = [
                TokenMetadata(**tm) for tm in record_data["token_metadata"]
            ]

            record = ActivationRecord(
                record_id=record_data["record_id"],
                prompt_id=record_data["prompt_id"],
                prompt_text=record_data["prompt_text"],
                timestep=record_data["timestep"],
                timestamp=record_data["timestamp"],
                layer=record_data["layer"],
                hook_mode=record_data["hook_mode"],
                activation_shape=tuple(record_data["activation_shape"]),
                token_metadata=token_metadata,
                activation_data_path=record_data["activation_data_path"],
                summary=record_data["summary"],
            )

            cache._records[record_id] = record
            cache._record_list.append(record_id)

        # 恢复索引
        for field, field_indices in data["indices"].items():
            cache._indices[field] = {k: list(v) for k, v in field_indices.items()}

        return cache

    def iter_records(self) -> Iterator[ActivationRecord]:
        """迭代所有记录"""
        for record_id in self._record_list:
            yield self._records[record_id]

    def iter_tokens(self) -> Iterator[Tuple[torch.Tensor, TokenMetadata]]:
        """迭代所有 token（带元数据）"""
        for record in self.iter_records():
            activations = record.get_activation(self.cache_path)
            for i, meta in enumerate(record.token_metadata):
                yield activations[i], meta

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, record_id: str) -> ActivationRecord:
        return self._records[record_id]


# ============================================================================
# 便捷函数
# ============================================================================

def create_cache(
    cache_dir: str = "./activation_cache",
    **kwargs,
) -> ActivationCache:
    """创建激活缓存"""
    config = ActivationCacheConfig(cache_dir=cache_dir, **kwargs)
    return ActivationCache(config)
