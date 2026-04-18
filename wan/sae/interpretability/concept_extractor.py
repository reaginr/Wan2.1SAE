"""
SAE概念提取器 - 通过正负提示词对比提取概念向量

核心思想：
1. 使用正例提示词（包含目标概念）和负例提示词（不包含目标概念）
2. 分别计算它们在SAE中的平均激活
3. 两者的差值即为该概念的"方向向量"
4. 该向量可以用于后续的干预生成

输出格式：
{
    "concept_name": "violence",
    "concept_vector": [...],  # SAE概念向量
    "metadata": {
        "layer_key": "block_out.layer15",
        "run_dir": "sae_runs/exp1",
        "extraction_method": "mean_diff",
        "positive_prompts_count": 100,
        "negative_prompts_count": 100,
    },
    "statistics": {
        "pos_mean_activation": [...],
        "neg_mean_activation": [...],
        "activation_difference": [...],
        "top_activated_features": [...],
    }
}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# 修复导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wan.modules.sae_new import SAEConfig, SparseAutoEncoder
from wan.sae.checkpoint_io import SAECheckpointIO
from wan.sae.logger import SAELogManager, get_test_logger
from wan.sae.sae_run_naming import SAERunLocator, load_json

logger = logging.getLogger(__name__)


##########################################################################################
# 参数配置区域
##########################################################################################

# --------------------------- 路径配置 ---------------------------
path_params = {
    "run_dir": "sae_runs/exp1",  # SAE训练目录
    "positive_prompts_file": "concepts/violence_positive.txt",  # 正例提示词文件
    "negative_prompts_file": "concepts/violence_negative.txt",  # 负例提示词文件
    "output_dir": "concept_vectors",  # 输出目录
}

# --------------------------- 层配置 ---------------------------
layer_params = {
    "hook_mode": "block_out",
    "hook_layers": "15,29",  # 要分析的层
}

# --------------------------- 提取配置 ---------------------------
extraction_params = {
    # 提取方法
    "method": "mean_diff",  # "mean_diff" | "max_diff" | "percentile_diff" | "contrastive"

    # 是否使用激活值的绝对值
    "use_abs": False,

    # 是否归一化概念向量
    "normalize": True,

    # 最小激活阈值（过滤噪声）
    "min_activation_threshold": 0.01,

    # 对比学习温度（用于contrastive方法）
    "contrastive_temp": 0.1,

    # 百分位数（用于percentile_diff方法）
    "percentile": 90,
}

# --------------------------- 统计配置 ---------------------------
stats_params = {
    # 保存top-k最激活特征
    "top_k_features": 50,

    # 计算特征选择度（selectivity）
    "compute_selectivity": True,

    # 选择度阈值
    "selectivity_threshold": 0.7,
}

# --------------------------- 系统配置 ---------------------------
system_params = {
    "device_id": 0,
    "seed": 0,
}


##########################################################################################
# 核心代码区域
##########################################################################################

@dataclass
class ConceptVector:
    """
    概念向量数据类
    """
    # 概念名称
    name: str = ""

    # 概念向量（SAE隐空间中的方向）
    vector: np.ndarray = field(default_factory=lambda: np.array([]))

    # 来源层
    layer_key: str = ""

    # 来源SAE运行目录
    run_dir: str = ""

    # 提取方法
    extraction_method: str = ""

    # 正例提示词列表
    positive_prompts: List[str] = field(default_factory=list)

    # 负例提示词列表
    negative_prompts: List[str] = field(default_factory=list)

    # 统计信息
    statistics: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "concept_name": self.name,
            "concept_vector": self.vector.tolist(),
            "layer_key": self.layer_key,
            "run_dir": self.run_dir,
            "extraction_method": self.extraction_method,
            "positive_prompts_count": len(self.positive_prompts),
            "negative_prompts_count": len(self.negative_prompts),
            "positive_prompts_sample": self.positive_prompts[:5] if self.positive_prompts else [],
            "negative_prompts_sample": self.negative_prompts[:5] if self.negative_prompts else [],
            "statistics": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.statistics.items()
            },
            "metadata": self.metadata,
        }

    def save(self, output_path: str) -> None:
        """保存概念向量"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存JSON（包含元信息）
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        # 保存numpy（仅向量，方便加载）
        np_path = output_path.with_suffix(".npy")
        np.save(np_path, self.vector)

        logger.info(f"概念向量已保存: {json_path}, {np_path}")

    @staticmethod
    def load(input_path: str) -> "ConceptVector":
        """加载概念向量"""
        input_path = Path(input_path)

        # 尝试加载JSON
        json_path = input_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 加载向量
            np_path = input_path.with_suffix(".npy")
            if np_path.exists():
                vector = np.load(np_path)
            else:
                vector = np.array(data.get("concept_vector", []))

            return ConceptVector(
                name=data.get("concept_name", ""),
                vector=vector,
                layer_key=data.get("layer_key", ""),
                run_dir=data.get("run_dir", ""),
                extraction_method=data.get("extraction_method", ""),
                statistics=data.get("statistics", {}),
                metadata=data.get("metadata", {}),
            )

        # 仅加载numpy
        np_path = input_path.with_suffix(".npy")
        if np_path.exists():
            vector = np.load(np_path)
            return ConceptVector(vector=vector)

        raise FileNotFoundError(f"找不到概念向量文件: {input_path}")


def load_prompts(file_path: str) -> List[str]:
    """从文件加载提示词（每行一条）"""
    path = Path(file_path)
    logger.debug(f"[LOAD] 开始加载提示词文件: {path}")

    if not path.exists():
        logger.error(f"[LOAD] 提示词文件不存在: {path}")
        raise FileNotFoundError(f"提示词文件不存在: {path}")

    prompts = []
    line_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)

    logger.info(f"从 {path} 加载了 {len(prompts)}/{line_count} 条有效提示词")
    logger.debug(f"[LOAD] 前3条示例: {prompts[:3] if prompts else 'N/A'}")
    return prompts


def compute_sae_activations(
    sae: SparseAutoEncoder,
    activations: np.ndarray,  # [T, L, C]
    device: torch.device,
    return_per_prompt: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    计算SAE激活值

    参数:
        sae: SAE模型
        activations: 输入激活 [T, L, C] 或 [B, T, L, C]
        device: 计算设备
        return_per_prompt: 是否返回每个提示词的激活（用于热力图）

    返回:
        默认: [d_hidden] 平均激活
        return_per_prompt=True: (z_mean, z_per_prompt)
            - z_mean: [d_hidden] 平均激活
            - z_per_prompt: [T, d_hidden] 每个提示词的激活
    """
    # 处理输入维度
    if activations.ndim == 3:
        T, L, C = activations.shape
        B = 1
        activations = activations.reshape(B, T, L, C)
    elif activations.ndim == 4:
        B, T, L, C = activations.shape
    else:
        raise ValueError(f"Unsupported activation shape: {activations.shape}")

    logger.debug(f"[COMPUTE] 输入激活: shape=[{B}, {T}, {L}, {C}], dtype={activations.dtype}")

    # 每个提示词单独编码
    z_per_prompt = []
    for i in range(T):
        # 提取第i个提示词的激活 [B, L, C] -> [B*L, C]
        x = torch.from_numpy(activations[:, i, :, :].reshape(-1, C)).float().to(device)

        with torch.no_grad():
            z, _, _ = sae.encode(x)  # [B*L, d_hidden]
            # 平均池化到 [d_hidden]
            z_mean_i = z.mean(dim=0).cpu().numpy()
            z_per_prompt.append(z_mean_i)

    # 转换为数组 [T, d_hidden]
    z_per_prompt = np.stack(z_per_prompt, axis=0)
    # 计算总体平均 [d_hidden]
    z_mean = z_per_prompt.mean(axis=0)

    logger.debug(f"[COMPUTE] SAE编码后: z_per_prompt.shape={z_per_prompt.shape}, sparsity={(z_per_prompt != 0).mean():.4f}")
    logger.debug(f"[COMPUTE] 平均激活: shape={z_mean.shape}, mean={z_mean.mean():.6f}, max={z_mean.max():.6f}")

    if return_per_prompt:
        return z_mean, z_per_prompt
    return z_mean


def extract_concept_vector_mean_diff(
    sae: SparseAutoEncoder,
    positive_activations: List[np.ndarray],
    negative_activations: List[np.ndarray],
    device: torch.device,
    use_abs: bool = False,
    normalize: bool = True,
    min_threshold: float = 0.01,
    save_per_prompt: bool = False,
    output_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    使用平均差分法提取概念向量

    concept_vector = mean(positive_activations) - mean(negative_activations)

    参数:
        save_per_prompt: 是否保存每个提示词的激活值
        output_dir: 保存目录（当save_per_prompt=True时必需）
    """
    logger.debug(f"[EXTRACT] mean_diff 开始: 正例={len(positive_activations)}, 负例={len(negative_activations)}, use_abs={use_abs}, normalize={normalize}, save_per_prompt={save_per_prompt}")

    # 计算正例平均激活
    logger.debug(f"[EXTRACT] 计算正例平均激活...")
    pos_activations = []
    pos_per_prompt_list = []  # 保存每个提示词的激活
    for i, act in enumerate(positive_activations):
        if save_per_prompt:
            z_mean, z_per = compute_sae_activations(sae, act, device, return_per_prompt=True)
            pos_per_prompt_list.append(z_per)
        else:
            z_mean = compute_sae_activations(sae, act, device)
        if use_abs:
            z_mean = np.abs(z_mean)
            if save_per_prompt:
                pos_per_prompt_list[-1] = np.abs(pos_per_prompt_list[-1])
        pos_activations.append(z_mean)
        logger.debug(f"[EXTRACT] 正例[{i}]: mean={z_mean.mean():.6f}, max={z_mean.max():.6f}")
    pos_mean = np.mean(pos_activations, axis=0)
    logger.debug(f"[EXTRACT] 正例平均: shape={pos_mean.shape}, mean={pos_mean.mean():.6f}, std={pos_mean.std():.6f}")

    # 计算负例平均激活
    logger.debug(f"[EXTRACT] 计算负例平均激活...")
    neg_activations = []
    neg_per_prompt_list = []  # 保存每个提示词的激活
    for i, act in enumerate(negative_activations):
        if save_per_prompt:
            z_mean, z_per = compute_sae_activations(sae, act, device, return_per_prompt=True)
            neg_per_prompt_list.append(z_per)
        else:
            z_mean = compute_sae_activations(sae, act, device)
        if use_abs:
            z_mean = np.abs(z_mean)
            if save_per_prompt:
                neg_per_prompt_list[-1] = np.abs(neg_per_prompt_list[-1])
        neg_activations.append(z_mean)
        logger.debug(f"[EXTRACT] 负例[{i}]: mean={z_mean.mean():.6f}, max={z_mean.max():.6f}")
    neg_mean = np.mean(neg_activations, axis=0)
    logger.debug(f"[EXTRACT] 负例平均: shape={neg_mean.shape}, mean={neg_mean.mean():.6f}, std={neg_mean.std():.6f}")

    # 计算差分
    concept_vector = pos_mean - neg_mean
    logger.debug(f"[EXTRACT] 差分向量: shape={concept_vector.shape}, mean={concept_vector.mean():.6f}, std={concept_vector.std():.6f}")

    # 应用阈值
    active_count_before = np.sum(concept_vector != 0)
    concept_vector[np.abs(concept_vector) < min_threshold] = 0
    active_count_after = np.sum(concept_vector != 0)
    logger.debug(f"[EXTRACT] 阈值过滤: threshold={min_threshold}, {active_count_before}->{active_count_after} 活跃特征")

    # 归一化
    if normalize:
        norm = np.linalg.norm(concept_vector)
        if norm > 0:
            concept_vector = concept_vector / norm
            logger.debug(f"[EXTRACT] 归一化: 原范数={norm:.6f}, 新范数={np.linalg.norm(concept_vector):.6f}")

    statistics = {
        "pos_mean_activation": pos_mean,
        "neg_mean_activation": neg_mean,
        "activation_difference": concept_vector,
        "pos_std": np.std(pos_activations, axis=0),
        "neg_std": np.std(neg_activations, axis=0),
    }

    # 保存每个提示词的激活值（用于热力图分析）
    if save_per_prompt and output_dir:
        logger.debug(f"[EXTRACT] 保存每个提示词的激活值到 {output_dir}...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 合并所有正例的 per-prompt 激活 [N_pos, d_hidden]
        if pos_per_prompt_list:
            pos_per_prompt = np.concatenate(pos_per_prompt_list, axis=0)  # [N_pos, d_hidden]
            neg_per_prompt = np.concatenate(neg_per_prompt_list, axis=0)  # [N_neg, d_hidden]

            # 保存为 .npz 文件（numpy压缩格式）
            per_prompt_path = output_path / "per_prompt_activations.npz"
            np.savez(
                per_prompt_path,
                pos_activations=pos_per_prompt,
                neg_activations=neg_per_prompt,
                pos_mean=pos_mean,
                neg_mean=neg_mean,
                concept_vector=concept_vector,
            )
            logger.info(f"Per-prompt activations saved: {per_prompt_path}")
            logger.debug(f"[EXTRACT] 正例激活矩阵: {pos_per_prompt.shape}, 负例激活矩阵: {neg_per_prompt.shape}")

            # 添加到统计信息
            statistics["pos_per_prompt"] = pos_per_prompt
            statistics["neg_per_prompt"] = neg_per_prompt
        else:
            logger.warning("[EXTRACT] pos_per_prompt_list is empty, nothing to save")

    logger.debug(f"[EXTRACT] mean_diff 完成: 输出shape={concept_vector.shape}")
    return concept_vector, statistics


def extract_concept_vector_contrastive(
    sae: SparseAutoEncoder,
    positive_activations: List[np.ndarray],
    negative_activations: List[np.ndarray],
    device: torch.device,
    temperature: float = 0.1,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    使用对比学习方法提取概念向量

    通过优化使得：
    - 正例在概念方向上投影较大
    - 负例在概念方向上投影较小
    """
    # 首先用mean_diff获得初始估计
    init_vector, _ = extract_concept_vector_mean_diff(
        sae, positive_activations, negative_activations, device, normalize=False
    )

    # 使用简单的迭代优化
    vector = torch.from_numpy(init_vector).float().to(device)
    vector.requires_grad = True

    optimizer = torch.optim.Adam([vector], lr=0.01)

    # 计算所有样本的激活
    pos_z = []
    for act in positive_activations[:50]:  # 限制样本数
        z = compute_sae_activations(sae, act, device)
        pos_z.append(torch.from_numpy(z).float().to(device))

    neg_z = []
    for act in negative_activations[:50]:
        z = compute_sae_activations(sae, act, device)
        neg_z.append(torch.from_numpy(z).float().to(device))

    # 对比学习优化
    for _ in range(100):
        optimizer.zero_grad()

        # 归一化方向向量
        v = vector / (torch.norm(vector) + 1e-8)

        # 正例投影（希望大）
        pos_proj = torch.stack([torch.dot(z, v) for z in pos_z])
        pos_loss = -torch.log(torch.sigmoid(pos_proj / temperature)).mean()

        # 负例投影（希望小）
        neg_proj = torch.stack([torch.dot(z, v) for z in neg_z])
        neg_loss = -torch.log(torch.sigmoid(-neg_proj / temperature)).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

    concept_vector = vector.detach().cpu().numpy()

    if normalize:
        norm = np.linalg.norm(concept_vector)
        if norm > 0:
            concept_vector = concept_vector / norm

    statistics = {
        "optimization_loss": loss.item(),
        "pos_mean_projection": torch.stack([torch.dot(z, v) for z in pos_z]).mean().item(),
        "neg_mean_projection": torch.stack([torch.dot(z, v) for z in neg_z]).mean().item(),
    }

    return concept_vector, statistics


def compute_feature_selectivity(
    concept_vector: np.ndarray,
    positive_activations: List[np.ndarray],
    negative_activations: List[np.ndarray],
    sae: SparseAutoEncoder,
    device: torch.device,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    计算特征选择度（selectivity）

    选择度 = (正例中激活该特征的比例) - (负例中激活该特征的比例)
    高选择度意味着该特征对该概念具有很强的区分性
    """
    d_hidden = len(concept_vector)

    # 统计正例激活
    pos_active = np.zeros(d_hidden)
    for act in positive_activations:
        z = compute_sae_activations(sae, act, device)
        pos_active += (np.abs(z) > 0.01).astype(float)
    pos_freq = pos_active / len(positive_activations)

    # 统计负例激活
    neg_active = np.zeros(d_hidden)
    for act in negative_activations:
        z = compute_sae_activations(sae, act, device)
        neg_active += (np.abs(z) > 0.01).astype(float)
    neg_freq = neg_active / len(negative_activations)

    # 计算选择度
    selectivity = pos_freq - neg_freq

    # 筛选高选择度特征
    high_selectivity_indices = np.where(np.abs(selectivity) > threshold)[0]

    return {
        "selectivity_scores": selectivity,
        "high_selectivity_indices": high_selectivity_indices.tolist(),
        "high_selectivity_count": len(high_selectivity_indices),
        "pos_activation_freq": pos_freq,
        "neg_activation_freq": neg_freq,
    }


def extract_concept_for_layer(
    run_dir: str,
    layer_key: str,
    positive_activations: List[np.ndarray],
    negative_activations: List[np.ndarray],
    positive_prompts: List[str],
    negative_prompts: List[str],
    concept_name: str,
    extraction_config: Dict[str, Any],
    stats_config: Dict[str, Any],
    device: torch.device,
    save_per_prompt: bool = False,
    per_prompt_output_dir: Optional[str] = None,
) -> ConceptVector:
    """
    为单层提取概念向量
    """
    logger.info(f"=" * 60)
    logger.info(f"提取概念 '{concept_name}' 从层 {layer_key}")
    logger.info(f"正例: {len(positive_activations)}, 负例: {len(negative_activations)}")
    logger.info(f"=" * 60)

    # 记录激活数据形状
    if positive_activations:
        logger.debug(f"[LAYER] 正例激活[0] shape: {positive_activations[0].shape}, dtype: {positive_activations[0].dtype}")
    if negative_activations:
        logger.debug(f"[LAYER] 负例激活[0] shape: {negative_activations[0].shape}, dtype: {negative_activations[0].dtype}")

    # 解析层信息
    hook_mode, layer_str = layer_key.split(".")
    layer_idx = int(layer_str.replace("layer", ""))
    logger.debug(f"[LAYER] 解析层: hook_mode={hook_mode}, layer_idx={layer_idx}")

    # 加载SAE（使用新的统一 IO 接口，自动兼容新旧格式）
    loc = SAERunLocator(run_dir=run_dir, hook_mode=hook_mode, layer_idx=layer_idx)
    logger.debug(f"[LAYER] SAE路径: {loc.latest_ckpt_path()}")

    try:
        io = SAECheckpointIO.load(loc, device=device, strict=True, allow_legacy=True)
        sae = io.sae
        sae_cfg = io.sae_config
        logger.debug(f"[LAYER] SAE配置来源: {getattr(io, '_config_source', 'checkpoint')}")
        if io._config_source == "json_fallback":
            logger.warning("从旧格式 .json 加载配置 [建议迁移]")
    except Exception as e:
        logger.error(f"加载SAE失败: {e}")
        raise

    logger.info(f"已加载SAE: d_model={sae_cfg.d_model}, d_hidden={sae_cfg.d_hidden}, sparsity={sae_cfg.sparsity}")
    logger.debug(f"[LAYER] SAE完整配置: {sae_cfg.to_dict()}")

    # 选择提取方法
    method = extraction_config["method"]

    if method == "mean_diff":
        concept_vector, statistics = extract_concept_vector_mean_diff(
            sae, positive_activations, negative_activations, device,
            use_abs=extraction_config["use_abs"],
            normalize=extraction_config["normalize"],
            min_threshold=extraction_config["min_activation_threshold"],
            save_per_prompt=save_per_prompt,
            output_dir=per_prompt_output_dir,
        )
    elif method == "contrastive":
        concept_vector, statistics = extract_concept_vector_contrastive(
            sae, positive_activations, negative_activations, device,
            temperature=extraction_config["contrastive_temp"],
            normalize=extraction_config["normalize"],
        )
    else:
        raise ValueError(f"未知的提取方法: {method}")

    # 计算top-k特征
    top_k = stats_config["top_k_features"]
    top_k_indices = np.argsort(np.abs(concept_vector))[-top_k:][::-1]
    top_k_values = concept_vector[top_k_indices]

    statistics["top_k_indices"] = top_k_indices
    statistics["top_k_values"] = top_k_values
    statistics["top_k_features"] = [
        {"index": int(idx), "value": float(val)}
        for idx, val in zip(top_k_indices, top_k_values)
    ]

    # 计算选择度
    if stats_config["compute_selectivity"]:
        logger.debug(f"[LAYER] 计算特征选择度... threshold={stats_config['selectivity_threshold']}")
        selectivity_stats = compute_feature_selectivity(
            concept_vector, positive_activations, negative_activations,
            sae, device, threshold=stats_config["selectivity_threshold"]
        )
        statistics["selectivity"] = selectivity_stats
        logger.debug(f"[LAYER] 选择度统计: high_selectivity_count={selectivity_stats.get('high_selectivity_count', 'N/A')}")

    # 创建概念向量对象
    logger.debug(f"[LAYER] 创建ConceptVector对象...")
    concept = ConceptVector(
        name=concept_name,
        vector=concept_vector,
        layer_key=layer_key,
        run_dir=run_dir,
        extraction_method=method,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        statistics=statistics,
        metadata={
            "sae_config": sae_cfg.to_dict(),
            "extraction_config": extraction_config,
            "stats_config": stats_config,
        }
    )

    logger.info(f"概念向量提取完成！")
    logger.info(f"  向量范数: {np.linalg.norm(concept_vector):.4f}")
    logger.info(f"  Top-5特征: {top_k_indices[:5].tolist()}")
    logger.info(f"  Top-5值: {top_k_values[:5].tolist()}")

    if stats_config["compute_selectivity"]:
        logger.info(f"  高选择度特征数: {selectivity_stats['high_selectivity_count']}")

    return concept


def pseudo_code_intervention():
    """
    干预生成的伪代码（供参考和实现）

    概念向量干预的基本思路：
    1. 在DiT生成过程中，hook特定层的输出
    2. 将hook的特征通过SAE编码
    3. 沿概念向量方向调整激活值
    4. 通过SAE解码回特征空间
    5. 继续正常生成
    """
    pseudo_code = '''
    # ========== 干预生成伪代码 ==========

    def intervene_generation(
        model,                    # DiT模型
        sae,                      # SAE模型
        concept_vector,           # 概念向量 [d_hidden]
        intervention_strength,    # 干预强度
        prompt,                   # 生成提示词
        num_steps,                # 扩散步数
    ):
        # 准备生成
        latents = initialize_noise(...)

        for t in timesteps:
            # 注册hook
            handles = register_hooks(model)

            # 前向传播
            with hook_activations() as activations:
                output = model(latents, t, prompt)

            # 获取hook的特征
            features = activations[layer_key]  # [B, L, C]

            # SAE编码
            z = sae.encode(features)  # [B, L, d_hidden]

            # 沿概念向量方向调整
            # 方式1: 加法干预
            z_intervened = z + intervention_strength * concept_vector

            # 方式2: 投影干预（沿方向调整）
            # projection = torch.dot(z, concept_vector)
            # z_intervened = z + intervention_strength * projection * concept_vector

            # 方式3: 条件干预（只在特定条件下调整）
            # if should_intervene(z):
            #     z_intervened = intervene(z, concept_vector, strength)

            # SAE解码
            features_intervened = sae.decode(z_intervened)

            # 替换原始特征（残差连接）
            activations[layer_key] = features + alpha * (features_intervened - features)

            # 继续生成
            latents = update_latents(output, latents, t)

            remove_hooks(handles)

        # 解码视频
        video = vae_decode(latents)
        return video

    # ========== 多概念干预 ==========

    def multi_concept_intervention(
        model,
        sae,
        concept_dict,             # {概念名: (概念向量, 干预强度)}
        prompt,
        num_steps,
    ):
        # 合并多个概念向量
        combined_vector = sum(
            strength * vector
            for vector, strength in concept_dict.values()
        )

        # 归一化
        combined_vector = combined_vector / len(concept_dict)

        return intervene_generation(
            model, sae, combined_vector, 1.0, prompt, num_steps
        )
    '''
    return pseudo_code


def main():
    parser = argparse.ArgumentParser(description="Extract concept vectors from SAE using contrastive prompts")
    parser.add_argument("--run_dir", type=str, default=path_params["run_dir"])
    parser.add_argument("--positive_file", type=str, default=path_params["positive_prompts_file"])
    parser.add_argument("--negative_file", type=str, default=path_params["negative_prompts_file"])
    parser.add_argument("--output_dir", type=str, default=path_params["output_dir"])
    parser.add_argument("--concept_name", type=str, required=True, help="概念名称，如'violence'")
    parser.add_argument("--hook_mode", type=str, default=layer_params["hook_mode"])
    parser.add_argument("--hook_layers", type=str, default=layer_params["hook_layers"])
    parser.add_argument("--method", type=str, default=extraction_params["method"])
    parser.add_argument("--device_id", type=int, default=system_params["device_id"])
    parser.add_argument("--save_pseudo_code", action="store_true", help="保存干预伪代码")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # 设置设备
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载提示词
    positive_prompts = load_prompts(args.positive_file)
    negative_prompts = load_prompts(args.negative_file)

    # TODO: 这里应该加载对应的激活值
    # 目前用随机数据作为占位符，实际使用时应从采集的数据加载
    logger.warning("注意：当前使用随机激活值作为示例，实际使用时请加载真实采集的激活值")

    # 加载SAE配置以获取维度
    layer_indices = [int(x.strip()) for x in args.hook_layers.split(",") if x.strip()]
    d_hidden = sae_params["d_hidden"]

    # 模拟加载激活值（实际使用时替换为真实数据）
    positive_activations = [np.random.randn(30, 100, 1536).astype(np.float32) for _ in positive_prompts]
    negative_activations = [np.random.randn(30, 100, 1536).astype(np.float32) for _ in negative_prompts]

    # 为每一层提取概念向量
    extraction_config = {
        "method": args.method,
        "use_abs": extraction_params["use_abs"],
        "normalize": extraction_params["normalize"],
        "min_activation_threshold": extraction_params["min_activation_threshold"],
        "contrastive_temp": extraction_params["contrastive_temp"],
    }

    stats_config = {
        "top_k_features": stats_params["top_k_features"],
        "compute_selectivity": stats_params["compute_selectivity"],
        "selectivity_threshold": stats_params["selectivity_threshold"],
    }

    for layer_idx in layer_indices:
        layer_key = f"{args.hook_mode}.layer{layer_idx}"

        try:
            concept = extract_concept_for_layer(
                run_dir=args.run_dir,
                layer_key=layer_key,
                positive_activations=positive_activations,
                negative_activations=negative_activations,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                concept_name=args.concept_name,
                extraction_config=extraction_config,
                stats_config=stats_config,
                device=device,
            )

            # 保存
            output_path = output_dir / f"{args.concept_name}_{layer_key}"
            concept.save(str(output_path))

        except Exception as e:
            logger.error(f"提取层 {layer_key} 失败: {e}")
            continue

    # 保存伪代码
    if args.save_pseudo_code:
        pseudo_code = pseudo_code_intervention()
        pseudo_path = output_dir / "intervention_pseudo_code.py"
        with open(pseudo_path, "w", encoding="utf-8") as f:
            f.write(pseudo_code)
        logger.info(f"干预伪代码已保存: {pseudo_path}")

    logger.info("=" * 60)
    logger.info("概念提取完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
