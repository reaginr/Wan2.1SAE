"""
[已弃用] SAE概念提取器（单文件版）- 请使用新的分离式模块

警告：此文件已弃用，仅用于向后兼容。新代码请使用以下分离式模块：

阶段一：激活值采集（GPU必需）
    python wan/sae/interpretability/concept_extractor_stage1.py \
        --model_path "./Wan2.1-T2V-1.3B" \
        --sae_run_dir "sae_runs/exp1" \
        --prompt_file "final_cleaned/pos_prompt_3" \
        --output_path "activations/violence_pos.npz" \
        --save_dit_states "15,29" \
        --save_sae_states "15,29" \
        --sampling_steps "0,5,10,15,20,25,29"

阶段二：概念提取（CPU即可，低内存）
    python wan/sae/interpretability/concept_extractor_stage2.py \
        --concept_name "violence" \
        --pos_activations "activations/violence_pos.npz" \
        --neg_activations "activations/violence_neg.npz" \
        --layer_key "block_out.layer15" \
        --output_dir "concept_vectors"

迁移说明：
1. 阶段一和阶段二已分离为独立文件，实现完全解耦
2. 阶段二不再依赖PyTorch或模型加载，仅需NumPy
3. 两个文件各自包含完整的参数配置和命令行接口
4. 此文件保留原有功能，但不再维护新功能

如需继续使用此文件，原有命令行参数仍然有效：
    python wan/sae/interpretability/concept_extractor_offline.py collect ...
    python wan/sae/interpretability/concept_extractor_offline.py extract ...
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.cuda.amp as amp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wan.configs.wan_t2v_1_3B import t2v_1_3B
from wan.modules.sae_new import SAEConfig, SparseAutoEncoder
from wan.sae.checkpoint_io import SAECheckpointIO
from wan.sae.hooking import HookMode, register_dit_hooks, remove_hooks
from wan.sae.prompt_io import PromptCleanConfig, load_prompts_from_dir
from wan.sae.sae_run_naming import SAERunLocator
from wan.text2video import WanT2V

logger = logging.getLogger(__name__)


##########################################################################################
# 阶段一：采集参数配置（类似sae_train_1_3b.py的代码内配置）
# 学术意义与建议值详见每个参数的注释
##########################################################################################

# --------------------------- 路径配置 ---------------------------
path_params = {
    # model_path: Wan 2.1 DiT模型权重目录
    # 学术意义: 预训练DiT权重，用于生成隐藏状态
    # 建议值: "./Wan2.1-T2V-1.3B"
    "model_path": "./Wan2.1-T2V-1.3B",

    # sae_run_dir: SAE训练输出目录
    # 学术意义: 从中加载训练好的SAE权重
    # 必须指定，用于SAE编码
    "sae_run_dir": "sae_runs/exp1",

    # prompt_file: 提示词文件路径（文件或目录）
    # 建议值: "final_cleaned/pos_prompt_3"
    "prompt_file": "final_cleaned/pos_prompt_3",

    # output_path: 激活值保存路径
    # 建议值: "activations/concept_name_pos.npz"
    "output_path": "activations/test_pos.npz",
}

# --------------------------- 内容采集配置 ---------------------------
content_params = {
    # save_dit_states: 保存哪些DiT层的隐藏状态
    # 格式: "15,29" 表示保存layer15和layer29
    # 空字符串 "" 表示不保存DiT状态
    # 学术意义: 原始DiT特征，用于分析或重新编码
    # 建议值: 与SAE训练时hook的层一致
    "save_dit_states": "15,29",

    # save_sae_states: 保存哪些SAE层的编码状态
    # 格式: "15,29" 表示对layer15和layer29的SAE进行编码并保存
    # 空字符串 "" 表示不保存SAE状态
    # 学术意义: SAE隐空间z，直接用于概念提取
    # 建议值: 与save_dit_states一致，或根据需求选择
    "save_sae_states": "15,29",

    # hook_mode: Hook模式
    # 可选值: "self_attn" | "cross_attn" | "self_and_cross" | "block_out"
    # 必须与SAE训练时一致
    "hook_mode": "block_out",
}

# --------------------------- 时间步配置 ---------------------------
timestep_params = {
    # sampling_steps: 采集哪些时间步的激活
    # 格式1: "0,5,10,15,20,25,29" - 指定具体步数
    # 格式2: "30" - 均匀采样30步
    # 格式3: "all" - 采集所有步数（不推荐，文件过大）
    # 学术意义: 覆盖扩散轨迹的不同噪声水平
    # 建议值: "0,5,10,15,20,25,29" 或 "30"
    "sampling_steps": "0,5,10,15,20,25,29",

    # frame_num: 生成帧数
    # 建议值: 81（与训练一致）
    "frame_num": 81,
}

# --------------------------- 尺寸配置 ---------------------------
size_params = {
    # 生成视频尺寸
    # 必须与SAE训练时一致以保证特征分布一致
    "size_w": 832,
    "size_h": 480,
}

# --------------------------- 批处理配置 ---------------------------
batch_params = {
    # batch_size: 每批处理的提示词数量
    # 建议值: 4-8（根据显存调整）
    "batch_size": 4,

    # max_prompts: 最大处理提示词数
    # 建议值: 全部（不限制）或指定数量用于测试
    "max_prompts": None,  # None表示不限制
}

# --------------------------- 系统配置 ---------------------------
system_params = {
    # device_id: GPU设备ID
    "device_id": 0,

    # seed: 随机种子
    "seed": 0,

    # min_len/max_len: 提示词长度过滤
    "min_len": 8,
    "max_len": 400,
}


# --------------------------- 阶段二：提取参数 ---------------------------
extract_params = {
    "run_dir": "sae_runs/exp1",
    "concept_name": "violence",
    "method": "mean_diff",  # "mean_diff" | "contrastive"
    "normalize": True,
    "batch_size": 32,  # 流式加载批次大小
    "min_threshold": 0.01,
}


##########################################################################################
# 工具函数
##########################################################################################

def compute_latent_shape(cfg, size_wh, frame_num: int, vae_z_dim: int) -> List[int]:
    """计算latent形状。"""
    w, h = size_wh
    F = frame_num
    t_lat = (F - 1) // cfg.vae_stride[0] + 1
    h_lat = h // cfg.vae_stride[1]
    w_lat = w // cfg.vae_stride[2]
    return [vae_z_dim, t_lat, h_lat, w_lat]


def compute_seq_len(cfg, latent_shape, sp_size: int) -> int:
    """计算序列长度。"""
    _, t_lat, h_lat, w_lat = latent_shape
    seq_len = math.ceil((h_lat * w_lat) / (cfg.patch_size[1] * cfg.patch_size[2]) * t_lat / sp_size) * sp_size
    return int(seq_len)


def parse_layers(s: Optional[str]) -> List[int]:
    """解析层索引字符串。"""
    if not s or s.strip() == "":
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_timesteps(s: str, max_steps: int = 1000) -> List[int]:
    """
    解析时间步配置字符串。

    格式:
    - "0,5,10,15" -> [0, 5, 10, 15]
    - "30" -> 均匀采样30步
    - "all" -> 所有时间步（不推荐）
    """
    if s.strip().lower() == "all":
        return list(range(max_steps))

    parts = [p.strip() for p in s.split(",") if p.strip()]

    # 如果只有一个数字，解释为步数，均匀采样
    if len(parts) == 1:
        num_steps = int(parts[0])
        # 均匀采样num_steps个时间步
        step_indices = np.linspace(0, max_steps - 1, num_steps, dtype=int).tolist()
        return step_indices

    # 否则解析具体步数
    return [int(p) for p in parts]


##########################################################################################
# 阶段一：联合采集器（DiT + SAE）
##########################################################################################

class JointActivationCollector:
    """
    联合激活采集器 - 同时采集DiT隐藏状态和SAE编码状态

    采集流程：
    1. 加载WanT2V（DiT）和SAE模型
    2. 对每个提示词，在指定时间步运行DiT前向传播
    3. Hook收集DiT隐藏状态 [B, L, C]
    4. 实时通过SAE编码得到隐状态z [B, L, d_hidden]
    5. 按配置保存DiT状态、SAE状态或两者
    """

    def __init__(
        self,
        model_path: str,
        sae_run_dir: str,
        hook_mode: str,
        dit_layers: List[int],
        sae_layers: List[int],
        device: torch.device,
        size_wh: Tuple[int, int] = (832, 480),
        frame_num: int = 81,
    ):
        self.model_path = model_path
        self.sae_run_dir = sae_run_dir
        self.hook_mode = hook_mode
        self.dit_layers = dit_layers
        self.sae_layers = sae_layers
        self.device = device
        self.size_wh = size_wh
        self.frame_num = frame_num

        # 加载DiT模型
        logger.info("=" * 60)
        logger.info("加载WanT2V模型...")
        cfg = t2v_1_3B
        self.wrapper = WanT2V(
            config=cfg,
            checkpoint_dir=model_path,
            device_id=device.index if device.index is not None else 0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
        )
        self.model = self.wrapper.model
        self.model.eval().requires_grad_(False).to(device)
        self.cfg = cfg

        # 计算latent形状
        vae_z_dim = self.wrapper.vae.model.z_dim
        self.latent_shape = compute_latent_shape(cfg, size_wh, frame_num, vae_z_dim)
        self.seq_len = compute_seq_len(cfg, self.latent_shape, self.wrapper.sp_size)
        logger.info(f"Latent shape={self.latent_shape}, seq_len={self.seq_len}")

        # 加载SAE模型
        self.saes = {}
        if sae_layers:
            logger.info("-" * 60)
            logger.info("加载SAE模型...")
            for layer_idx in sae_layers:
                loc = SAERunLocator(run_dir=sae_run_dir, hook_mode=hook_mode, layer_idx=layer_idx)
                try:
                    io = SAECheckpointIO.load(loc, device=device, strict=True, allow_legacy=True)
                    io.sae.eval()
                    self.saes[layer_idx] = io.sae
                    logger.info(f"  已加载SAE layer{layer_idx}: d_hidden={io.sae_config.d_hidden}")
                except Exception as e:
                    logger.error(f"  加载SAE layer{layer_idx}失败: {e}")
                    raise
            logger.info("-" * 60)

        # 显存信息
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            logger.info(f"模型加载后显存: {allocated:.2f}GB / {total:.2f}GB")
        logger.info("=" * 60)

    def collect_single_prompt_timesteps(
        self,
        prompt: str,
        timesteps: List[int],
    ) -> Dict[str, Any]:
        """
        采集单个提示词在多个时间步的激活。

        返回: {
            "dit": {layer_idx: [T, L, C]},  # 如果save_dit_states
            "sae": {layer_idx: [T, L, d_hidden]},  # 如果save_sae_states
            "metadata": {...}
        }
        """
        # 文本编码（只编码一次，复用）
        self.wrapper.text_encoder.model.to(self.device)
        context = self.wrapper.text_encoder([prompt], self.device)

        # 初始化结果收集器
        dit_results = {layer_idx: [] for layer_idx in self.dit_layers}
        sae_results = {layer_idx: [] for layer_idx in self.sae_layers}

        # 对每个时间步进行前向传播
        for t_idx, t_val in enumerate(timesteps):
            logger.debug(f"  时间步 {t_idx+1}/{len(timesteps)}: t={t_val}")

            # 准备噪声latent
            t = torch.tensor([t_val], device=self.device, dtype=torch.long)
            x = torch.randn(1, *self.latent_shape, device=self.device, dtype=torch.float32)
            x_list = [x]

            # Hook收集DiT激活
            raw: Dict[str, torch.Tensor] = {}

            def on_tensor(k: str, v: torch.Tensor):
                """Hook回调"""
                raw[k] = v  # [1, L, C]

            # 注册hooks
            handles = register_dit_hooks(
                self.model,
                hook_layers=self.dit_layers,
                hook_mode=self.hook_mode,
                on_tensor=on_tensor
            )

            try:
                with torch.no_grad(), amp.autocast(dtype=self.cfg.param_dtype):
                    _ = self.model(x_list, t=t, context=context, seq_len=self.seq_len)
            finally:
                remove_hooks(handles)

            # 处理收集到的激活
            for layer_idx in self.dit_layers:
                layer_key = f"{self.hook_mode}.layer{layer_idx}"
                if layer_key not in raw:
                    logger.warning(f"  未收集到 {layer_key}")
                    continue

                h = raw[layer_key]  # [1, L, C]
                h_np = h.cpu().numpy()[0]  # [L, C]

                # 保存DiT状态
                if layer_idx in self.dit_layers:
                    dit_results[layer_idx].append(h_np)

                # SAE编码并保存
                if layer_idx in self.saes:
                    sae = self.saes[layer_idx]
                    with torch.no_grad():
                        z, _, _ = sae.encode(h)  # [1, L, d_hidden]
                        z_np = z.cpu().numpy()[0]  # [L, d_hidden]
                    sae_results[layer_idx].append(z_np)

        # 合并时间步
        output = {
            "dit": {},
            "sae": {},
            "metadata": {
                "timesteps": timesteps,
                "num_timesteps": len(timesteps),
            }
        }

        for layer_idx, acts in dit_results.items():
            if acts:
                output["dit"][layer_idx] = np.stack(acts, axis=0)  # [T, L, C]

        for layer_idx, acts in sae_results.items():
            if acts:
                output["sae"][layer_idx] = np.stack(acts, axis=0)  # [T, L, d_hidden]

        return output

    def collect_all(
        self,
        prompts: List[str],
        timesteps: List[int],
    ) -> Dict[str, Any]:
        """
        采集所有提示词的激活值。

        返回: {
            "dit_{layer_idx}": [N, T, L, C],
            "sae_{layer_idx}": [N, T, L, d_hidden],
            "prompts": [...],
            "timesteps": [...],
            "metadata": {...}
        }
        """
        N = len(prompts)
        T = len(timesteps)

        logger.info(f"开始采集 {N} 个提示词，每个 {T} 个时间步")
        logger.info(f"DiT层: {self.dit_layers if self.dit_layers else 'None'}")
        logger.info(f"SAE层: {self.sae_layers if self.sae_layers else 'None'}")

        # 初始化结果收集器
        dit_results = {layer_idx: [] for layer_idx in self.dit_layers}
        sae_results = {layer_idx: [] for layer_idx in self.sae_layers}

        for i, prompt in enumerate(prompts):
            logger.info(f"[{i+1}/{N}] 处理: {prompt[:50]}...")

            result = self.collect_single_prompt_timesteps(prompt, timesteps)

            for layer_idx, act in result["dit"].items():
                dit_results[layer_idx].append(act)

            for layer_idx, act in result["sae"].items():
                sae_results[layer_idx].append(act)

            # 显存状态
            if torch.cuda.is_available() and (i + 1) % 5 == 0:
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(f"  显存使用: {allocated:.2f} GB")

        # 合并所有样本
        output = {}

        for layer_idx, acts_list in dit_results.items():
            if acts_list:
                key = f"dit_{self.hook_mode}_layer{layer_idx}"
                output[key] = np.stack(acts_list, axis=0)  # [N, T, L, C]
                logger.info(f"{key}: shape={output[key].shape}")

        for layer_idx, acts_list in sae_results.items():
            if acts_list:
                key = f"sae_{self.hook_mode}_layer{layer_idx}"
                output[key] = np.stack(acts_list, axis=0)  # [N, T, L, d_hidden]
                logger.info(f"{key}: shape={output[key].shape}")

        # 元数据
        output["prompts"] = np.array(prompts, dtype=object)
        output["timesteps"] = np.array(timesteps)

        metadata = {
            "num_prompts": N,
            "num_timesteps": T,
            "timesteps": timesteps,
            "dit_layers": self.dit_layers,
            "sae_layers": self.sae_layers,
            "hook_mode": self.hook_mode,
            "model_path": self.model_path,
            "sae_run_dir": self.sae_run_dir,
            "size_wh": self.size_wh,
            "frame_num": self.frame_num,
        }
        output["metadata_json"] = json.dumps(metadata)

        return output

    def save_activations(
        self,
        activations: Dict[str, Any],
        output_path: str,
    ):
        """保存激活值到npz文件。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 计算总大小
        total_size_mb = 0
        for key, arr in activations.items():
            if isinstance(arr, np.ndarray):
                size_mb = arr.nbytes / 1024**2
                total_size_mb += size_mb
                logger.info(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, size={size_mb:.2f}MB")

        # 保存
        np.savez_compressed(output_path, **activations)

        logger.info("=" * 60)
        logger.info(f"激活值已保存: {output_path}")
        logger.info(f"总大小: {total_size_mb:.2f} MB (压缩后)")
        logger.info("=" * 60)


##########################################################################################
# 阶段二：流式概念提取（保持不变）
##########################################################################################

class StreamingActivationLoader:
    """流式激活值加载器（从npz文件流式加载SAE状态）"""

    def __init__(self, npz_path: str, layer_key: str, state_type: str = "sae", batch_size: int = 32):
        """
        Args:
            npz_path: npz文件路径
            layer_key: 如 "block_out.layer15"
            state_type: "dit" 或 "sae"
            batch_size: 流式批次大小
        """
        self.npz_path = Path(npz_path)
        self.layer_key = layer_key
        self.state_type = state_type
        self.batch_size = batch_size

        # 构建存储key
        hook_mode, layer_str = layer_key.split(".")
        layer_idx = int(layer_str.replace("layer", ""))
        self.save_key = f"{state_type}_{hook_mode}_layer{layer_idx}"

        # 延迟加载文件信息
        with np.load(self.npz_path, mmap_mode='r') as data:
            if self.save_key not in data:
                available = [k for k in data.keys() if not k.endswith("_json")]
                raise KeyError(f"文件中找不到 {self.save_key}，可用keys: {available}")

            self.activations = data[self.save_key]  # 内存映射
            self.shape = self.activations.shape
            self.dtype = self.activations.dtype
            self.num_samples = self.shape[0]

            # 加载提示词
            self.prompts = list(data["prompts"]) if "prompts" in data else []

            # 解析元信息
            if "metadata_json" in data:
                self.metadata = json.loads(str(data["metadata_json"]))
            else:
                self.metadata = {}

        logger.info(f"流式加载器: {npz_path}")
        logger.info(f"  类型: {state_type}, 层: {layer_key}")
        logger.info(f"  shape: {self.shape}, 样本数: {self.num_samples}")

    def __len__(self) -> int:
        return self.num_samples

    def iter_batches(self) -> Iterator[Tuple[np.ndarray, List[str]]]:
        """批量迭代"""
        for i in range(0, self.num_samples, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_samples)
            batch_acts = np.array(self.activations[i:end_idx])
            batch_prompts = self.prompts[i:end_idx] if self.prompts else []
            yield batch_acts, batch_prompts


@dataclass
class RunningMean:
    """增量计算均值"""
    count: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.array([]))

    def update(self, new_values: np.ndarray) -> None:
        """new_values: [B, D]"""
        batch_count = new_values.shape[0]
        batch_mean = new_values.mean(axis=0)

        if self.count == 0:
            self.mean = batch_mean
        else:
            delta = batch_mean - self.mean
            self.mean += delta * batch_count / (self.count + batch_count)

        self.count += batch_count

    def get_mean(self) -> np.ndarray:
        return self.mean


def extract_concept_streaming(
    pos_loader: StreamingActivationLoader,
    neg_loader: StreamingActivationLoader,
    method: str = "mean_diff",
    normalize: bool = True,
    min_threshold: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    流式概念提取。
    注意：这里直接加载SAE隐状态z，无需再次编码。
    """
    logger.info("=" * 60)
    logger.info("开始流式概念提取")
    logger.info(f"正例: {len(pos_loader)}, 负例: {len(neg_loader)}")
    logger.info(f"方法: {method}, 归一化: {normalize}")
    logger.info("=" * 60)

    # 流式计算均值
    pos_stats = RunningMean()
    neg_stats = RunningMean()

    # 处理正例
    logger.info("处理正例...")
    for batch_acts, _ in pos_loader.iter_batches():
        # batch_acts: [B, T, L, d_hidden]
        # 平均池化到 [B, d_hidden]
        B = batch_acts.shape[0]
        z_batch = batch_acts.reshape(B, -1, batch_acts.shape[-1]).mean(axis=1)
        pos_stats.update(z_batch)

    # 处理负例
    logger.info("处理负例...")
    for batch_acts, _ in neg_loader.iter_batches():
        B = batch_acts.shape[0]
        z_batch = batch_acts.reshape(B, -1, batch_acts.shape[-1]).mean(axis=1)
        neg_stats.update(z_batch)

    # 计算概念向量
    pos_mean = pos_stats.get_mean()
    neg_mean = neg_stats.get_mean()
    concept_vector = pos_mean - neg_mean

    logger.info(f"正例均值: shape={pos_mean.shape}")
    logger.info(f"负例均值: shape={neg_mean.shape}")

    # 阈值过滤
    active_before = np.sum(np.abs(concept_vector) >= min_threshold)
    concept_vector[np.abs(concept_vector) < min_threshold] = 0
    active_after = np.sum(concept_vector != 0)
    logger.info(f"阈值过滤: {active_before} -> {active_after} 活跃特征")

    # 归一化
    if normalize:
        norm = np.linalg.norm(concept_vector)
        if norm > 0:
            concept_vector = concept_vector / norm

    statistics = {
        "pos_count": pos_stats.count,
        "neg_count": neg_stats.count,
        "active_features": int(active_after),
    }

    return concept_vector, statistics


##########################################################################################
# 主流程
##########################################################################################

def stage_collect(args):
    """阶段一：联合采集DiT和SAE隐状态"""
    logger.info("=" * 60)
    logger.info("阶段一：联合采集DiT和SAE隐状态")
    logger.info("=" * 60)

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # 解析层配置
    dit_layers = parse_layers(args.save_dit_states)
    sae_layers = parse_layers(args.save_sae_states)
    timesteps = parse_timesteps(args.sampling_steps)

    if not dit_layers and not sae_layers:
        raise ValueError("必须至少保存DiT状态或SAE状态之一")

    logger.info(f"DiT层: {dit_layers}")
    logger.info(f"SAE层: {sae_layers}")
    logger.info(f"时间步: {timesteps[:10]}... (共{len(timesteps)}个)")

    # 加载提示词
    prompt_path = Path(args.prompt_file)
    if prompt_path.is_dir():
        clean_cfg = PromptCleanConfig(min_len=args.min_len, max_len=args.max_len)
        prompts = load_prompts_from_dir(str(prompt_path), clean_cfg=clean_cfg)
    else:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    logger.info(f"加载了 {len(prompts)} 条提示词")

    # 初始化采集器
    collector = JointActivationCollector(
        model_path=args.model_path,
        sae_run_dir=args.sae_run_dir,
        hook_mode=args.hook_mode,
        dit_layers=dit_layers,
        sae_layers=sae_layers,
        device=device,
        size_wh=(args.size_w, args.size_h),
        frame_num=args.frame_num,
    )

    # 采集激活
    activations = collector.collect_all(
        prompts=prompts,
        timesteps=timesteps,
    )

    # 保存
    collector.save_activations(activations, args.output_path)

    logger.info("阶段一完成！")
    return args.output_path


def stage_extract(args):
    """阶段二：概念提取（仅CPU，不加载DiT模型）"""
    logger.info("=" * 60)
    logger.info("阶段二：概念向量提取")
    logger.info("=" * 60)

    # 阶段二只需要CPU，因为只进行简单的向量运算
    # 但如果要加载SAE进行验证，可能需要GPU

    # 流式加载器
    pos_loader = StreamingActivationLoader(
        args.pos_activations, args.layer_key, state_type="sae", batch_size=args.batch_size
    )
    neg_loader = StreamingActivationLoader(
        args.neg_activations, args.layer_key, state_type="sae", batch_size=args.batch_size
    )

    # 提取概念向量
    concept_vector, statistics = extract_concept_streaming(
        pos_loader=pos_loader,
        neg_loader=neg_loader,
        method=args.method,
        normalize=args.normalize,
        min_threshold=args.min_threshold,
    )

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{args.concept_name}_{args.layer_key}"
    npy_path = output_dir / f"{output_name}.npy"
    json_path = output_dir / f"{output_name}.json"

    np.save(npy_path, concept_vector)

    # Top-k特征
    top_k = 50
    top_indices = np.argsort(np.abs(concept_vector))[-top_k:][::-1]

    result = {
        "concept_name": args.concept_name,
        "layer_key": args.layer_key,
        "method": args.method,
        "vector_shape": list(concept_vector.shape),
        "norm": float(np.linalg.norm(concept_vector)),
        "top_k_features": [
            {"index": int(idx), "value": float(concept_vector[idx])}
            for idx in top_indices[:top_k]
        ],
        "statistics": statistics,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info("=" * 60)
    logger.info("概念提取完成！")
    logger.info(f"概念向量: {npy_path}")
    logger.info(f"范数: {result['norm']:.4f}")
    logger.info(f"活跃特征: {statistics['active_features']}")
    logger.info("=" * 60)


def print_collect_config():
    """打印采集配置"""
    logger.info("=" * 60)
    logger.info("阶段一配置（采集）")
    logger.info("=" * 60)
    for name, params in [
        ("路径配置", path_params),
        ("内容采集配置", content_params),
        ("时间步配置", timestep_params),
        ("尺寸配置", size_params),
        ("批处理配置", batch_params),
        ("系统配置", system_params),
    ]:
        logger.info(f"\n【{name}】")
        for k, v in params.items():
            logger.info(f"  {k}: {v}")


def main():
    import warnings
    warnings.warn(
        "concept_extractor_offline.py 已弃用，请使用分离式模块：\n"
        "  阶段一: concept_extractor_stage1.py\n"
        "  阶段二: concept_extractor_stage2.py",
        DeprecationWarning,
        stacklevel=2
    )

    parser = argparse.ArgumentParser(description="SAE概念提取（两阶段离线分离式，支持DiT+SAE联合采集）[已弃用]")
    subparsers = parser.add_subparsers(dest="stage", help="阶段选择")

    # ========== 阶段一：采集 ==========
    collect_parser = subparsers.add_parser("collect", help="阶段一：采集激活值")

    # 路径参数
    collect_parser.add_argument("--model_path", type=str, default=path_params["model_path"])
    collect_parser.add_argument("--sae_run_dir", type=str, default=path_params.get("sae_run_dir"))
    collect_parser.add_argument("--prompt_file", type=str, default=path_params["prompt_file"])
    collect_parser.add_argument("--output_path", type=str, default=path_params["output_path"])

    # 内容采集参数
    collect_parser.add_argument("--save_dit_states", type=str, default=content_params["save_dit_states"])
    collect_parser.add_argument("--save_sae_states", type=str, default=content_params["save_sae_states"])
    collect_parser.add_argument("--hook_mode", type=str, default=content_params["hook_mode"])

    # 时间步参数
    collect_parser.add_argument("--sampling_steps", type=str, default=timestep_params["sampling_steps"])
    collect_parser.add_argument("--frame_num", type=int, default=timestep_params["frame_num"])

    # 尺寸参数
    collect_parser.add_argument("--size_w", type=int, default=size_params["size_w"])
    collect_parser.add_argument("--size_h", type=int, default=size_params["size_h"])

    # 批处理参数
    collect_parser.add_argument("--batch_size", type=int, default=batch_params["batch_size"])
    collect_parser.add_argument("--max_prompts", type=int, default=batch_params["max_prompts"])

    # 系统参数
    collect_parser.add_argument("--device_id", type=int, default=system_params["device_id"])
    collect_parser.add_argument("--seed", type=int, default=system_params["seed"])
    collect_parser.add_argument("--min_len", type=int, default=system_params["min_len"])
    collect_parser.add_argument("--max_len", type=int, default=system_params["max_len"])

    # 打印配置
    collect_parser.add_argument("--print_config", action="store_true", help="打印配置后退出")

    # ========== 阶段二：提取 ==========
    extract_parser = subparsers.add_parser("extract", help="阶段二：提取概念向量")
    extract_parser.add_argument("--run_dir", type=str, default=extract_params["run_dir"])
    extract_parser.add_argument("--concept_name", type=str, required=True)
    extract_parser.add_argument("--pos_activations", type=str, required=True)
    extract_parser.add_argument("--neg_activations", type=str, required=True)
    extract_parser.add_argument("--layer_key", type=str, required=True)
    extract_parser.add_argument("--output_dir", type=str, default=extract_params["run_dir"])
    extract_parser.add_argument("--method", type=str, default=extract_params["method"])
    extract_parser.add_argument("--normalize", action="store_true", default=extract_params["normalize"])
    extract_parser.add_argument("--batch_size", type=int, default=extract_params["batch_size"])
    extract_parser.add_argument("--min_threshold", type=float, default=extract_params["min_threshold"])

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.stage == "collect":
        if args.print_config:
            print_collect_config()
            return
        stage_collect(args)
    elif args.stage == "extract":
        stage_extract(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
