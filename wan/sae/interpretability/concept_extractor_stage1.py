"""
SAE概念提取 - 阶段一：激活值采集（GPU必需）

功能：
1. 加载已训练的SAE模型（通过SAECheckpointIO，兼容新旧格式）
2. 配对处理正负提示词（逐对提取，保证紧凑）
3. 采集SAE隐藏层状态（z）和可选的DiT隐藏状态
4. 按层/类/极性分层存储
5. 支持增量采集

【重要设计决策】使用简单Euler而非原生调度器的原因：

1. 调度器状态管理过于复杂：
   - UniPC/DPM++调度器有复杂的内部状态（step_index, model_outputs, sigmas等）
   - 多步求解器（2-3阶）对调用顺序极其敏感
   - CFG需要两次model()调用，会干扰调度器的历史记录

2. 调度器与Hook机制冲突：
   - Hook回调中的任何操作可能干扰调度器的 delicate 状态
   - 已发现多起因索引越界导致的崩溃（见fm_solvers.py中的多处修复注释）

3. Euler方法的优势：
   - 无内部状态，稳定可靠
   - 与SAE训练代码完全一致（特征分布匹配）
   - 概念提取不需要生成视频，只关心隐藏状态分布
   - 相对差异（pos-neg）不受采样方法影响

4. 参考：sae_train_t2v_1_3b.py同样使用Euler而非调度器

使用示例（基础版）：
    python wan/sae/interpretability/concept_extractor_stage1.py \
        --model_path "./Wan2.1-T2V-1.3B" \
        --sae_run_dir "sae_runs/exp1" \
        --pos_prompts "final_cleaned/pos_prompt_3.txt" \
        --neg_prompts "final_cleaned/neg_prompt_3.txt" \
        --category "violence" \
        --output_root "activations" \
        --sae_layers "15,29" \
        --sampling_steps 30

使用示例（推荐开启CFG，特征更准确）：
    python wan/sae/interpretability/concept_extractor_stage1.py \
        --model_path "./Wan2.1-T2V-1.3B" \
        --sae_run_dir "sae_runs/exp1" \
        --pos_prompts "final_cleaned/pos_prompt_3.txt" \
        --neg_prompts "final_cleaned/neg_prompt_3.txt" \
        --category "violence" \
        --output_root "activations" \
        --sae_layers "15,29" \
        --use_cfg \
        --guide_scale 5.0 \
        --sampling_steps 30

存储结构：
    activations/
    ├── sae_layer15/
    │   └── violence/
    │       ├── pos/
    │       │   ├── activations.npy      # [N, T, L, d_hidden]
    │       │   ├── metadata.json        # [{idx, prompt, pair_idx}, ...]
    │       │   └── checkpoint.json      # 增量采集断点
    │       └── neg/
    │           ├── activations.npy
    │           └── metadata.json
    ├── dit_layer15/
    │   └── violence/
    │       ├── pos/
    │       └── neg/
    └── extraction_config.json           # 全局配置
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.cuda.amp as amp

# 复用训练代码的路径处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wan.configs.wan_t2v_1_3B import t2v_1_3B
from wan.modules.sae_new import SparseAutoEncoder
from wan.sae.checkpoint_io import SAECheckpointIO
from wan.sae.hooking import HookMode, register_dit_hooks, remove_hooks
from wan.sae.path_utils import resolve_dir, resolve_path, get_project_root
from wan.sae.prompt_io import PromptCleanConfig, load_prompts_from_dir
from wan.sae.sae_run_naming import SAERunLocator, load_json, save_json
from wan.text2video import WanT2V
from wan.sae.interpretability.activation_io import ActivationIO, SampleMetadata, ExtractionCheckpoint

logger = logging.getLogger(__name__)


##########################################################################################
# 参数配置（复用sae_train_t2v_1_3b.py的设计：代码内配置 + 命令行覆盖）
##########################################################################################

# --------------------------- 路径配置 ---------------------------
path_params = {
    # model_path: Wan 2.1 DiT模型权重目录
    # 必须与训练SAE时使用的模型一致
    "model_path": "./Wan2.1-T2V-1.3B",

    # sae_run_dir: SAE训练输出目录
    # 从中加载训练好的SAE权重
    "sae_run_dir": "sae_runs/exp1",

    # pos_prompts: 正样本提示词文件（包含概念）
    "pos_prompts": "final_cleaned/pos_prompt_3.txt",

    # neg_prompts: 负样本提示词文件（不包含概念）
    "neg_prompts": "final_cleaned/neg_prompt_3.txt",

    # output_root: 激活值输出根目录
    "output_root": "activations",
}

# --------------------------- 概念配置 ---------------------------
concept_params = {
    # category: 概念类别名称（如"violence", "pornography"）
    "category": "violence",

    # sae_layers: 要采集的SAE层（用SAE编码）
    # 格式: "15,29" 表示第15层和第29层
    "sae_layers": "15,29",

    # save_dit_layers: 要保存的DiT原始层（可选）
    # 格式: "15" 或 "15,29"，空字符串表示不保存
    "save_dit_layers": "",

    # hook_mode: Hook位置
    # 可选值: "self_attn" | "cross_attn" | "self_and_cross" | "block_out"
    # 必须与SAE训练时一致
    "hook_mode": "block_out",
}

# --------------------------- 扩散采样配置 ---------------------------
# 注意：本实现使用简单Euler方法替代Wan原生的UniPC/DPM++调度器
# 但保留CFG（Classifier-Free Guidance）支持
# 原因：Euler稳定无状态问题，与Hook机制兼容；CFG可提升特征质量
sampling_params = {
    # sampling_steps: 采样步数（时间步数）
    # Flow Matching通常30-50步
    "sampling_steps": 30,

    # use_cfg: 是否使用Classifier-Free Guidance
    # True = 运行两次前向（条件+无条件），False = 只运行一次
    # 建议：概念提取建议开启CFG，特征分布更准确
    "use_cfg": False,

    # guide_scale: CFG guidance scale（use_cfg=True时有效）
    "guide_scale": 5.0,

    # negative_prompt: 负提示词（use_cfg=True时用于无条件分支）
    # 空字符串表示使用模型默认负提示词
    "negative_prompt": "",
}

# --------------------------- 生成尺寸配置 ---------------------------
generation_params = {
    # 必须与SAE训练时一致
    "size_w": 832,
    "size_h": 480,
    "frame_num": 81,  # 必须是4n+1
}

# --------------------------- 增量采集配置 ---------------------------
resume_params = {
    # enabled: 是否启用增量采集
    "enabled": False,

    # checkpoint_file: 断点文件路径
    # 空字符串表示自动检测
    "checkpoint_file": "",
}

# --------------------------- 系统配置 ---------------------------
system_params = {
    "device_id": 0,
    "seed": 0,
}

# --------------------------- 提示词清洗配置 ---------------------------
prompt_clean_params = {
    "min_len": 8,
    "max_len": 400,
}


##########################################################################################
# 工具函数（复用训练代码）
##########################################################################################

def compute_latent_shape(cfg, size_wh: Tuple[int, int], frame_num: int, vae_z_dim: int) -> List[int]:
    """计算latent形状（复用sae_train_t2v_1_3b.py）"""
    w, h = size_wh
    F = frame_num
    t_lat = (F - 1) // cfg.vae_stride[0] + 1
    h_lat = h // cfg.vae_stride[1]
    w_lat = w // cfg.vae_stride[2]
    return [vae_z_dim, t_lat, h_lat, w_lat]


def compute_seq_len(cfg, latent_shape: List[int], sp_size: int) -> int:
    """计算序列长度（复用sae_train_t2v_1_3b.py）"""
    _, t_lat, h_lat, w_lat = latent_shape
    seq_len = math.ceil(
        (h_lat * w_lat) / (cfg.patch_size[1] * cfg.patch_size[2]) * t_lat / sp_size
    ) * sp_size
    return int(seq_len)


def parse_layers(s: str) -> List[int]:
    """解析层索引字符串（复用sae_train_t2v_1_3b.py）"""
    if not s or s.strip() == "":
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def compute_timesteps(num_steps: int, num_train_timesteps: int = 1000) -> np.ndarray:
    """
    计算采样时间步（复用训练代码逻辑）。

    从num_train_timesteps-1线性递减到0，均匀采样num_steps个点。
    例如：num_steps=30, num_train_timesteps=1000 -> [999, 966, 932, ..., 0]
    """
    return np.linspace(num_train_timesteps - 1, 0, num_steps, dtype=np.int32)


##########################################################################################
# 核心类：配对激活采集器
##########################################################################################

class PairedActivationCollector:
    """
    配对激活采集器。

    核心设计：
    1. 逐对处理正负提示词（保证紧凑）
    2. 先正后负（或在一个batch中同时处理）
    3. 支持增量采集（断点续传）
    4. SAE加载复用SAECheckpointIO（兼容新旧格式）
    """

    def __init__(
        self,
        model_path: str,
        sae_run_dir: str,
        hook_mode: HookMode,
        sae_layers: List[int],
        save_dit_layers: List[int],
        device: torch.device,
        size_wh: Tuple[int, int] = (832, 480),
        frame_num: int = 81,
        use_cfg: bool = False,
        guide_scale: float = 5.0,
        seed: int = 0,
    ):
        self.model_path = model_path
        self.sae_run_dir = sae_run_dir
        self.hook_mode = hook_mode
        self.sae_layers = sae_layers
        self.save_dit_layers = save_dit_layers
        self.device = device
        self.size_wh = size_wh
        self.frame_num = frame_num
        self.use_cfg = use_cfg
        self.guide_scale = guide_scale
        self.seed = seed

        # 加载WanT2V（复用训练代码逻辑）
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

        # 计算所有hook层（SAE层 + DiT层，去重）
        all_hook_layers = list(set(sae_layers + save_dit_layers))
        self.all_hook_layers = sorted(all_hook_layers)

        # 加载SAE模型（复用SAECheckpointIO，兼容新旧格式）
        self.saes: Dict[int, SparseAutoEncoder] = {}
        if sae_layers:
            logger.info("-" * 60)
            logger.info("加载SAE模型（复用SAECheckpointIO接口）...")
            for layer_idx in sae_layers:
                loc = SAERunLocator(
                    run_dir=sae_run_dir,
                    hook_mode=hook_mode,
                    layer_idx=layer_idx,
                )
                try:
                    # 复用现有接口：自动兼容新旧格式
                    io = SAECheckpointIO.load(
                        loc,
                        device=device,
                        strict=True,
                        allow_legacy=True,  # 允许旧格式回退
                    )
                    io.sae.eval()
                    self.saes[layer_idx] = io.sae
                    logger.info(
                        f"  已加载SAE layer{layer_idx}: "
                        f"d_hidden={io.sae_config.d_hidden}, "
                        f"source={io._config_source if hasattr(io, '_config_source') else 'unknown'}"
                    )
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

    def _collect_single_prompt(
        self,
        prompt: str,
        timesteps: np.ndarray,
        pair_idx: int = 0,
        negative_prompt: str = "",
    ) -> Dict[str, np.ndarray]:
        """
        采集单个提示词在所有时间步的激活。

        采样方法：
        - 使用简单Euler方法（而非Wan原生的UniPC/DPM++调度器）
          原因：Euler无复杂状态管理，与Hook机制兼容，与SAE训练代码一致
        - 支持CFG（可选）：开启时运行两次前向（条件+无条件），提升特征质量

        Returns:
            {
                "sae_layer{idx}": [T, L, d_hidden],
                "dit_layer{idx}": [T, L, 1536],  # 如果配置了保存
            }
        """
        # 文本编码
        self.wrapper.text_encoder.model.to(self.device)
        context = self.wrapper.text_encoder([prompt], self.device)

        # CFG准备
        context_null = None
        if self.use_cfg:
            neg_p = negative_prompt if negative_prompt else self.wrapper.sample_neg_prompt
            context_null = self.wrapper.text_encoder([neg_p], self.device)

        # 初始化结果收集器
        sae_results = {layer_idx: [] for layer_idx in self.sae_layers}
        dit_results = {layer_idx: [] for layer_idx in self.save_dit_layers}

        # 生成初始噪声（复用训练代码逻辑）
        # 使用pair_idx作为种子的一部分，确保同一pair正负样本使用不同的噪声
        seed_g = torch.Generator(device=self.device)
        seed_val = (self.seed + pair_idx * 1000) % (2**32)  # 避免溢出
        seed_g.manual_seed(seed_val)
        latent = torch.randn(
            self.latent_shape[0],
            self.latent_shape[1],
            self.latent_shape[2],
            self.latent_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g,
        )
        # 保持与训练代码一致：使用list
        latents = [latent]

        # 遍历所有时间步
        for t_val in timesteps:
            t_tensor = torch.tensor([t_val], device=self.device, dtype=torch.long)

            # Hook收集
            raw: Dict[str, torch.Tensor] = {}

            def on_tensor(k: str, v: torch.Tensor):
                raw[k] = v  # [1, L, C]

            handles = register_dit_hooks(
                self.model,
                hook_layers=self.all_hook_layers,
                hook_mode=self.hook_mode,
                on_tensor=on_tensor,
            )

            try:
                # 使用简单Euler方法（稳定可靠，与Hook机制兼容）
                with torch.no_grad(), amp.autocast(dtype=self.cfg.param_dtype):
                    # 条件分支
                    noise_pred_cond = self.model(
                        latents, t=t_tensor, context=context, seq_len=self.seq_len
                    )

                    # CFG无条件分支
                    if self.use_cfg and context_null is not None:
                        noise_pred_uncond = self.model(
                            latents, t=t_tensor, context=context_null, seq_len=self.seq_len
                        )
                        # 合并预测
                        pred = [
                            u + self.guide_scale * (c - u)
                            for c, u in zip(noise_pred_cond, noise_pred_uncond)
                        ]
                    else:
                        pred = noise_pred_cond

                    # Euler更新: z_next = z - pred * dt
                    dt = 1.0 / len(timesteps)
                    new_latents = []
                    for p, z in zip(pred, latents):
                        z_next = z - p * dt
                        new_latents.append(z_next)
                    latents = new_latents

            finally:
                remove_hooks(handles)

            # 处理收集到的激活
            for layer_idx in self.all_hook_layers:
                layer_key = f"{self.hook_mode}.layer{layer_idx}"
                if layer_key not in raw:
                    continue

                h = raw[layer_key]  # [1, L, C]
                # 保持原始形状用于保存 [L, C]
                h_np = h.cpu().numpy()[0]  # [L, C]

                # 保存DiT状态（如果配置了）- 保存原始DiT输出
                if layer_idx in dit_results:
                    dit_results[layer_idx].append(h_np)

                # SAE编码并保存（如果配置了）
                if layer_idx in self.saes:
                    sae = self.saes[layer_idx]
                    with torch.no_grad():
                        # SAE期望2D输入 [N, d_model]，需要reshape
                        # h: [1, L, C] -> [L, C] (即 [N, d_model])
                        h_2d = h.reshape(-1, h.shape[-1])  # [L, C]
                        z, _, _ = sae.encode(h_2d)  # [L, d_hidden]
                        z_np = z.cpu().numpy()  # [L, d_hidden]
                    sae_results[layer_idx].append(z_np)

        # 合并时间步 [T, L, D]
        output = {}
        for layer_idx, acts in sae_results.items():
            if acts:
                output[f"sae_layer{layer_idx}"] = np.stack(acts, axis=0)
        for layer_idx, acts in dit_results.items():
            if acts:
                output[f"dit_layer{layer_idx}"] = np.stack(acts, axis=0)

        return output

    def collect_pair(
        self,
        pos_prompt: str,
        neg_prompt: str,
        pair_idx: int,
        timesteps: np.ndarray,
        negative_prompt: str = "",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        采集一对正负提示词的激活。

        采样方法：简单Euler（稳定，与Hook机制兼容）
        CFG支持：如果use_cfg=True，每个样本会运行两次前向（条件+无条件）

        Returns:
            (pos_activations, neg_activations)
            每个dict包含sae_layer{idx}和dit_layer{idx}
        """
        logger.info(f"  采集pair {pair_idx}: 正样本")
        pos_acts = self._collect_single_prompt(
            pos_prompt, timesteps, pair_idx,
            negative_prompt=negative_prompt if self.use_cfg else ""
        )

        logger.info(f"  采集pair {pair_idx}: 负样本")
        neg_acts = self._collect_single_prompt(
            neg_prompt, timesteps, pair_idx,
            negative_prompt=negative_prompt if self.use_cfg else ""
        )

        return pos_acts, neg_acts


##########################################################################################
# 存储管理器（使用ActivationIO统一接口）
##########################################################################################

class ActivationStorage:
    """
    激活值存储管理器。

    基于ActivationIO的薄包装，提供阶段一特定的存储逻辑。
    目录结构：{output_root}/sae_layer{idx}/{category}/{polarity}/
    """

    def __init__(
        self,
        output_root: str,
        category: str,
        sae_layers: List[int],
        save_dit_layers: List[int],
    ):
        self.io = ActivationIO(output_root)
        self.category = category
        self.sae_layers = sae_layers
        self.save_dit_layers = save_dit_layers

        # 创建目录结构
        self._create_directories()

    def _create_directories(self):
        """创建所有需要的目录结构"""
        for layer_idx in self.sae_layers:
            self.io.ensure_layer_structure("sae", layer_idx, self.category)
        for layer_idx in self.save_dit_layers:
            self.io.ensure_layer_structure("dit", layer_idx, self.category)

    def save_activations_incremental(
        self,
        layer_type: str,
        layer_idx: int,
        polarity: str,
        new_activations: np.ndarray,
        new_metadata: List[Dict[str, Any]],
    ):
        """增量保存激活值和元信息"""
        # 转换元信息格式
        metadata_objs = [SampleMetadata(**m) for m in new_metadata]

        # 使用ActivationIO保存
        self.io.save_activations(
            layer_type, layer_idx, self.category, polarity,
            new_activations, append=True
        )
        self.io.save_metadata(
            layer_type, layer_idx, self.category, polarity,
            metadata_objs, append=True
        )

    def save_checkpoint(
        self,
        layer_idx: int,
        polarity: str,
        checkpoint: ExtractionCheckpoint,
    ):
        """保存增量采集断点"""
        self.io.save_checkpoint("sae", layer_idx, self.category, polarity, checkpoint)

    def load_checkpoint(
        self,
        layer_idx: int,
        polarity: str,
    ) -> Optional[ExtractionCheckpoint]:
        """加载增量采集断点"""
        return self.io.load_checkpoint("sae", layer_idx, self.category, polarity)


##########################################################################################
# 主流程
##########################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="SAE概念提取 - 阶段一：激活值采集（配对提取，GPU必需）"
    )

    # 路径参数
    parser.add_argument("--model_path", type=str, default=path_params["model_path"])
    parser.add_argument("--sae_run_dir", type=str, default=path_params["sae_run_dir"])
    parser.add_argument("--pos_prompts", type=str, default=path_params["pos_prompts"])
    parser.add_argument("--neg_prompts", type=str, default=path_params["neg_prompts"])
    parser.add_argument("--output_root", type=str, default=path_params["output_root"])

    # 概念参数
    parser.add_argument("--category", type=str, default=concept_params["category"])
    parser.add_argument("--sae_layers", type=str, default=concept_params["sae_layers"])
    parser.add_argument("--save_dit_layers", type=str, default=concept_params["save_dit_layers"])
    parser.add_argument("--hook_mode", type=str, default=concept_params["hook_mode"],
                        choices=["self_attn", "cross_attn", "self_and_cross", "block_out"])

    # 采样参数
    parser.add_argument(
        "--sampling_steps", type=int, default=sampling_params["sampling_steps"],
        help="采样步数（Euler方法），建议30-50步"
    )
    parser.add_argument(
        "--use_cfg", action="store_true", default=sampling_params["use_cfg"],
        help="是否使用Classifier-Free Guidance（推荐开启，特征更准确）"
    )
    parser.add_argument(
        "--guide_scale", type=float, default=sampling_params["guide_scale"],
        help="CFG guidance scale（use_cfg=True时有效）"
    )
    parser.add_argument(
        "--negative_prompt", type=str, default=sampling_params["negative_prompt"],
        help="负提示词（use_cfg=True时用于无条件分支，空=使用默认）"
    )

    # 生成尺寸
    parser.add_argument("--size_w", type=int, default=generation_params["size_w"])
    parser.add_argument("--size_h", type=int, default=generation_params["size_h"])
    parser.add_argument("--frame_num", type=int, default=generation_params["frame_num"])

    # 增量采集
    parser.add_argument("--resume", action="store_true", default=resume_params["enabled"])

    # 系统
    parser.add_argument("--device_id", type=int, default=system_params["device_id"])
    parser.add_argument("--seed", type=int, default=system_params["seed"])

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # 设置设备
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # 解析层配置
    sae_layers = parse_layers(args.sae_layers)
    save_dit_layers = parse_layers(args.save_dit_layers)

    if not sae_layers:
        raise ValueError("必须至少指定一个SAE层（--sae_layers）")

    logger.info("=" * 60)
    logger.info("阶段一：激活值采集（配对提取，使用简单Euler方法）")
    logger.info("=" * 60)
    logger.info(f"概念类别: {args.category}")
    logger.info(f"SAE层: {sae_layers}")
    logger.info(f"DiT层（可选）: {save_dit_layers if save_dit_layers else 'None'}")
    logger.info(f"Hook模式: {args.hook_mode}")
    logger.info(f"采样方法: 简单Euler（稳定可靠，与Hook机制兼容）")
    logger.info(f"CFG: {args.use_cfg}（{'启用' if args.use_cfg else '禁用'}）")
    if args.use_cfg:
        logger.info(f"  guide_scale: {args.guide_scale}")
    logger.info(f"时间步数: {args.sampling_steps}")
    logger.info("=" * 60)

    # 加载提示词
    clean_cfg = PromptCleanConfig(
        min_len=prompt_clean_params["min_len"],
        max_len=prompt_clean_params["max_len"],
    )

    # 从文件加载（支持单文件或目录）
    pos_path = Path(args.pos_prompts)
    if pos_path.is_dir():
        pos_prompts = load_prompts_from_dir(str(pos_path), clean_cfg=clean_cfg)
    else:
        with open(pos_path, "r", encoding="utf-8") as f:
            pos_prompts = [line.strip() for line in f if line.strip()]

    neg_path = Path(args.neg_prompts)
    if neg_path.is_dir():
        neg_prompts = load_prompts_from_dir(str(neg_path), clean_cfg=clean_cfg)
    else:
        with open(neg_path, "r", encoding="utf-8") as f:
            neg_prompts = [line.strip() for line in f if line.strip()]

    if len(pos_prompts) != len(neg_prompts):
        raise ValueError(f"正负提示词数量不匹配: pos={len(pos_prompts)}, neg={len(neg_prompts)}")

    num_pairs = len(pos_prompts)
    logger.info(f"加载了 {num_pairs} 对提示词")

    # 计算时间步
    timesteps = compute_timesteps(args.sampling_steps)
    logger.info(f"时间步: {timesteps[:5]}...{timesteps[-5:]}")

    # 初始化采集器（使用简单Euler方法，支持可选CFG）
    collector = PairedActivationCollector(
        model_path=args.model_path,
        sae_run_dir=args.sae_run_dir,
        hook_mode=args.hook_mode,
        sae_layers=sae_layers,
        save_dit_layers=save_dit_layers,
        device=device,
        size_wh=(args.size_w, args.size_h),
        frame_num=args.frame_num,
        use_cfg=args.use_cfg,
        guide_scale=args.guide_scale,
        seed=args.seed,
    )

    # 初始化存储
    storage = ActivationStorage(
        output_root=args.output_root,
        category=args.category,
        sae_layers=sae_layers,
        save_dit_layers=save_dit_layers,
    )

    # 检查增量采集
    completed_pairs: set = set()
    if args.resume:
        for layer_idx in sae_layers:
            ckpt = storage.load_checkpoint(layer_idx, "pos")
            if ckpt:
                completed_pairs.update(ckpt.completed_pair_indices)
                logger.info(f"从断点恢复layer{layer_idx}: 已完成 {len(ckpt.completed_pair_indices)}/{num_pairs}")

    # 保存全局配置（使用ActivationIO统一接口）
    config_data = {
        "category": args.category,
        "sae_run_dir": args.sae_run_dir,
        "sae_checkpoints": {
            f"layer{idx}": str(SAERunLocator(
                run_dir=args.sae_run_dir,
                hook_mode=args.hook_mode,
                layer_idx=idx,
            ).latest_ckpt_path())
            for idx in sae_layers
        },
        "hook_mode": args.hook_mode,
        "sae_layers": sae_layers,
        "save_dit_layers": save_dit_layers,
        "timesteps": timesteps.tolist(),
        "num_timesteps": len(timesteps),
        "sampling_method": "simple_euler",  # 明确标注使用简单Euler
        "sampling_method_note": "使用简单Euler而非原生调度器（UniPC/DPM++），避免与Hook机制冲突",
        "use_cfg": args.use_cfg,
        "guide_scale": args.guide_scale if args.use_cfg else None,
        "size_wh": [args.size_w, args.size_h],
        "frame_num": args.frame_num,
        "num_pairs": num_pairs,
        "extraction_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    storage.io.save_config(config_data)

    # 主循环：逐对提取
    logger.info("=" * 60)
    logger.info("开始配对提取...")
    logger.info("=" * 60)

    for pair_idx in range(num_pairs):
        if pair_idx in completed_pairs:
            logger.info(f"[{pair_idx+1}/{num_pairs}] 已跳过（断点）")
            continue

        pos_p = pos_prompts[pair_idx]
        neg_p = neg_prompts[pair_idx]

        logger.info(f"[{pair_idx+1}/{num_pairs}] 处理pair {pair_idx}")
        logger.info(f"  正: {pos_p[:50]}...")
        logger.info(f"  负: {neg_p[:50]}...")

        # 采集（使用简单Euler方法，支持可选CFG）
        pos_acts, neg_acts = collector.collect_pair(
            pos_prompt=pos_p,
            neg_prompt=neg_p,
            pair_idx=pair_idx,
            timesteps=timesteps,
            negative_prompt=args.negative_prompt,
        )

        # 准备元信息
        pos_meta = [{
            "idx": pair_idx,
            "pair_idx": pair_idx,
            "prompt": pos_p,
            "category": args.category,
            "polarity": "pos",
        }]
        neg_meta = [{
            "idx": pair_idx,
            "pair_idx": pair_idx,
            "prompt": neg_p,
            "category": args.category,
            "polarity": "neg",
        }]

        # 保存SAE激活
        for layer_idx in sae_layers:
            key = f"sae_layer{layer_idx}"
            if key in pos_acts:
                storage.save_activations_incremental(
                    "sae", layer_idx, "pos",
                    pos_acts[key][np.newaxis, ...],  # [1, T, L, D]
                    pos_meta,
                )
            if key in neg_acts:
                storage.save_activations_incremental(
                    "sae", layer_idx, "neg",
                    neg_acts[key][np.newaxis, ...],
                    neg_meta,
                )

        # 保存DiT激活（如果有）
        for layer_idx in save_dit_layers:
            key = f"dit_layer{layer_idx}"
            if key in pos_acts:
                storage.save_activations_incremental(
                    "dit", layer_idx, "pos",
                    pos_acts[key][np.newaxis, ...],
                    pos_meta,
                )
            if key in neg_acts:
                storage.save_activations_incremental(
                    "dit", layer_idx, "neg",
                    neg_acts[key][np.newaxis, ...],
                    neg_meta,
                )

        # 更新断点
        completed_pairs.add(pair_idx)
        for layer_idx in sae_layers:
            ckpt = ExtractionCheckpoint(
                completed_pair_indices=sorted(list(completed_pairs)),
                total_pairs=num_pairs,
            )
            storage.save_checkpoint(layer_idx, "pos", ckpt)
            storage.save_checkpoint(layer_idx, "neg", ckpt)

        # 显存状态
        if torch.cuda.is_available() and (pair_idx + 1) % 10 == 0:
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            logger.info(f"  显存使用: {allocated:.2f} GB")

    logger.info("=" * 60)
    logger.info("阶段一完成！")
    logger.info(f"输出目录: {args.output_root}")
    logger.info(f"总对数: {num_pairs}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
