"""
SAE干预生成器 - 通过概念向量干预视频生成

功能：
1. 加载已训练的SAE和概念向量
2. 在视频生成过程中对特定层进行干预
3. 支持多个概念向量的组合干预
4. 持久化干预配置和生成结果

干预配置格式：
{
    "prompt": "生成提示词",
    "video_path": "output/video_001.mp4",
    "interventions": [
        {
            "concept_name": "violence",
            "layer_key": "block_out.layer15",
            "strength": 0.5,           # 干预强度
            "method": "additive",      # "additive" | "multiplicative" | "projection"
            "timestep_range": [0, 30]  # 干预的时间步范围
        },
        ...
    ],
    "metadata": {
        "run_dir": "sae_runs/exp1",
        "checkpoint_dir": "./Wan2.1-T2V-1.3B",
        "timestamp": "2025-04-01T12:00:00",
        "seed": 42,
    }
}
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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.cuda.amp as amp

# 修复导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wan.configs.wan_t2v_1_3B import t2v_1_3B
from wan.modules.sae_new import SAEConfig, SparseAutoEncoder
from wan.sae.checkpoint_io import SAECheckpointIO
from wan.sae.hooking import HookMode, register_dit_hooks, remove_hooks
from wan.sae.logger import SAELogManager, get_steering_logger
from wan.sae.sae_run_naming import SAERunLocator, load_json
from wan.text2video import WanT2V

logger = logging.getLogger(__name__)


##########################################################################################
# 参数配置区域
##########################################################################################

# --------------------------- 路径配置 ---------------------------
path_params = {
    "checkpoint_dir": "~/Wan/Wan2.1-T2V-1.3B",
    "run_dir": "sae_runs/exp1",  # SAE训练目录
    "concept_vectors_dir": "concept_vectors",  # 概念向量目录
    "output_dir": "steering_outputs",  # 输出目录
}

# --------------------------- 生成配置 ---------------------------
generation_params = {
    "prompt": "A person fighting in the street",  # 提示词
    "size_w": 832,
    "size_h": 480,
    "frame_num": 81,
    "sampling_steps": 30,
    "shift": 5.0,
    "seed": 0,
}

# --------------------------- 干预配置 ---------------------------
steering_params = {
    # 干预列表
    "interventions": [
        # 示例干预配置
        # {
        #     "concept_name": "violence",
        #     "layer_key": "block_out.layer15",
        #     "strength": 0.3,
        #     "method": "additive",
        #     "timestep_range": [0, 30],
        # }
    ],

    # 干预方法说明：
    # - "additive": z_new = z + strength * concept_vector
    # - "multiplicative": z_new = z * (1 + strength * concept_vector)
    # - "projection": 只保留沿concept_vector方向的分量并调整
    # - "clamp": 限制concept_vector方向的激活值范围

    # 全局干预强度缩放
    "global_strength_scale": 1.0,

    # 是否动态调整干预强度（随时间步衰减）
    "dynamic_strength": False,
    "strength_decay": "linear",  # "linear" | "exponential" | "cosine"
}

# --------------------------- 系统配置 ---------------------------
system_params = {
    "device_id": 0,
    "save_intermediate": False,  # 是否保存中间帧
    "offload_text_encoder": True,
}


##########################################################################################
# 核心代码区域
##########################################################################################

@dataclass
class InterventionConfig:
    """
    单个干预配置
    """
    concept_name: str = ""  # 概念名称
    layer_key: str = ""  # 目标层，如"block_out.layer15"
    strength: float = 0.0  # 干预强度（可正可负）
    method: str = "additive"  # 干预方法
    timestep_range: List[int] = field(default_factory=lambda: [0, 1000])  # 时间步范围 [start, end]

    # 高级选项
    feature_mask: Optional[List[int]] = None  # 只干预特定特征索引
    apply_to: str = "all"  # "all" | "spatial" | "temporal"  # 应用范围

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_name": self.concept_name,
            "layer_key": self.layer_key,
            "strength": self.strength,
            "method": self.method,
            "timestep_range": self.timestep_range,
            "feature_mask": self.feature_mask,
            "apply_to": self.apply_to,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InterventionConfig":
        return InterventionConfig(
            concept_name=data.get("concept_name", ""),
            layer_key=data.get("layer_key", ""),
            strength=data.get("strength", 0.0),
            method=data.get("method", "additive"),
            timestep_range=data.get("timestep_range", [0, 1000]),
            feature_mask=data.get("feature_mask"),
            apply_to=data.get("apply_to", "all"),
        )


@dataclass
class SteeringSession:
    """
    干预会话配置（用于持久化）
    """
    prompt: str = ""
    video_path: str = ""
    interventions: List[InterventionConfig] = field(default_factory=list)

    # 生成参数
    size_w: int = 832
    size_h: int = 480
    frame_num: int = 81
    sampling_steps: int = 30
    seed: int = 0

    # 源配置
    run_dir: str = ""
    checkpoint_dir: str = ""
    concept_vectors_dir: str = ""

    # 元数据
    timestamp: str = ""
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "video_path": self.video_path,
            "interventions": [i.to_dict() for i in self.interventions],
            "size_w": self.size_w,
            "size_h": self.size_h,
            "frame_num": self.frame_num,
            "sampling_steps": self.sampling_steps,
            "seed": self.seed,
            "run_dir": self.run_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "concept_vectors_dir": self.concept_vectors_dir,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }

    def save(self, output_path: str) -> None:
        """保存会话配置"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"会话配置已保存: {path}")

    @staticmethod
    def load(input_path: str) -> "SteeringSession":
        """加载会话配置"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return SteeringSession(
            prompt=data.get("prompt", ""),
            video_path=data.get("video_path", ""),
            interventions=[InterventionConfig.from_dict(i) for i in data.get("interventions", [])],
            size_w=data.get("size_w", 832),
            size_h=data.get("size_h", 480),
            frame_num=data.get("frame_num", 81),
            sampling_steps=data.get("sampling_steps", 30),
            seed=data.get("seed", 0),
            run_dir=data.get("run_dir", ""),
            checkpoint_dir=data.get("checkpoint_dir", ""),
            concept_vectors_dir=data.get("concept_vectors_dir", ""),
            timestamp=data.get("timestamp", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            metadata=data.get("metadata", {}),
        )


class ConceptVectorManager:
    """
    概念向量管理器

    管理多个概念向量的加载和缓存
    """

    def __init__(self, concept_vectors_dir: str):
        self.concept_vectors_dir = Path(concept_vectors_dir)
        self.cache: Dict[str, np.ndarray] = {}

    def load_concept_vector(self, concept_name: str, layer_key: str) -> Optional[np.ndarray]:
        """
        加载概念向量

        文件名格式: {concept_name}_{layer_key}.npy
        例如: violence_block_out.layer15.npy
        """
        cache_key = f"{concept_name}_{layer_key}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # 尝试多种文件名格式
        possible_names = [
            f"{concept_name}_{layer_key}.npy",
            f"{concept_name}_{layer_key.replace('.', '_')}.npy",
            f"{concept_name}/{layer_key}.npy",
        ]

        for name in possible_names:
            path = self.concept_vectors_dir / name
            if path.exists():
                vector = np.load(path)
                self.cache[cache_key] = vector
                logger.info(f"加载概念向量: {path}")
                return vector

        logger.error(f"找不到概念向量: {concept_name} / {layer_key}")
        return None

    def get_concept_vector(self, concept_name: str, layer_key: str) -> Optional[np.ndarray]:
        """获取概念向量（带缓存）"""
        return self.load_concept_vector(concept_name, layer_key)


class SAEIntervener:
    """
    SAE干预器

    在DiT生成过程中执行SAE干预
    """

    def __init__(
        self,
        sae: SparseAutoEncoder,
        concept_vector: np.ndarray,
        intervention_config: InterventionConfig,
        global_scale: float = 1.0,
    ):
        self.sae = sae
        self.concept_vector = torch.from_numpy(concept_vector).float().to(sae.config.d_model)
        self.config = intervention_config
        self.global_scale = global_scale

        # 确保维度匹配
        if len(concept_vector) != sae.config.d_hidden:
            raise ValueError(
                f"概念向量维度 {len(concept_vector)} 与SAE d_hidden {sae.config.d_hidden} 不匹配"
            )

    def compute_strength(self, timestep: int, total_steps: int) -> float:
        """计算当前时间步的干预强度"""
        base_strength = self.config.strength * self.global_scale

        # 检查时间步范围
        start_t, end_t = self.config.timestep_range
        if timestep < start_t or timestep > end_t:
            return 0.0

        # 动态强度调整
        if steering_params["dynamic_strength"]:
            progress = 1.0 - (timestep / total_steps)
            decay_type = steering_params["strength_decay"]

            if decay_type == "linear":
                scale = progress
            elif decay_type == "exponential":
                scale = math.exp(-3 * (1 - progress))
            elif decay_type == "cosine":
                scale = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
            else:
                scale = 1.0

            return base_strength * scale

        return base_strength

    def intervene(self, features: torch.Tensor, timestep: int, total_steps: int) -> torch.Tensor:
        """
        对特征进行干预

        features: [B, L, C]
        返回: 干预后的特征 [B, L, C]
        """
        strength = self.compute_strength(timestep, total_steps)
        if abs(strength) < 1e-6:
            return features

        B, L, C = features.shape
        x = features.reshape(-1, C)  # [B*L, C]

        # SAE编码
        with torch.no_grad():
            z, _, _ = self.sae.encode(x)  # [B*L, d_hidden]

        # 应用干预
        if self.config.method == "additive":
            z_intervened = z + strength * self.concept_vector

        elif self.config.method == "multiplicative":
            z_intervened = z * (1 + strength * self.concept_vector)

        elif self.config.method == "projection":
            # 计算沿概念向量方向的投影
            projection = torch.sum(z * self.concept_vector, dim=-1, keepdim=True)
            z_intervened = z + strength * projection * self.concept_vector

        elif self.config.method == "clamp":
            # 限制概念向量方向的值
            projection = torch.sum(z * self.concept_vector, dim=-1, keepdim=True)
            clamped_projection = torch.clamp(projection, -abs(strength), abs(strength))
            z_intervened = z - projection * self.concept_vector + clamped_projection * self.concept_vector

        else:
            raise ValueError(f"未知的干预方法: {self.config.method}")

        # 特征掩码（如果指定）
        if self.config.feature_mask is not None:
            mask = torch.zeros_like(z_intervened)
            mask[:, self.config.feature_mask] = 1
            z_intervened = z * (1 - mask) + z_intervened * mask

        # SAE解码
        with torch.no_grad():
            x_intervened = self.sae.decode(z_intervened)

        # 应用残差连接
        alpha = min(abs(strength), 1.0)
        x_output = x + alpha * (x_intervened - x)

        return x_output.reshape(B, L, C)


def compute_latent_shape(cfg, size_wh, frame_num: int, vae_z_dim: int) -> List[int]:
    """计算latent形状"""
    w, h = size_wh
    F = frame_num
    t_lat = (F - 1) // cfg.vae_stride[0] + 1
    h_lat = h // cfg.vae_stride[1]
    w_lat = w // cfg.vae_stride[2]
    return [vae_z_dim, t_lat, h_lat, w_lat]


def compute_seq_len(cfg, latent_shape, sp_size: int) -> int:
    """计算序列长度"""
    _, t_lat, h_lat, w_lat = latent_shape
    seq_len = math.ceil((h_lat * w_lat) / (cfg.patch_size[1] * cfg.patch_size[2]) * t_lat / sp_size) * sp_size
    return int(seq_len)


def generate_with_intervention(
    wrapper: WanT2V,
    prompt: str,
    interventions: List[InterventionConfig],
    concept_manager: ConceptVectorManager,
    run_dir: str,
    device: torch.device,
    size_wh: Tuple[int, int] = (832, 480),
    frame_num: int = 81,
    sampling_steps: int = 30,
    shift: float = 5.0,
    seed: int = 0,
) -> torch.Tensor:
    """
    带干预的视频生成

    返回: 生成的视频张量 [C, T, H, W]
    """
    cfg = wrapper.config
    model = wrapper.model

    # 计算latent形状
    vae_z_dim = wrapper.vae.model.z_dim
    latent_shape = compute_latent_shape(cfg, size_wh, frame_num, vae_z_dim)
    seq_len = compute_seq_len(cfg, latent_shape, wrapper.sp_size)

    logger.info(f"Latent shape: {latent_shape}, seq_len: {seq_len}")

    # 准备干预器
    interveners: Dict[str, SAEIntervener] = {}

    for interv_config in interventions:
        layer_key = interv_config.layer_key

        # 加载SAE（使用新的统一 IO 接口，自动兼容新旧格式）
        hook_mode, layer_str = layer_key.split(".")
        layer_idx = int(layer_str.replace("layer", ""))

        loc = SAERunLocator(run_dir=run_dir, hook_mode=hook_mode, layer_idx=layer_idx)

        try:
            io = SAECheckpointIO.load(loc, device=device, strict=True, allow_legacy=True)
            sae = io.sae
            sae_cfg = io.sae_config
            if io._config_source == "json_fallback":
                logger.warning(f"从旧格式 .json 加载配置 [建议迁移]")
        except Exception as e:
            logger.error(f"加载SAE失败: {e}")
            raise

        # 加载概念向量
        concept_vector = concept_manager.get_concept_vector(
            interv_config.concept_name, layer_key
        )

        if concept_vector is None:
            logger.warning(f"跳过干预: 找不到概念向量 {interv_config.concept_name}")
            continue

        # 创建干预器
        intervener = SAEIntervener(
            sae=sae,
            concept_vector=concept_vector,
            intervention_config=interv_config,
            global_scale=steering_params["global_strength_scale"],
        )

        interveners[layer_key] = intervener
        logger.info(f"已加载干预: {layer_key} / {interv_config.concept_name}")

    # 文本编码
    wrapper.text_encoder.model.to(device)
    context = wrapper.text_encoder([prompt], device)

    if system_params["offload_text_encoder"]:
        wrapper.text_encoder.model.cpu()

    # 初始化噪声
    torch.manual_seed(seed)
    latent = torch.randn(
        latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3],
        dtype=torch.float32, device=device
    )

    # 时间步
    timesteps = torch.linspace(
        cfg.num_train_timesteps - 1, 0, sampling_steps,
        device=device, dtype=torch.long
    )

    # 生成循环
    for step_idx, t in enumerate(timesteps):
        logger.info(f"生成步骤 {step_idx + 1}/{sampling_steps}, timestep={t.item()}")

        # 收集hook的特征
        hooked_features: Dict[str, torch.Tensor] = {}

        def make_hook_callback(key: str):
            def callback(module, inp, out):
                hooked_features[key] = out.detach().clone()
                # 应用干预
                if key in interveners:
                    out_intervened = interveners[key].intervene(
                        out, int(t.item()), cfg.num_train_timesteps
                    )
                    return out_intervened
                return out
            return callback

        # 注册hooks
        handles = []
        for layer_key in interveners.keys():
            hook_mode, layer_str = layer_key.split(".")
            layer_idx = int(layer_str.replace("layer", ""))

            block = model.blocks[layer_idx]

            if hook_mode == "block_out":
                handle = block.register_forward_hook(make_hook_callback(layer_key))
                handles.append(handle)
            elif hook_mode == "self_attn":
                handle = block.self_attn.register_forward_hook(make_hook_callback(layer_key))
                handles.append(handle)
            elif hook_mode == "cross_attn":
                handle = block.cross_attn.register_forward_hook(make_hook_callback(layer_key))
                handles.append(handle)

        try:
            # 前向传播
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=cfg.param_dtype):
                    timestep = torch.stack([t])
                    noise_pred = model([latent], t=timestep, context=context, seq_len=seq_len)

                    # 更新latent
                    dt = 1.0 / sampling_steps
                    latent = latent - noise_pred[0] * dt

        finally:
            for handle in handles:
                handle.remove()

    # VAE解码
    logger.info("解码视频...")
    video = wrapper.vae.decode(latent.unsqueeze(0))

    return video


def main():
    parser = argparse.ArgumentParser(description="Generate video with SAE concept steering")
    parser.add_argument("--config", type=str, default="", help="JSON配置文件路径")
    parser.add_argument("--checkpoint_dir", type=str, default=path_params["checkpoint_dir"])
    parser.add_argument("--run_dir", type=str, default=path_params["run_dir"])
    parser.add_argument("--concept_dir", type=str, default=path_params["concept_vectors_dir"])
    parser.add_argument("--output_dir", type=str, default=path_params["output_dir"])
    parser.add_argument("--prompt", type=str, default=generation_params["prompt"])
    parser.add_argument("--size_w", type=int, default=generation_params["size_w"])
    parser.add_argument("--size_h", type=int, default=generation_params["size_h"])
    parser.add_argument("--frame_num", type=int, default=generation_params["frame_num"])
    parser.add_argument("--steps", type=int, default=generation_params["sampling_steps"])
    parser.add_argument("--seed", type=int, default=generation_params["seed"])
    parser.add_argument("--device_id", type=int, default=system_params["device_id"])
    parser.add_argument("--save_session", action="store_true", default=True)

    args = parser.parse_args()

    # 加载配置文件（如果提供）
    interventions = []
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        interventions = [InterventionConfig.from_dict(i) for i in config_data.get("interventions", [])]
        if "prompt" in config_data:
            args.prompt = config_data["prompt"]

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

    # 生成视频文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"steered_{timestamp}.mp4"
    video_path = output_dir / video_filename

    # 创建概念向量管理器
    concept_manager = ConceptVectorManager(args.concept_dir)

    # 加载WanT2V
    logger.info("加载WanT2V...")
    cfg = t2v_1_3B
    wrapper = WanT2V(
        config=cfg,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    logger.info("WanT2V加载完成")

    # 生成
    start_time = time.time()

    video = generate_with_intervention(
        wrapper=wrapper,
        prompt=args.prompt,
        interventions=interventions,
        concept_manager=concept_manager,
        run_dir=args.run_dir,
        device=device,
        size_wh=(args.size_w, args.size_h),
        frame_num=args.frame_num,
        sampling_steps=args.steps,
        seed=args.seed,
    )

    # 保存视频
    # TODO: 实现视频保存（需要添加适当的编码器）
    logger.info(f"视频生成完成: {video.shape}")

    duration = time.time() - start_time

    # 保存会话配置
    if args.save_session:
        session = SteeringSession(
            prompt=args.prompt,
            video_path=str(video_path),
            interventions=interventions,
            size_w=args.size_w,
            size_h=args.size_h,
            frame_num=args.frame_num,
            sampling_steps=args.steps,
            seed=args.seed,
            run_dir=args.run_dir,
            checkpoint_dir=args.checkpoint_dir,
            concept_vectors_dir=args.concept_dir,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            metadata={
                "global_strength_scale": steering_params["global_strength_scale"],
                "dynamic_strength": steering_params["dynamic_strength"],
            }
        )

        session_path = output_dir / f"session_{timestamp}.json"
        session.save(str(session_path))

    logger.info("=" * 60)
    logger.info("干预生成完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"生成时间: {duration:.1f}秒")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
