"""
带 SAE 干预的视频生成模块

在 Wan DiT 采样过程中，对 Layer29 激活应用 SAE 概念干预

核心流程:
1. 加载 Wan 模型
2. 加载 SAE 和概念向量
3. 注册 hook 到 Layer29
4. 在 DiT forward 中：
   - 获取激活 → RMSNorm → SAE encode → 干预 → SAE decode → 替换激活
5. 运行完整采样生成视频

作者：Claude
日期：2026-05-11
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class VideoGenerationConfig:
    """视频生成配置"""

    # 模型路径
    model_path: str = "./Wan2.1-T2V-1.3B"
    sae_checkpoint: str = "./sae_init_layer29.pt"

    # 概念向量
    vector_dir: str = "./outputs/concept_vectors"
    concept: str = "sex"  # "sex" or "violence"
    vector_type: str = "sparse"

    # 干预配置
    layer_idx: int = 29
    gamma: float = 0.5  # 干预强度

    # SAE 配置
    d_model: int = 1536
    d_hidden: int = 12288
    top_k: int = 128

    # 视频配置
    frame_num: int = 81
    size: Tuple[int, int] = (832, 480)
    sampling_steps: int = 30
    shift: float = 5.0
    guide_scale: float = 5.0

    # 输出
    output_dir: str = "./outputs/generated_videos"

    # 设备
    device: str = "cuda"
    seed: int = 42


# ============================================================================
# Timestep 采样器 (保持与 extract_latents 一致)
# ============================================================================

class TruncatedGaussianSampler:
    """
    截断高斯 timestep 采样器

    严格约束:
    - t ∈ [150, 800]
    - μ = 300
    - σ = 80
    """

    def __init__(
        self,
        min_t: int = 150,
        max_t: int = 800,
        mu: float = 300.0,
        sigma: float = 80.0,
    ):
        self.min_t = min_t
        self.max_t = max_t
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """采样 n 个 timestep"""
        if seed is not None:
            np.random.seed(seed)

        samples = []
        while len(samples) < n:
            t = np.random.normal(self.mu, self.sigma)
            if self.min_t <= t <= self.max_t:
                samples.append(int(round(t)))

        return np.array(samples[:n])


# ============================================================================
# SAE 模型
# ============================================================================

class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder"""

    def __init__(
        self,
        d_model: int = 1536,
        d_hidden: int = 12288,
        top_k: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.top_k = top_k

        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码并返回 TopK 稀疏表示"""
        z = F.relu(self.encoder(x))
        topk_val, topk_idx = torch.topk(z, k=self.top_k, dim=-1, largest=True)

        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_val)

        return z_sparse, topk_idx, topk_val

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码"""
        return self.decoder(z)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = "cuda") -> "TopKSAE":
        """从 checkpoint 加载"""
        ckpt = torch.load(checkpoint_path, map_location=device)

        if "Wenc" in ckpt and "Wdec" in ckpt:
            # sae_mixed_init 格式
            Wenc = ckpt["Wenc"]
            Wdec = ckpt["Wdec"]
            d_hidden, d_model = Wenc.shape

            sae = cls(d_model=d_model, d_hidden=d_hidden)
            sae.encoder.weight.data = Wenc.float()
            sae.decoder.weight.data = Wdec.float()

        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            d_hidden = state_dict["encoder.weight"].shape[0]
            d_model = state_dict["encoder.weight"].shape[1]

            sae = cls(d_model=d_model, d_hidden=d_hidden)
            sae.load_state_dict(state_dict, strict=False)

        else:
            d_hidden = ckpt["encoder.weight"].shape[0]
            d_model = ckpt["encoder.weight"].shape[1]

            sae = cls(d_model=d_model, d_hidden=d_hidden)
            sae.load_state_dict(ckpt, strict=False)

        return sae.to(device)


# ============================================================================
# RMSNorm
# ============================================================================

def per_token_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token RMSNorm"""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    return x_norm, rms


# ============================================================================
# 干预逻辑
# ============================================================================

def project_onto_concept(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """将 z 投影到概念向量 v 上: proj_v(z) = (z·v)v"""
    v_norm = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
    z_flat = z.reshape(-1, z.size(-1))
    proj_coef = z_flat @ v_norm
    projection = proj_coef.unsqueeze(-1) * v_norm.unsqueeze(0)
    return projection.reshape(z.shape)


def apply_intervention(z: torch.Tensor, v: torch.Tensor, gamma: float) -> torch.Tensor:
    """应用干预: z' = z - γ * proj_v(z)"""
    projection = project_onto_concept(z, v)
    return z - gamma * projection


# ============================================================================
# 干预 Hook
# ============================================================================

class InterventionHook:
    """
    Layer29 干预 Hook

    在 DiT forward 过程中：
    1. 获取 Layer29 输出
    2. RMSNorm
    3. SAE encode
    4. 应用概念干预
    5. SAE decode
    6. 替换原始输出
    """

    def __init__(
        self,
        sae: TopKSAE,
        concept_vector: torch.Tensor,
        gamma: float,
        layer_idx: int = 29,
    ):
        self.sae = sae
        self.concept_vector = concept_vector.to(next(sae.parameters()).device)
        self.gamma = gamma
        self.layer_idx = layer_idx

        # 统计
        self.intervention_count = 0
        self.total_projection_norm = 0.0

    def __call__(self, module, input, output):
        """
        Hook 函数

        注意：这里需要修改 output，所以返回新的 output
        """
        # 获取激活
        act = output  # [B, L, D]

        # 保存原始形状
        original_shape = act.shape

        # RMSNorm
        act_flat = act.reshape(-1, act.size(-1))
        act_norm, rms = per_token_rms_norm(act_flat)

        # SAE encode
        with torch.no_grad():
            z_sparse, _, _ = self.sae.encode(act_norm)

        # 应用干预
        z_intervened = apply_intervention(z_sparse, self.concept_vector, self.gamma)

        # SAE decode
        act_recon = self.sae.decode(z_intervened)

        # 反归一化
        act_intervened = act_recon * rms

        # 恢复形状
        act_intervened = act_intervened.reshape(original_shape)

        # 统计
        self.intervention_count += 1
        self.total_projection_norm += (z_sparse - z_intervened).norm().item()

        # 返回干预后的激活
        return act_intervened

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "intervention_count": self.intervention_count,
            "mean_projection_norm": self.total_projection_norm / max(1, self.intervention_count),
        }


# ============================================================================
# 带干预的视频生成器
# ============================================================================

class VideoGeneratorWithIntervention:
    """
    带 SAE 概念干预的视频生成器

    流程:
    1. 加载 Wan T2V 模型
    2. 加载 SAE
    3. 注册干预 hook 到 Layer29
    4. 运行 DiT 采样
    5. VAE decode 生成视频帧
    """

    def __init__(self, config: VideoGenerationConfig):
        self.config = config
        self.device = config.device

        # 加载模型
        self._load_models()

        # 干预 hook handler
        self.hook_handler = None
        self.intervention_hook = None

    def _load_models(self):
        """加载 Wan 模型和 SAE"""
        logger.info(f"Loading Wan model from {self.config.model_path}")

        from wan.configs.wan_t2v_1_3B import t2v_1_3B
        from wan.text2video import WanT2V

        self.t2v = WanT2V(
            config=t2v_1_3B,
            checkpoint_dir=self.config.model_path,
            device_id=0,
            t5_cpu=True,
        )

        # 加载 SAE
        logger.info(f"Loading SAE from {self.config.sae_checkpoint}")
        self.sae = TopKSAE.load_from_checkpoint(
            self.config.sae_checkpoint,
            device=self.device,
        )
        self.sae.eval()

        # 加载概念向量
        vector_file = Path(self.config.vector_dir) / f"{self.config.concept}_vector_{self.config.vector_type}.pt"

        if vector_file.exists():
            data = torch.load(vector_file, map_location=self.device)
            self.concept_vector = data["vector"].to(self.device)
            logger.info(f"Loaded concept vector for '{self.config.concept}', shape: {self.concept_vector.shape}")
        else:
            logger.warning(f"Concept vector not found: {vector_file}, using zero vector")
            self.concept_vector = torch.zeros(self.config.d_hidden, device=self.device)

        logger.info("Models loaded successfully")

    def register_intervention_hook(self, gamma: float):
        """
        注册干预 hook

        参数:
            gamma: 干预强度
        """
        # 移除已有的 hook
        self.remove_intervention_hook()

        # 创建 intervention hook
        self.intervention_hook = InterventionHook(
            sae=self.sae,
            concept_vector=self.concept_vector,
            gamma=gamma,
            layer_idx=self.config.layer_idx,
        )

        # 注册到 Layer29
        target_layer = self.t2v.model.blocks[self.config.layer_idx]
        self.hook_handler = target_layer.register_forward_hook(self.intervention_hook)

        logger.info(f"Registered intervention hook at Layer {self.config.layer_idx} with gamma={gamma}")

    def remove_intervention_hook(self):
        """移除干预 hook"""
        if self.hook_handler is not None:
            self.hook_handler.remove()
            self.hook_handler = None
            self.intervention_hook = None

    def generate(
        self,
        prompt: str,
        gamma: float = 0.0,
        seed: Optional[int] = None,
        return_frames: bool = True,
    ) -> Dict[str, Any]:
        """
        生成视频

        参数:
            prompt: 提示词
            gamma: 干预强度 (0.0 表示无干预)
            seed: 随机种子
            return_frames: 是否返回帧数据

        返回:
            Dict: {
                "prompt": str,
                "gamma": float,
                "frames": List[np.ndarray] or None,
                "latent_shape": tuple,
                "stats": Dict,
            }
        """
        if seed is None:
            seed = self.config.seed

        # 设置随机种子
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # 注册或移除干预 hook
        if gamma > 0:
            self.register_intervention_hook(gamma)
        else:
            self.remove_intervention_hook()

        # 运行生成
        try:
            # 使用 Wan 的 generate 方法
            frames = self.t2v.generate(
                input_prompt=prompt,
                frame_num=self.config.frame_num,
                size=self.config.size,
                sampling_steps=self.config.sampling_steps,
                shift=self.config.shift,
                guide_scale=self.config.guide_scale,
                seed=seed,
            )

            # 获取统计信息
            stats = {}
            if self.intervention_hook is not None:
                stats = self.intervention_hook.get_stats()

            result = {
                "prompt": prompt,
                "gamma": gamma,
                "frames": frames if return_frames else None,
                "frame_count": len(frames) if frames is not None else 0,
                "stats": stats,
            }

            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "prompt": prompt,
                "gamma": gamma,
                "frames": None,
                "frame_count": 0,
                "error": str(e),
            }

    def generate_batch(
        self,
        prompts: List[str],
        gammas: List[float],
        max_per_condition: int = 5,
        seed_start: int = 42,
    ) -> Dict[str, List[Dict]]:
        """
        批量生成视频

        参数:
            prompts: 提示词列表
            gammas: 干预强度列表
            max_per_condition: 每个条件最多生成数量
            seed_start: 起始随机种子

        返回:
            Dict[gamma, List[results]]
        """
        results = {gamma: [] for gamma in gammas}

        # 限制数量
        prompts = prompts[:max_per_condition]

        total = len(prompts) * len(gammas)
        pbar = tqdm(total=total, desc="Generating videos")

        for gamma in gammas:
            for i, prompt in enumerate(prompts):
                result = self.generate(
                    prompt=prompt,
                    gamma=gamma,
                    seed=seed_start + i,
                    return_frames=True,
                )
                results[gamma].append(result)
                pbar.update(1)

        pbar.close()

        return results


# ============================================================================
# 视频保存
# ============================================================================

def save_video_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 16,
):
    """
    保存视频帧为视频文件

    参数:
        frames: 帧列表 [H, W, C]
        output_path: 输出路径 (.mp4)
        fps: 帧率
    """
    try:
        import imageio

        # 创建目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 保存为 mp4
        with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
            for frame in frames:
                writer.append_data(frame)

        logger.info(f"Saved video to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save video: {e}")


def save_frames_as_images(
    frames: List[np.ndarray],
    output_dir: str,
    prefix: str = "frame",
):
    """
    保存帧为图片

    参数:
        frames: 帧列表
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    try:
        from PIL import Image

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(f"{output_dir}/{prefix}_{i:04d}.png")

        logger.info(f"Saved {len(frames)} frames to {output_dir}")

    except Exception as e:
        logger.error(f"Failed to save frames: {e}")


# ============================================================================
# 主流程
# ============================================================================

def load_prompts(prompt_file: str, max_prompts: Optional[int] = None) -> List[str]:
    """加载提示词"""
    prompts = []

    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line) >= 8:
                prompts.append(line)

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    logger.info(f"Loaded {len(prompts)} prompts from {prompt_file}")

    return prompts


def main():
    parser = argparse.ArgumentParser(description="Video Generation with SAE Intervention")

    # 模型路径
    parser.add_argument("--model_path", type=str, required=True,
                        help="Wan model path")
    parser.add_argument("--sae_checkpoint", type=str, required=True,
                        help="SAE checkpoint path")
    parser.add_argument("--vector_dir", type=str, default="./outputs/concept_vectors",
                        help="Concept vector directory")

    # 概念配置
    parser.add_argument("--concept", type=str, default="sex",
                        choices=["sex", "violence"],
                        help="Concept to intervene")
    parser.add_argument("--vector_type", type=str, default="sparse",
                        choices=["dense", "sparse"])

    # 干预配置
    parser.add_argument("--gamma", type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.8, 1.0],
                        help="Intervention strength(s)")
    parser.add_argument("--layer", type=int, default=29,
                        help="Layer to intervene")

    # 视频配置
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--size", type=str, default="832x480",
                        help="Video size (WxH)")
    parser.add_argument("--sampling_steps", type=int, default=30)

    # 提示词
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Prompt file path")
    parser.add_argument("--max_prompts", type=int, default=5,
                        help="Max prompts per condition")

    # 输出
    parser.add_argument("--output_dir", type=str, default="./outputs/generated_videos",
                        help="Output directory")

    # 其他
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 解析 size
    size = tuple(map(int, args.size.lower().split('x')))

    # 创建配置
    config = VideoGenerationConfig(
        model_path=args.model_path,
        sae_checkpoint=args.sae_checkpoint,
        vector_dir=args.vector_dir,
        concept=args.concept,
        vector_type=args.vector_type,
        layer_idx=args.layer,
        frame_num=args.frame_num,
        size=size,
        sampling_steps=args.sampling_steps,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )

    # 加载提示词
    prompts = load_prompts(args.prompt_file, args.max_prompts)

    # 创建生成器
    generator = VideoGeneratorWithIntervention(config)

    # 生成视频
    results = generator.generate_batch(
        prompts=prompts,
        gammas=args.gamma,
        max_per_condition=args.max_prompts,
        seed_start=args.seed,
    )

    # 保存结果
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存视频和元数据
    for gamma, gamma_results in results.items():
        gamma_dir = output_path / f"gamma_{gamma:.1f}"
        gamma_dir.mkdir(parents=True, exist_ok=True)

        for i, result in enumerate(gamma_results):
            if result.get("frames") is not None:
                # 保存视频
                video_path = gamma_dir / f"video_{i:03d}.mp4"
                save_video_frames(result["frames"], str(video_path))

                # 保存帧为图片 (可选)
                frames_dir = gamma_dir / f"frames_{i:03d}"
                save_frames_as_images(result["frames"], str(frames_dir))

            # 保存元数据
            meta = {
                "prompt": result["prompt"],
                "gamma": result["gamma"],
                "frame_count": result.get("frame_count", 0),
                "stats": result.get("stats", {}),
            }

            if result.get("error"):
                meta["error"] = result["error"]

            with open(gamma_dir / f"meta_{i:03d}.json", 'w') as f:
                json.dump(meta, f, indent=2)

    # 保存汇总
    summary = {
        "concept": args.concept,
        "gammas": args.gamma,
        "n_prompts": len(prompts),
        "results_summary": {
            gamma: {
                "n_generated": len(r),
                "n_success": sum(1 for x in r if x.get("frames") is not None),
            }
            for gamma, r in results.items()
        }
    }

    with open(output_path / "generation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nGeneration completed! Results saved to {output_path}")


if __name__ == "__main__":
    main()
