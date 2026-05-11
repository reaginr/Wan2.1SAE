"""
Layer29 SAE Latent 提取 Pipeline

严格按照 TODO_list_v4.md (紧急版) 规范

核心约束：
- 只处理 Layer29
- Timestep ∈ [150, 800]
- Truncated Gaussian Sampling: μ=300, σ=80
- 禁止 oversample 和 decorrelation

输出：
- 每个prompt的SAE latent
- 保存为 .pt 文件

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
from typing import Any, Dict, List, Optional, Tuple

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
class LatentExtractionConfig:
    """Latent 提取配置"""

    # 模型配置
    model_path: str = "./Wan2.1-T2V-1.3B"
    sae_checkpoint: str = ""  # Layer29 SAE 初始化权重

    # 层配置 (固定)
    layer_idx: int = 29
    hook_type: str = "block_out"

    # Timestep 配置 (严格约束)
    min_timestep: int = 150
    max_timestep: int = 800
    timestep_mu: float = 300.0
    timestep_sigma: float = 80.0
    num_timesteps: int = 5  # 每prompt采样几个timestep

    # Token 配置
    tokens_per_timestep: int = 1536  # 每timestep保留多少token
    spatial_stride: int = 1  # 固定为1

    # SAE 配置
    d_model: int = 1536
    d_hidden: int = 12288  # 8x expansion
    top_k: int = 128

    # 数据集配置
    prompt_dir: str = "./datasets"
    output_dir: str = "./outputs/layer29_latents"

    # 运行配置
    device: str = "cuda"
    seed: int = 42
    batch_size: int = 1  # 每次处理1个prompt

    # 类别
    categories: List[str] = field(default_factory=lambda: [
        "sex_positive", "sex_negative",
        "violence_positive", "violence_negative",
        "clean_prompts"
    ])

    def validate(self):
        """验证配置"""
        assert self.layer_idx == 29, "只处理 Layer29"
        assert self.min_timestep >= 150, f"min_timestep 必须 >= 150, got {self.min_timestep}"
        assert self.max_timestep <= 800, f"max_timestep 必须 <= 800, got {self.max_timestep}"
        assert self.spatial_stride == 1, "训练阶段 stride 必须为 1"


# ============================================================================
# Timestep 采样器
# ============================================================================

class TruncatedGaussianSampler:
    """
    截断高斯 timestep 采样器

    严格按照 TODO 规范:
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
        """
        采样 n 个 timestep

        返回:
            np.ndarray: shape [n], dtype int
        """
        if seed is not None:
            np.random.seed(seed)

        samples = []
        while len(samples) < n:
            # 从高斯分布采样
            t = np.random.normal(self.mu, self.sigma)

            # 截断
            if self.min_t <= t <= self.max_t:
                samples.append(int(round(t)))

        return np.array(samples[:n])


# ============================================================================
# SAE 模型 (简化版，支持加载初始化权重)
# ============================================================================

class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder

    支持加载初始化阶段的权重
    """

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

        # Encoder: d_model -> d_hidden
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)

        # Decoder: d_hidden -> d_model
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码

        参数:
            x: [N, D] 输入 (已归一化)

        返回:
            z_sparse: [N, d_hidden] 稀疏编码
            topk_idx: [N, k] TopK索引
            topk_val: [N, k] TopK值
        """
        # Encoder forward
        z = self.encoder(x)  # [N, d_hidden]
        z = F.relu(z)  # 激活

        # TopK
        topk_val, topk_idx = torch.topk(z, k=self.top_k, dim=1, largest=True)

        # 构建稀疏表示
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(1, topk_idx, topk_val)

        return z_sparse, topk_idx, topk_val

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码

        参数:
            z: [N, d_hidden] 稀疏编码

        返回:
            x_hat: [N, D] 重建
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        完整前向

        返回:
            x_hat: 重建
            info: 信息字典
        """
        z_sparse, topk_idx, topk_val = self.encode(x)
        x_hat = self.decode(z_sparse)

        info = {
            "topk_idx": topk_idx,
            "topk_val": topk_val,
            "z_sparse": z_sparse,
        }

        return x_hat, info

    @classmethod
    def load_from_init(cls, checkpoint_path: str, device: str = "cuda") -> "TopKSAE":
        """
        从初始化阶段的 checkpoint 加载

        支持格式:
        - sae_mixed_init.py 输出: {"Wenc": ..., "Wdec": ...}
        - TopKSAE.save_pretrained 输出: {"state_dict": ...}
        - 标准 state_dict
        """
        ckpt = torch.load(checkpoint_path, map_location=device)

        # 判断格式
        if "Wenc" in ckpt and "Wdec" in ckpt:
            # 格式1: sae_mixed_init.py 输出
            Wenc = ckpt["Wenc"]  # [d_hidden, d_model]
            Wdec = ckpt["Wdec"]  # [d_model, d_hidden]

            d_hidden, d_model = Wenc.shape

            sae = cls(d_model=d_model, d_hidden=d_hidden)

            # nn.Linear weight 形状是 [out_features, in_features]
            # encoder: Linear(d_model, d_hidden), weight 形状 [d_hidden, d_model]
            sae.encoder.weight.data = Wenc.float()
            sae.decoder.weight.data = Wdec.float()

            logger.info(f"Loaded init checkpoint (mixed_init format): {checkpoint_path}")

        elif "state_dict" in ckpt:
            # 格式2: TopKSAE.save_pretrained 输出
            state_dict = ckpt["state_dict"]

            d_hidden = state_dict["encoder.weight"].shape[0]
            d_model = state_dict["encoder.weight"].shape[1]

            sae = cls(d_model=d_model, d_hidden=d_hidden)
            sae.load_state_dict(state_dict, strict=False)

            logger.info(f"Loaded init checkpoint (TopKSAE format): {checkpoint_path}")

        else:
            # 格式3: 标准 state_dict
            d_hidden = ckpt["encoder.weight"].shape[0]
            d_model = ckpt["encoder.weight"].shape[1]

            sae = cls(d_model=d_model, d_hidden=d_hidden)
            sae.load_state_dict(ckpt, strict=False)

            logger.info(f"Loaded init checkpoint (standard format): {checkpoint_path}")

        return sae.to(device)


# ============================================================================
# RMSNorm
# ============================================================================

def per_token_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token RMSNorm

    参数:
        x: [N, D] 输入

    返回:
        x_norm: 归一化后的张量
        rms: RMS值 [N, 1]
    """
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    return x_norm, rms


# ============================================================================
# Latent 提取器
# ============================================================================

class Layer29LatentExtractor:
    """
    Layer29 SAE Latent 提取器

    流程:
    1. 加载 Wan 模型
    2. 加载 SAE
    3. 对每个 prompt:
       - 运行 DiT 采样
       - Hook Layer29 激活
       - RMSNorm
       - SAE encode
       - 保存 latent
    """

    def __init__(self, config: LatentExtractionConfig):
        self.config = config
        self.device = config.device

        # Timestep 采样器
        self.timestep_sampler = TruncatedGaussianSampler(
            min_t=config.min_timestep,
            max_t=config.max_timestep,
            mu=config.timestep_mu,
            sigma=config.timestep_sigma,
        )

        # 加载模型
        self._load_models()

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
        if self.config.sae_checkpoint:
            logger.info(f"Loading SAE from {self.config.sae_checkpoint}")
            self.sae = TopKSAE.load_from_init(
                self.config.sae_checkpoint,
                device=self.device
            )
        else:
            logger.warning("No SAE checkpoint specified, using random init")
            self.sae = TopKSAE(
                d_model=self.config.d_model,
                d_hidden=self.config.d_hidden,
                top_k=self.config.top_k,
            ).to(self.device)

        logger.info("Models loaded successfully")

    def _sample_timesteps(self, n: int, seed: Optional[int] = None) -> List[int]:
        """采样 timesteps"""
        return self.timestep_sampler.sample(n, seed=seed).tolist()

    def extract_single_prompt(
        self,
        prompt: str,
        timesteps: Optional[List[int]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        提取单个 prompt 的 latent

        参数:
            prompt: 提示词
            timesteps: 指定的 timesteps (可选)
            seed: 随机种子

        返回:
            {
                "prompt": str,
                "latents": List[Dict],  # 每个 timestep 的 latent
            }
        """
        if timesteps is None:
            timesteps = self._sample_timesteps(self.config.num_timesteps, seed=seed)

        # Hook 存储
        hook_outputs = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                hook_outputs[layer_idx] = output.detach()
            return hook

        # 注册 hook
        target_layer = self.t2v.model.blocks[self.config.layer_idx]
        hook_handler = target_layer.register_forward_hook(make_hook(self.config.layer_idx))

        latents = []

        try:
            for t in timesteps:
                # 重置 hook 存储
                hook_outputs.clear()

                # 运行单步 DiT forward
                # (简化版：只运行一次 forward 获取激活)
                self._run_single_dit_step(prompt, t, seed=seed)

                if self.config.layer_idx not in hook_outputs:
                    logger.warning(f"No activation captured for t={t}")
                    continue

                # 获取激活
                act = hook_outputs[self.config.layer_idx]  # [B, L, D]

                # RMSNorm
                act_flat = act.reshape(-1, act.size(-1))  # [B*L, D]
                act_norm, rms = per_token_rms_norm(act_flat)

                # 采样 token
                if len(act_norm) > self.config.tokens_per_timestep:
                    indices = torch.randperm(len(act_norm))[:self.config.tokens_per_timestep]
                    act_norm = act_norm[indices]

                # SAE encode
                with torch.no_grad():
                    z_sparse, topk_idx, topk_val = self.sae.encode(act_norm)

                # 保存
                latents.append({
                    "timestep": t,
                    "topk_idx": topk_idx.cpu(),  # [N, k]
                    "topk_val": topk_val.cpu(),  # [N, k]
                    "z_sparse": z_sparse.cpu(),  # [N, d_hidden] (可选)
                    "n_tokens": len(act_norm),
                })

        finally:
            hook_handler.remove()

        return {
            "prompt": prompt,
            "latents": latents,
        }

    def _run_single_dit_step(
        self,
        prompt: str,
        timestep: int,
        size: tuple = (832, 480),
        frame_num: int = 81,
        seed: int = 42,
    ):
        """
        运行单步 DiT forward

        为了获取特定 timestep 的激活
        """
        import torch.cuda.amp as amp
        from diffusers import FlowMatchEulerDiscreteScheduler

        config = self.t2v.config
        device = self.t2v.device

        # 计算 latent shape
        target_shape = (
            self.t2v.vae.model.z_dim,
            (frame_num - 1) // self.t2v.vae_stride[0] + 1,
            size[1] // self.t2v.vae_stride[1],
            size[0] // self.t2v.vae_stride[2],
        )

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (self.t2v.patch_size[1] * self.t2v.patch_size[2]) *
            target_shape[1] / self.t2v.sp_size
        ) * self.t2v.sp_size

        # 随机种子
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)

        # 文本编码
        if not self.t2v.t5_cpu:
            self.t2v.text_encoder.model.to(device)
            context = self.t2v.text_encoder([prompt], device)
        else:
            context = self.t2v.text_encoder([prompt], torch.device('cpu'))
            context = [t.to(device) for t in context]

        # 初始噪声
        noise = torch.randn(
            target_shape[0], target_shape[1], target_shape[2], target_shape[3],
            dtype=torch.float32,
            device=device,
            generator=seed_g
        )

        # 调度器
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=5.0,
        )

        # 获取目标 timestep
        t = torch.tensor([timestep], device=device, dtype=torch.long)

        # DiT forward
        with amp.autocast(dtype=self.t2v.param_dtype), torch.no_grad():
            self.t2v.model.to(device)
            _ = self.t2v.model([noise], t=t, context=context, seq_len=seq_len)

    def extract_category(
        self,
        category: str,
        prompts: List[str],
        seed_start: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        提取一个类别所有 prompt 的 latent

        参数:
            category: 类别名
            prompts: 提示词列表
            seed_start: 起始随机种子

        返回:
            List[Dict]: 每个 prompt 的结果
        """
        results = []

        for i, prompt in enumerate(tqdm(prompts, desc=f"Extracting {category}")):
            try:
                result = self.extract_single_prompt(
                    prompt=prompt,
                    seed=seed_start + i,
                )
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i+1}/{len(prompts)} prompts for {category}")

            except Exception as e:
                logger.warning(f"Failed to extract prompt {i}: {e}")
                continue

        return results


# ============================================================================
# 主流程
# ============================================================================

def load_prompts_by_category(prompt_dir: str) -> Dict[str, List[str]]:
    """
    按类别加载 prompts

    目录结构:
    datasets/
    ├── sex_positive.txt
    ├── sex_negative.txt
    ├── violence_positive.txt
    ├── violence_negative.txt
    └── clean_prompts.txt
    """
    prompt_path = Path(prompt_dir)
    categories = {}

    for txt_file in prompt_path.glob("*.txt"):
        category = txt_file.stem
        with open(txt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip() and len(line.strip()) >= 8]

        categories[category] = prompts
        logger.info(f"Loaded {len(prompts)} prompts for category: {category}")

    return categories


def save_latents(
    results: List[Dict[str, Any]],
    output_path: str,
    category: str,
):
    """保存 latent 结果"""
    output_file = Path(output_path) / f"{category}_latents.pt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(results, output_file)
    logger.info(f"Saved {len(results)} latent results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Layer29 SAE Latent Extraction")

    # 路径配置
    parser.add_argument("--model_path", type=str, required=True,
                        help="Wan model path")
    parser.add_argument("--sae_checkpoint", type=str, required=True,
                        help="Layer29 SAE checkpoint")
    parser.add_argument("--prompt_dir", type=str, default="./datasets",
                        help="Prompt directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/layer29_latents",
                        help="Output directory")

    # 提取配置
    parser.add_argument("--num_timesteps", type=int, default=5,
                        help="Timesteps per prompt")
    parser.add_argument("--tokens_per_timestep", type=int, default=1536,
                        help="Tokens per timestep")
    parser.add_argument("--categories", type=str, default="all",
                        help="Categories to extract (comma-separated or 'all')")

    # Timestep 配置
    parser.add_argument("--min_timestep", type=int, default=150)
    parser.add_argument("--max_timestep", type=int, default=800)
    parser.add_argument("--timestep_mu", type=float, default=300.0)
    parser.add_argument("--timestep_sigma", type=float, default=80.0)

    # 其他
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 创建配置
    config = LatentExtractionConfig(
        model_path=args.model_path,
        sae_checkpoint=args.sae_checkpoint,
        prompt_dir=args.prompt_dir,
        output_dir=args.output_dir,
        num_timesteps=args.num_timesteps,
        tokens_per_timestep=args.tokens_per_timestep,
        min_timestep=args.min_timestep,
        max_timestep=args.max_timestep,
        timestep_mu=args.timestep_mu,
        timestep_sigma=args.timestep_sigma,
        device=args.device,
        seed=args.seed,
    )

    config.validate()

    # 加载 prompts
    all_prompts = load_prompts_by_category(args.prompt_dir)

    # 确定要处理的类别
    if args.categories == "all":
        categories = list(all_prompts.keys())
    else:
        categories = [c.strip() for c in args.categories.split(",")]

    logger.info(f"Categories to process: {categories}")

    # 创建提取器
    extractor = Layer29LatentExtractor(config)

    # 提取每个类别
    for category in categories:
        if category not in all_prompts:
            logger.warning(f"Category not found: {category}")
            continue

        prompts = all_prompts[category]

        logger.info(f"\n{'='*70}")
        logger.info(f"Processing category: {category} ({len(prompts)} prompts)")
        logger.info(f"{'='*70}")

        results = extractor.extract_category(
            category=category,
            prompts=prompts,
            seed_start=config.seed,
        )

        # 保存
        save_latents(results, args.output_dir, category)

    logger.info("\nLatent extraction completed!")


if __name__ == "__main__":
    main()
