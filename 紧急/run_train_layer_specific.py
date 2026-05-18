"""
Layer-wise SAE 训练脚本 - 符合 TODO_list_v4 规范

使用统一配置文件 config.py，支持三阶段配置

使用方法:
    # 参数测试阶段 (快速验证代码正确性)
    python run_train_layer_specific.py --config 紧急/config_test.py

    # 预训练阶段 (中等规模，验证超参数)
    python run_train_layer_specific.py --config 紧急/config_pretrain.py

    # 正式训练阶段 (完整训练)
    python run_train_layer_specific.py --config 紧急/config_formal.py

    # 命令行覆盖配置文件参数
    python run_train_layer_specific.py --config 紧急/config_test.py --steps 200 --max_prompts 20

    # nohup 后台运行
    nohup python -u run_train_layer_specific.py --config 紧急/config_formal.py \\
        > formal_train.log 2>&1 &

三阶段说明:
    1. 参数测试 (config_test.py): 步数少、数据少、无预热、无EMA
    2. 预训练 (config_pretrain.py): 中等步数、中等数据、轻量预热、启用EMA
    3. 正式训练 (config_formal.py): 充足步数、完整数据、标准预热、完整EMA

作者：Claude
日期：2026-05-17
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入统一配置
from config import (
    PATH_PARAMS,
    TIMESTEP_PARAMS,
    SAE_PARAMS,
    TRAINING_PARAMS,
    SAMPLING_PARAMS,
    VALIDATION_PARAMS,
    PARAM_TEST_SAMPLING_PARAMS,
    validate_all_configs,
)

# 导入监控模块
from training_monitor import (
    DeadNeuronTracker,
    SparsityCalculator,
    TrainingVisualizer,
    create_monitoring_suite,
)

# 导入采样器模块
try:
    from 初始化.samplers import (
        ParamTestTokenSampler,
        ParamTestSamplerConfig,
        TrainTokenSampler,
        TrainSamplerConfig,
        TruncatedGaussianTimestepSampler,
        create_param_test_sampler,
        create_train_sampler,
        UnifiedSampler,
        per_token_rms_norm,
    )
    SAMPLERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Samplers module not available: {e}")
    SAMPLERS_AVAILABLE = False

# ============================================================================
# 日志配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 验证配置
validate_all_configs()


# ============================================================================
# Truncated Gaussian Timestep 采样器
# ============================================================================

class TruncatedGaussianSampler:
    """
    截断高斯 timestep 采样器

    严格按照 TODO_list_v4 规范:
    - t ∈ [min_t, max_t]
    - 从 N(μ, σ²) 采样，截断到指定范围
    """

    def __init__(
        self,
        min_t: int = 150,
        max_t: int = 800,
        mu: float = 300.0,
        sigma: float = 80.0,
        seed: Optional[int] = None,
    ):
        self.min_t = min_t
        self.max_t = max_t
        self.mu = mu
        self.sigma = sigma

        if seed is not None:
            np.random.seed(seed)

    def sample(self, n: int = 1) -> np.ndarray:
        """采样 n 个 timestep，返回 int 数组"""
        samples = []
        while len(samples) < n:
            t = np.random.normal(self.mu, self.sigma)
            if self.min_t <= t <= self.max_t:
                samples.append(int(round(t)))
        return np.array(samples[:n])

    def sample_layer_specific(
        self,
        layer_idx: int,
        n: int = 1,
    ) -> np.ndarray:
        """使用层特定参数采样"""
        params = TIMESTEP_PARAMS.layer_params.get(layer_idx, {
            "mu": self.mu,
            "sigma": self.sigma,
            "min_t": self.min_t,
            "max_t": self.max_t,
        })

        sampler = TruncatedGaussianSampler(
            min_t=params["min_t"],
            max_t=params["max_t"],
            mu=params["mu"],
            sigma=params["sigma"],
        )
        return sampler.sample(n)


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
        """编码，返回 (z_sparse, topk_idx, topk_val)"""
        z = F.relu(self.encoder(x))
        topk_val, topk_idx = torch.topk(z, k=self.top_k, dim=-1, largest=True)

        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_val)

        return z_sparse, topk_idx, topk_val

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z_sparse, topk_idx, topk_val = self.encode(x)
        x_hat = self.decode(z_sparse)
        return x_hat, {"topk_idx": topk_idx, "topk_val": topk_val, "z_sparse": z_sparse}

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str, device: str = "cuda") -> "TopKSAE":
        """从 checkpoint 加载"""
        ckpt = torch.load(ckpt_path, map_location=device)

        if "Wenc" in ckpt and "Wdec" in ckpt:
            # sae_mixed_init 格式
            Wenc, Wdec = ckpt["Wenc"], ckpt["Wdec"]
            d_hidden, d_model = Wenc.shape
            sae = cls(d_model, d_hidden)
            sae.encoder.weight.data = Wenc.float()
            sae.decoder.weight.data = Wdec.float()
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            d_hidden = sd["encoder.weight"].shape[0]
            d_model = sd["encoder.weight"].shape[1]
            sae = cls(d_model, d_hidden)
            sae.load_state_dict(sd, strict=False)
        else:
            d_hidden = ckpt["encoder.weight"].shape[0]
            d_model = ckpt["encoder.weight"].shape[1]
            sae = cls(d_model, d_hidden)
            sae.load_state_dict(ckpt, strict=False)

        return sae.to(device)


# ============================================================================
# RMSNorm
# ============================================================================

def per_token_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token RMSNorm"""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms, rms


# ============================================================================
# 激活数据集
# ============================================================================

@dataclass
class ActivationRecord:
    """单条激活记录（包含采样信息）"""
    prompt: str
    prompt_idx: int
    layer_idx: int
    timestep: int
    activation: torch.Tensor  # [N, D]
    n_tokens: int


class ActivationDataset(Dataset):
    """激活数据集"""

    def __init__(self, records: List[ActivationRecord]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "activation": r.activation,
            "timestep": r.timestep,
            "prompt_idx": r.prompt_idx,
        }


# ============================================================================
# Wan 模型加载与激活提取
# ============================================================================

class WanActivationExtractor:
    """
    从 Wan DiT 提取激活

    核心优化：只运行 DiT，不运行 VAE 解码

    采样模式:
    - 'train': 训练阶段采样 (全局随机，无 bias)
    - 'param_test': 参数测试阶段采样 (时空局部性 + soft bias)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        t5_cpu: bool = True,
        sampling_mode: str = "train",
        sampling_config: Optional[Dict] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.t5_cpu = t5_cpu
        self.sampling_mode = sampling_mode
        self.sampling_config = sampling_config or {}

        logger.info(f"Loading Wan model from {model_path}")
        from wan.configs.wan_t2v_1_3B import t2v_1_3B
        from wan.text2video import WanT2V

        self.t2v = WanT2V(
            config=t2v_1_3B,
            checkpoint_dir=model_path,
            device_id=0,
            t5_cpu=t5_cpu,
        )
        logger.info("Wan model loaded successfully")

        # 初始化采样器 (如果可用)
        self._param_test_sampler = None
        if SAMPLERS_AVAILABLE and sampling_mode == "param_test":
            config = ParamTestSamplerConfig(
                tokens_per_timestep=sampling_config.get("tokens_per_timestep", 1536),
                num_timesteps_per_prompt=sampling_config.get("num_timesteps_per_prompt", 5),
                temporal_chunk_size=sampling_config.get("temporal_chunk_size", 3),
                spatial_block_size=sampling_config.get("spatial_block_size", 8),
                num_spatial_blocks=sampling_config.get("num_spatial_blocks", 24),
                norm_bias_enabled=sampling_config.get("norm_bias_enabled", True),
                norm_bias_strength=sampling_config.get("norm_bias_strength", 0.3),
                decorrelation_enabled=sampling_config.get("decorrelation_enabled", True),
                decorrelation_threshold=sampling_config.get("decorrelation_threshold", 0.7),
            )
            self._param_test_sampler = ParamTestTokenSampler(config)
            logger.info(f"  Using param_test sampling mode with temporal_chunk={config.temporal_chunk_size}, "
                       f"spatial_block={config.spatial_block_size}x{config.spatial_block_size}")

    def extract_layer_activations(
        self,
        prompts: List[str],
        layer_indices: List[int],
        num_timesteps_per_prompt: int = 5,
        tokens_per_timestep: int = 1536,
        sampling_steps: int = 30,
        seed_start: int = 42,
    ) -> Dict[int, List[ActivationRecord]]:
        """
        提取多层激活

        参数:
            prompts: 提示词列表
            layer_indices: 层索引列表
            num_timesteps_per_prompt: 每个 prompt 采样的 timestep 数
            tokens_per_timestep: 每个 timestep 保留的 token 数
            sampling_steps: DiT 扩散步数
            seed_start: 随机种子起始值

        返回:
            {layer_idx: List[ActivationRecord]}
        """
        # 初始化各层的 timestep 采样器
        if SAMPLERS_AVAILABLE:
            layer_samplers = {
                layer: TruncatedGaussianTimestepSampler(
                    layer_params={layer: TIMESTEP_PARAMS.layer_params[layer]},
                    seed=seed_start,
                )
                for layer in layer_indices
            }
        else:
            # 回退到内部采样器
            layer_samplers = {
                layer: TruncatedGaussianSampler(
                    mu=TIMESTEP_PARAMS.layer_params[layer]["mu"],
                    sigma=TIMESTEP_PARAMS.layer_params[layer]["sigma"],
                    min_t=TIMESTEP_PARAMS.layer_params[layer]["min_t"],
                    max_t=TIMESTEP_PARAMS.layer_params[layer]["max_t"],
                )
                for layer in layer_indices
            }

        # 存储各层记录
        layer_records = {layer: [] for layer in layer_indices}

        # 注册 hooks
        hook_outputs = {}
        hook_handlers = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                hook_outputs[layer_idx] = output.detach()
            return hook

        for layer_idx in layer_indices:
            target_layer = self.t2v.model.blocks[layer_idx]
            hook_handlers.append(
                target_layer.register_forward_hook(make_hook(layer_idx))
            )

        logger.info(f"Registered hooks for layers: {layer_indices}")
        logger.info(f"  Sampling mode: {self.sampling_mode}")

        try:
            for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Extracting activations")):
                # 为每个层采样不同的 timestep
                layer_timesteps = {}
                for layer in layer_indices:
                    if SAMPLERS_AVAILABLE:
                        timesteps = layer_samplers[layer].sample(layer, num_timesteps_per_prompt)
                    else:
                        timesteps = layer_samplers[layer].sample(num_timesteps_per_prompt)
                    layer_timesteps[layer] = timesteps if isinstance(timesteps, list) else timesteps.tolist()

                # 运行 DiT 采样
                self._run_dit_with_timesteps(
                    prompt=prompt,
                    layer_timesteps=layer_timesteps,
                    layer_indices=layer_indices,
                    sampling_steps=sampling_steps,
                    seed=seed_start + prompt_idx,
                )

                # 收集各层激活
                for layer_idx in layer_indices:
                    if layer_idx not in hook_outputs:
                        continue

                    act = hook_outputs[layer_idx]  # [B, L, D]

                    # 根据采样模式处理激活
                    if self.sampling_mode == "param_test" and self._param_test_sampler is not None:
                        # 参数测试阶段采样
                        act_sampled, sample_meta = self._param_test_sampler.sample(
                            activations=act,
                            timestep=layer_timesteps[layer_idx][0] if layer_timesteps[layer_idx] else None,
                            grid_size=(11, 30, 52),  # latent shape
                            layer_idx=layer_idx,
                        )
                        act_norm = act_sampled
                        n_tokens = len(act_norm)
                    else:
                        # 训练阶段采样 (全局随机)
                        act_flat = act.reshape(-1, act.size(-1))
                        act_norm, rms = per_token_rms_norm(act_flat)

                        # 随机采样 token
                        if len(act_norm) > tokens_per_timestep:
                            indices = torch.randperm(len(act_norm))[:tokens_per_timestep]
                            act_norm = act_norm[indices]
                        n_tokens = len(act_norm)

                    # 为每个 timestep 创建记录
                    for t_idx, t in enumerate(layer_timesteps[layer_idx]):
                        record = ActivationRecord(
                            prompt=prompt,
                            prompt_idx=prompt_idx,
                            layer_idx=layer_idx,
                            timestep=t,
                            activation=act_norm.cpu().clone(),
                            n_tokens=n_tokens,
                        )
                        layer_records[layer_idx].append(record)

                hook_outputs.clear()

                if (prompt_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {prompt_idx + 1}/{len(prompts)} prompts")

        finally:
            for h in hook_handlers:
                h.remove()

        # 打印统计
        for layer, records in layer_records.items():
            logger.info(f"Layer {layer}: {len(records)} activation records")

        return layer_records

    def _run_dit_with_timesteps(
        self,
        prompt: str,
        layer_timesteps: Dict[int, List[int]],
        layer_indices: List[int],
        sampling_steps: int = 30,
        seed: int = 42,
    ):
        """运行 DiT 采样（优化版，不运行 VAE）"""
        import torch.cuda.amp as amp
        from diffusers import FlowMatchEulerDiscreteScheduler

        config = self.t2v.config
        device = self.t2v.device

        # 计算 latent shape
        target_shape = (
            self.t2v.vae.model.z_dim,
            81 // self.t2v.vae_stride[0] + 1,
            480 // self.t2v.vae_stride[1],
            832 // self.t2v.vae_stride[2],
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
            context = self.t2v.text_encoder([prompt], device)
            context_null = self.t2v.text_encoder([""], device)
        else:
            context = self.t2v.text_encoder([prompt], torch.device('cpu'))
            context_null = self.t2v.text_encoder([""], device)
            context = [t.to(device) for t in context]
            context_null = [t.to(device) for t in context_null]

        # 初始噪声
        noise = [torch.randn(
            target_shape[0], target_shape[1], target_shape[2], target_shape[3],
            dtype=torch.float32, device=device, generator=seed_g
        )]

        # 调度器
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=5.0,
        )
        scheduler.set_timesteps(sampling_steps, device=device)
        timesteps = scheduler.timesteps

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        with amp.autocast(dtype=self.t2v.param_dtype), torch.no_grad():
            self.t2v.model.to(device)
            for t in timesteps:
                latent_model_input = noise
                timestep = torch.stack([t])

                noise_pred_cond = self.t2v.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.t2v.model(latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + 5.0 * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = scheduler.step(
                    noise_pred.unsqueeze(0), t, noise[0].unsqueeze(0),
                    return_dict=False, generator=seed_g
                )[0]
                noise = [temp_x0.squeeze(0)]


# ============================================================================
# 训练循环
# ============================================================================

class LayerSpecificTrainer:
    """单层 SAE 训练器 (含可视化监控)"""

    def __init__(
        self,
        layer_idx: int,
        sae: TopKSAE,
        device: str = "cuda",
        lr: float = 6e-5,
        min_lr: float = 1e-5,
        warmup_steps: int = 400,
        total_steps: int = 2000,
        grad_clip: float = 0.3,
        ema_decay: float = 0.999,
        use_ema: bool = True,
        output_dir: str = "./outputs",
    ):
        self.layer_idx = layer_idx
        self.sae = sae
        self.device = device
        self.grad_clip = grad_clip
        self.output_dir = output_dir
        self.use_ema = use_ema

        # 优化器
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.95, 0.999))

        # 学习率调度器
        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr / lr + (1 - min_lr / lr) * 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # EMA (仅在启用时初始化)
        self.ema_weights = {}
        if self.use_ema:
            for name, param in sae.named_parameters():
                self.ema_weights[name] = param.data.clone()
        self.ema_decay = ema_decay

        # 统计
        self.step = 0
        self.best_mse = float('inf')

        # ========== 新增：监控组件 ==========
        d_hidden = sae.d_hidden
        top_k = sae.top_k

        self.dead_tracker = DeadNeuronTracker(d_hidden=d_hidden, window_size=100)
        self.sparsity_calc = SparsityCalculator(d_hidden=d_hidden, top_k=top_k)
        self.visualizer = TrainingVisualizer(output_dir)
        self.visualizer.register_layer(layer_idx, d_hidden)

        logger.info(f"  [Monitor] Initialized for layer {layer_idx}: d_hidden={d_hidden}, top_k={top_k}")

    def update_ema(self):
        """更新 EMA 权重"""
        if not self.use_ema:
            return
        with torch.no_grad():
            for name, param in self.sae.named_parameters():
                self.ema_weights[name].mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """单步训练 (含监控)"""
        self.sae.train()
        self.optimizer.zero_grad()

        x = batch["activation"].to(self.device)

        # Forward
        x_hat, info = self.sae(x)
        z_sparse = info["z_sparse"]

        # Loss: MSE
        loss = F.mse_loss(x_hat, x)

        # Backward
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.grad_clip)

        self.optimizer.step()
        self.scheduler.step()
        self.update_ema()

        self.step += 1

        # ========== 新增：更新监控 ==========
        # 更新死神经元追踪
        self.dead_tracker.update(z_sparse.detach())

        # 记录到可视化器
        self.visualizer.log_train_step(
            layer_idx=self.layer_idx,
            step=self.step,
            loss=loss.item(),
            mse=loss.item(),
            lr=self.scheduler.get_last_lr()[0],
            z_sparse=z_sparse.detach(),
            dead_tracker=self.dead_tracker,
        )

        return {
            "loss": loss.item(),
            "mse": loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证 (含监控)"""
        self.sae.eval()
        total_loss = 0
        total_z_sparse = []
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["activation"].to(self.device)
                x_hat, info = self.sae(x)
                loss = F.mse_loss(x_hat, x)
                total_loss += loss.item()
                total_z_sparse.append(info["z_sparse"].detach())
                n_batches += 1

        avg_mse = total_loss / max(1, n_batches)
        self.best_mse = min(self.best_mse, avg_mse)

        # ========== 新增：记录验证指标 ==========
        if total_z_sparse:
            z_all = torch.cat(total_z_sparse, dim=0)
            self.visualizer.log_validation(
                layer_idx=self.layer_idx,
                step=self.step,
                mse=avg_mse,
                z_sparse=z_all,
                dead_tracker=self.dead_tracker,
            )

        return {"val_mse": avg_mse, "best_mse": self.best_mse}

    def generate_plots(self):
        """生成可视化图表"""
        self.visualizer.generate_plots(self.layer_idx)
        self.visualizer.save_statistics(self.layer_idx)
        logger.info(f"  Generated training plots for layer {self.layer_idx}")

    def save_checkpoint(self, path: str):
        """保存 checkpoint"""
        ckpt = {
            "step": self.step,
            "state_dict": self.sae.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_mse": self.best_mse,
            "layer_idx": self.layer_idx,
            "use_ema": self.use_ema,
            "timestep_params": TIMESTEP_PARAMS.layer_params[self.layer_idx],
        }
        # 仅在启用 EMA 时保存 EMA 权重
        if self.use_ema:
            ckpt["ema_weights"] = self.ema_weights
        torch.save(ckpt, path)
        logger.info(f"  Saved checkpoint to {path}")


# ============================================================================
# 主训练流程
# ============================================================================

def train_all_layers(
    layer_indices: List[int],
    prompts: List[str],
    config: Dict,
    init_dir: Optional[str] = None,
    run_dir: str = "sae_runs/exp",
) -> Dict[str, Any]:
    """
    训练所有层（顺序执行）

    参数:
        layer_indices: 层索引列表
        prompts: 提示词列表
        config: 配置字典
        init_dir: 初始化权重目录
        run_dir: 输出目录
    """
    results = {}
    start_time = time.time()

    # 加载 Wan 模型
    logger.info("\n" + "=" * 70)
    logger.info("Loading Wan Model")
    logger.info("=" * 70)

    # 获取采样模式和配置
    sampling_mode = config.get("sampling_mode", "train")
    sampling_config = {
        "tokens_per_timestep": config.get("tokens_per_timestep", 1536),
        "num_timesteps_per_prompt": config.get("num_timesteps_per_prompt", 5),
        "temporal_chunk_size": config.get("temporal_chunk_size", 3),
        "spatial_block_size": config.get("spatial_block_size", 8),
        "num_spatial_blocks": config.get("num_spatial_blocks", 24),
        "norm_bias_enabled": config.get("norm_bias_enabled", True),
        "norm_bias_strength": config.get("norm_bias_strength", 0.3),
        "decorrelation_enabled": config.get("decorrelation_enabled", True),
        "decorrelation_threshold": config.get("decorrelation_threshold", 0.7),
    }

    extractor = WanActivationExtractor(
        model_path=config["model_path"],
        device="cuda",
        t5_cpu=True,
        sampling_mode=sampling_mode,
        sampling_config=sampling_config,
    )

    # 提取所有层的激活
    logger.info("\n" + "=" * 70)
    logger.info(f"Extracting Activations (Mode: {sampling_mode})")
    logger.info("=" * 70)

    for layer in layer_indices:
        params = TIMESTEP_PARAMS.layer_params[layer]
        logger.info(f"  Layer {layer}: μ={params['mu']}, σ={params['sigma']}")

    layer_records = extractor.extract_layer_activations(
        prompts=prompts,
        layer_indices=layer_indices,
        num_timesteps_per_prompt=config["num_timesteps_per_prompt"],
        tokens_per_timestep=config["tokens_per_timestep"],
        sampling_steps=config["sampling_steps"],
    )

    # 逐层训练
    for layer_idx in layer_indices:
        logger.info("\n" + "=" * 70)
        logger.info(f"Training Layer {layer_idx}")
        logger.info("=" * 70)

        if layer_idx not in layer_records or not layer_records[layer_idx]:
            logger.error(f"No records for layer {layer_idx}, skipping")
            continue

        records = layer_records[layer_idx]

        # 分割训练/验证
        random.shuffle(records)
        split = int(len(records) * 0.9)
        train_records = records[:split]
        val_records = records[split:]

        logger.info(f"  Train records: {len(train_records)}")
        logger.info(f"  Val records: {len(val_records)}")

        # 创建数据加载器
        train_loader = DataLoader(
            ActivationDataset(train_records),
            batch_size=config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            ActivationDataset(val_records),
            batch_size=config["batch_size"],
        )

        # 创建 SAE
        sae = TopKSAE(
            d_model=SAE_PARAMS.d_model,
            d_hidden=SAE_PARAMS.d_hidden,
            top_k=SAE_PARAMS.top_k,
        ).to("cuda")

        # 加载初始化权重
        if init_dir:
            init_path = Path(init_dir) / f"sae_init_layer{layer_idx}.pt"
            if init_path.exists():
                sae = TopKSAE.load_from_checkpoint(str(init_path), device="cuda")
                logger.info(f"  Loaded init weights from {init_path}")

        # 创建训练器 (传入 output_dir 用于可视化)
        trainer = LayerSpecificTrainer(
            layer_idx=layer_idx,
            sae=sae,
            lr=config["lr"],
            min_lr=config["min_lr"],
            warmup_steps=config.get("warmup_steps", 0),
            total_steps=config["steps"],
            grad_clip=config["grad_clip"],
            ema_decay=config["ema_decay"],
            use_ema=config.get("use_ema", True),
            output_dir=run_dir,
        )

        # 训练循环
        layer_start = time.time()
        pbar = tqdm(range(config["steps"]), desc=f"Layer {layer_idx}")

        for step in pbar:
            batch = next(iter(train_loader))
            metrics = trainer.train_step(batch)

            pbar.set_postfix({
                "loss": f"{metrics['mse']:.4f}",
                "lr": f"{metrics['lr']:.2e}",
            })

            # 验证
            if (step + 1) % config["val_interval"] == 0:
                val_metrics = trainer.validate(val_loader)
                logger.info(f"  Step {step+1}: val_mse={val_metrics['val_mse']:.6f}")

            # 保存 checkpoint
            if (step + 1) % config["checkpoint_interval"] == 0:
                ckpt_dir = Path(run_dir) / f"layer{layer_idx}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                trainer.save_checkpoint(str(ckpt_dir / f"sae_step{step+1}.pt"))

        layer_elapsed = time.time() - layer_start

        # ========== 新增：生成可视化图表 ==========
        logger.info(f"  Generating training visualizations...")
        trainer.generate_plots()

        # 保存最终 checkpoint
        ckpt_dir = Path(run_dir) / f"layer{layer_idx}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(ckpt_dir / "sae_latest.pt"))

        # 保存采样信息
        sampling_info = {
            "layer_idx": layer_idx,
            "timestep_params": TIMESTEP_PARAMS.layer_params[layer_idx],
            "n_train_records": len(train_records),
            "n_val_records": len(val_records),
            "n_prompts": len(prompts),
            "steps": config["steps"],
            "final_mse": trainer.best_mse,
            "elapsed_seconds": layer_elapsed,
        }

        with open(ckpt_dir / "sampling_info.json", 'w') as f:
            json.dump(sampling_info, f, indent=2)

        results[layer_idx] = {
            "steps": config["steps"],
            "best_mse": trainer.best_mse,
            "elapsed_seconds": layer_elapsed,
        }

        logger.info(f"  Layer {layer_idx} completed: best_mse={trainer.best_mse:.6f}")

    total_elapsed = time.time() - start_time

    # 保存总结
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_hours": total_elapsed / 3600,
        "layer_timestep_params": TIMESTEP_PARAMS.layer_params,
        "layers": results,
    }

    with open(Path(run_dir) / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nTotal training time: {total_elapsed / 3600:.2f} hours")

    return summary


# ============================================================================
# 主入口
# ============================================================================

def load_stage_config(config_path: str) -> Dict:
    """
    加载阶段配置文件

    支持:
        - config_test.py: 参数测试阶段
        - config_pretrain.py: 预训练阶段
        - config_formal.py: 正式训练阶段

    返回:
        配置字典
    """
    import importlib.util

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # 动态加载模块
    spec = importlib.util.spec_from_file_location("stage_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 调用 get_xxx_config 函数
    if hasattr(module, 'get_test_config'):
        return module.get_test_config()
    elif hasattr(module, 'get_pretrain_config'):
        return module.get_pretrain_config()
    elif hasattr(module, 'get_formal_config'):
        return module.get_formal_config()
    else:
        # 尝试查找任何 get_ 开头的函数
        for name in dir(module):
            if name.startswith('get_') and name.endswith('_config'):
                return getattr(module, name)()

        logger.error(f"No get_xxx_config function found in {config_path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Layer-Specific SAE Training (TODO_list_v4 Compliant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 参数测试阶段 (快速验证)
  python run_train_layer_specific.py --config 紧急/config_test.py

  # 预训练阶段
  python run_train_layer_specific.py --config 紧急/config_pretrain.py

  # 正式训练阶段
  python run_train_layer_specific.py --config 紧急/config_formal.py

  # 命令行覆盖配置文件参数
  python run_train_layer_specific.py --config 紧急/config_test.py --steps 200 --max_prompts 20
  python run_train_layer_specific.py --config 紧急/config_formal.py --tokens_per_timestep 1024 --lr 1e-4
        """,
    )

    # 配置文件
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (config_test.py, config_pretrain.py, config_formal.py)")

    # 模式 (向后兼容，不使用 --config 时使用)
    parser.add_argument("--mode", type=str, default="test",
                        choices=["test", "train"],
                        help="test: 快速验证, train: 完整训练 (仅在不使用 --config 时生效)")

    # ========== 路径参数 ==========
    parser.add_argument("--model_path", type=str, default=None,
                        help="DiT 模型路径")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="提示词文件路径")
    parser.add_argument("--init_dir", type=str, default=None,
                        help="SAE 初始化权重目录")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="输出目录")

    # ========== 训练参数 ==========
    parser.add_argument("--steps", type=int, default=None,
                        help="训练步数")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="预热步数")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率")
    parser.add_argument("--use_ema", action="store_true", default=None,
                        help="启用 EMA")
    parser.add_argument("--no_ema", action="store_true",
                        help="禁用 EMA")

    # ========== 数据/采样参数 ==========
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="最大提示词数量")
    parser.add_argument("--num_timesteps_per_prompt", type=int, default=None,
                        help="每个 prompt 采样的 timestep 数")
    parser.add_argument("--tokens_per_timestep", type=int, default=None,
                        help="每个 timestep 保留的 token 数 (解释见下)")

    # ========== 验证参数 ==========
    parser.add_argument("--val_interval", type=int, default=None,
                        help="验证间隔")
    parser.add_argument("--layers", type=str, default="14,19,24,29",
                        help="训练的层，逗号分隔")

    # ========== 其他 ==========
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    # ========== 加载配置 ==========
    if args.config:
        # 从配置文件加载
        logger.info(f"Loading config from: {args.config}")
        config = load_stage_config(args.config)

        # 确定 stage 名称
        config_name = Path(args.config).stem
        stage_name = {
            "config_test": "参数测试",
            "config_pretrain": "预训练",
            "config_formal": "正式训练",
        }.get(config_name, config_name)
    else:
        # 向后兼容：从默认配置构建
        config = {
            "model_path": PATH_PARAMS.model_path,
            "sae_init_dir": PATH_PARAMS.sae_init_dir,
            "prompt_file": PATH_PARAMS.prompt_file,
            "run_dir": PATH_PARAMS.run_dir,
            "steps": 50 if args.mode == "test" else TRAINING_PARAMS.steps,
            "warmup_steps": TRAINING_PARAMS.warmup_steps,
            "batch_size": TRAINING_PARAMS.batch_size,
            "accum_steps": TRAINING_PARAMS.accum_steps,
            "lr": TRAINING_PARAMS.lr,
            "min_lr": TRAINING_PARAMS.min_lr,
            "grad_clip": TRAINING_PARAMS.grad_clip,
            "ema_decay": TRAINING_PARAMS.ema_decay,
            "use_ema": TRAINING_PARAMS.use_ema,
            "num_timesteps_per_prompt": SAMPLING_PARAMS.num_timesteps_per_prompt,
            "tokens_per_timestep": SAMPLING_PARAMS.tokens_per_timestep,
            "sampling_steps": SAMPLING_PARAMS.sampling_steps,
            "val_interval": VALIDATION_PARAMS.val_interval,
            "checkpoint_interval": VALIDATION_PARAMS.checkpoint_interval,
            "max_prompts": 10 if args.mode == "test" else 80,
            "seed": args.seed,
            "d_model": SAE_PARAMS.d_model,
            "d_hidden": SAE_PARAMS.d_hidden,
            "top_k": SAE_PARAMS.top_k,
        }
        stage_name = "参数测试" if args.mode == "test" else "训练"

    # ========== 命令行覆盖配置文件 ==========
    # 路径参数
    if args.model_path:
        config["model_path"] = args.model_path
    if args.prompt_file:
        config["prompt_file"] = args.prompt_file
    if args.init_dir:
        config["sae_init_dir"] = args.init_dir
    if args.run_dir:
        config["run_dir"] = args.run_dir

    # 训练参数
    if args.steps:
        config["steps"] = args.steps
    if args.warmup_steps is not None:
        config["warmup_steps"] = args.warmup_steps
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["lr"] = args.lr
    if args.use_ema:
        config["use_ema"] = True
    if args.no_ema:
        config["use_ema"] = False

    # 数据/采样参数
    if args.max_prompts:
        config["max_prompts"] = args.max_prompts
    if args.num_timesteps_per_prompt:
        config["num_timesteps_per_prompt"] = args.num_timesteps_per_prompt
    if args.tokens_per_timestep:
        config["tokens_per_timestep"] = args.tokens_per_timestep

    # 验证参数
    if args.val_interval:
        config["val_interval"] = args.val_interval

    # 确保 init_dir 存在
    if "sae_init_dir" not in config:
        config["sae_init_dir"] = PATH_PARAMS.sae_init_dir

    # 解析层
    layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    # ========== 打印配置 ==========
    logger.info("\n" + "=" * 70)
    logger.info(f"Layer-Specific SAE Training - {stage_name}阶段")
    logger.info("=" * 70)
    logger.info(f"  Config: {args.config or '默认配置'}")
    logger.info(f"  Layers: {layer_indices}")
    logger.info("-" * 70)
    logger.info("  Timestep Parameters (Layer-Specific):")
    for layer in layer_indices:
        params = TIMESTEP_PARAMS.layer_params[layer]
        logger.info(f"    Layer {layer}: μ={params['mu']}, σ={params['sigma']}")
    logger.info("-" * 70)
    logger.info("  Training Parameters:")
    logger.info(f"    steps: {config['steps']}")
    logger.info(f"    warmup_steps: {config.get('warmup_steps', 0)}")
    logger.info(f"    batch_size: {config['batch_size']}")
    logger.info(f"    accum_steps: {config['accum_steps']}")
    logger.info(f"    lr: {config['lr']}")
    logger.info(f"    use_ema: {config.get('use_ema', True)}")
    logger.info("-" * 70)
    logger.info("  Data Parameters:")
    logger.info(f"    max_prompts: {config['max_prompts']}")
    logger.info(f"    num_timesteps_per_prompt: {config['num_timesteps_per_prompt']}")
    logger.info(f"    tokens_per_timestep: {config['tokens_per_timestep']}")
    logger.info("-" * 70)
    logger.info("  Paths:")
    logger.info(f"    model: {config['model_path']}")
    logger.info(f"    prompts: {config['prompt_file']}")
    logger.info(f"    output: {config['run_dir']}")
    logger.info("=" * 70)

    # 加载提示词
    prompt_path = Path(config["prompt_file"])
    if not prompt_path.exists():
        logger.error(f"Prompt file not found: {prompt_path}")
        sys.exit(1)

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip() and len(line.strip()) >= 8]

    random.seed(config.get("seed", args.seed))
    random.shuffle(prompts)
    prompts = prompts[:config["max_prompts"]]

    logger.info(f"Loaded {len(prompts)} prompts")

    # 创建输出目录
    Path(config["run_dir"]).mkdir(parents=True, exist_ok=True)

    # 训练
    summary = train_all_layers(
        layer_indices=layer_indices,
        prompts=prompts,
        config=config,
        init_dir=config.get("sae_init_dir"),
        run_dir=config["run_dir"],
    )

    logger.info("\n" + "=" * 70)
    logger.info(f"{stage_name}阶段完成")
    logger.info("=" * 70)

    # 打印输出文件位置
    logger.info("\nOutput files:")
    for layer in layer_indices:
        layer_dir = Path(config["run_dir"]) / f"layer{layer}"
        if layer_dir.exists():
            logger.info(f"  Layer {layer}:")
            logger.info(f"    Checkpoint: {layer_dir / 'sae_latest.pt'}")
            logger.info(f"    Visualizations: {layer_dir}/")
            logger.info(f"      - loss_curve.png")
            logger.info(f"      - sparsity_trend.png")
            logger.info(f"      - dead_neurons.png")
            logger.info(f"      - learning_rate.png")
            logger.info(f"      - dashboard.png")


if __name__ == "__main__":
    main()
